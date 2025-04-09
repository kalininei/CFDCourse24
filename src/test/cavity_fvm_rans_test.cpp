#include "cfd24_test.hpp"
#include "cfd24/grid/vtk.hpp"
#include "cfd24/grid/regular_grid2d.hpp"
#include "cfd24/mat/lodmat.hpp"
#include "cfd24/mat/sparse_matrix_solver.hpp"
#include "cfd24/debug/printer.hpp"
#include "utils/vecmat.hpp"
#include "cfd24/debug/tictoc.hpp"
#include "cfd24/debug/saver.hpp"
#include "cfd24/fvm/fvm_assembler.hpp"
#include "utils/filesystem.hpp"
#include "cfd24/grid/unstructured_grid2d.hpp"
#include <list>

using namespace cfd;

namespace{

double MC(double r){
	//return std::max(0.0, std::min(std::min(2.0*r, (1+r)/2.0), 2.0));
	return std::max(1.0, std::min(0.0, r));
}

double tvd_upwind_weight(
		const std::vector<double>& u,
		const std::vector<Vector>& grad_u,
		Vector cij, size_t i, size_t j){

	// treat boundary: take boundary value
	size_t n_cells = grad_u.size();
	if (i >= n_cells){
		return 1.0;
	} else if (j >= n_cells){
		return 0.0;
	}

	// internals
	double dudc = dot_product(grad_u[i], cij);
	double up = u[j] - 2 * dudc;
	double denum = u[i] - u[j];
	if (denum == 0) denum = 1e-12;
	double r = (u[i] - up) / denum;
	double phi = MC(r);
	return 1 - phi/2;
}


struct Cavity2DFvmSimpleWorker{
	Cavity2DFvmSimpleWorker(const IGrid& grid, double Re, double E);
	void initialize_saver(std::string stem);
	double step();
	void save_current_fields(size_t iter);

	size_t vec_size() const{
		return _collocations.size();
	}
private:
	const IGrid& _grid;
	const double _Re;
	const double _tau;
	const double _alpha_p;
	const FvmExtendedCollocations _collocations;
	const FvmFacesDn _dfdn_computer;
	const FvmCellGradient_Gauss _grad_computer;

	struct BoundaryInfo{
		std::vector<size_t> bnd_colloc;
		std::vector<size_t> bnd_colloc_u0;
		std::vector<size_t> bnd_colloc_u1;
	};
	BoundaryInfo _boundary_info;

	std::vector<double> _p;
	std::vector<double> _u;
	std::vector<double> _v;
	std::vector<double> _un_face;
	std::vector<double> _dpdn_face;
	std::vector<Vector> _grad_p;

	AmgcMatrixSolver _p_stroke_solver;
	AmgcMatrixSolver _u_solver, _v_solver;

	CsrMatrix _mat_u, _mat_v;
	std::vector<double> _rhs_u;
	std::vector<double> _rhs_v;
	double _d;

	std::shared_ptr<VtkUtils::TimeSeriesWriter> _writer;

	double to_next_iteration();
	void gather_boundary_collocations();
	void assemble_p_stroke_solver();
	CsrMatrix assemble_u_lhs(const std::vector<double>& u) const;
	void assemble_uv_slae();

	std::vector<double> compute_u_star();
	std::vector<double> compute_v_star();
	std::vector<double> compute_un_star_face_rhie_chow(const std::vector<double>& u_star, const std::vector<double>& v_star);
	std::vector<double> compute_un_stroke_face(const std::vector<double>& dpstroke_dn);
	std::vector<double> compute_p_stroke(const std::vector<double>& un_star_face);
	std::vector<double> compute_u_stroke(const std::vector<Vector>& grad_p_stroke);
	std::vector<double> compute_v_stroke(const std::vector<Vector>& grad_p_stroke);

	static double compute_tau(const IGrid& grid, double Re, double E);
};

double Cavity2DFvmSimpleWorker::compute_tau(const IGrid& grid, double Re, double E){
	double h2 = grid.cell_volume(0);
	for (size_t i=1; i<grid.n_cells(); ++i){
		h2 = std::max(h2, grid.cell_volume(i));
	}
	return E*Re*h2/4.0;
}

Cavity2DFvmSimpleWorker::Cavity2DFvmSimpleWorker(const IGrid& grid, double Re, double E):
	_grid(grid),
	_Re(Re),
	_tau(compute_tau(grid, Re, E)),
	_alpha_p(1.0/(1.0 + E)),
	_collocations(grid),
	_dfdn_computer(grid, _collocations),
	_grad_computer(grid, _collocations)
{
	_d = 1.0/(1 + E);
	gather_boundary_collocations();
	assemble_p_stroke_solver();

	_u = std::vector<double>(vec_size(), 0);
	_v = std::vector<double>(vec_size(), 0);
	_p = std::vector<double>(vec_size(), 0);
	_dpdn_face = std::vector<double>(_grid.n_faces(), 0);
	_un_face = std::vector<double>(_grid.n_faces(), 0);
	_grad_p = std::vector<Vector>(_grid.n_cells(), {0, 0, 0});
	to_next_iteration();
}

void Cavity2DFvmSimpleWorker::gather_boundary_collocations(){
	BoundaryInfo& bi = _boundary_info;

	std::list<std::pair<size_t, size_t>> colloc_faces;
	for (size_t icolloc: _collocations.face_collocations){
		size_t iface = _collocations.face_index(icolloc);
		bi.bnd_colloc.push_back(icolloc);
		if (std::abs(_grid.face_center(iface).y() - 1) < 1e-6){
			bi.bnd_colloc_u1.push_back(icolloc);
		} else {
			bi.bnd_colloc_u0.push_back(icolloc);
		}
	}
}

void Cavity2DFvmSimpleWorker::initialize_saver(std::string stem){
	_writer.reset(new VtkUtils::TimeSeriesWriter(stem));
};

double Cavity2DFvmSimpleWorker::to_next_iteration(){
	assemble_uv_slae();

	// residual vectors
	std::vector<double> res_u = compute_residual_vec(_mat_u, _rhs_u, _u);
	std::vector<double> res_v = compute_residual_vec(_mat_v, _rhs_v, _v);
	for (size_t icell=0; icell < _grid.n_cells(); ++icell){
		double coef = 1.0 / _tau / _grid.cell_volume(icell);
		res_u[icell] *= coef;
		res_v[icell] *= coef;
	}

	// norm
	double res = 0;
	for (size_t icell=0; icell < _grid.n_cells(); ++icell){
		res = std::max(res, std::max(res_u[icell], res_v[icell]));
	}
	return res;
};

double Cavity2DFvmSimpleWorker::step(){
	// Predictor step: U-star
	std::vector<double> u_star = compute_u_star();
	std::vector<double> v_star = compute_v_star();
	std::vector<double> un_star_face = compute_un_star_face_rhie_chow(u_star, v_star);
	// Pressure correction
	std::vector<double> p_stroke = compute_p_stroke(un_star_face);
	std::vector<Vector> grad_p_stroke = _grad_computer.compute(p_stroke);
	std::vector<double> dpstroke_dn_face = _dfdn_computer.compute(p_stroke);
	// Velocity correction
	std::vector<double> u_stroke = compute_u_stroke(grad_p_stroke);
	std::vector<double> v_stroke = compute_v_stroke(grad_p_stroke);
	std::vector<double> un_stroke_face = compute_un_stroke_face(dpstroke_dn_face);

	// Set final values
	_u = vector_sum(u_star, 1.0, u_stroke);
	_v= vector_sum(v_star, 1.0, v_stroke);
	_un_face = vector_sum(un_star_face, 1.0, un_stroke_face);
	_p = vector_sum(_p, _alpha_p, p_stroke);
	_grad_p = vector_sum(_grad_p, _alpha_p, grad_p_stroke);
	_dpdn_face = vector_sum(_dpdn_face, _alpha_p, dpstroke_dn_face);

	return to_next_iteration();
}

void Cavity2DFvmSimpleWorker::save_current_fields(size_t iter){
	if (_writer){
		std::string filepath = _writer->add(iter);
		_grid.save_vtk(filepath);
		VtkUtils::add_cell_data(_p, "pressure", filepath, _grid.n_cells());
		VtkUtils::add_cell_vector(_u, _v, "velocity", filepath, _grid.n_cells());
	}
}

void Cavity2DFvmSimpleWorker::assemble_p_stroke_solver(){
	LodMatrix mat(vec_size());
	for (size_t iface = 0; iface < _grid.n_faces(); ++iface){
		size_t negative_colloc = _collocations.tab_face_colloc(iface)[0];
		size_t positive_colloc = _collocations.tab_face_colloc(iface)[1];
		double area = _grid.face_area(iface);

		for (const std::pair<const size_t, double>& iter: _dfdn_computer.linear_combination(iface)){
			size_t column = iter.first;
			double coef = -_d * area * iter.second;
			mat.add_value(negative_colloc, column, coef);
			mat.add_value(positive_colloc, column, -coef);
		}
	}
	mat.set_unit_row(0);
	_p_stroke_solver.set_matrix(mat.to_csr());
}

CsrMatrix Cavity2DFvmSimpleWorker::assemble_u_lhs(const std::vector<double>& u) const{
	std::vector<Vector> grad_u = _grad_computer.compute(u);

	// =============== Left side
	LodMatrix mat(vec_size());
	// u
	for (size_t icell = 0; icell < _grid.n_cells(); ++icell){
		mat.add_value(icell, icell, _grid.cell_volume(icell));
	}
	for (size_t iface = 0; iface < _grid.n_faces(); ++iface){
		size_t negative_colloc = _collocations.tab_face_colloc(iface)[0];
		size_t positive_colloc = _collocations.tab_face_colloc(iface)[1];
		double area = _grid.face_area(iface);
		double un = _un_face[iface];

		// - tau/Re * Laplace(u)
		for (const std::pair<const size_t, double>& iter: _dfdn_computer.linear_combination(iface)){
			size_t column = iter.first;
			double coef = -_tau/_Re * area * iter.second;
			mat.add_value(negative_colloc, column, coef);
			mat.add_value(positive_colloc, column, -coef);
		}
		
		// + nonlinear tvd convection
		{
			size_t i = negative_colloc; // upwind cell
			size_t j = positive_colloc; // downwind cell
			double coef = _tau * area * un;
			if (un < 0){
				std::swap(i, j);
				coef *= -1;
			};
			Vector cij = _collocations.points[j] - _collocations.points[i];

			double wi = tvd_upwind_weight(u, grad_u, cij, i, j);
			double wj = 1 - wi;
			mat.add_value(i, i,  wi*coef);
			mat.add_value(i, j,  wj*coef);
			mat.add_value(j, j, -wj*coef);
			mat.add_value(j, i, -wi*coef);
		}
	}
	// bnd
	for (size_t icolloc: _boundary_info.bnd_colloc){
		mat.set_unit_row(icolloc);
	}

	return mat.to_csr();
};

void Cavity2DFvmSimpleWorker::assemble_uv_slae(){
	_mat_u = assemble_u_lhs(_u);
	_mat_v = assemble_u_lhs(_v);

	_u_solver.set_matrix(_mat_u);
	_v_solver.set_matrix(_mat_v);

	// ============== right side
	_rhs_u.resize(vec_size());
	_rhs_v.resize(vec_size());
	for (size_t icell = 0; icell < _grid.n_cells(); ++icell){
		_rhs_u[icell] = (_u[icell] -_tau * _grad_p[icell].x()) * _grid.cell_volume(icell);
		_rhs_v[icell] = (_v[icell] -_tau * _grad_p[icell].y()) * _grid.cell_volume(icell);
	}
	// bnd
	for (size_t icolloc: _boundary_info.bnd_colloc_u0){
		_rhs_u[icolloc] = 0;
		_rhs_v[icolloc] = 0;
	}
	for (size_t icolloc: _boundary_info.bnd_colloc_u1){
		_rhs_u[icolloc] = 1;
		_rhs_v[icolloc] = 0;
	}
}

std::vector<double> Cavity2DFvmSimpleWorker::compute_u_star(){
	std::vector<double> u_star(_u);
	_u_solver.solve(_rhs_u, u_star);
	return u_star;
}

std::vector<double> Cavity2DFvmSimpleWorker::compute_v_star(){
	std::vector<double> v_star(_v);
	_v_solver.solve(_rhs_v, v_star);
	return v_star;
}

std::vector<double> Cavity2DFvmSimpleWorker::compute_un_star_face_rhie_chow(
		const std::vector<double>& u_star,
		const std::vector<double>& v_star){

	std::vector<double> ret(_grid.n_faces());

	for (size_t iface=0; iface<_grid.n_faces(); ++iface){
		size_t ci = _grid.tab_face_cell(iface)[0];
		size_t cj = _grid.tab_face_cell(iface)[1];
		if (ci == INVALID_INDEX || cj == INVALID_INDEX){
			ret[iface] = 0;
		} else {
			// Rhie-Chow interpolation
			Vector normal = _grid.face_normal(iface);
			Vector uvec_i = Vector(u_star[ci], v_star[ci]);
			Vector uvec_j = Vector(u_star[cj], v_star[cj]);
			
			double ustar_i = dot_product(uvec_i, normal);
			double ustar_j = dot_product(uvec_j, normal);
			double dpdn_i = dot_product(_grad_p[ci], normal);
			double dpdn_j = dot_product(_grad_p[cj], normal);
			double dpdn_ij = _dpdn_face[iface];

			ret[iface] = 0.5*(ustar_i + ustar_j)
			           + 0.5*_tau*(dpdn_i + dpdn_j - 2*dpdn_ij);
		}
	}

	return ret;
}

std::vector<double> Cavity2DFvmSimpleWorker::compute_un_stroke_face(const std::vector<double>& dpstroke_dn){
	std::vector<double> un(_grid.n_faces());
	for (size_t iface=0; iface<_grid.n_faces(); ++iface){
		un[iface] = -_tau * _d * dpstroke_dn[iface];
	}
	return un;
}

std::vector<double> Cavity2DFvmSimpleWorker::compute_p_stroke(const std::vector<double>& un_star_face){
	std::vector<double> rhs(vec_size(), 0.0);
	// internal
	for (size_t iface = 0; iface < _grid.n_faces(); ++iface){
		double coef = -_grid.face_area(iface) / _tau * un_star_face[iface];
		size_t neg = _grid.tab_face_cell(iface)[0];
		size_t pos = _grid.tab_face_cell(iface)[1];
		if (neg != INVALID_INDEX) rhs[neg] += coef;
		if (pos != INVALID_INDEX) rhs[pos] -= coef;
	}
	rhs[0] = 0;
	// solve
	std::vector<double> p_stroke;
	_p_stroke_solver.solve(rhs, p_stroke);
	return p_stroke;
}

std::vector<double> Cavity2DFvmSimpleWorker::compute_u_stroke(const std::vector<Vector>& grad_p_stroke){
	std::vector<double> u_stroke(vec_size(), 0.0);
	for (size_t i=0; i<_grid.n_cells(); ++i){
		u_stroke[i] = -_tau * _d * grad_p_stroke[i].x();
	}
	return u_stroke;
}

std::vector<double> Cavity2DFvmSimpleWorker::compute_v_stroke(const std::vector<Vector>& grad_p_stroke){
	std::vector<double> v_stroke(vec_size(), 0.0);
	for (size_t i=0; i<_grid.n_cells(); ++i){
		v_stroke[i] = -_tau * _d * grad_p_stroke[i].y();
	}
	return v_stroke;
}

}

TEST_CASE("Cavity 2D, FVM-SIMPLE-Laminar algorithm", "[cavity-fvm-rans-laminar]"){
	std::cout << std::endl << "--- cfd24_test [cavity-fvm-rans-laminar] --- " << std::endl;

	// problem parameters
	double Re = 50'000;
	size_t max_it = 10'000;
	double eps = 1e-2;
	double E = 0.1;

	// worker initialization
	RegularGrid2D grid(0, 1, 0, 1, 30, 30);
	Cavity2DFvmSimpleWorker worker(grid, Re, E);
	worker.initialize_saver("cavity-fvm-rans-laminar");

	// iterations loop
	size_t it = 0;
	for (it=1; it < max_it; ++it){
		double nrm = worker.step();

		// print norm and friction force 
		std::cout << it << " " << nrm << std::endl;

		// export solution to vtk
		worker.save_current_fields(it);

		// break if residual is low enough
		if (nrm < eps){
			break;
		}
	}
	CHECK(it == 26);
}


///////////////////////////////////////////////////////////////////////
// K-Epsilon
///////////////////////////////////////////////////////////////////////

namespace{

struct Cavity2DFvmSimpleWorker_KEps{
	Cavity2DFvmSimpleWorker_KEps(const IGrid& grid, double Re, double E);
	void initialize_saver(std::string stem);
	double step();
	void save_current_fields(size_t iter);

	size_t vec_size() const{
		return _collocations.size();
	}
private:
	const IGrid& _grid;
	const double _Re;
	const double _tau;
	const double _alpha_p;
	const FvmExtendedCollocations _collocations;
	const FvmFacesDn _dfdn_computer;
	const FvmCellGradient_Gauss _grad_computer;

	const double _Cmu = 0.09;
	const double _sigma_k = 1.0;
	const double _sigma_e = 1.3;
	const double _C1 = 1.44;
	const double _C2 = 1.92;
	const double delta_nuT = 0.01;   // nuT >= delta_nuT * 1/Re

	struct BoundaryInfo{
		std::vector<size_t> bnd_colloc;
		std::vector<size_t> bnd_colloc_u0;
		std::vector<size_t> bnd_colloc_u1;
		std::map<size_t, std::vector<size_t>> bnd_cell_to_colloc;
	};
	BoundaryInfo _boundary_info;

	std::vector<double> _p;
	std::vector<double> _u;
	std::vector<double> _v;
	std::vector<double> _un_face;
	std::vector<double> _dpdn_face;
	std::vector<Vector> _grad_p;
	std::vector<double> _k;
	std::vector<double> _epsilon;
	std::vector<double> _nuT;

	AmgcMatrixSolver _p_stroke_solver;
	AmgcMatrixSolver _u_solver, _v_solver;

	CsrMatrix _mat_u, _mat_v;
	std::vector<double> _rhs_u;
	std::vector<double> _rhs_v;
	double _d;

	std::shared_ptr<VtkUtils::TimeSeriesWriter> _writer;

	double to_next_iteration();
	void gather_boundary_collocations();
	void assemble_p_stroke_solver();
	CsrMatrix assemble_convdiff_lhs(
			const std::vector<double>& u,
			const std::vector<double>& nu_face,
			const std::vector<double>& diag={}) const;
	void assemble_uv_slae();

	std::vector<double> compute_u_star();
	std::vector<double> compute_v_star();
	std::vector<double> compute_un_star_face_rhie_chow(const std::vector<double>& u_star, const std::vector<double>& v_star);
	std::vector<double> compute_un_stroke_face(const std::vector<double>& dpstroke_dn);
	std::vector<double> compute_p_stroke(const std::vector<double>& un_star_face);
	std::vector<double> compute_u_stroke(const std::vector<Vector>& grad_p_stroke);
	std::vector<double> compute_v_stroke(const std::vector<Vector>& grad_p_stroke);
	std::vector<double> compute_nueff_face(double sigma=1.0) const;
	std::vector<double> compute_G() const;
	std::vector<double> compute_gamma() const;
	std::vector<double> compute_nuT() const;
	std::vector<double> compute_epsilon(const std::vector<double>& G, const std::vector<double>& gamma) const;
	std::vector<double> compute_k(const std::vector<double>& G, const std::vector<double>& gamma) const;

	static double compute_tau(const IGrid& grid, double Re, double E);
};

double Cavity2DFvmSimpleWorker_KEps::compute_tau(const IGrid& grid, double Re, double E){
	double h2 = grid.cell_volume(0);
	for (size_t i=1; i<grid.n_cells(); ++i){
		h2 = std::max(h2, grid.cell_volume(i));
	}
	return E*Re*h2/4.0;
}

Cavity2DFvmSimpleWorker_KEps::Cavity2DFvmSimpleWorker_KEps(const IGrid& grid, double Re, double E):
	_grid(grid),
	_Re(Re),
	_tau(compute_tau(grid, Re, E)),
	_alpha_p(1.0/(1.0 + E)),
	_collocations(grid),
	_dfdn_computer(grid, _collocations),
	_grad_computer(grid, _collocations)
{
	_d = 1.0/(1 + E);
	gather_boundary_collocations();
	assemble_p_stroke_solver();

	_u = std::vector<double>(vec_size(), 0);
	_v = std::vector<double>(vec_size(), 0);
	_p = std::vector<double>(vec_size(), 0);
	_k = std::vector<double>(vec_size(), 0);
	_epsilon = std::vector<double>(vec_size(), 0);
	_nuT = std::vector<double>(vec_size(), 0);

	_dpdn_face = std::vector<double>(_grid.n_faces(), 0);
	_un_face = std::vector<double>(_grid.n_faces(), 0);
	_grad_p = std::vector<Vector>(_grid.n_cells(), {0, 0, 0});
	to_next_iteration();
}

void Cavity2DFvmSimpleWorker_KEps::gather_boundary_collocations(){
	BoundaryInfo& bi = _boundary_info;

	std::list<std::pair<size_t, size_t>> colloc_faces;
	for (size_t icolloc: _collocations.face_collocations){
		size_t iface = _collocations.face_index(icolloc);
		bi.bnd_colloc.push_back(icolloc);
		if (std::abs(_grid.face_center(iface).y() - 1) < 1e-6){
			bi.bnd_colloc_u1.push_back(icolloc);
		} else {
			bi.bnd_colloc_u0.push_back(icolloc);
		}
	}

	for (size_t icolloc: bi.bnd_colloc){
		size_t iface = _collocations.face_index(icolloc);
		size_t icell = _grid.tab_face_cell(iface)[0];
		if (icell == INVALID_INDEX) icell = _grid.tab_face_cell(iface)[1];
		auto fnd = bi.bnd_cell_to_colloc.find(icell);
		if (fnd == bi.bnd_cell_to_colloc.end()){
			bi.bnd_cell_to_colloc[icell] = {};
		}
		bi.bnd_cell_to_colloc[icell].push_back(icolloc);
	}
}

void Cavity2DFvmSimpleWorker_KEps::initialize_saver(std::string stem){
	_writer.reset(new VtkUtils::TimeSeriesWriter(stem));
};

double Cavity2DFvmSimpleWorker_KEps::to_next_iteration(){
	assemble_uv_slae();

	// residual vectors
	std::vector<double> res_u = compute_residual_vec(_mat_u, _rhs_u, _u);
	std::vector<double> res_v = compute_residual_vec(_mat_v, _rhs_v, _v);
	for (size_t icell=0; icell < _grid.n_cells(); ++icell){
		double coef = 1.0 / _tau / _grid.cell_volume(icell);
		res_u[icell] *= coef;
		res_v[icell] *= coef;
	}

	// norm
	double res = 0;
	for (size_t icell=0; icell < _grid.n_cells(); ++icell){
		res = std::max(res, std::max(res_u[icell], res_v[icell]));
	}
	return res;
};

double Cavity2DFvmSimpleWorker_KEps::step(){
	// Predictor step: U-star
	std::vector<double> u_star = compute_u_star();
	std::vector<double> v_star = compute_v_star();
	std::vector<double> un_star_face = compute_un_star_face_rhie_chow(u_star, v_star);
	// Pressure correction
	std::vector<double> p_stroke = compute_p_stroke(un_star_face);
	std::vector<Vector> grad_p_stroke = _grad_computer.compute(p_stroke);
	std::vector<double> dpstroke_dn_face = _dfdn_computer.compute(p_stroke);
	// Velocity correction
	std::vector<double> u_stroke = compute_u_stroke(grad_p_stroke);
	std::vector<double> v_stroke = compute_v_stroke(grad_p_stroke);
	std::vector<double> un_stroke_face = compute_un_stroke_face(dpstroke_dn_face);
	// Set final values
	_u = vector_sum(u_star, 1.0, u_stroke);
	_v = vector_sum(v_star, 1.0, v_stroke);
	_un_face = vector_sum(un_star_face, 1.0, un_stroke_face);
	_p = vector_sum(_p, _alpha_p, p_stroke);
	_grad_p = vector_sum(_grad_p, _alpha_p, grad_p_stroke);
	_dpdn_face = vector_sum(_dpdn_face, _alpha_p, dpstroke_dn_face);
	// Generation
	std::vector<double> G = compute_G();
	// gamma
	std::vector<double> gamma = compute_gamma();
	// epsilon
	_epsilon = compute_epsilon(G, gamma);
	// k
	_k = compute_k(G, gamma);
	// turbulent viscosity
	_nuT = compute_nuT();

	return to_next_iteration();
}

void Cavity2DFvmSimpleWorker_KEps::save_current_fields(size_t iter){
	if (_writer){
		std::string filepath = _writer->add(iter);
		_grid.save_vtk(filepath);
		VtkUtils::add_cell_data(_p, "pressure", filepath, _grid.n_cells());
		VtkUtils::add_cell_vector(_u, _v, "velocity", filepath, _grid.n_cells());
		VtkUtils::add_cell_data(_k, "k", filepath, _grid.n_cells());
		VtkUtils::add_cell_data(_epsilon, "epsilon", filepath, _grid.n_cells());
	}
}

void Cavity2DFvmSimpleWorker_KEps::assemble_p_stroke_solver(){
	LodMatrix mat(vec_size());
	for (size_t iface = 0; iface < _grid.n_faces(); ++iface){
		size_t negative_colloc = _collocations.tab_face_colloc(iface)[0];
		size_t positive_colloc = _collocations.tab_face_colloc(iface)[1];
		double area = _grid.face_area(iface);

		for (const std::pair<const size_t, double>& iter: _dfdn_computer.linear_combination(iface)){
			size_t column = iter.first;
			double coef = -_d * area * iter.second;
			mat.add_value(negative_colloc, column, coef);
			mat.add_value(positive_colloc, column, -coef);
		}
	}
	mat.set_unit_row(0);
	_p_stroke_solver.set_matrix(mat.to_csr());
}

CsrMatrix Cavity2DFvmSimpleWorker_KEps::assemble_convdiff_lhs(
		const std::vector<double>& u,
		const std::vector<double>& nu_face,
		const std::vector<double>& diag) const{
	std::vector<Vector> grad_u = _grad_computer.compute(u);

	// =============== Left side
	LodMatrix mat(vec_size());
	// u
	for (size_t icell = 0; icell < _grid.n_cells(); ++icell){
		mat.add_value(icell, icell, _grid.cell_volume(icell));
	}
	for (size_t iface = 0; iface < _grid.n_faces(); ++iface){
		size_t negative_colloc = _collocations.tab_face_colloc(iface)[0];
		size_t positive_colloc = _collocations.tab_face_colloc(iface)[1];
		double area = _grid.face_area(iface);
		double un = _un_face[iface];

		// - tau * nu * Laplace(u)
		for (const std::pair<const size_t, double>& iter: _dfdn_computer.linear_combination(iface)){
			size_t column = iter.first;
			double coef = -_tau * nu_face[iface] * area * iter.second;
			mat.add_value(negative_colloc, column, coef);
			mat.add_value(positive_colloc, column, -coef);
		}
		
		// + nonlinear tvd convection
		{
			size_t i = negative_colloc; // upwind cell
			size_t j = positive_colloc; // downwind cell
			double coef = _tau * area * un;
			if (un < 0){
				std::swap(i, j);
				coef *= -1;
			};
			Vector cij = _collocations.points[j] - _collocations.points[i];

			double wi = tvd_upwind_weight(u, grad_u, cij, i, j);
			double wj = 1 - wi;
			mat.add_value(i, i,  wi*coef);
			mat.add_value(i, j,  wj*coef);
			mat.add_value(j, j, -wj*coef);
			mat.add_value(j, i, -wi*coef);
		}
	}
	// + diag
	if (diag.size() > 0)
	for (size_t icell = 0; icell < _grid.n_cells(); ++icell){
		mat.add_value(icell, icell, diag[icell] * _grid.cell_volume(icell));
	}

	return mat.to_csr();
};

void Cavity2DFvmSimpleWorker_KEps::assemble_uv_slae(){
	std::vector<double> nu = compute_nueff_face();
	_mat_u = assemble_convdiff_lhs(_u, nu);
	_mat_v = assemble_convdiff_lhs(_v, nu);
	
	// bnd
	for (size_t icolloc: _boundary_info.bnd_colloc){
		_mat_u.set_unit_row(icolloc);
		_mat_v.set_unit_row(icolloc);
	}

	_u_solver.set_matrix(_mat_u);
	_v_solver.set_matrix(_mat_v);

	// ============== right side
	_rhs_u.resize(vec_size());
	_rhs_v.resize(vec_size());
	for (size_t icell = 0; icell < _grid.n_cells(); ++icell){
		_rhs_u[icell] = (_u[icell] -_tau * _grad_p[icell].x()) * _grid.cell_volume(icell);
		_rhs_v[icell] = (_v[icell] -_tau * _grad_p[icell].y()) * _grid.cell_volume(icell);
	}
	// bnd
	for (size_t icolloc: _boundary_info.bnd_colloc_u0){
		_rhs_u[icolloc] = 0;
		_rhs_v[icolloc] = 0;
	}
	for (size_t icolloc: _boundary_info.bnd_colloc_u1){
		_rhs_u[icolloc] = 1;
		_rhs_v[icolloc] = 0;
	}
}


std::vector<double> Cavity2DFvmSimpleWorker_KEps::compute_u_star(){
	std::vector<double> u_star(_u);
	_u_solver.solve(_rhs_u, u_star);
	return u_star;
}

std::vector<double> Cavity2DFvmSimpleWorker_KEps::compute_v_star(){
	std::vector<double> v_star(_v);
	_v_solver.solve(_rhs_v, v_star);
	return v_star;
}

std::vector<double> Cavity2DFvmSimpleWorker_KEps::compute_un_star_face_rhie_chow(
		const std::vector<double>& u_star,
		const std::vector<double>& v_star){

	std::vector<double> ret(_grid.n_faces());

	for (size_t iface=0; iface<_grid.n_faces(); ++iface){
		size_t ci = _grid.tab_face_cell(iface)[0];
		size_t cj = _grid.tab_face_cell(iface)[1];
		if (ci == INVALID_INDEX || cj == INVALID_INDEX){
			ret[iface] = 0;
		} else {
			// Rhie-Chow interpolation
			Vector normal = _grid.face_normal(iface);
			Vector uvec_i = Vector(u_star[ci], v_star[ci]);
			Vector uvec_j = Vector(u_star[cj], v_star[cj]);
			
			double ustar_i = dot_product(uvec_i, normal);
			double ustar_j = dot_product(uvec_j, normal);
			double dpdn_i = dot_product(_grad_p[ci], normal);
			double dpdn_j = dot_product(_grad_p[cj], normal);
			double dpdn_ij = _dpdn_face[iface];

			ret[iface] = 0.5*(ustar_i + ustar_j)
			           + 0.5*_tau*(dpdn_i + dpdn_j - 2*dpdn_ij);
		}
	}

	return ret;
}

std::vector<double> Cavity2DFvmSimpleWorker_KEps::compute_un_stroke_face(const std::vector<double>& dpstroke_dn){
	std::vector<double> un(_grid.n_faces());
	for (size_t iface=0; iface<_grid.n_faces(); ++iface){
		un[iface] = -_tau * _d * dpstroke_dn[iface];
	}
	return un;
}

std::vector<double> Cavity2DFvmSimpleWorker_KEps::compute_p_stroke(const std::vector<double>& un_star_face){
	std::vector<double> rhs(vec_size(), 0.0);
	// internal
	for (size_t iface = 0; iface < _grid.n_faces(); ++iface){
		double coef = -_grid.face_area(iface) / _tau * un_star_face[iface];
		size_t neg = _grid.tab_face_cell(iface)[0];
		size_t pos = _grid.tab_face_cell(iface)[1];
		if (neg != INVALID_INDEX) rhs[neg] += coef;
		if (pos != INVALID_INDEX) rhs[pos] -= coef;
	}
	rhs[0] = 0;
	// solve
	std::vector<double> p_stroke;
	_p_stroke_solver.solve(rhs, p_stroke);
	return p_stroke;
}

std::vector<double> Cavity2DFvmSimpleWorker_KEps::compute_u_stroke(const std::vector<Vector>& grad_p_stroke){
	std::vector<double> u_stroke(vec_size(), 0.0);
	for (size_t i=0; i<_grid.n_cells(); ++i){
		u_stroke[i] = -_tau * _d * grad_p_stroke[i].x();
	}
	return u_stroke;
}

std::vector<double> Cavity2DFvmSimpleWorker_KEps::compute_v_stroke(const std::vector<Vector>& grad_p_stroke){
	std::vector<double> v_stroke(vec_size(), 0.0);
	for (size_t i=0; i<_grid.n_cells(); ++i){
		v_stroke[i] = -_tau * _d * grad_p_stroke[i].y();
	}
	return v_stroke;
}

std::vector<double> Cavity2DFvmSimpleWorker_KEps::compute_nueff_face(double sigma) const{
	// TODO: nueff_face = 1/Re  + nuT_face / sigma
	return std::vector<double>(_grid.n_faces(), 1.0/_Re);
}

std::vector<double> Cavity2DFvmSimpleWorker_KEps::compute_G() const{
	// TODO
	return std::vector<double>(vec_size(), 0.0);
};

std::vector<double> Cavity2DFvmSimpleWorker_KEps::compute_gamma() const{
	// TODO
	return std::vector<double>(vec_size(), 0.0);
}

std::vector<double> Cavity2DFvmSimpleWorker_KEps::compute_nuT() const{
	// TODO
	return std::vector<double>(vec_size(), 0.0);
}

std::vector<double> Cavity2DFvmSimpleWorker_KEps::compute_k(const std::vector<double>& G, const std::vector<double>& gamma) const{
	// left side
	std::vector<double> nu = compute_nueff_face(_sigma_k);
	CsrMatrix mat = assemble_convdiff_lhs(_k, nu, gamma);
	
	//  right side
	std::vector<double> rhs(vec_size(), 0.0);
	for (size_t icell = 0; icell < _grid.n_cells(); ++icell){
		rhs[icell] = (_k[icell] +_tau * G[icell]) * _grid.cell_volume(icell);
	}

	// solve
	std::vector<double> new_k(_k);
	AmgcMatrixSolver::solve_slae(mat, rhs, new_k);
	return new_k;
}

std::vector<double> Cavity2DFvmSimpleWorker_KEps::compute_epsilon(const std::vector<double>& G, const std::vector<double>& gamma) const{
	std::vector<double> nu = compute_nueff_face(_sigma_e);
	CsrMatrix mat = assemble_convdiff_lhs(_epsilon, nu, std::vector<double>(vec_size(), _C2));
	
	// ============== right side
	std::vector<double> rhs(vec_size(), 0.0);
	for (size_t icell = 0; icell < _grid.n_cells(); ++icell){
		double v = _C1 * G[icell] * gamma[icell];
		rhs[icell] = (_epsilon[icell] +_tau * v) * _grid.cell_volume(icell);
	}

	//boundary condition
	for (const auto& it: _boundary_info.bnd_cell_to_colloc){
		size_t icell = it.first;
		rhs[icell] = 0;
		mat.set_unit_row(icell);

		double w = 1.0 / it.second.size();
		
		for (size_t icolloc: it.second){
			//size_t iface = _collocations.face_index(icolloc);
			double eps_i = 0; // TODO
			// on boundary
			mat.set_unit_row(icolloc);
			rhs[icolloc] = eps_i;
			// on near-boundary cell
			rhs[icell] = w * eps_i;
		}
	}
	
	// solve
	std::vector<double> new_e(_epsilon);
	AmgcMatrixSolver::solve_slae(mat, rhs, new_e);
	return new_e;
}

}

TEST_CASE("Cavity 2D with k-e model, FVM-SIMPLE-Laminar algorithm", "[cavity-fvm-rans-ke]"){
	std::cout << std::endl << "--- cfd24_test [cavity-fvm-rans-laminar-ke] --- " << std::endl;

	// problem parameters
	double Re = 50'000;
	size_t max_it = 10'000;
	double eps = 1e-2;
	double E = 0.1;

	// worker initialization
	RegularGrid2D grid(0, 1, 0, 1, 30, 30);
	Cavity2DFvmSimpleWorker_KEps worker(grid, Re, E);
	worker.initialize_saver("cavity-fvm-rans-ke");

	// iterations loop
	size_t it = 0;
	for (it=1; it < max_it; ++it){
		double nrm = worker.step();

		// print norm and friction force 
		std::cout << it << " " << nrm << std::endl;

		// export solution to vtk
		worker.save_current_fields(it);

		// break if residual is low enough
		if (nrm < eps){
			break;
		}
	}
	CHECK(it == 26);
}
