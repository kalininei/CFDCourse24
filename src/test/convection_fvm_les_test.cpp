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
	return std::max(0.0, std::min(std::min(2.0*r, (1+r)/2.0), 2.0));
	//return std::min(1.0, std::max(0.0, r));
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


struct ConvectionFvmWorker{
	ConvectionFvmWorker(const IGrid& grid, double Re, double Pe, double E);
	void initialize_saver(std::string stem);
	double step();
	void save_current_fields(size_t iter);

	size_t vec_size() const{
		return _collocations.size();
	}
private:
	const IGrid& _grid;
	const double _Re;
	const double _Pe;
	const double _tau;
	const double _alpha_p;
	const FvmExtendedCollocations _collocations;
	const FvmFacesDn _dfdn_computer;
	const FvmCellGradient_Gauss _grad_computer;

	struct BoundaryInfo{
		std::vector<size_t> bnd_colloc;
		std::vector<size_t> bnd_colloc_t01;
		std::vector<size_t> bnd_colloc_t0;
		std::vector<size_t> bnd_colloc_t1;
	};
	BoundaryInfo _boundary_info;

	std::vector<double> _p;
	std::vector<double> _u;
	std::vector<double> _v;
	std::vector<double> _t;
	std::vector<double> _un_face;
	std::vector<double> _dpdn_face;
	std::vector<Vector> _grad_p;
	std::vector<double> _nu_velocity, _nu_temperature;

	AmgcMatrixSolver _p_stroke_solver;
	AmgcMatrixSolver _u_solver, _v_solver, _t_solver;

	CsrMatrix _mat_u, _mat_v, _mat_t;
	std::vector<double> _rhs_u;
	std::vector<double> _rhs_v;
	std::vector<double> _rhs_t;
	double _d;

	std::shared_ptr<VtkUtils::TimeSeriesWriter> _writer;

	double to_next_iteration();
	void gather_boundary_collocations();
	void assemble_p_stroke_solver();
	CsrMatrix assemble_u_lhs(const std::vector<double>& nu, const std::vector<double>& u, const std::vector<size_t>& dir_colloc) const;
	void assemble_uvt_slae();
	void assemble_nu();

	std::vector<double> compute_u_star();
	std::vector<double> compute_v_star();
	std::vector<double> compute_temperature();
	std::vector<double> compute_un_star_face_rhie_chow(const std::vector<double>& u_star, const std::vector<double>& v_star);
	std::vector<double> compute_un_stroke_face(const std::vector<double>& dpstroke_dn);
	std::vector<double> compute_p_stroke(const std::vector<double>& un_star_face);
	std::vector<double> compute_u_stroke(const std::vector<Vector>& grad_p_stroke);
	std::vector<double> compute_v_stroke(const std::vector<Vector>& grad_p_stroke);

	static double compute_tau(const IGrid& grid, double Re, double E);
};

double ConvectionFvmWorker::compute_tau(const IGrid& grid, double Re, double E){
	double h2 = grid.cell_volume(0);
	for (size_t i=1; i<grid.n_cells(); ++i){
		h2 = std::max(h2, grid.cell_volume(i));
	}
	return E*Re*h2/4.0;
}

ConvectionFvmWorker::ConvectionFvmWorker(const IGrid& grid, double Re, double Pe, double E):
	_grid(grid),
	_Re(Re),
	_Pe(Pe),
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
	_t = std::vector<double>(vec_size(), 0);
	_dpdn_face = std::vector<double>(_grid.n_faces(), 0);
	_un_face = std::vector<double>(_grid.n_faces(), 0);
	_grad_p = std::vector<Vector>(_grid.n_cells(), {0, 0, 0});
	to_next_iteration();
}

void ConvectionFvmWorker::gather_boundary_collocations(){
	BoundaryInfo& bi = _boundary_info;
	std::list<std::pair<size_t, size_t>> colloc_faces;
	for (size_t icolloc: _collocations.face_collocations){
		size_t iface = _collocations.face_index(icolloc);
		bi.bnd_colloc.push_back(icolloc);
		if (std::abs(_grid.face_center(iface).y() - 1) < 1e-6){
			// upper: t=0
			bi.bnd_colloc_t0.push_back(icolloc);
			bi.bnd_colloc_t01.push_back(icolloc);
		} else if (std::abs(_grid.face_center(iface).y()) < 1e-6){
			// lower: t=1
			bi.bnd_colloc_t1.push_back(icolloc);
			bi.bnd_colloc_t01.push_back(icolloc);
		}
	}
}

void ConvectionFvmWorker::initialize_saver(std::string stem){
	_writer.reset(new VtkUtils::TimeSeriesWriter(stem));
};

double ConvectionFvmWorker::to_next_iteration(){
	assemble_nu();
	assemble_uvt_slae();

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

double ConvectionFvmWorker::step(){
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
	// Temperature solution
	std::vector<double> t = compute_temperature();

	// Set final values
	_u = vector_sum(u_star, 1.0, u_stroke);
	_v= vector_sum(v_star, 1.0, v_stroke);
	_un_face = vector_sum(un_star_face, 1.0, un_stroke_face);
	_p = vector_sum(_p, _alpha_p, p_stroke);
	_grad_p = vector_sum(_grad_p, _alpha_p, grad_p_stroke);
	_dpdn_face = vector_sum(_dpdn_face, _alpha_p, dpstroke_dn_face);
	_t = t;

	return to_next_iteration();
}

void ConvectionFvmWorker::save_current_fields(size_t iter){
	if (_writer){
		std::string filepath = _writer->add(iter);
		_grid.save_vtk(filepath);
		VtkUtils::add_cell_data(_p, "pressure", filepath, _grid.n_cells());
		VtkUtils::add_cell_data(_t, "temperature", filepath, _grid.n_cells());
		VtkUtils::add_cell_vector(_u, _v, "velocity", filepath, _grid.n_cells());
	}
}

void ConvectionFvmWorker::assemble_p_stroke_solver(){
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

CsrMatrix ConvectionFvmWorker::assemble_u_lhs(const std::vector<double>& nu, const std::vector<double>& u, const std::vector<size_t>& dir_colloc) const{
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
			double coef = -_tau * nu[iface] * area * iter.second;
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
	for (size_t icolloc: dir_colloc){
		mat.set_unit_row(icolloc);
	}

	return mat.to_csr();
};

void ConvectionFvmWorker::assemble_nu(){
	_nu_velocity.resize(_grid.n_faces(), 1.0/_Re);
	_nu_temperature.resize(_grid.n_faces(), 1.0/_Pe);
	// TODO: add les
}

void ConvectionFvmWorker::assemble_uvt_slae(){
	_mat_u = assemble_u_lhs(_nu_velocity, _u, _boundary_info.bnd_colloc);
	_mat_v = assemble_u_lhs(_nu_velocity, _v, _boundary_info.bnd_colloc);
	_mat_t = assemble_u_lhs(_nu_temperature, _t, _boundary_info.bnd_colloc_t01);

	_u_solver.set_matrix(_mat_u);
	_v_solver.set_matrix(_mat_v);
	_t_solver.set_matrix(_mat_t);

	// ============== right side velocity
	_rhs_u.resize(vec_size());
	_rhs_v.resize(vec_size());
	for (size_t icell = 0; icell < _grid.n_cells(); ++icell){
		_rhs_u[icell] = (_u[icell] -_tau * _grad_p[icell].x()) * _grid.cell_volume(icell);
		_rhs_v[icell] = (_v[icell] -_tau * _grad_p[icell].y() + _tau * _t[icell]) * _grid.cell_volume(icell);
	}
	// bnd
	for (size_t icolloc: _boundary_info.bnd_colloc){
		_rhs_u[icolloc] = 0;
		_rhs_v[icolloc] = 0;
	}

	// ============== right side temperature
	_rhs_t.resize(vec_size());
	for (size_t icell = 0; icell < _grid.n_cells(); ++icell){
		_rhs_t[icell] = _t[icell] * _grid.cell_volume(icell);
	}
	// bnd
	for (size_t icolloc: _boundary_info.bnd_colloc_t0){
		_rhs_t[icolloc] = 0;
	}
	for (size_t icolloc: _boundary_info.bnd_colloc_t1){
		_rhs_t[icolloc] = 1;
	}
}

std::vector<double> ConvectionFvmWorker::compute_u_star(){
	std::vector<double> u_star(_u);
	_u_solver.solve(_rhs_u, u_star);
	return u_star;
}

std::vector<double> ConvectionFvmWorker::compute_v_star(){
	std::vector<double> v_star(_v);
	_v_solver.solve(_rhs_v, v_star);
	return v_star;
}

std::vector<double> ConvectionFvmWorker::compute_temperature(){
	std::vector<double> t(_t);
	_t_solver.solve(_rhs_t, t);
	return t;
}

std::vector<double> ConvectionFvmWorker::compute_un_star_face_rhie_chow(
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

std::vector<double> ConvectionFvmWorker::compute_un_stroke_face(const std::vector<double>& dpstroke_dn){
	std::vector<double> un(_grid.n_faces());
	for (size_t iface=0; iface<_grid.n_faces(); ++iface){
		un[iface] = -_tau * _d * dpstroke_dn[iface];
	}
	return un;
}

std::vector<double> ConvectionFvmWorker::compute_p_stroke(const std::vector<double>& un_star_face){
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

std::vector<double> ConvectionFvmWorker::compute_u_stroke(const std::vector<Vector>& grad_p_stroke){
	std::vector<double> u_stroke(vec_size(), 0.0);
	for (size_t i=0; i<_grid.n_cells(); ++i){
		u_stroke[i] = -_tau * _d * grad_p_stroke[i].x();
	}
	return u_stroke;
}

std::vector<double> ConvectionFvmWorker::compute_v_stroke(const std::vector<Vector>& grad_p_stroke){
	std::vector<double> v_stroke(vec_size(), 0.0);
	for (size_t i=0; i<_grid.n_cells(); ++i){
		v_stroke[i] = -_tau * _d * grad_p_stroke[i].y();
	}
	return v_stroke;
}

}

TEST_CASE("Convection, FVM-SIMPLE algorithm", "[convection-fvm]"){
	std::cout << std::endl << "--- cfd24_test [convection-fvm] --- " << std::endl;

	// problem parameters
	double Re = 1000;
	double Pe = 1000;
	size_t max_it = 10;
	double eps = 1e-6;
	double E = 0.1;
	double L = 2;

	// worker initialization
	RegularGrid2D grid(0, L, 0, 1, 30*L, 30);
	ConvectionFvmWorker worker(grid, Re, Pe, E);
	worker.initialize_saver("convection-fvm");

	// iterations loop
	size_t it = 0;
	double nrm = 0;
	for (it=1; it < max_it; ++it){
		nrm = worker.step();

		// print norm and friction force 
		std::cout << it << " " << nrm << std::endl;

		// export solution to vtk
		if (it % 10 == 0) worker.save_current_fields(it);

		// break if residual is low enough
		if (nrm < eps){
			break;
		}
	}
	CHECK(nrm == Approx(0.12425).margin(1e-4));
}
