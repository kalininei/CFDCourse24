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
#include "cfd24/fvm/fvm_dpdn_boundary.hpp"
#include <iomanip>
#include <list>

using namespace cfd;

struct CylinderFvmSimpleWorker{
	CylinderFvmSimpleWorker(const IGrid& grid, double Re, double E, double timestep);
	void initialize_saver(std::string stem, double save_time_step);

	double step();
	double to_next_time_step();
	void save_current_fields(double time) const;

	size_t vec_size() const{
		return _collocations.size();
	}
private:
	const IGrid& _grid;
	const double _Re;
	const double _tau;
	const double _alpha_p;
	const double _time_step;
	const FvmExtendedCollocations _collocations;
	const FvmFacesDn _dfdn_computer;
	const FvmCellGradient_LeastSquares _grad_computer;

	struct BoundaryInfo{
		std::vector<size_t> all;
		std::vector<size_t> cyl;
		std::vector<size_t> input;
		std::vector<size_t> output;
		std::vector<size_t> sym;
	};
	BoundaryInfo _boundary_info;

	std::vector<double> _p;
	std::vector<double> _u;
	std::vector<double> _v;
	std::vector<double> _un_face;
	std::vector<double> _dpdn_face;
	std::vector<Vector> _grad_p;
	std::vector<double> _u_old;
	std::vector<double> _v_old;

	AmgcMatrixSolver _p_stroke_solver;
	AmgcMatrixSolver _uv_solver;

	CsrMatrix _mat_uv;
	std::vector<double> _rhs_u;
	std::vector<double> _rhs_v;
	double _d;

	std::shared_ptr<VtkUtils::TimeSeriesWriter> _writer;

	double to_next_iteration();
	void gather_boundary_collocations();
	void assemble_p_stroke_solver();
	CsrMatrix assemble_uv_lhs(double coef_u, double coef_conv, double coef_diff) const;
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

double CylinderFvmSimpleWorker::compute_tau(const IGrid& grid, double Re, double E){
	double h2 = grid.cell_volume(0);
	for (size_t i=1; i<grid.n_cells(); ++i){
		h2 = std::min(h2, grid.cell_volume(i));
	}
	return E*Re*h2/4.0;
}

CylinderFvmSimpleWorker::CylinderFvmSimpleWorker(const IGrid& grid, double Re, double E, double time_step):
	_grid(grid),
	_Re(Re),
	_tau(compute_tau(grid, Re, E)),
	_alpha_p(1.0/(1.0 + _tau/time_step + E)),
	_time_step(time_step),
	_collocations(grid),
	_dfdn_computer(grid, _collocations),
	_grad_computer(grid, _collocations)
{
	_d = 1.0/(1 + _tau/time_step + E);
	gather_boundary_collocations();
	assemble_p_stroke_solver();

	_u = std::vector<double>(vec_size(), 0);
	_v = std::vector<double>(vec_size(), 0);
	_p = std::vector<double>(vec_size(), 0);
	_grad_p = std::vector<Vector>(_grid.n_cells(), {0, 0, 0});
	_un_face = std::vector<double>(_grid.n_faces(), 0);
	_dpdn_face = std::vector<double>(_grid.n_faces(), 0);
	to_next_time_step();
}

void CylinderFvmSimpleWorker::gather_boundary_collocations(){
	double xmin = _grid.point(0).x(); double xmax = _grid.point(0).x();
	double ymin = _grid.point(0).y(); double ymax = _grid.point(0).y();
	for (size_t i=1; i<_grid.n_points(); ++i){
		Point p = _grid.point(i);
		xmin = std::min(xmin, p.x()); ymin = std::min(ymin, p.y());
		xmax = std::max(xmax, p.x()); ymax = std::max(ymax, p.y());
	}

	BoundaryInfo& bi = _boundary_info;
	for (size_t icolloc: _collocations.face_collocations){
		size_t iface = _collocations.face_index(icolloc);
		bi.all.push_back(icolloc);
		Point fc = _grid.face_center(iface);
		if (std::abs(fc.y() - ymin) < 1e-6){
			bi.sym.push_back(icolloc);
		} else if (std::abs(fc.y() - ymax) < 1e-6){
			bi.sym.push_back(icolloc);
		} else if (std::abs(fc.x() - xmin) < 1e-6){
			bi.input.push_back(icolloc);
		} else if (std::abs(fc.x() - xmax) < 1e-6){
			bi.output.push_back(icolloc);
		} else {
			bi.cyl.push_back(icolloc);
		}
	}
}

void CylinderFvmSimpleWorker::initialize_saver(std::string stem, double save_time_step){
	_writer.reset(new VtkUtils::TimeSeriesWriter(stem));
	_writer->set_time_step(save_time_step);
};

double CylinderFvmSimpleWorker::to_next_iteration(){
	assemble_uv_slae();
	// residual vectors
	std::vector<double> res_u = compute_residual_vec(_mat_uv, _rhs_u, _u);
	std::vector<double> res_v = compute_residual_vec(_mat_uv, _rhs_v, _v);
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

double CylinderFvmSimpleWorker::step(){
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
	_dpdn_face = vector_sum(_dpdn_face, _alpha_p, dpstroke_dn_face);
	_grad_p = vector_sum(_grad_p, _alpha_p, grad_p_stroke);

	return to_next_iteration();
}

void CylinderFvmSimpleWorker::save_current_fields(double time) const{
	if (_writer){
		std::string filepath = _writer->add(time);
		if (filepath.empty()){
			return;
		}
		_grid.save_vtk(filepath);
		VtkUtils::add_cell_data(_p, "pressure", filepath, _grid.n_cells());
		VtkUtils::add_cell_vector(_u, _v, "velocity", filepath, _grid.n_cells());
	}
}

void CylinderFvmSimpleWorker::assemble_p_stroke_solver(){
	LodMatrix mat(vec_size());
	// internal
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

CsrMatrix CylinderFvmSimpleWorker::assemble_uv_lhs(double coef_u, double coef_conv, double coef_diff) const{
	LodMatrix mat(vec_size());
	// coef_u * u
	for (size_t icell = 0; icell < _grid.n_cells(); ++icell){
		mat.add_value(icell, icell, coef_u*_grid.cell_volume(icell));
	}
	for (size_t iface = 0; iface < _grid.n_faces(); ++iface){
		size_t negative_colloc = _collocations.tab_face_colloc(iface)[0];
		size_t positive_colloc = _collocations.tab_face_colloc(iface)[1];
		double area = _grid.face_area(iface);

		// - coef_diff * Laplace(u)
		for (const std::pair<const size_t, double>& iter: _dfdn_computer.linear_combination(iface)){
			size_t column = iter.first;
			double coef = -coef_diff * area * iter.second;
			mat.add_value(negative_colloc, column, coef);
			mat.add_value(positive_colloc, column, -coef);
		}
		
		// + coef_conv * convection
		{
			double coef = coef_conv * area * _un_face[iface]/2.0;
			mat.add_value(negative_colloc, negative_colloc,  coef);
			mat.add_value(negative_colloc, positive_colloc,  coef);
			mat.add_value(positive_colloc, positive_colloc, -coef);
			mat.add_value(positive_colloc, negative_colloc, -coef);
		}
	}
	// input, cylinder boundary: dirichlet
	for (size_t icolloc: _boundary_info.input) mat.set_unit_row(icolloc);
	for (size_t icolloc: _boundary_info.cyl) mat.set_unit_row(icolloc);
	// output boundary: du/dt + un*du/dn = 0
	for (size_t icolloc: _boundary_info.output){
		mat.remove_row(icolloc);
		size_t iface = _collocations.face_index(icolloc);
		double sgn = (_grid.tab_face_cell(iface)[1] == INVALID_INDEX) ? 1 : -1;
		mat.add_value(icolloc, icolloc, coef_u);
		for (auto it: _dfdn_computer.linear_combination(iface)){
			size_t icolumn = it.first;
			double value = coef_conv * sgn * it.second;
			mat.add_value(icolloc, icolumn, value);
		}
	}
	return mat.to_csr();
}

void CylinderFvmSimpleWorker::assemble_uv_slae(){
	// =============== LHS
	_mat_uv = assemble_uv_lhs(1+_tau/_time_step, _tau, _tau/_Re);
	_uv_solver.set_matrix(_mat_uv);

	// ============== RHS
	_rhs_u.resize(vec_size());
	_rhs_v.resize(vec_size());
	for (size_t icell = 0; icell < _grid.n_cells(); ++icell){
		_rhs_u[icell] = (_u[icell] + _tau/_time_step*_u_old[icell] -_tau * _grad_p[icell].x()) * _grid.cell_volume(icell);
		_rhs_v[icell] = (_v[icell] + _tau/_time_step*_v_old[icell] -_tau * _grad_p[icell].y()) * _grid.cell_volume(icell);
	}
	// bnd
	for (size_t icolloc: _boundary_info.input){
		_rhs_u[icolloc] = 1;
		_rhs_v[icolloc] = 0;
	}
	for (size_t icolloc: _boundary_info.cyl){
		_rhs_u[icolloc] = 0;
		_rhs_v[icolloc] = 0;
	}
	for (size_t icolloc: _boundary_info.output){
		_rhs_u[icolloc] = _u[icolloc] + _u_old[icolloc] * _tau / _time_step;
		_rhs_v[icolloc] = _v[icolloc] + _v_old[icolloc] * _tau / _time_step;
	}
}

std::vector<double> CylinderFvmSimpleWorker::compute_u_star(){
	std::vector<double> u_star(_u);
	_uv_solver.solve(_rhs_u, u_star);
	return u_star;
}

std::vector<double> CylinderFvmSimpleWorker::compute_v_star(){
	std::vector<double> v_star(_v);
	_uv_solver.solve(_rhs_v, v_star);
	return v_star;
}

std::vector<double> CylinderFvmSimpleWorker::compute_un_star_face_rhie_chow(
		const std::vector<double>& u_star,
		const std::vector<double>& v_star){

	std::vector<double> ret(_grid.n_faces());

	for (size_t iface=0; iface<_grid.n_faces(); ++iface){
		size_t ci = _grid.tab_face_cell(iface)[0];
		size_t cj = _grid.tab_face_cell(iface)[1];
		if (ci == INVALID_INDEX || cj == INVALID_INDEX){
			// Boundary. Default zero.
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
	// input boundary 
	for (size_t icoll: _boundary_info.input){
		size_t iface = _collocations.face_index(icoll);
		ret[iface] = (_grid.tab_face_cell(iface)[0] == INVALID_INDEX) ? 1 : -1;
	}
	// output boundary
	for (size_t icoll: _boundary_info.output){
		size_t iface = _collocations.face_index(icoll);
		ret[iface] = (_grid.tab_face_cell(iface)[0] == INVALID_INDEX) ? -1 : 1;
	}

	return ret;
}

std::vector<double> CylinderFvmSimpleWorker::compute_un_stroke_face(
		const std::vector<double>& dpstroke_dn_face){
	std::vector<double> un(_grid.n_faces());
	for (size_t iface=0; iface<_grid.n_faces(); ++iface){
		un[iface] = -_tau * _d * dpstroke_dn_face[iface];
	}
	return un;
}

std::vector<double> CylinderFvmSimpleWorker::compute_p_stroke(const std::vector<double>& un_star_face){
	// rhs
	std::vector<double> rhs(vec_size(), 0.0);
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

std::vector<double> CylinderFvmSimpleWorker::compute_u_stroke(const std::vector<Vector>& grad_p_stroke){
	std::vector<double> u_stroke(vec_size(), 0.0);
	for (size_t i=0; i<_grid.n_cells(); ++i){
		u_stroke[i] = -_tau * _d * grad_p_stroke[i].x();
	}
	return u_stroke;
}

std::vector<double> CylinderFvmSimpleWorker::compute_v_stroke(const std::vector<Vector>& grad_p_stroke){
	std::vector<double> v_stroke(vec_size(), 0.0);
	for (size_t i=0; i<_grid.n_cells(); ++i){
		v_stroke[i] = -_tau * _d * grad_p_stroke[i].y();
	}
	return v_stroke;
}

double CylinderFvmSimpleWorker::to_next_time_step(){
	_u_old = _u;
	_v_old = _v;
	return to_next_iteration();
}

namespace {

std::string convergence_report(double time, int it){
	std::ostringstream oss;
	oss << std::setprecision(2) << std::fixed;
	oss << "Time = " << std::setw(5) << time << " converged in " << it << " iterations" << std::endl;
	return oss.str();
}

}

TEST_CASE("Cylinder 2D, FVM-SIMPLE algorithm", "[cylinder2-fvm-simple]"){
	std::cout << std::endl << "--- cfd24_test [cylinder2-fvm-simple] --- " << std::endl;

	// problem parameters
	double Re = 100;
	size_t max_it = 1'000;
	double eps = 1e-1;
	double E = 4;
	double time_step = 0.1;
	double end_time = 0.1;
	//double end_time = 100.0;

	// worker initialization
	auto grid = UnstructuredGrid2D::vtk_read(test_directory_file("cylgrid_5k.vtk"));
	CylinderFvmSimpleWorker worker(grid, Re, E, time_step);
	worker.initialize_saver("cylinder2-fvm", 0.5);

	// time loop
	size_t it = 0;
	for (double time=time_step; time<end_time+1e-6; time+=time_step){
		for (it=1; it < max_it; ++it){
			double nrm = worker.step();
			// break inner iterations if residual is low enough
			if (nrm < eps){
				break;
			} else if (it == max_it-1) {
				std::cout << "WARNING: internal SIMPLE interations not converged with nrm = "
				          << nrm << std::endl;
			}
		}
		// uvp_old = uvp
		worker.to_next_time_step();

		// save and report
		std::cout << convergence_report(time, it);
		worker.save_current_fields(time);
	}

	CHECK(it == 16);
}
