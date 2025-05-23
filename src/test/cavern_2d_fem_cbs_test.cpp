#include "cfd24_test.hpp"
#include "cfd24/grid/vtk.hpp"
#include "cfd24/grid/regular_grid2d.hpp"
#include "cfd24/mat/lodmat.hpp"
#include "cfd24/mat/sparse_matrix_solver.hpp"
#include "utils/vecmat.hpp"
#include "cfd24/debug/tictoc.hpp"
#include "cfd24/debug/saver.hpp"
#include "cfd24/fvm/fvm_assembler.hpp"
#include "utils/filesystem.hpp"
#include "cfd24/grid/unstructured_grid2d.hpp"
#include "cfd24/fvm/fvm_dpdn_boundary.hpp"
#include "cfd24/fem/fem_assembler.hpp"
#include "cfd24/fem/fem_assembler.hpp"
#include "cfd24/fem/elem1d/segment_linear.hpp"
#include "cfd24/fem/elem2d/triangle_linear.hpp"
#include "cfd24/fem/elem2d/triangle_quadratic.hpp"
#include "cfd24/fem/elem2d/quadrangle_linear.hpp"
#include "cfd24/fem/elem2d/quadrangle_quadratic.hpp"
#include "cfd24/numeric_integration/segment_quadrature.hpp"
#include "cfd24/numeric_integration/square_quadrature.hpp"
#include "cfd24/numeric_integration/triangle_quadrature.hpp"
#include "cfd24/fem/fem_numeric_integrals.hpp"
#include "cfd24/debug/printer.hpp"
#include "cfd24/fem/fem_sorted_cell_info.hpp"
#include <list>
#include <iomanip>

using namespace cfd;

namespace {

// ========================== ACavern2DFemWorker ========================
struct ACavern2DCbsWorker{
	ACavern2DCbsWorker(const IGrid& grid, double Re, double tau_coef);
	void initialize_saver(std::string stem);
	void step();
	void save_current_fields(size_t iter);
	void convergence_report(size_t iter);
	bool is_converged(double eps) const;
protected:
	const IGrid& _grid;
	const double _Re;
	double _tau_coef = 0.7;
	double _beta_min = 10;
	FemAssembler _fem;
	std::vector<size_t> _stiff_wall_bases, _moving_wall_bases;
	std::vector<double> _alen_e, _alen;
	double _theta1 = 1.0;

	double _n2_dudt = 0;
	double _n2_dvdt = 0;
	double _n2_dpdt = 0;

	std::vector<double> _p;
	std::vector<double> _u;
	std::vector<double> _v;

	std::shared_ptr<VtkUtils::TimeSeriesWriter> _writer;

	static FemAssembler build_fem(unsigned power, const IGrid& grid);

	virtual void to_next_iteration(
			const std::vector<double>& tau,
			const std::vector<double>& beta,
			std::vector<double>& u_new,
			std::vector<double>& v_new,
			std::vector<double>& p_new);
	virtual std::vector<double> calculate_beta() const = 0;
	virtual std::vector<double> calculate_timestep(const std::vector<double>& beta) const = 0;

	virtual void compute_delta_u_1(
			const std::vector<double>& tau,
			std::vector<double>& delta_u, std::vector<double>& delta_v) const;
	virtual void compute_delta_u_2(
			const std::vector<double>& tau,
			const std::vector<double>& p_new,
			std::vector<double>& delta_u, std::vector<double>& delta_v) const;
	virtual void apply_velocity_bc(
			double u_top,
			std::vector<double>& u, std::vector<double>& v) const;

	virtual void compute_pressure(
			const std::vector<double>& tau,
			const std::vector<double>& beta,
			const std::vector<double>& delta_u, const std::vector<double>& delta_v,
			std::vector<double>& pnew) const = 0;
};

ACavern2DCbsWorker::ACavern2DCbsWorker(const IGrid& grid, double Re, double tau_coef):
		_grid(grid), _Re(Re), _tau_coef(tau_coef), _fem(build_fem(1, _grid)){
	// wall bases
	for (size_t ibas = 0; ibas < _fem.n_bases(); ++ibas){
		auto p = _fem.reference_point(ibas);
		if (std::abs(p.x()) < 1e-6 || std::abs(p.x()-1) < 1e-6 || std::abs(p.y()) < 1e-6){
			_stiff_wall_bases.push_back(ibas);
		} else if (std::abs(p.y() - 1) < 1e-6){
			_moving_wall_bases.push_back(ibas);
		}
	}

	// init vectors
	_u = std::vector<double>(_fem.n_bases(), 0);
	_v = std::vector<double>(_fem.n_bases(), 0);
	_p = std::vector<double>(_fem.n_bases(), 0);

	// lengths
	_alen = std::vector<double>(_fem.n_bases(), 1e12);
	_alen_e = std::vector<double>(_fem.n_elements(), 1e12);
	for (size_t icell=0; icell < _grid.n_cells(); ++icell){
		std::vector<size_t> ipoints = _grid.tab_cell_point(icell);
		for (size_t i=0; i<ipoints.size(); ++i){
			size_t im = (i==0) ? ipoints.size()-1 : i-1;
			size_t ip  = (i == ipoints.size()-1) ? 0 : i+1;
			Point pm = _grid.point(ipoints[im]);
			Point pp = _grid.point(ipoints[ip]);
			Vector v1 = pp - _grid.point(ipoints[i]);
			Vector v2 = pm - _grid.point(ipoints[i]);
			double ar = std::abs(cross_product_2d(v1, v2));
			double h = ar / vector_abs(pm - pp);
			_alen_e[icell] = std::min(_alen_e[icell], h);
		}

		for (size_t ibas: _fem.tab_elem_basis(icell)){
			_alen[ibas] = std::min(_alen_e[icell], _alen[ibas]);
		}
	}
}

void ACavern2DCbsWorker::initialize_saver(std::string stem){
	_writer.reset(new VtkUtils::TimeSeriesWriter(stem));
};

void ACavern2DCbsWorker::save_current_fields(size_t iter){
	if (_writer){
		std::string filepath = _writer->add(iter);
		_grid.save_vtk(filepath);
		VtkUtils::add_point_data(_p, "pressure", filepath, _grid.n_points());
		VtkUtils::add_point_vector(_u, _v, "velocity", filepath, _grid.n_points());
	}
}

FemAssembler ACavern2DCbsWorker::build_fem(unsigned power, const IGrid& grid){
	std::vector<FemElement> elements;
	std::vector<std::vector<size_t>> tab_elem_basis;

	if (power > 2) throw std::runtime_error("power should equal 1 or 2");

	std::shared_ptr<IElementBasis> basis3, basis4;
	if (power == 1){
		basis3.reset(new TriangleLinearBasis());
		basis4.reset(new QuadrangleLinearBasis());
	} else {
		basis3.reset(new TriangleQuadraticBasis());
		basis4.reset(new QuadrangleQuadraticBasis());
	}
	// elements
	int n_quad_cells = 0;
	for (size_t icell=0; icell < grid.n_cells(); ++icell){
		PolygonElementInfo cell_info(grid, icell);


		std::shared_ptr<IElementGeometry> geom;
		std::shared_ptr<IElementBasis> basis;
		const Quadrature* quadrature;

		std::vector<size_t> ipoints = grid.tab_cell_point(icell);
		Point p0 = grid.point(ipoints[0]);
		Point p1 = grid.point(ipoints[1]);
		Point p2 = grid.point(ipoints[2]);

		tab_elem_basis.push_back(ipoints);

		if (ipoints.size() == 3){
			geom = std::make_shared<TriangleLinearGeometry>(p0, p1, p2);
			basis = basis3;
			quadrature = quadrature_triangle_gauss3();
			if (power == 2){
				tab_elem_basis.back().push_back(grid.n_points() + cell_info.ifaces[0]);
				tab_elem_basis.back().push_back(grid.n_points() + cell_info.ifaces[1]);
				tab_elem_basis.back().push_back(grid.n_points() + cell_info.ifaces[2]);
			}
		} else if (ipoints.size() == 4){
			Point p3 = grid.point(ipoints[3]);

			geom = std::make_shared<QuadrangleLinearGeometry>(p0, p1, p2, p3);
			basis = basis4;
			quadrature = quadrature_square_gauss3();
			if (power == 2){
				tab_elem_basis.back().push_back(grid.n_points() + cell_info.ifaces[0]);
				tab_elem_basis.back().push_back(grid.n_points() + cell_info.ifaces[1]);
				tab_elem_basis.back().push_back(grid.n_points() + cell_info.ifaces[2]);
				tab_elem_basis.back().push_back(grid.n_points() + cell_info.ifaces[3]);
				tab_elem_basis.back().push_back(grid.n_points() + grid.n_faces() + n_quad_cells);
			}
			n_quad_cells++;
		} else {
			throw std::runtime_error("invalid fem grid");
		}
		auto integrals = std::make_shared<NumericElementIntegrals>(quadrature, geom, basis);
		elements.push_back(FemElement{geom, basis, integrals});
	}

	size_t n_bases = grid.n_points();
	if (power == 2) n_bases += grid.n_faces() + n_quad_cells;
	return FemAssembler(n_bases, elements, tab_elem_basis);
}

void ACavern2DCbsWorker::step(){
	std::vector<double> beta = calculate_beta();
	std::vector<double> tau = calculate_timestep(beta);
	
	// step 1
	std::vector<double> delta_u_1(_fem.n_bases());
	std::vector<double> delta_v_1(_fem.n_bases());
	compute_delta_u_1(tau, delta_u_1, delta_v_1);

	// step 2
	std::vector<double> p_new(_fem.n_bases());
	compute_pressure(tau, beta, delta_u_1, delta_v_1, p_new);

	// step 3
	std::vector<double> delta_u_2(_fem.n_bases());
	std::vector<double> delta_v_2(_fem.n_bases());
	compute_delta_u_2(tau, p_new, delta_u_2, delta_v_2);

	// u_new
	std::vector<double> u_new = _u;
	std::vector<double> v_new = _v;
	for (size_t i=0; i<_fem.n_bases(); ++i){
		u_new[i] += delta_u_1[i] + delta_u_2[i];
		v_new[i] += delta_v_1[i] + delta_v_2[i];
	}
	//bc
	apply_velocity_bc(1.0, u_new, v_new);

	to_next_iteration(tau, beta, u_new, v_new, p_new);
}

void ACavern2DCbsWorker::to_next_iteration(
		const std::vector<double>& tau,
		const std::vector<double>& beta,
		std::vector<double>& u_new,
		std::vector<double>& v_new,
		std::vector<double>& p_new){

	// convergence d_/dt
	std::vector<double> diag(_fem.n_bases(), 0);
	double ar = 0;
	for (size_t ielem=0; ielem < _fem.n_elements(); ++ielem){
		auto local = _fem.element(ielem).integrals->load_vector();
		//_fem.add_to_global_vector(1.0/tau[ielem], ielem, local, diag);
		_fem.add_to_global_vector(1.0, ielem, local, diag);
		ar += _grid.cell_volume(ielem);
	}

	_n2_dudt = 0;
	_n2_dvdt = 0;
	_n2_dpdt = 0;
	for (size_t i=0; i<_fem.n_bases(); ++i){
		double diff_u = (u_new[i] - _u[i]);
		double diff_v = (v_new[i] - _v[i]);
		double diff_p = (p_new[i] - _p[i]);
		_n2_dudt += diff_u * diff_u * diag[i];
		_n2_dvdt += diff_v * diff_v * diag[i];
		_n2_dpdt += diff_p * diff_p * diag[i];
	}
	_n2_dudt = std::sqrt(_n2_dudt/ar);
	_n2_dvdt = std::sqrt(_n2_dvdt/ar);
	_n2_dpdt = std::sqrt(_n2_dpdt/ar);

	// swap
	std::swap(_u, u_new);
	std::swap(_v, v_new);
	std::swap(_p, p_new);
}

void ACavern2DCbsWorker::convergence_report(size_t iter){
	std::cout << std::scientific;
	std::cout << "it = " << std::setw(5) << std::left << iter << "  ";
	std::cout << "|dudt| = " << std::setprecision(3)  << _n2_dudt << "  ";
	std::cout << "|dvdt| = " << std::setprecision(3)  << _n2_dvdt << "  ";
	std::cout << "|dpdt| = " << std::setprecision(3)  << _n2_dpdt << "  ";
	std::cout << std::endl;
	
}

bool ACavern2DCbsWorker::is_converged(double eps) const{
	return eps > _n2_dudt 
		&& eps > _n2_dvdt
		&& eps > _n2_dpdt;
}

void ACavern2DCbsWorker::apply_velocity_bc(double u_top, std::vector<double>& u, std::vector<double>& v) const {
	for (size_t ibas: _stiff_wall_bases){
		u[ibas] = 0.0;
		v[ibas] = 0.0;
	}
	for (size_t ibas: _moving_wall_bases){
		u[ibas] = u_top;
		v[ibas] = 0.0;
	}
}

void ACavern2DCbsWorker::compute_delta_u_1(
		const std::vector<double>& tau,
		std::vector<double>& delta_u,
		std::vector<double>& delta_v) const{

	std::vector<double> rhs_u(_fem.n_bases(), 0.0);
	std::vector<double> rhs_v(_fem.n_bases(), 0.0);
	std::vector<double> lhs(_fem.n_bases(), 0.0);
	for (size_t ielem=0; ielem < _fem.n_elements(); ++ielem){
		const FemElement& elem = _fem.element(ielem);
		auto vx = _fem.local_vector(ielem, _u);
		auto vy = _fem.local_vector(ielem, _v);
		// convection
		{
			std::vector<double> local = elem.integrals->transport_matrix(vx, vy);
			_fem.add_to_global_vector(-1.0, ielem, local, _u, rhs_u);
			_fem.add_to_global_vector(-1.0, ielem, local, _v, rhs_v);
		}
		// cbs stabilization
		{
			std::vector<double> local = elem.integrals->transport_matrix_stab_supg(vx, vy);
			_fem.add_to_global_vector(-tau[ielem]/2.0, ielem, local, _u, rhs_u);
			_fem.add_to_global_vector(-tau[ielem]/2.0, ielem, local, _v, rhs_v);
		}
		// diffusion
		{
			std::vector<double> local = elem.integrals->stiff_matrix();
			_fem.add_to_global_vector(-1.0/_Re, ielem, local, _u, rhs_u);
			_fem.add_to_global_vector(-1.0/_Re, ielem, local, _v, rhs_v);
		}
		// lhs = lumped_mass/tau
		{
			std::vector<double> local = elem.integrals->load_vector();
			_fem.add_to_global_vector(1.0/tau[ielem], ielem, local, lhs);
		}
	}

	// explicit solver
	for (size_t i=0; i<_fem.n_bases(); ++i){
		delta_u[i] = rhs_u[i] / lhs[i];
		delta_v[i] = rhs_v[i] / lhs[i];
	}
}

void ACavern2DCbsWorker::compute_delta_u_2(
		const std::vector<double>& tau,
		const std::vector<double>& p_new,
		std::vector<double>& delta_u, std::vector<double>& delta_v) const {

	std::vector<double> rhs_u(_fem.n_bases(), 0);
	std::vector<double> rhs_v(_fem.n_bases(), 0);
	std::vector<double> lhs(_fem.n_bases(), 0);
	for (size_t ielem=0; ielem<_fem.n_elements(); ++ielem){
		auto elem = _fem.element(ielem);
		std::vector<double> vx = _fem.local_vector(ielem, _u);
		std::vector<double> vy = _fem.local_vector(ielem, _v);
		// -dp/dx * phi
		{
			auto local = elem.integrals->dx_matrix();
			_fem.add_to_global_vector(-1.0, ielem, local, p_new, rhs_u);
		}
		// -tau/2 * dp/dx v*nabla(phi)
		{
			auto local = elem.integrals->dx_matrix_stab_supg(vx, vy);
			_fem.add_to_global_vector(-tau[ielem]/2, ielem, local, _p, rhs_u);
		}
		// -dp/dy * phi
		{
			auto local = elem.integrals->dy_matrix();
			_fem.add_to_global_vector(-1.0, ielem, local, p_new, rhs_v);
		}
		// -tau/2 * dp/dy v*nabla(phi)
		{
			auto local = elem.integrals->dy_matrix_stab_supg2(vx, vy);
			_fem.add_to_global_vector(-tau[ielem]/2, ielem, local, _p, rhs_v);
		}
		// lhs = lumped_mass/tau
		{
			auto local = elem.integrals->load_vector();
			_fem.add_to_global_vector(1.0/tau[ielem], ielem, local, lhs);
		}
	}

	for (size_t i=0; i<_fem.n_bases(); ++i){
		delta_u[i] = rhs_u[i]/lhs[i];
		delta_v[i] = rhs_v[i]/lhs[i];
	}
}

// =================================== Semi-Implicit scheme
class SemiImplicitCbsWorker: public ACavern2DCbsWorker{
public:
	SemiImplicitCbsWorker(const IGrid& grid, double Re, double tau_coef=0.7):
			ACavern2DCbsWorker(grid, Re, tau_coef){

		// pressure solver
		CsrMatrix pmat(_fem.stencil());
		for (size_t ielem=0; ielem < _fem.n_elements(); ++ielem){
			auto local = _fem.element(ielem).integrals->stiff_matrix();
			_fem.add_to_global_matrix(1.0, ielem, local, pmat.vals());
		}
		pmat.set_unit_row(0);
		_p_solver.set_matrix(pmat);
	};
private:
	AmgcMatrixSolver _p_solver;

	std::vector<double> calculate_beta() const override;
	std::vector<double> calculate_timestep(const std::vector<double>&) const override;
	void compute_pressure(
			const std::vector<double>& tau,
			const std::vector<double>& beta,
			const std::vector<double>& delta_u, const std::vector<double>& delta_v,
			std::vector<double>& pnew) const override;
};

std::vector<double> SemiImplicitCbsWorker::calculate_beta() const{
	return {};
}

std::vector<double> SemiImplicitCbsWorker::calculate_timestep(const std::vector<double>&) const{
	std::vector<double> tau(_fem.n_elements());
	for (size_t ielem=0; ielem < _fem.n_elements(); ++ielem){
		double vel2= 1e-6;
		for (size_t ipoint: _grid.tab_cell_point(ielem)){
			double v2 = _u[ipoint] * _u[ipoint] + _v[ipoint] + _v[ipoint];
			vel2 = std::max(vel2, v2);
		}
		double amin = _alen_e[ielem];
		tau[ielem] = _tau_coef*amin*std::min(amin*_Re/2.0, 1.0/sqrt(vel2));
	}
	double tau_min = *std::min_element(tau.begin(), tau.end());
	return std::vector<double>(_fem.n_elements(), tau_min);
}

void SemiImplicitCbsWorker::compute_pressure(
		const std::vector<double>& tau,
		const std::vector<double>&,
		const std::vector<double>& delta_u, const std::vector<double>& delta_v,
		std::vector<double>& pnew) const{

	std::vector<double> rhs(_fem.n_bases(), 0);
	for (size_t ielem = 0; ielem < _fem.n_elements(); ++ielem){
		const FemElement& elem = _fem.element(ielem);
		std::vector<double> vx = _fem.local_vector(ielem, _u);
		std::vector<double> vy = _fem.local_vector(ielem, _v);
		std::vector<double> delta_vx = _fem.local_vector(ielem, delta_u);
		std::vector<double> delta_vy = _fem.local_vector(ielem, delta_v);
		// div(u)
		{
			std::vector<double> local = elem.integrals->divergence_vector(vx, vy);
			_fem.add_to_global_vector(-1.0/tau[ielem]/_theta1, ielem, local, rhs);
		}
		// div(u_star - u)
		{
			std::vector<double> local = elem.integrals->divergence_vector_byparts(delta_vx, delta_vy);
			_fem.add_to_global_vector(+1.0/tau[ielem], ielem, local, rhs);
		}
	}
	rhs[0] = 0;
	_p_solver.solve(rhs, pnew);
}

TEST_CASE("Cavern 2D, FEM CBS algorithm", "[cavern2-fem-si-cbs]"){
	std::cout << std::endl << "--- cfd24_test [cavern2-fem-si-cbs] --- " << std::endl;

	double Re = 100;
	size_t max_it = 10'000;
	double eps = 1e-3;
	double tau_coef = 0.6;

	cfd::RegularGrid2D grid(0, 1, 0, 1, 20, 20);

	SemiImplicitCbsWorker worker(grid, Re, tau_coef);
	worker.initialize_saver("cavern2-cbs");

	size_t it = 0;
	for (it=0; it<max_it; ++it){
		worker.step();
		if (it % 100 == 0){
			worker.save_current_fields(it);
		}
		if (it % 20 == 0){
			worker.convergence_report(it);
		}
		if (worker.is_converged(eps)){
			std::cout << "======== Converged" << std::endl;
			worker.convergence_report(it);
			break;
		}
	}
	CHECK(it == 51);
}


// ======================================= Fully explicit scheme
class ExplicitCbsWorker: public ACavern2DCbsWorker{
public:
	ExplicitCbsWorker(const IGrid& grid, double Re, double tau_coef=0.7):
			ACavern2DCbsWorker(grid, Re, tau_coef){ };
private:
	std::vector<double> calculate_beta() const override;
	std::vector<double> calculate_timestep(const std::vector<double>&) const override;
	void compute_pressure(
			const std::vector<double>& tau,
			const std::vector<double>& beta,
			const std::vector<double>& delta_u, const std::vector<double>& delta_v,
			std::vector<double>& pnew) const override;
};

std::vector<double> ExplicitCbsWorker::calculate_beta() const{
	std::vector<double> beta(_fem.n_elements());

	for (size_t ielem=0; ielem < _fem.n_elements(); ++ielem){
		double vel2= 1e-6;
		for (size_t ipoint: _grid.tab_cell_point(ielem)){
			double v2 = _u[ipoint] * _u[ipoint] + _v[ipoint] + _v[ipoint];
			vel2 = std::max(vel2, v2);
		}
		beta[ielem] = std::max(_beta_min, std::max(std::sqrt(vel2), 1.0/_Re/_alen_e[ielem]));
	}
	return beta;
}

std::vector<double> ExplicitCbsWorker::calculate_timestep(const std::vector<double>& beta) const{
	std::vector<double> tau(_fem.n_elements());
	for (size_t ielem=0; ielem < _fem.n_elements(); ++ielem){
		double vel2= 1e-6;
		for (size_t ipoint: _grid.tab_cell_point(ielem)){
			double v2 = _u[ipoint] * _u[ipoint] + _v[ipoint] + _v[ipoint];
			vel2 = std::max(vel2, v2);
		}
		double amin = _alen_e[ielem];
		tau[ielem] = _tau_coef*amin/(std::sqrt(vel2) + beta[ielem]);
	}
	return tau;
}

void ExplicitCbsWorker::compute_pressure(
		const std::vector<double>& tau,
		const std::vector<double>& beta,
		const std::vector<double>& delta_u, const std::vector<double>& delta_v,
		std::vector<double>& pnew) const{

	std::vector<double> rhs(_fem.n_bases(), 0);
	std::vector<double> lhs(_fem.n_bases(), 0);
	for (size_t ielem = 0; ielem < _fem.n_elements(); ++ielem){
		const FemElement& elem = _fem.element(ielem);
		std::vector<double> vx = _fem.local_vector(ielem, _u);
		std::vector<double> vy = _fem.local_vector(ielem, _v);
		std::vector<double> delta_vx = _fem.local_vector(ielem, delta_u);
		std::vector<double> delta_vy = _fem.local_vector(ielem, delta_v);
		// rhs = -∇⋅u⃗
		{
			std::vector<double> local = elem.integrals->divergence_vector(vx, vy);
			_fem.add_to_global_vector(-1.0, ielem, local, rhs);
		}
		//       -θ₁∇⋅(u⃗' - u⃗), by_parts
		{
			std::vector<double> local = elem.integrals->divergence_vector_byparts(delta_vx, delta_vy);
			_fem.add_to_global_vector(_theta1, ielem, local, rhs);
		}
		//       +τθ₁∇²p
		{
			auto local = elem.integrals->stiff_matrix();
			_fem.add_to_global_vector(-_theta1*tau[ielem], ielem, local, _p, rhs);
		}
		// lhs = lumped_mass/tau/beta^2
		{
			std::vector<double> local = elem.integrals->load_vector();
			_fem.add_to_global_vector(1.0/tau[ielem]/beta[ielem]/beta[ielem], ielem, local, lhs);
		}
	}
	pnew[0] = 0;
	for (size_t i=1; i<_fem.n_bases(); ++i){
		pnew[i] = _p[i] + rhs[i]/lhs[i];
	}
}

TEST_CASE("Cavern 2D, FEM Explicit CBS algorithm", "[cavern2-fem-ex-cbs]"){
	std::cout << std::endl << "--- cfd24_test [cavern2-fem-ex-cbs] --- " << std::endl;

	double Re = 100;
	size_t max_it = 10'000;
	double eps = 1e-2;
	double tau_coef = 0.6;

	cfd::RegularGrid2D grid(0, 1, 0, 1, 20, 20);

	ExplicitCbsWorker worker(grid, Re, tau_coef);
	worker.initialize_saver("cavern2-cbs");

	size_t it = 0;
	for (it=0; it<max_it; ++it){
		worker.step();
		if (it % 100 == 0){
			worker.save_current_fields(it);
		}
		if (it % 20 == 0){
			worker.convergence_report(it);
		}
		if (worker.is_converged(eps)){
			std::cout << "======== Converged" << std::endl;
			worker.convergence_report(it);
			break;
		}
	}
	CHECK(it == 121);
}

class A: public ExplicitCbsWorker{
public:
	A(const IGrid& grid, double Re, double a): ExplicitCbsWorker(grid, Re, a){
		_rho.resize(_fem.n_bases(), 1.0);
		_beta_min = 1;
	}

	void apply_velocity_bc(double u_top, std::vector<double>& u, std::vector<double>& v) const override {
		for (size_t ibas: _stiff_wall_bases){
			u[ibas] = 0.0;
			v[ibas] = 0.0;
		}
		for (size_t ibas: _moving_wall_bases){
			u[ibas] = u_top / _rho[ibas];
			v[ibas] = 0.0;
		}
	}
	void compute_delta_u_1(
			const std::vector<double>& tau,
			std::vector<double>& delta_u,
			std::vector<double>& delta_v) const override{

		std::vector<double> velo_x = _u;
		std::vector<double> velo_y = _v;
		for (size_t i=0; i<velo_x.size(); ++i){
			velo_x[i] /= _rho[i];
			velo_y[i] /= _rho[i];
		}

		std::vector<double> rhs_u(_fem.n_bases(), 0.0);
		std::vector<double> rhs_v(_fem.n_bases(), 0.0);
		std::vector<double> lhs(_fem.n_bases(), 0.0);
		for (size_t ielem=0; ielem < _fem.n_elements(); ++ielem){
			const FemElement& elem = _fem.element(ielem);
			auto vx = _fem.local_vector(ielem, velo_x);
			auto vy = _fem.local_vector(ielem, velo_y);
			// convection
			{
				std::vector<double> local = elem.integrals->transport_matrix(vx, vy);
				_fem.add_to_global_vector(-1.0, ielem, local, _u, rhs_u);
				_fem.add_to_global_vector(-1.0, ielem, local, _v, rhs_v);
			}
			// cbs stabilization
			{
				std::vector<double> local = elem.integrals->transport_matrix_stab_supg(vx, vy);
				_fem.add_to_global_vector(-tau[ielem]/2.0, ielem, local, _u, rhs_u);
				_fem.add_to_global_vector(-tau[ielem]/2.0, ielem, local, _v, rhs_v);
			}
			// diffusion
			{
				std::vector<double> local = elem.integrals->stiff_matrix();
				_fem.add_to_global_vector(-1.0/_Re, ielem, local, velo_x, rhs_u);
				_fem.add_to_global_vector(-1.0/_Re, ielem, local, velo_y, rhs_v);
			}
			// lhs = lumped_mass/tau
			{
				std::vector<double> local = elem.integrals->load_vector();
				_fem.add_to_global_vector(1.0/tau[ielem], ielem, local, lhs);
			}
		}

		// explicit solver
		for (size_t i=0; i<_fem.n_bases(); ++i){
			delta_u[i] = rhs_u[i] / lhs[i];
			delta_v[i] = rhs_v[i] / lhs[i];
		}
	}

	void compute_delta_u_2(
			const std::vector<double>& tau,
			const std::vector<double>& p_new,
			std::vector<double>& delta_u, std::vector<double>& delta_v) const override{

		std::vector<double> velo_x = _u;
		std::vector<double> velo_y = _v;
		for (size_t i=0; i<velo_x.size(); ++i){
			velo_x[i] /= _rho[i];
			velo_y[i] /= _rho[i];
		}

		std::vector<double> rhs_u(_fem.n_bases(), 0);
		std::vector<double> rhs_v(_fem.n_bases(), 0);
		std::vector<double> lhs(_fem.n_bases(), 0);
		for (size_t ielem=0; ielem<_fem.n_elements(); ++ielem){
			auto elem = _fem.element(ielem);
			std::vector<double> vx = _fem.local_vector(ielem, velo_x);
			std::vector<double> vy = _fem.local_vector(ielem, velo_y);
			// -dp/dx * phi
			{
				auto local = elem.integrals->dx_matrix();
				_fem.add_to_global_vector(-1.0, ielem, local, p_new, rhs_u);
			}
			// -tau/2 * dp/dx v*nabla(phi)
			{
				auto local = elem.integrals->dx_matrix_stab_supg(vx, vy);
				_fem.add_to_global_vector(-tau[ielem]/2, ielem, local, _p, rhs_u);
			}
			// -dp/dy * phi
			{
				auto local = elem.integrals->dy_matrix();
				_fem.add_to_global_vector(-1.0, ielem, local, p_new, rhs_v);
			}
			// -tau/2 * dp/dy v*nabla(phi)
			{
				auto local = elem.integrals->dy_matrix_stab_supg2(vx, vy);
				_fem.add_to_global_vector(-tau[ielem]/2, ielem, local, _p, rhs_v);
			}
			// lhs = lumped_mass/tau
			{
				auto local = elem.integrals->load_vector();
				_fem.add_to_global_vector(1.0/tau[ielem], ielem, local, lhs);
			}
		}

		for (size_t i=0; i<_fem.n_bases(); ++i){
			delta_u[i] = rhs_u[i]/lhs[i];
			delta_v[i] = rhs_v[i]/lhs[i];
		}
	}

	void to_next_iteration(
			const std::vector<double>& tau,
			const std::vector<double>& beta,
			std::vector<double>& u_new,
			std::vector<double>& v_new,
			std::vector<double>& p_new) override{

		// rho update
		std::vector<double> lhs(_fem.n_bases(), 0);
		std::vector<double> rhs(_fem.n_bases(), 0);
		for (size_t ielem=0; ielem<_fem.n_elements(); ++ielem){
			auto local = _fem.element(ielem).integrals->load_vector();
			_fem.add_to_global_vector(1.0/tau[ielem], ielem, local, lhs);
			_fem.add_to_global_vector(1.0/tau[ielem]/beta[ielem]/beta[ielem], ielem, local, rhs);
		}
		for (size_t i=0; i<_fem.n_bases(); ++i){
			_rho[i] += rhs[i] / lhs[i] * (p_new[i] - _p[i]);
		}
		//auto r1 = *std::min_element(rhs.begin(), rhs.end());
		//auto r2 = *std::max_element(rhs.begin(), rhs.end());
		//std::cout << r1 << " " << r2 << std::endl;

		ExplicitCbsWorker::to_next_iteration(tau, beta, u_new, v_new, p_new);
	}
private:
	std::vector<double> _rho;
};

//TEST_CASE("Cavern 2D, FEM Explicit2 CBS algorithm", "[cavern2-fem-a-cbs]"){
//        std::cout << std::endl << "--- cfd24_test [cavern2-fem-a-ex-cbs] --- " << std::endl;

//        double Re = 100;
//        size_t max_it = 10'000;
//        //double eps = 1e-2;

//        std::string grid_fn = tmp_directory_file("grid3_zi.vtk");
//        cfd::UnstructuredGrid2D grid = cfd::UnstructuredGrid2D::vtk_read(grid_fn);
//        //cfd::RegularGrid2D grid(0, 1, 0, 1, 30, 30);

//        A worker(grid, Re, 0.6);
//        worker.initialize_saver("cavern2-cbs");

//        for (size_t it=0; it<max_it; ++it){
//                worker.step();
//                //std::cout << it << " " << nrm << std::endl;
//                //double e1 = worker.residual_divergence();
//                //Vector e2 = worker.residual_momentum();
//                //std::cout << e1 << " " << e2.x() << " " << e2.y() << std::endl;
//                if (it % 100 == 0){
//                        worker.save_current_fields(it);
//                }
//                if (it % 20 == 0){
//                        worker.convergence_report(it);
//                }
//                //if (it > 1000){
//                //        break;
//                //}
//        }
//}

}
