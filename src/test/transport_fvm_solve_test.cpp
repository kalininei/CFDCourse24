#include "cfd24_test.hpp"
#include "cfd24/grid/unstructured_grid2d.hpp"
#include "cfd24/grid/grid1d.hpp"
#include "cfd24/grid/regular_grid2d.hpp"
#include "cfd24/mat/csrmat.hpp"
#include "cfd24/mat/sparse_matrix_solver.hpp"
#include "cfd24/grid/vtk.hpp"
#include "cfd24/mat/lodmat.hpp"
#include "utils/filesystem.hpp"
#include "cfd24/debug/printer.hpp"

using namespace cfd;

namespace{

class ATestTransport2FvmWorker{
public:
	double init_solution(Point p) const{
		double x = p.x();
		double y = p.y();
		return (x >= 0.05 && x <= 0.25 && y >= 0.4 && y <= 0.6) ? 1.0 : 0.0;
	}

	Vector velocity(Point p) const{
		return {1, 0};
	}

	double exact_solution(Point p) const{
		return init_solution(p-_time*Point(1, 0));
	}

	ATestTransport2FvmWorker(const IGrid& g): _grid(g), _u(_grid.n_cells()){
		// initial solution
		for (size_t i=0; i<_grid.n_cells(); ++i){
			_u[i] = init_solution(_grid.cell_center(i));
		}
	}

	void step(double tau){
		_time += tau;
		impl_step(tau);
	}

	void save_vtk(const std::string& filename){
		// save grid
		_grid.save_vtk(filename);
		
		// save numerical solution
		VtkUtils::add_cell_data(_u, "numerical", filename);

		// save exact solution
		std::vector<double> exact(_grid.n_cells());
		for (size_t i=0; i<_grid.n_cells(); ++i){
			exact[i] = exact_solution(_grid.cell_center(i));
		}
		VtkUtils::add_cell_data(exact, "exact", filename);
	}

	double current_time() const {
		return _time;
	}

	double compute_norm2() const{
		// sum
		double sum = 0;
		double vol = 0;
		for (size_t i=0; i<_grid.n_cells(); ++i){
			double diff = _u[i] - exact_solution(_grid.cell_center(i));
			sum += diff*diff*_grid.cell_volume(i);
			vol += _grid.cell_volume(i);
		}

		return std::sqrt(sum / vol);
	}

	double compute_norm_max() const{
		return 1.0 - *std::max_element(_u.begin(), _u.end());
	};
protected:
	const IGrid& _grid;
	std::vector<double> _u;
	double _time = 0;

	virtual void impl_step(double tau) = 0;
};

///////////////////////////////////////////////////////////////////////////////
// Explicit upwind transport solver
///////////////////////////////////////////////////////////////////////////////

class TestTransport2FvmUpwindExplicit: public ATestTransport2FvmWorker{
public:
	TestTransport2FvmUpwindExplicit(const IGrid& g): ATestTransport2FvmWorker(g){}
private:
	void impl_step(double tau) override {
		std::vector<double> unew = _u;
		for (size_t iface=0; iface<_grid.n_faces(); ++iface){
			size_t cell_left  = _grid.tab_face_cell(iface)[0];
			size_t cell_right = _grid.tab_face_cell(iface)[1];
			// no flux from boundary
			if (cell_left == INVALID_INDEX || cell_right == INVALID_INDEX){
				continue;
			}
			// compute velocity at face center
			Point face_center = _grid.face_center(iface);
			Vector vel = velocity(face_center);
			Vector nij = _grid.face_normal(iface);
			double Un = dot_product(vel, nij);
			if (Un < 0){
				std::swap(cell_left, cell_right);
				Un *= -1;
			}
			double uij = _u[cell_left];   // upwind scheme
			double f = uij * _grid.face_area(iface) * Un;

			// add to adjasting cells
			unew[cell_left]  -= tau * f / _grid.cell_volume(cell_left);
			unew[cell_right] += tau * f / _grid.cell_volume(cell_right);
		}
		std::swap(unew, _u);
	}
};

TEST_CASE("Transport 2D fvm solver, upwind explicit", "[transport2-fvm-upwind-explicit]"){
	std::cout << std::endl << "--- cfd24_test [transport2-fvm-upwind-explicit] --- " << std::endl;
	double tend = 0.5;

	// solver
	RegularGrid2D g(0, 1, 0, 1, 40, 40);
	TestTransport2FvmUpwindExplicit worker(g);
	double tau = 0.02;

	// saver
	VtkUtils::TimeSeriesWriter writer("transport2-fvm-upwind-explicit");
	std::string out_filename = writer.add(0);
	worker.save_vtk(out_filename);

	double n2=0, nm=0;
	while (worker.current_time() < tend - 1e-6) {
		// solve problem
		worker.step(tau);
		// export solution to vtk
		out_filename = writer.add(worker.current_time());

		worker.save_vtk(out_filename);

		n2 = worker.compute_norm2();
		nm = worker.compute_norm_max();

		std::cout << worker.current_time() << " " << n2 << " " << nm << std::endl;
	};
	CHECK(nm == Approx(0.0447215951).margin(1e-4));
}
}
