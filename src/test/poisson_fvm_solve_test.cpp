#include "cfd24_test.hpp"
#include "cfd24/grid/regular_grid2d.hpp"
#include "cfd24/grid/unstructured_grid2d.hpp"
#include "cfd24/grid/vtk.hpp"
#include "cfd24/grid/grid_partition.hpp"
#include "cfd24/mat/csrmat.hpp"
#include "cfd24/mat/lodmat.hpp"
#include "cfd24/mat/sparse_matrix_solver.hpp"
#include "cfd24/debug/printer.hpp"
#include "cfd24/debug/tictoc.hpp"
#include "test/utils/filesystem.hpp"
#include "cfd24/fvm/fvm_assembler.hpp"
#include "cfd24/debug/nan_signal_handler.hpp"

using namespace cfd;

///////////////////////////////////////////////////////////////////////////////
// Fvm
///////////////////////////////////////////////////////////////////////////////

struct TestPoisson2FvmWorker{
	static double exact_solution(Point p){
		double x = p.x();
		double y = p.y();
		return cos(10*x*x)*sin(10*y) + sin(10*x*x)*cos(10*x);
	}
	static double exact_rhs(Point p){
		double x = p.x();
		double y = p.y();
		return (20*sin(10*x*x)+(400*x*x+100)*cos(10*x*x))*sin(10*y)
				+(400*x*x+100)*cos(10*x)*sin(10*x*x)
				+(400*x*sin(10*x)-20*cos(10*x))*cos(10*x*x);
	}

	TestPoisson2FvmWorker(const IGrid& grid);
	double solve();
	void save_vtk(const std::string& filename) const;
	const std::vector<double>& u() const {return _u; }
protected:
	const IGrid& _grid;
	std::vector<size_t> _internal_faces;
	struct DirichletFace{
		size_t iface;
		size_t icell;
		double value;
		Vector outer_normal;
	};
	std::vector<DirichletFace> _dirichlet_faces;
	std::vector<double> _u;

	virtual CsrMatrix approximate_lhs() const;
	virtual std::vector<double> approximate_rhs() const;
	double compute_norm2() const;
};

TestPoisson2FvmWorker::TestPoisson2FvmWorker(const IGrid& grid): _grid(grid){
	// assemble face lists
	for (size_t iface=0; iface<_grid.n_faces(); ++iface){
		size_t icell_negative = _grid.tab_face_cell(iface)[0];
		size_t icell_positive = _grid.tab_face_cell(iface)[1];
		if (icell_positive != INVALID_INDEX && icell_negative != INVALID_INDEX){
			// internal faces list
			_internal_faces.push_back(iface);
		} else {
			// dirichlet faces list
			DirichletFace dir_face;
			dir_face.iface = iface;
			dir_face.value = exact_solution(_grid.face_center(iface));
			if (icell_positive == INVALID_INDEX){
				dir_face.icell = icell_negative;
				dir_face.outer_normal = _grid.face_normal(iface);
			} else {
				dir_face.icell = icell_positive;
				dir_face.outer_normal = -_grid.face_normal(iface);
			}
			_dirichlet_faces.push_back(dir_face);
		}
	}
}

double TestPoisson2FvmWorker::solve(){
	// 1. build SLAE
	CsrMatrix mat = approximate_lhs();
	std::vector<double> rhs = approximate_rhs();
	// 2. solve SLAE
	AmgcMatrixSolver solver;
	solver.set_matrix(mat);
	solver.solve(rhs, _u);
	// 3. compute norm2
	return compute_norm2();
}

// saves numerical and exact solution into the vtk format
void TestPoisson2FvmWorker::save_vtk(const std::string& filename) const{
	// save grid
	_grid.save_vtk(filename);
	// save numerical solution
	VtkUtils::add_cell_data(_u, "numerical", filename, _grid.n_cells());
	// save exact solution
	std::vector<double> exact(_grid.n_cells());
	for (size_t i=0; i<_grid.n_cells(); ++i){
		exact[i] = exact_solution(_grid.cell_center(i));
	}
	VtkUtils::add_cell_data(exact, "exact", filename);
}

CsrMatrix TestPoisson2FvmWorker::approximate_lhs() const{
	LodMatrix mat(_grid.n_cells());
	// internal faces
	for (size_t iface: _internal_faces){
		Vector normal = _grid.face_normal(iface);
		size_t negative_side_cell = _grid.tab_face_cell(iface)[0];
		size_t positive_side_cell = _grid.tab_face_cell(iface)[1];
		Point ci = _grid.cell_center(negative_side_cell);
		Point cj = _grid.cell_center(positive_side_cell);
		double h = dot_product(normal, cj-ci);
		double coef = _grid.face_area(iface) / h;

		mat.add_value(negative_side_cell, negative_side_cell, coef);
		mat.add_value(positive_side_cell, positive_side_cell, coef);
		mat.add_value(negative_side_cell, positive_side_cell, -coef);
		mat.add_value(positive_side_cell, negative_side_cell, -coef);
	}
	// dirichlet faces
	for (const DirichletFace& dir_face: _dirichlet_faces){
		size_t icell = dir_face.icell;
		size_t iface = dir_face.iface;
		Point gs = _grid.face_center(iface);
		Point ci = _grid.cell_center(icell);
		Vector normal = dir_face.outer_normal;
		double h = dot_product(normal, gs-ci);
		double coef = _grid.face_area(iface) / h;
		mat.add_value(icell, icell, coef);
	}
	return mat.to_csr();
}

std::vector<double> TestPoisson2FvmWorker::approximate_rhs() const{
	std::vector<double> rhs(_grid.n_cells(), 0.0);
	// internal
	for (size_t icell=0; icell < _grid.n_cells(); ++icell){
		double value = exact_rhs(_grid.cell_center(icell));
		double volume = _grid.cell_volume(icell);
		rhs[icell] = value * volume;
	}
	// dirichlet faces
	for (const DirichletFace& dir_face: _dirichlet_faces){
		size_t icell = dir_face.icell;
		size_t iface = dir_face.iface;
		Point gs = _grid.face_center(iface);
		Point ci = _grid.cell_center(icell);
		Vector normal = dir_face.outer_normal;
		double h = dot_product(normal, gs-ci);
		double coef = _grid.face_area(iface) / h;
		rhs[icell] += dir_face.value * coef;
	}
	return rhs;
}

double TestPoisson2FvmWorker::compute_norm2() const{
	double norm2 = 0;
	double full_area = 0;
	for (size_t icell=0; icell<_grid.n_cells(); ++icell){
		double diff = _u[icell] - exact_solution(_grid.cell_center(icell));
		norm2 += _grid.cell_volume(icell) * diff * diff;
		full_area += _grid.cell_volume(icell);
	}
	return std::sqrt(norm2/full_area);
}

TEST_CASE("Poisson-fvm 2D solver", "[poisson2-fvm]"){
	std::cout << std::endl << "--- cfd24_test [poisson2-fvm] --- " << std::endl;

	size_t nx = 20;
	RegularGrid2D grid(0.0, 1.0, 0.0, 1.0, nx, nx);
	TestPoisson2FvmWorker worker(grid);
	double nrm = worker.solve();
	worker.save_vtk("poisson2_fvm.vtk");
	std::cout << grid.n_cells() << " " << nrm << std::endl;

	CHECK(nrm == Approx(0.04371).margin(1e-4));
};


///////////////////////////////////////////////////////////////////////////////
// SkewFvm
///////////////////////////////////////////////////////////////////////////////

struct TestPoisson2SkewFvmWorker: public TestPoisson2FvmWorker{
	TestPoisson2SkewFvmWorker(const IGrid& grid) : TestPoisson2FvmWorker(grid), _collocations(grid){ }
private:
	FvmExtendedCollocations _collocations;
	CsrMatrix approximate_lhs() const override;
	std::vector<double> approximate_rhs() const override;
};

CsrMatrix TestPoisson2SkewFvmWorker::approximate_lhs() const{
	LodMatrix mat(_collocations.size());
	FvmFacesDn dudn(_grid, _collocations);

	// internal
	for (size_t iface = 0; iface < _grid.n_faces(); ++iface){
		size_t negative_colloc = _collocations.tab_face_colloc(iface)[0];
		size_t positive_colloc = _collocations.tab_face_colloc(iface)[1];
		double area = _grid.face_area(iface);

		for (const std::pair<const size_t, double>& iter: dudn.linear_combination(iface)){
			size_t column = iter.first;
			double coef = area * iter.second;
			mat.add_value(positive_colloc, column, coef);
			mat.add_value(negative_colloc, column, -coef);
		}
	}

	// dirichlet bc
	for (size_t icolloc: _collocations.face_collocations){
		mat.set_unit_row(icolloc);
	}

	return mat.to_csr();
}


std::vector<double> TestPoisson2SkewFvmWorker::approximate_rhs() const{
	std::vector<double> rhs(_collocations.size(), 0.0);

	// f
	for (size_t icell=0; icell < _grid.n_cells(); ++icell){
		double value = exact_rhs(_grid.cell_center(icell));
		double volume = _grid.cell_volume(icell);
		rhs[icell] = value * volume;
	}

	// dirichlet bc
	for (size_t icolloc: _collocations.face_collocations){
		rhs[icolloc] = exact_solution(_collocations.points[icolloc]);
	}

	return rhs;
}

TEST_CASE("Poisson-fvm 2D solver, skewgrid", "[poisson2-fvm-skew]"){
	std::cout << std::endl << "--- cfd24_test [poisson2-fvm-skew] --- " << std::endl;

	std::string grid_fn = test_directory_file("tetragrid_500.vtk");
	UnstructuredGrid2D grid = UnstructuredGrid2D::vtk_read(grid_fn);
	TestPoisson2SkewFvmWorker worker(grid);
	double nrm = worker.solve();
	worker.save_vtk("poisson2_fvm_skew.vtk");
	std::cout << grid.n_cells() << " " << nrm << std::endl;

	CHECK(nrm == Approx(0.0422449151).margin(1e-4));
}


///////////////////////////////////////////////////////////////////////////////
// DDM
///////////////////////////////////////////////////////////////////////////////

namespace {

struct TestPoissonFvmSubAssembler{
public:
	static double exact_solution(Point p){
		double x = p.x();
		double y = p.y();
		return cos(10*x*x)*sin(10*y) + sin(10*x*x)*cos(10*x);
	}
	static double exact_rhs(Point p){
		double x = p.x();
		double y = p.y();
		return (20*sin(10*x*x)+(400*x*x+100)*cos(10*x*x))*sin(10*y)
				+(400*x*x+100)*cos(10*x)*sin(10*x*x)
				+(400*x*sin(10*x)-20*cos(10*x))*cos(10*x*x);
	}

	TestPoissonFvmSubAssembler(const SubGrid& grid):
			_grid(grid),
			_collocations(grid, grid.domain_connection_faces()){
	}

	std::vector<double> assemble_rhs() const;
	CsrMatrix assemble_lhs() const;
private:
	const SubGrid& _grid;
	FvmExtendedCollocations _collocations;
};

CsrMatrix TestPoissonFvmSubAssembler::assemble_lhs() const{
	LodMatrix mat(_collocations.size());
	FvmFacesDn dudn(_grid, _collocations);

	// internal
	for (size_t iface = 0; iface < _grid.n_faces(); ++iface){
		size_t negative_colloc = _collocations.tab_face_colloc(iface)[0];
		size_t positive_colloc = _collocations.tab_face_colloc(iface)[1];
		if (negative_colloc == INVALID_INDEX || positive_colloc == INVALID_INDEX){
			continue;
		}
		double area = _grid.face_area(iface);

		for (const std::pair<const size_t, double>& iter: dudn.linear_combination(iface)){
			size_t column = iter.first;
			double coef = area * iter.second;
			mat.add_value(positive_colloc, column, coef);
			mat.add_value(negative_colloc, column, -coef);
		}
	}

	// outside dirichlet bc
	for (size_t icolloc: _collocations.face_collocations){
		mat.set_unit_row(icolloc);
	}

	// ghost cells dirichlet bc
	for (size_t ighost: _grid.ghost_cells()){
		mat.set_unit_row(ighost);
	}

	return mat.to_csr();
}

std::vector<double> TestPoissonFvmSubAssembler::assemble_rhs() const{
	std::vector<double> rhs(_collocations.size(), 0);
	// internal
	for (size_t icell=0; icell < _grid.n_cells(); ++icell){
		double value = exact_rhs(_grid.cell_center(icell));
		rhs[icell] = value * _grid.cell_volume(icell);
	}
	// outside dirichlet bc
	for (size_t icolloc: _collocations.face_collocations){
		size_t iface = _collocations.face_index(icolloc);
		rhs[icolloc] = exact_solution(_grid.face_center(iface));
	}
	return rhs;
}

double compute_residual(const CsrMatrix& mat, const std::vector<double>& u, const std::vector<double>& rhs){
	double ret = 0;
	for (size_t icell=0; icell < u.size(); ++icell){
		double r = std::abs(mat.mult_vec(icell, u) - rhs[icell]);
		ret = std::max(ret, r);
	}
	return ret;
}
}

TEST_CASE("Poisson-fvm, DDM", "[poisson2-fvm-ddm]"){
	std::cout << std::endl << "--- cfd24_test [poisson2-fvm-ddm] --- " << std::endl;

	// read grid
	std::string grid_fn = test_directory_file("trigrid_500.vtk");
	auto grid = std::make_shared<UnstructuredGrid2D>(UnstructuredGrid2D::vtk_read(grid_fn));
	
	
	// find solution of the full problem
	TestPoisson2SkewFvmWorker full_worker(*grid);
	full_worker.solve();
	std::vector<double> error(grid->n_cells(), 0);

	// grid parition
	std::vector<double> global_u(grid->n_cells(), 0);
	GridPartition gridpart = GridPartition::build_uniform<2>(grid, {1, 2}, 2, 1);
	std::vector<std::shared_ptr<SubGrid>> local_grid;
	for (size_t idom=0; idom<gridpart.n_domains(); ++idom){
		local_grid.push_back(gridpart.subgrid(idom));
	}

	// restriction functions
	std::vector<std::vector<double>> local_Xi;
	for (size_t idom=0; idom < gridpart.n_domains(); ++idom){
		local_Xi.push_back(gridpart.restriction_weights(idom, GridPartition::Restriction::AVERAGE));
	}

	// assemble subproblems
	std::vector<CsrMatrix> local_mat;
	std::vector<std::vector<double>> local_rhs;
	std::vector<std::vector<double>> local_u;
	std::vector<std::shared_ptr<AmgcMatrixSolver>> local_solver;
	for (size_t idom=0; idom < gridpart.n_domains(); ++idom){
		TestPoissonFvmSubAssembler wrk(*gridpart.subgrid(idom));
		local_mat.push_back(wrk.assemble_lhs());
		local_rhs.push_back(wrk.assemble_rhs());
		local_u.emplace_back(local_rhs.back().size(), 0);
		local_solver.emplace_back(new AmgcMatrixSolver(1000, 1e-12));
		local_solver.back()->set_matrix(local_mat.back());
	}

	// solution vtk writer initialization
	VtkUtils::TimeSeriesWriter writer("schwarz_ddm_poisson.vtk");

	// Schwarz iterations
	size_t it;
	double global_residual;
	for (it=0; it < 1'000; ++it){
		global_residual = 0;

		// serial method
		for (size_t idom=0; idom < gridpart.n_domains(); ++idom){
			// bc
			for (size_t local_ghost: local_grid[idom]->ghost_cells()){
				size_t global_ghost = local_grid[idom]->to_global_cell(local_ghost);
				local_rhs[idom][local_ghost] = global_u[global_ghost];
			}
			// residual
			double local_residual = compute_residual(local_mat[idom], local_u[idom], local_rhs[idom]);
			global_residual = std::max(global_residual, local_residual);

			// local solution
			local_solver[idom]->solve(local_rhs[idom], local_u[idom]);
		
			// update global solution
			std::fill(global_u.begin(), global_u.end(), 0.0);
			for (size_t idom2=0; idom2 < gridpart.n_domains(); ++idom2){
				// Xi * u
				std::vector<double> xi_u(local_grid[idom2]->n_cells());
				for (size_t i=0; i<xi_u.size(); ++i) xi_u[i] = local_u[idom2][i] * local_Xi[idom2][i];

				// E(Xi * u)
				std::vector<double> local_u_expanded = local_grid[idom2]->expand_cell_vector(xi_u);

				// add to result
				for (size_t i=0; i<global_u.size(); ++i){
					global_u[i] += local_u_expanded[i];
				}
			}
		}

		// info
		std::cout << "it=" << it <<", residual=" << global_residual << std::endl;

		// save solution
		std::string out_filename = writer.add(it);
		grid->save_vtk(out_filename);
		VtkUtils::add_cell_data(global_u, "solution", out_filename);
		for (size_t i=0; i<grid->n_cells(); ++i) error[i] = std::abs(full_worker.u()[i] - global_u[i]);
		VtkUtils::add_cell_data(error, "error", out_filename);

		// exit condition
		if (global_residual < 1e-6){
			break;
		}
	}

	CHECK(it == 8);
	CHECK(global_residual == Approx(1.81678e-07).margin(1e-9));
}

///////////////////////////////////////////////////////////////////////////////
// Preconditioned probmels
///////////////////////////////////////////////////////////////////////////////
namespace {

double exact_solution(Point p){
	double x = p.x();
	double y = p.y();
	return cos(10*x*x)*sin(10*y) + sin(10*x*x)*cos(10*x);
}

double exact_rhs(Point p){
	double x = p.x();
	double y = p.y();
	return (20*sin(10*x*x)+(400*x*x+100)*cos(10*x*x))*sin(10*y)
			+(400*x*x+100)*cos(10*x)*sin(10*x*x)
			+(400*x*sin(10*x)-20*cos(10*x))*cos(10*x*x);
}

double absnrm(const std::vector<double>& r){
	double ret = 0;
	for (size_t i=0; i<r.size(); ++i){
		ret = std::max(ret, std::abs(r[i]));
	}
	return ret;
};

double average(const std::vector<double>& r){
	double s = 0;
	for (size_t i=0 ;i<r.size(); ++i){
		s += std::abs(r[i]);
	}
	return s;
};


class IProblemWithPrecond{
public:
	virtual size_t size() const = 0;
	virtual std::vector<double> A(const std::vector<double>& x) const = 0;
	virtual std::vector<double> residual(const std::vector<double>& x) const = 0;
	virtual std::vector<double> Minv_res(const std::vector<double>& r) const = 0;
};

class IPoissonWithPrecond: public IProblemWithPrecond{
public:
	IPoissonWithPrecond(
			std::shared_ptr<IGrid> grid,
			const std::vector<size_t>& ignored_bnd_faces={},
			const std::vector<size_t>& dirichlet_cells={}):
		_grid(grid),
		_collocations(*_grid, ignored_bnd_faces){

		// assemble SLAE
		// lhs
		LodMatrix mat(_collocations.size());
		FvmFacesDn dudn(*_grid, _collocations);
		for (size_t iface = 0; iface < grid->n_faces(); ++iface){
			size_t negative_colloc = _collocations.tab_face_colloc(iface)[0];
			size_t positive_colloc = _collocations.tab_face_colloc(iface)[1];
			if (negative_colloc == INVALID_INDEX || positive_colloc == INVALID_INDEX){
				continue;
			}
			double area = grid->face_area(iface);

			for (const std::pair<const size_t, double>& iter: dudn.linear_combination(iface)){
				size_t column = iter.first;
				double coef = area * iter.second;
				mat.add_value(positive_colloc, column, coef);
				mat.add_value(negative_colloc, column, -coef);
			}
		}
		for (size_t icolloc: _collocations.face_collocations){
			mat.set_unit_row(icolloc);
		}
		// rhs
		_rhs.resize(_collocations.size(), 0);
		for (size_t icell=0; icell < _grid->n_cells(); ++icell){
			double value = exact_rhs(_grid->cell_center(icell));
			_rhs[icell] = value * _grid->cell_volume(icell);
		}
		for (size_t icolloc: _collocations.face_collocations){
			size_t iface = _collocations.face_index(icolloc);
			_rhs[icolloc] = exact_solution(grid->face_center(iface));
		}
		// cell dirichlet
		for (size_t icell: dirichlet_cells){
			mat.set_unit_row(icell);
			_rhs[icell] = exact_solution(grid->cell_center(icell));
		}

		_lhs = mat.to_csr();
	}
	size_t size() const override{
		return _rhs.size();
	}
	std::vector<double> residual(const std::vector<double>& x) const override{
		std::vector<double> Ax = A(x);
		std::vector<double> ret = _rhs;
		for (size_t i=0; i<size(); ++i){
			ret[i] -= Ax[i];
		}
		return ret;
	};
	std::vector<double> A(const std::vector<double>& x) const override{
		return _lhs.mult_vec(x);
	};
protected:
	std::shared_ptr<IGrid> _grid;
	FvmExtendedCollocations _collocations;
	CsrMatrix _lhs;
	std::vector<double> _rhs;
};

class PoissonWithPrecond_Constant: public IPoissonWithPrecond{
public:
	PoissonWithPrecond_Constant(std::shared_ptr<IGrid> grid, double gamma=1.0): IPoissonWithPrecond(grid), _gamma(gamma){}
	std::vector<double> Minv_res(const std::vector<double>& x) const override{
		std::vector<double> ret(size());
		for (size_t i=0; i<size(); ++i){
			ret[i] = x[i] / _gamma;
		}
		return ret;
	}
private:
	const double _gamma;
};

class PoissonWithPrecond_Jacobi: public IPoissonWithPrecond{
public:
	PoissonWithPrecond_Jacobi(std::shared_ptr<IGrid> grid): IPoissonWithPrecond(grid), _diag(_lhs.diagonal()){}
	std::vector<double> Minv_res(const std::vector<double>& x) const override{
		std::vector<double> ret(size());
		for (size_t i=0; i<size(); ++i){
			ret[i] = x[i] / _diag[i];
		}
		return ret;
	}
private:
	std::vector<double> _diag;
};

class PoissonWithPrecond_Seidel: public IPoissonWithPrecond{
public:
	PoissonWithPrecond_Seidel(std::shared_ptr<IGrid> grid): IPoissonWithPrecond(grid), _diag(_lhs.diagonal()){}
	std::vector<double> Minv_res(const std::vector<double>& x) const override{
		std::vector<double> ret(size());
		for (size_t i=0; i<size(); ++i){
			ret[i] = x[i];
			for (size_t a = _lhs.addr()[i]; a < _lhs.addr()[i+1]; ++a){
				size_t j = _lhs.cols()[a];
				if (j < i){
					double a_ij = _lhs.vals()[a];
					ret[i] -= a_ij * ret[j];
				}
			}
			ret[i] /= _diag[i];
		}
		return ret;
	}
private:
	std::vector<double> _diag;
};

class PoissonWithPrecond_LocalDDM: public IPoissonWithPrecond{
public:
	PoissonWithPrecond_LocalDDM(std::shared_ptr<SubGrid> subgrid, const std::vector<double>& Xi)
			: IPoissonWithPrecond(subgrid, subgrid->domain_connection_faces(), subgrid->ghost_cells()),
			  _Xi(Xi){
		_solver.reset(new AmgcMatrixSolver(1000, 1e-12));
		_solver->set_matrix(_lhs);
		_global_size = subgrid->basic_grid()->n_cells() + subgrid->basic_grid()->boundary_faces().size();

		for (size_t icolloc: _collocations.cell_collocations){
			size_t icell = _collocations.cell_index(icolloc);
			size_t iglobal_cell = subgrid->to_global_cell(icell);
			size_t iglobal_colloc = iglobal_cell;
			_global_collocs.push_back(iglobal_colloc);
		}
		std::vector<size_t> global_bnd_faces = subgrid->basic_grid()->boundary_faces();
		for (size_t icolloc: _collocations.face_collocations){
			size_t iface = _collocations.face_index(icolloc);
			size_t iglobal_face = subgrid->to_global_face(iface);
			auto fnd = std::find(global_bnd_faces.begin(), global_bnd_faces.end(), iglobal_face);
			if (fnd == global_bnd_faces.end()){
				throw std::runtime_error("assembling error");
			}
			size_t ibnd_face = fnd - global_bnd_faces.begin();
			size_t iglobal_colloc = subgrid->basic_grid()->n_cells() + ibnd_face;
			_global_collocs.push_back(iglobal_colloc);

			auto icells = subgrid->tab_face_cell(iface);
			size_t icell = icells[0];
			if (icell == INVALID_INDEX) icell = icells[1];
			_Xi.push_back(_Xi[icell]);
		}
	}

	std::vector<double> Minv_res(const std::vector<double>& x) const override{
		std::vector<double> ret;
		std::vector<double> rhs(x);
		for (size_t i: std::dynamic_pointer_cast<SubGrid>(_grid)->ghost_cells()){
			rhs[i] = 0;
		}
		_solver->solve(rhs, ret);
		return ret;
	}

	std::vector<double> restrict_to_local(const std::vector<double>& global_x) const{
		std::vector<double> ret(_global_collocs.size());
		for (size_t i=0; i<ret.size(); ++i){
			ret[i] = global_x[_global_collocs[i]];
		}
		return ret;
	};

	std::vector<double> extend_to_global(const std::vector<double>& x, bool weighted=false) const{
		std::vector<double> ret(_global_size, 0.0);
		for (size_t i=0; i<x.size(); ++i){
			double value = x[i];
			if (weighted) value *= _Xi[i];
			ret[_global_collocs[i]] = value;
		}
		return ret;
	};

	const std::vector<double>& Xi() const {
		return _Xi;
	}
	
	std::vector<double> weigh(const std::vector<double>& u) const{
		std::vector<double> ret(u.size());
		for (size_t i=0; i<ret.size(); ++i){
			ret[i] = u[i] * _Xi[i];
		}
		return ret;
	}

	size_t global_colloc(size_t local_colloc) const {
		return  _global_collocs[local_colloc];
	}
private:
	std::vector<double> _Xi;
	std::shared_ptr<AmgcMatrixSolver> _solver;
	std::vector<size_t> _global_collocs;
	size_t _global_size;
};

class PoissonWithPrecond_DDM: public IProblemWithPrecond{
public:
	PoissonWithPrecond_DDM(std::shared_ptr<IGrid2D> grid, std::array<int, 2> part, size_t nbuf): _grid(grid){
		GridPartition gridpart = GridPartition::build_uniform<2>(grid, part, nbuf, 1);
		for (size_t idom=0; idom<gridpart.n_domains(); ++idom){
			_subproblems.emplace_back(
				gridpart.subgrid(idom),
				gridpart.restriction_weights(idom, GridPartition::Restriction::AVERAGE)
			);
		}
		_size = grid->n_cells() + grid->boundary_faces().size();
	};
	size_t size() const override{
		return _size;
	}
	std::vector<double> A(const std::vector<double>& x) const override{
		auto local_method = [this](size_t idom, const std::vector<double>& local){
			return _subproblems[idom].A(local);
		};
		return assemble_from_locals(local_method, x);
	};

	std::vector<double> residual(const std::vector<double>& x) const override{
		auto local_method = [this](size_t idom, const std::vector<double>& local){
			return _subproblems[idom].residual(local);
		};
		return assemble_from_locals(local_method, x);
	};

	std::vector<double> Minv_res(const std::vector<double>& x) const override{
		auto local_method = [this](size_t idom, const std::vector<double>& local){
			return _subproblems[idom].Minv_res(local);
		};
		return assemble_from_locals(local_method, x);
	};

	const std::vector<PoissonWithPrecond_LocalDDM>& subproblems(){
		return _subproblems;
	}

	std::vector<double> reduce_to_global(const std::vector<std::vector<double>>& local_x) const{
		std::vector<double> ret(_size, 0);
		for (size_t idom=0; idom<_subproblems.size(); ++idom){
			std::vector<double> extended_local_x = _subproblems[idom].extend_to_global(local_x[idom], true);
			for (size_t i=0; i<ret.size(); ++i){
				ret[i] += extended_local_x[i];
			}
		}
		return ret;
	};

	void exchange_bc(std::vector<std::vector<double>>& u) const {
		std::vector<double> global_u = reduce_to_global(u);
		for (size_t idom=0; idom < u.size(); ++idom){
			u[idom] = _subproblems[idom].restrict_to_local(global_u);
		}
	}

private:
	std::shared_ptr<IGrid2D> _grid;
	std::vector<PoissonWithPrecond_LocalDDM> _subproblems;
	size_t _size;

	std::vector<double> assemble_from_locals(
			std::function<std::vector<double>(size_t, const std::vector<double>&)> method,
			const std::vector<double>& x) const{

		std::vector<std::vector<double>> local_ret;
		for (size_t idom=0; idom<_subproblems.size(); ++idom){
			std::vector<double> local_x = _subproblems[idom].restrict_to_local(x);
			local_ret.push_back(method(idom, local_x));
		}
		return reduce_to_global(local_ret);
	};

};

std::vector<double> fixed_point_iterator(std::shared_ptr<IProblemWithPrecond> prob, size_t max_it){
	std::vector<double> x(prob->size(), 0.0);
	std::vector<double> r = prob->residual(x);

	for (size_t it=0; it < max_it; ++it){
		double absr = absnrm(r);
		std::cout << "it=" << it <<", nrm=" << absr << std::endl;
		if (absr < 1e-6){
			break;
		}

		std::vector<double> r_prime = prob->Minv_res(r);
		for (size_t i=0; i<prob->size(); ++i){
			x[i] += r_prime[i];
		}
		r = prob->residual(x);
	}
	return x;
}

std::vector<double> bicgstab(std::shared_ptr<IProblemWithPrecond> prob, size_t max_it){
	std::vector<double> x(prob->size(), 0.0);
	std::vector<double> r_prime = prob->Minv_res(prob->residual(x));
	std::vector<double> r_tilde = r_prime;
	std::vector<double> p_prime(prob->size(), 0);
	std::vector<double> v_prime(prob->size(), 0);
	double alpha = 0;
	double rho = 1;
	double omega = 1;

	auto scalar_product = [](const std::vector<double>& u, const std::vector<double>& v)->double{
		return std::inner_product(u.begin(), u.end(), v.begin(), 0.0);
	};

	for (size_t it=0; it < max_it; ++it){
		// 1. rho
		double rho_next = scalar_product(r_tilde, r_prime);
	
		// 2. beta
		double beta = rho_next / rho * alpha / omega;
		rho = rho_next;

		// 3. p'
		for (size_t i=0; i<p_prime.size(); ++i){
			p_prime[i] = r_prime[i] + beta * (p_prime[i] - omega * v_prime[i]);
		}

		// 4. v'
		v_prime = prob->Minv_res(prob->A(p_prime));

		// 5. alpha
		alpha = rho / scalar_product(v_prime, r_tilde);

		// 6. s'
		std::vector<double> s_prime(r_prime.size());
		for (size_t i=0; i<s_prime.size(); ++i){
			s_prime[i] = r_prime[i] - alpha * v_prime[i];
		}

		// 7. t'
		std::vector<double> t_prime = prob->Minv_res(prob->A(s_prime));

		// 8. omega
		omega = scalar_product(t_prime, s_prime) / scalar_product(t_prime, t_prime);

		// 9. x, r_prime
		for (size_t i=0; i<prob->size(); ++i){
			x[i] += alpha * p_prime[i] + omega * s_prime[i];
			r_prime[i] = s_prime[i] - omega * t_prime[i];
		}

		// 10. r
		double absr = absnrm(prob->residual(x));
		std::cout << "it=" << it <<", nrm=" << absr << std::endl;
		if (absr < 1e-6){
			break;
		}
	}
	return x;
}

}

TEST_CASE("Poisson-fvm, fixed point iterations", "[poisson2-fvm-fpi]"){
	std::cout << std::endl << "--- cfd24_test [poisson2-fvm-fpi] --- " << std::endl;

	std::string grid_fn = test_directory_file("trigrid_500.vtk");
	auto grid = std::make_shared<UnstructuredGrid2D>(UnstructuredGrid2D::vtk_read(grid_fn));

	// Jacobi preconditioner
	{
		std::cout << "jacobi preconditioner" << std::endl;
		std::shared_ptr<IProblemWithPrecond> prob(new PoissonWithPrecond_Jacobi(grid));
		std::vector<double> x = fixed_point_iterator(prob, 10);
		CHECK(absnrm(prob->residual(x)) == Approx(0.368666).margin(1e-5));
	}

	// Seidel preconditioner
	{
		std::cout << "seidel preconditioner" << std::endl;
		std::shared_ptr<IProblemWithPrecond> prob(new PoissonWithPrecond_Seidel(grid));
		std::vector<double> x = fixed_point_iterator(prob, 10);
		CHECK(absnrm(prob->residual(x)) == Approx(0.275442185).margin(1e-5));
	}

	// DDM preconditioner
	{
		std::cout << "ddm preconditioner" << std::endl;
		std::shared_ptr<IProblemWithPrecond> prob(new PoissonWithPrecond_DDM(grid, {1, 2}, 2));
		std::vector<double> x = fixed_point_iterator(prob, 10);
		CHECK(absnrm(prob->residual(x)) == Approx(2.59956e-05).margin(1e-9));
	}
}

TEST_CASE("Poisson-fvm, bicgstab", "[poisson2-fvm-bicgstab]"){
	std::cout << std::endl << "--- cfd24_test [poisson2-fvm-bicgstab] --- " << std::endl;

	std::string grid_fn = test_directory_file("trigrid_500.vtk");
	auto grid = std::make_shared<UnstructuredGrid2D>(UnstructuredGrid2D::vtk_read(grid_fn));

	// no preconditioner
	{
		std::cout << "no preconditioner" << std::endl;
		std::shared_ptr<IProblemWithPrecond> prob(new PoissonWithPrecond_Constant(grid));
		std::vector<double> x = bicgstab(prob, 10);

		CHECK(absnrm(prob->residual(x)) == Approx(0.0277606).margin(1e-5));
	}
	
	// DDM preconditioner
	{
		std::cout << "ddm preconditioner" << std::endl;
		std::shared_ptr<PoissonWithPrecond_DDM> prob(new PoissonWithPrecond_DDM(grid, {1, 2}, 2));
		std::vector<double> x = bicgstab(prob, 10);
		CHECK(absnrm(prob->residual(x)) == Approx(6.56013e-09).margin(1e-12));
	}
}
