#include "cfd24_test.hpp"
#include "cfd24/geom/point.hpp"
#include "cfd24/grid/regular_grid2d.hpp"
#include "cfd24/grid/unstructured_grid2d.hpp"
#include "cfd24/fvm/fvm_assembler.hpp"
#include "cfd24/grid/vtk.hpp"
#include "cfd24/debug/saver.hpp"
#include "utils/filesystem.hpp"

using namespace cfd;

namespace{

double func(Point p){
	double x = p.x();
	double y = p.y();
	return cos(10*x) + sin(5*x*y) + cos(10*y);
}

double dfdx(Point p){
	double x = p.x();
	double y = p.y();
	return -10*sin(10*x) + 5*y*cos(5*x*y);
}

double dfdy(Point p){
	double x = p.x();
	double y = p.y();
	return 5*x*cos(5*x*y) - 10*sin(10*y);
}

}

TEST_CASE("Fvm gradient-least squares", "[fvm-gradient-ls]"){
	std::shared_ptr<IGrid> grid;
	int test_no;
	SECTION("Regular grid"){
		test_no = 0;
		grid = std::make_shared<RegularGrid2D>(0, 1, 0, 1, 20, 20);
	}
	SECTION("Unstructured grid"){
		test_no = 1;
		std::string fn = test_directory_file("tetragrid_500.vtk");
		grid = std::make_shared<UnstructuredGrid2D>(UnstructuredGrid2D::vtk_read(fn, true));
	}

	FvmExtendedCollocations collocations(*grid);
	std::vector<double> f;
	std::vector<Vector> exact_grad_f;
	for (Point p: collocations.points){
		f.push_back(func(p));
	}
	for (size_t i=0; i<grid->n_cells(); ++i){
		Point p = grid->cell_center(i);
		exact_grad_f.push_back(Vector{dfdx(p), dfdy(p)});
	}
	std::vector<Vector> grad_f = FvmCellGradient_LeastSquares(*grid, collocations).compute(f);

	double sum = 0;
	for (size_t i=0; i<grid->n_cells(); ++i){
		double diff_x = grad_f[i].x() - exact_grad_f[i].x();
		double diff_y = grad_f[i].y() - exact_grad_f[i].y();
		double diff2 = diff_x*diff_x + diff_y*diff_y;
		sum += grid->cell_volume(i) * diff2;
	}
	double residual = std::sqrt(sum);

	switch (test_no){
		case 0: CHECK(residual == Approx(0.855929).margin(1e-6)); break;
		case 1: CHECK(residual == Approx(0.903186).margin(1e-6)); break;
	}
}

TEST_CASE("Fvm gradient gauss", "[fvm-gradient-gauss]"){
	std::shared_ptr<IGrid> grid;
	int test_no;
	SECTION("Regular grid"){
		test_no = 0;
		grid = std::make_shared<RegularGrid2D>(0, 1, 0, 1, 20, 20);
	}
	SECTION("Unstructured grid"){
		test_no = 1;
		std::string fn = test_directory_file("tetragrid_500.vtk");
		grid = std::make_shared<UnstructuredGrid2D>(UnstructuredGrid2D::vtk_read(fn, true));
	}

	FvmExtendedCollocations collocations(*grid);
	std::vector<double> f;
	std::vector<Vector> exact_grad_f;
	for (Point p: collocations.points){
		f.push_back(func(p));
	}

	for (size_t i=0; i<grid->n_cells(); ++i){
		Point p = grid->cell_center(i);
		exact_grad_f.push_back(Vector{dfdx(p), dfdy(p)});
	}
	std::vector<Vector> grad_f = FvmCellGradient_Gauss(*grid, collocations).compute(f);

	double sum = 0;
	for (size_t i=0; i<grid->n_cells(); ++i){
		double diff_x = grad_f[i].x() - exact_grad_f[i].x();
		double diff_y = grad_f[i].y() - exact_grad_f[i].y();
		double diff2 = diff_x*diff_x + diff_y*diff_y;
		sum += grid->cell_volume(i) * diff2;
	}
	double residual = std::sqrt(sum);

	switch (test_no){
		case 0: CHECK(residual == Approx(0.4810939144).margin(1e-6)); break;
		case 1: CHECK(residual == Approx(2.3812139187).margin(1e-6)); break;
	}
}
