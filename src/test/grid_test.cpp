#include "cfd24_test.hpp"
#include "cfd24/grid/grid1d.hpp"
#include "cfd24/grid/regular_grid2d.hpp"
#include "cfd24/grid/unstructured_grid2d.hpp"
#include "cfd24/grid/vtk.hpp"
#include "cfd24/grid/grid_partition.hpp"
#include "utils/filesystem.hpp"

using namespace cfd;


TEST_CASE("Grid1d", "[grid1]"){
	Grid1D grid(0, 1, 3);

	CHECK(grid.n_points() == 4);
	CHECK(grid.n_cells() == 3);
	CHECK(grid.n_faces() == 4);

	grid.save_vtk("out1.vtk");
	VtkUtils::add_point_data(std::vector<double>{0, 1, 2, 3}, "data1", "out1.vtk");
}

TEST_CASE("RegularGrid2d", "[reggrid2]"){
	RegularGrid2D grid(0, 1, 1, 3, 3, 2);

	CHECK(grid.n_points() == 12);
	CHECK(grid.n_cells() == 6);

	CHECK(grid.to_split_point_index(8)[0] == 0);
	CHECK(grid.to_split_point_index(8)[1] == 2);
	CHECK(grid.to_split_point_index(11)[0] == 3);
	CHECK(grid.to_split_point_index(11)[1] == 2);
	CHECK(grid.to_linear_point_index({0, 0}) == 0);
	CHECK(grid.to_linear_point_index({2, 0}) == 2);
	CHECK(grid.to_linear_point_index({3, 1}) == 7);
	CHECK(grid.cell_center(0).x() == Approx(0.1666).margin(1e-2));
	CHECK(grid.cell_center(0).y() == Approx(1.5).margin(1e-2));
	CHECK(grid.cell_center(4).x() == Approx(0.5).margin(1e-2));
	CHECK(grid.cell_center(4).y() == Approx(2.5).margin(1e-2));

	grid.save_vtk("out2.vtk");
	VtkUtils::add_point_data(std::vector<double>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, "data1", "out2.vtk");
}

TEST_CASE("RegularGrid2d-nonuniform", "[reggrid2-nonuni]"){
	std::vector<double> x{10, 12, 15, 16};
	std::vector<double> y{-10, -8, -3};
	RegularGrid2D grid(x, y);

	CHECK(grid.Lx() == Approx(6));
	CHECK(grid.Ly() == Approx(7));
	CHECK(grid.cell_center(4).x() == Approx(13.5));
	CHECK(grid.cell_center(4).y() == Approx(-5.5));

	CHECK(grid.n_faces() == 17);
	CHECK(grid.face_center(8).x() == Approx(15.5));
	CHECK(grid.face_center(8).y() == Approx(-3));
	CHECK(grid.face_center(13).x() == Approx(10));
	CHECK(grid.face_center(13).y() == Approx(-5.5));
	CHECK(grid.face_center(16).x() == Approx(16));
	CHECK(grid.face_center(16).y() == Approx(-5.5));
	CHECK(grid.face_normal(4).x() == 0.0);
	CHECK(grid.face_normal(4).y() == 1.0);
	CHECK(grid.face_normal(10).x() == 1.0);
	CHECK(grid.face_normal(10).y() == 0.0);
	CHECK(grid.face_area(6) == Approx(2));
	CHECK(grid.face_area(12) == Approx(2));
}

TEST_CASE("UnstructuredGrid2d", "[unstructured-grid2]"){
	std::vector<double> x{10, 12, 15, 16};
	std::vector<double> y{-10, -8, -3};
	RegularGrid2D grid(x, y);
	UnstructuredGrid2D ugrid(grid);

	CHECK(ugrid.n_cells() == 6);
	CHECK(ugrid.n_points() == 12);
	CHECK(ugrid.n_faces() == 17);


	CHECK(ugrid.cell_center(4).x() == Approx(13.5));
	CHECK(ugrid.cell_center(4).y() == Approx(-5.5));

	CHECK(ugrid.face_center(16).x() == Approx(15.5));
	CHECK(ugrid.face_center(16).y() == Approx(-3));
	CHECK(ugrid.face_center(8).x() == Approx(10));
	CHECK(ugrid.face_center(8).y() == Approx(-5.5));
	CHECK(ugrid.face_center(13).x() == Approx(16));
	CHECK(ugrid.face_center(13).y() == Approx(-5.5));
	CHECK(ugrid.face_normal(9).x() == 0.0);
	CHECK(ugrid.face_normal(9).y() == -1.0);
	CHECK(ugrid.face_normal(3).x() == 1.0);
	CHECK(ugrid.face_normal(3).y() == 0.0);
	CHECK(ugrid.face_area(14) == Approx(2));
	CHECK(ugrid.face_area(6) == Approx(2));
}

TEST_CASE("UnstructuredGrid2d, read from vtk", "[unstructured2-vtk]"){
	std::string fn = test_directory_file("hexagrid_50.vtk");
	UnstructuredGrid2D grid = UnstructuredGrid2D::vtk_read(fn);

	CHECK(grid.n_points() == 106);
	CHECK(grid.n_cells() == 52);
	CHECK(grid.cell_volume(0) == Approx(0.00595238).margin(1e-6));
	CHECK(grid.cell_volume(34) == Approx(0.0238095).margin(1e-6));
	CHECK(grid.tab_face_cell(53)[0] == 5);
	CHECK(grid.tab_face_cell(53)[1] == 21);
	CHECK(grid.cell_center(36).x() == Approx(0.714286).margin(1e-6));
	CHECK(grid.cell_center(36).y() == Approx(0.583333).margin(1e-6));
	CHECK(grid.cell_center(0).x() == Approx(0.037037037).margin(1e-6));
	CHECK(grid.cell_center(0).y() == Approx(0.037037037).margin(1e-6));
	CHECK(grid.cell_center(7).y() == Approx((grid.point(12).y() + grid.point(13).y())/2.0).margin(1e-6));

	CHECK_NOTHROW(grid.save_vtk("hexa.vtk"));
}

TEST_CASE("Load grid from windows build", "[unstructured2-win]"){
	auto grid = UnstructuredGrid2D::vtk_read(test_directory_file("pebigrid_from_win.vtk"));
	CHECK(grid.n_cells() == 1022);
}

TEST_CASE("Grid partition, 2d", "[grid-partition-2d]"){
	std::string grid_fn = test_directory_file("tetragrid_500.vtk");
	auto grid = std::make_shared<UnstructuredGrid2D>(UnstructuredGrid2D::vtk_read(grid_fn));
	{
		// no buffer, no ghost
		GridPartition gridpart = GridPartition::build_uniform<2>(grid, {2, 1}, 0, 0);
		CHECK(gridpart.n_domains() == 2);
		std::shared_ptr<SubGrid> sgrid0 = gridpart.subgrid(0);
		std::shared_ptr<SubGrid> sgrid1 = gridpart.subgrid(1);

		CHECK(grid->n_cells() == 498);
		CHECK(sgrid0->n_cells() == 252);
		CHECK(sgrid1->n_cells() == 246);

		CHECK(grid->n_points() == 472);
		CHECK(sgrid0->n_points() == 252);
		CHECK(sgrid1->n_points() == 246);
	}
	{
		// no buffer, with ghost
		GridPartition gridpart = GridPartition::build_uniform<2>(grid, {1, 2}, 0, 1);
		std::shared_ptr<SubGrid> sgrid0 = gridpart.subgrid(0);
		std::shared_ptr<SubGrid> sgrid1 = gridpart.subgrid(1);

		CHECK(grid->n_cells() == 498);
		CHECK(sgrid0->n_cells() == 283);
		CHECK(sgrid1->n_cells() == 278);

		CHECK(grid->n_points() == 472);
		CHECK(sgrid0->n_points() == 283);
		CHECK(sgrid1->n_points() == 278);

		CHECK(sgrid0->to_global_cell(77) == 154);
		CHECK(sgrid0->to_global_cell(254) == 16);
		CHECK(sgrid1->to_global_cell(165) == 344);
		CHECK(sgrid1->to_global_cell(274) == 326);

		CHECK(grid->face_center(sgrid0->to_global_face(189)) == sgrid0->face_center(189));
		CHECK(grid->face_center(sgrid1->to_global_face(78)) == sgrid1->face_center(78));
		size_t dcf0 = sgrid0->domain_connection_faces()[3];
		size_t dcf = sgrid0->to_global_face(dcf0);
		CHECK((sgrid0->tab_face_cell(dcf0)[0] == INVALID_INDEX || sgrid0->tab_face_cell(dcf0)[1] == INVALID_INDEX));
		CHECK((grid->tab_face_cell(dcf)[0] != INVALID_INDEX && grid->tab_face_cell(dcf)[1] != INVALID_INDEX));

		std::vector<double> r0 = gridpart.restriction_weights(0, GridPartition::Restriction::UNITY);
		std::vector<double> r1 = gridpart.restriction_weights(0, GridPartition::Restriction::AVERAGE);

		CHECK(r0[89] == 1.0);
		CHECK(r0[257] == 0.0);
		CHECK(r1[89] == 1.0);
		CHECK(r1[257] == 0.0);
	}

	{
		// buffer=2, whith ghosts
		GridPartition gridpart = GridPartition::build_uniform<2>(grid, {1, 2}, 2, 1);
		std::shared_ptr<SubGrid> sgrid0 = gridpart.subgrid(0);
		std::shared_ptr<SubGrid> sgrid1 = gridpart.subgrid(1);

		std::vector<double> r0 = gridpart.restriction_weights(0, GridPartition::Restriction::UNITY);
		std::vector<double> r1 = gridpart.restriction_weights(0, GridPartition::Restriction::AVERAGE);

		CHECK(r0[343] == 0.0);
		CHECK(r0[306] == 1.0);
		CHECK(r0[142] == 1.0);
		CHECK(r0[110] == 1.0);

		CHECK(r1[343] == 0.0);
		CHECK(r1[306] == 0.5);
		CHECK(r1[142] == 0.5);
		CHECK(r1[110] == 1.0);


		grid->save_vtk("g.vtk");
		sgrid0->save_vtk("g0.vtk");
		sgrid1->save_vtk("g1.vtk");
		VtkUtils::add_cell_data(r0, "unity_restrict", "g0.vtk");
		VtkUtils::add_cell_data(r1, "average_restrict", "g0.vtk");
	}
	{
		// 2x2 with buffer and ghosts
		GridPartition gridpart = GridPartition::build_uniform<2>(grid, {2, 2}, 2, 1);
		std::shared_ptr<SubGrid> sgrid0 = gridpart.subgrid(0);
		std::shared_ptr<SubGrid> sgrid1 = gridpart.subgrid(1);
		std::shared_ptr<SubGrid> sgrid2 = gridpart.subgrid(2);
		std::shared_ptr<SubGrid> sgrid3 = gridpart.subgrid(3);
		std::vector<double> r1 = gridpart.restriction_weights(0, GridPartition::Restriction::AVERAGE);

		sgrid0->save_vtk("g0.vtk");
		VtkUtils::add_cell_data(r1, "average_restrict", "g0.vtk");
		std::vector<double> bli(sgrid0->n_cells());
		for (size_t i=0; i<sgrid0->n_cells(); ++i){
			bli[i] = (double)sgrid0->buffer_layer_index(i);
		}
		VtkUtils::add_cell_data(bli, "bli", "g0.vtk");

		CHECK(r1[189] == 1.0/4.0);
		CHECK(r1[182] == 1.0/3.0);
		CHECK(r1[149] == 1.0/2.0);
		CHECK(r1[202] == 0.0);
		CHECK(r1[87] == 1.0);
	}

}
