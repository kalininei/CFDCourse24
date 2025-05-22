set (SRC
	cfd24.cpp

	mat/csrmat.cpp
	mat/lodmat.cpp
	mat/densemat.cpp
	mat/sparse_matrix_solver.cpp

	geom/simplex.cpp
	geom/searcher.cpp
	geom/jacobi.cpp

	grid/vtk.cpp
	grid/i_grid.cpp
	grid/grid1d.cpp
	grid/regular_grid2d.cpp
	grid/unstructured_grid2d.cpp
	grid/grid_partition.cpp

	numeric_integration/quadrature.cpp
	numeric_integration/segment_quadrature.cpp
	numeric_integration/square_quadrature.cpp
	numeric_integration/triangle_quadrature.cpp

	fvm/fvm_assembler.cpp
	fvm/fvm_dpdn_boundary.cpp

	fem/fem_assembler.cpp
	fem/fem_numeric_integrals.cpp
        fem/fem_sorted_cell_info.cpp
	fem/elem1d/segment_linear.cpp
	fem/elem1d/segment_quadratic.cpp
	fem/elem1d/segment_cubic.cpp
	fem/elem2d/triangle_linear.cpp
	fem/elem2d/quadrangle_linear.cpp
	fem/elem2d/triangle_quadratic.cpp
	fem/elem2d/quadrangle_quadratic.cpp
	fem/elem2d/triangle_cubic.cpp

	debug/printer.cpp
	debug/saver.cpp
	debug/tictoc.cpp
)

set (HEADERS
	cfd24.hpp
	cfd_common.hpp
	macros.hpp

	mat/i_mat.hpp
	mat/i_sparse_mat.hpp
	mat/csrmat.hpp
	mat/lodmat.hpp
	mat/sparse_matrix_solver.hpp
	mat/densemat.hpp

	geom/point.hpp
	geom/simplex.hpp
	geom/searcher.hpp
	geom/jacobi.hpp
	geom/i_point_function.hpp

	grid/vtk.hpp
	grid/i_grid.hpp
	grid/grid1d.hpp
	grid/regular_grid2d.hpp
	grid/unstructured_grid2d.hpp

	numeric_integration/quadrature.hpp
	numeric_integration/segment_quadrature.hpp
	numeric_integration/square_quadrature.hpp
	numeric_integration/triangle_quadrature.hpp

	fvm/fvm_assembler.hpp
	fvm/fvm_dpdn_boundary.hpp

	fem/fem_assembler.hpp
	fem/fem_element.hpp
	fem/fem_numeric_integrals.hpp
        fem/fem_sorted_cell_info.hpp
	fem/elem1d/segment_linear.hpp
	fem/elem1d/segment_quadratic.hpp
	fem/elem1d/segment_cubic.hpp
	fem/elem2d/triangle_linear.hpp
	fem/elem2d/quadrangle_linear.hpp
	fem/elem2d/triangle_quadratic.hpp
	fem/elem2d/quadrangle_quadratic.hpp
	fem/elem2d/triangle_cubic.hpp

	debug/printer.hpp
	debug/saver.hpp
	debug/tictoc.hpp
)

add_library(${CFD24_TARGET} STATIC ${SRC} ${HEADERS})
