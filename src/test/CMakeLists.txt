set (SRC
	cfd24_test.cpp
	grid_test.cpp
	mat_test.cpp
	geom_test.cpp
	poisson_solve_test.cpp
	transport_solve_test.cpp
	transport_fvm_solve_test.cpp
	transport_fem_solve_test.cpp
	cavern_2d_simple_test.cpp
	cavern_2d_fvm_simple_test.cpp
	cavern_2d_fem_simple_test.cpp
	cavern_2d_fem_cbs_test.cpp
	obstacle_2d_simple_test.cpp
	obstacle_nonstat_2d_simple_test.cpp
	poiseuille_2d_simple_test.cpp
	linear_2d_simple_test.cpp
	poisson_fvm_solve_test.cpp
	fvm_gradient_test.cpp
	cylinder_fvm_simple_test.cpp
	poisson_fem_solve_test.cpp
	poisson_hermite_solve_test.cpp
	interp_test.cpp
	convdiff_fem_test.cpp
	cavity_fvm_rans_test.cpp

	utils/filesystem.cpp
	utils/vecmat.cpp
)

set (HEADERS
	cfd24_test.hpp

	utils/filesystem.hpp
	utils/vecmat.hpp
)

add_executable(cfd24_test ${SRC} ${HEADERS})
target_link_libraries(cfd24_test PRIVATE ${CATCH2_FRAMEWORK} ${CFD24_TARGET})
target_include_directories(cfd24_test PRIVATE ${CMAKE_SOURCE_DIR}/src/test)
target_compile_definitions(cfd24_test PUBLIC TEST_DIRECTORY="${CMAKE_SOURCE_DIR}/test_data/")
