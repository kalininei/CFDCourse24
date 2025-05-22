set (SRC
	mpi_test.cpp
	ddm_poisson_test.cpp
)

set (HEADERS
	mpi_test.hpp
)

add_executable(mpi_cfd24_test ${SRC} ${HEADERS})
target_include_directories(mpi_cfd24_test PRIVATE ${CMAKE_SOURCE_DIR}/src/mpi_test)
target_link_libraries(mpi_cfd24_test PRIVATE ${CFD24_TARGET})
target_compile_definitions(mpi_cfd24_test PUBLIC TEST_DIRECTORY="${CMAKE_SOURCE_DIR}/test_data/")
target_link_libraries(mpi_cfd24_test PUBLIC MPI::MPI_CXX)
