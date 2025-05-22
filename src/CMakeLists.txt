set(CFD24_TARGET cfd24)

add_subdirectory(cfd24)
add_subdirectory(test)
if (MPI_TESTS)
	add_subdirectory(mpi_test)
endif()
