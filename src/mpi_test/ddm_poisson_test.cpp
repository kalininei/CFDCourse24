#include "mpi_test.hpp"
#include "cfd24/grid/unstructured_grid2d.hpp"

using namespace cfd;

void schwarz_ddm_poisson_test(){
	int mpi_rank, mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	std::string grid_fn = test_directory_file("tetragrid_500.vtk");
	auto grid = std::make_shared<UnstructuredGrid2D>(UnstructuredGrid2D::vtk_read(grid_fn, true));
	if (mpi_rank == 0){
		CHECK(grid != nullptr);
		CHECK(grid->n_cells() == 498);
		std::cout << "Reading grid: " << grid->n_cells() << " cells" << std::endl;
	}
}
