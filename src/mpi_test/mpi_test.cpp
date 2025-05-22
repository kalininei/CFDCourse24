#include "mpi_test.hpp"

#ifndef TEST_DIRECTORY
#define TEST_DIRECTORY "./"
#endif

std::string test_directory_file(std::string path){
	// check current dir
	{
		std::ifstream ofs(path);
		if (ofs) return path;
	}
	{
		std::ifstream ofs(TEST_DIRECTORY + path);
		if (ofs) return TEST_DIRECTORY + path;
	}
	throw std::runtime_error(path + " was not found");
}

void simple_mpi_test(){
	int mpi_rank, mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	// складывает индексы рангов: 0 + 1 + ... + (mpi_size-1).
	// результат кладет в переменную sumv на ранге 0.
	int v = mpi_rank;
	int sumv;
	MPI_Reduce(
		&v,             // [in] Указатель на отправляемые данные (на каждом процессе)
		&sumv,          // [out] Указатель на буфер для результата (только на root)
		1,              // [in] Количество элементов в sendbuf
		MPI_INT,        // [in] Тип данных (MPI_INT, MPI_FLOAT и т. д.)
		MPI_SUM,        // [in] Операция редукции (MPI_SUM, MPI_MAX и т. д.)
		0,              // [in] Ранг процесса-получателя результата
		MPI_COMM_WORLD  // [in] Коммуникатор (обычно MPI_COMM_WORLD)
	);

	if (mpi_rank == 0){
		int sum = 0;
		for (size_t r=0; r < mpi_size; ++r) sum += r;
		CHECK(sum == sumv);
	}
}

int main(int argc, char* argv[]){
	MPI_Init(&argc, &argv);

	int mpi_rank, mpi_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	simple_mpi_test();
	schwarz_ddm_poisson_test();

	if (mpi_rank == 0){
		std::cout << "DONE" << std::endl;
	}

	MPI_Finalize();
}
