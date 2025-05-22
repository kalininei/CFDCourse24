// =======================================================================================================================
// unew[i] - u[i]        u[i] - u[i-1]
// -------------   +  V  ------------- = 0
//     tau                    h 

// serial version
void explicit_step(double tau, double h, double V,
                   double bc_left, std::vector<double>& u){

	std::vector<double> unew(u.size());
	unew[0] = bc_left;
	for (size_t i=1; i<u.size(); ++i){
		unew[i] = u[i] - tau * V / h * (u[i] - u[i-1]);
	}
	std::swap(unew, u);
}

// openmp version
void explicit_step(double tau, double h, double V,
                  double bc_left, std::vector<double>& u){

	std::vector<double> unew(u.size());
	unew[0] = bc_left;
	#pragma omp parallel for
	for (size_t i=1; i<u.size(); ++i){
		unew[i] = u[i] - tau * V / h * (u[i] - u[i-1]);
	}
	std::swap(unew, u);
}

// std::thread version
void explicit_step(double tau, double h, double V,
                   double bc_left, std::vector<double>& u) {

	constexpr size_t num_threads = 4;  // 4 потока
	std::vector<double> unew(u.size());
	unew[0] = bc_left;

	// тело цикла
	auto worker = [&](size_t start, size_t end) {
		for (size_t i = start; i < end; ++i) {
			unew[i] = u[i] - tau * V / h * (u[i] - u[i-1]);
		}
	};

	// Разделяем работу между потоками: первая четверть узлов идёт в первый поток,
	// вторая -- во второй и т.д.
	size_t chunk_size = (u.size() - 1) / num_threads;
	std::vector<std::thread> threads;
	for (size_t t = 0; t < num_threads; ++t) {
		size_t start = t * chunk_size + 1;
		size_t end = (t == num_threads - 1) ? u.size()
		                                    : (t + 1) * chunk_size + 1;
		threads.emplace_back(worker, start, end);
	}

	// Дожидаемся завершения всех потоков
	for (auto& thread : threads) {
		thread.join();
	}

	std::swap(unew, u);
}


// ===========================================================================================
// std::async
std::vector<double> compute_u_star(const std::vector<double>& uold,
                                   const std::vector<double>& vold,
                                   const std::vector<double>& wold){
	//...
}
std::vector<double> compute_v_star(...
std::vector<double> compute_w_star(...

std::vector<double> compute_p_prime(const std::vector<double>& u_star,
                                    const std::vector<double>& v_star,
                                    const std::vector<double>& w_star){
	//...
}

void simple_step(){
	std::future<std::vector<double>> u_star = std::async(
			std::launch::async, compute_u_star, u_old, v_old, w_old);
	std::future<std::vector<double>> v_star = std::async(
			std::launch::async, compute_v_star, u_old, v_old, w_old);
	std::future<std::vector<double>> w_star = std::async(
			std::launch::async, compute_w_star, u_old, v_old, w_old);

	std::vector<double> p_prime = compute_p_prime(
			u_star.get(), v_star.get(), w_star.get());
	//...
};//


// ==========================================================================================
// open mpi
int main(int argc, char** argv){
	int mpi_rank, mpi_size;
	
	// Инициализация MPI
	MPI_Init(&argc, &argv);
	// получить ранг и общее количество рангов
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);  
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	// построить сетку в области
	Grid grid = build_grid();

	// глобальные вектора
	std::vector<double> rhs = build_rhs(grid); // includes global bc
	std::vector<double> u = std::vector<double>(grid.n_cells(), 0);

	// построить локальную сетку в домене
	SubGrid local_grid = grid.subgrid(mpi_rank);

	// построить локальную матрицу левой части
	Matrix local_mat = build_lhs(local_grid); // includes global and local bc
	MatrixSolver local_solver = build_local_solver(local_mat);
	// локальные вектора
	std::vector<double> local_rhs = local_grid.project(rhs);
	std::vector<double> local_u = local_grid.project(u);

	for (size_t it=0; it < 1'000; ++it){
		// установка граничных условий в ghost-узлах
		for (size_t local_ghost: local_grid.ghost_cells()){
			size_t global_ghost = local_grid.to_global(local_ghost);
			local_rhs[local_ghost] = u[global_ghost];
		}
	
		// невязка и условие выхода
		double local_residual = local_mat.residual(local_u, local_rhs);
		double global_residual;
		MPI_Allreduce(
			&local_residual,  // Отправляемый буфер
			&global_residual, // Принимающий буфер 
			1,                // Количество элементов
			MPI_DOUBLE,       // Тип данных
			MPI_MAX,          // Операция
			MPI_COMM_WORLD);
		if (global_residual < 1e-6) {
			MPI_Barrier(MPI_COMM_WORLD); // Синхронизация перед выходом
			break;
		}

		// Решение локальной системы
		local_u = local_solver.solve(local_rhs);

		// Расширяем локальный вектор до глобального с помощью оператора продолжения
		std::vector<double> local_u_extended = local_grid.extend_with_xi(local_u);

		// Записать итог
		MPI_Allreduce(
			local_u_extended.data(),  // Отправляемый буфер
			u.data(),                 // Принимающий буфер
			u.size(),                 // Количество элементов
			MPI_DOUBLE,               // Тип данных
			MPI_SUM,                  // Операция
			MPI_COMM_WORLD);

		// информация
		if (mpi_rank == 0){
			std::cout << it << " res= " << global_residual << std::endl;
			grid.save_data(u, "data" + std::to_string(it));
		}
	}

	MPI_Finalize();
}
