// CeramicComposite_v1.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//

#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include <iostream>
#include <cmath>
#include <ctime>
#include <omp.h>
#include<locale.h>


using namespace boost::numeric::ublas;


//структура данных "трехдиагональная матрица" - каждая диагональ хранится в виде отдельного вектора
template<class __DataType>
class TridiagonalMatrix {

	int mat_size;
	vector<__DataType> main_diag_vals;
	vector<__DataType> low_diag_vals;
	vector<__DataType> up_diag_vals;
public:
	TridiagonalMatrix() :mat_size(0), main_diag_vals(zero_vector<__DataType>(0)), low_diag_vals(zero_vector<__DataType>(0)),
		up_diag_vals(zero_vector<__DataType>(0)) {};
	TridiagonalMatrix(int p_mat_size, vector<__DataType> p_main_diag_vals, vector<__DataType> p_low_diag_vals, vector<__DataType> p_up_diag_vals):mat_size(p_mat_size) {
		main_diag_vals = p_main_diag_vals;
		low_diag_vals = p_low_diag_vals;
		up_diag_vals = p_up_diag_vals;
	}
	~TridiagonalMatrix() {};
	TridiagonalMatrix<__DataType>& operator=(const TridiagonalMatrix& p_mat) {
		if (this == &p_mat) return (*this);
		mat_size = p_mat.mat_size;
		main_diag_vals = p_mat.main_diag_vals;
		low_diag_vals = p_mat.low_diag_vals;
		up_diag_vals = p_mat.up_diag_vals;
		return (*this);
	}
	// доступ к элементам
	__DataType At(int idx_i, int idx_j) {
		if ((idx_i >= mat_size) || (idx_j >= mat_size)) {
			throw std::range_error("Missing key");
		}
		else if (idx_i == idx_j) {
			return main_diag_vals(idx_i);
		}
		else if (idx_i == idx_j - 1) {
			return up_diag_vals(idx_i);
		}
		else if (idx_i == idx_j + 1) {
			return low_diag_vals(idx_i - 1);
		}
		else {
			return 0;
		}
	}
	int size() {
		return mat_size;
	}
};


// метод прогонки с использованием описанной структуры данных
template<class __DataType>
vector<__DataType> TridiagonalAlgorithm(TridiagonalMatrix<__DataType> A, vector<__DataType> b) {
	vector<__DataType> X = zero_vector<__DataType>(A.size());
	vector<__DataType> L = zero_vector<__DataType>(A.size() - 1);
	vector<__DataType> M = zero_vector<__DataType>(A.size());
	L(0) = -A.At(0, 1) / A.At(0, 0);
	M(0) = b(0) / A.At(0, 0);
	int i, j, k = X.size() - 1;
	double time;
	clock_t start = clock();
	omp_set_num_threads(4);
#pragma omp parallel
	{
#pragma omp  for private(i) shared(L)
		for (i = 1; i < L.size(); ++i)
		{
			L(i) = -A.At(i, i + 1) / (L(i - 1) * A.At(i, i - 1) + A.At(i, i));
		}
#pragma omp for private(j) shared(M)
		for (j = 1; j < M.size(); ++j)
		{
			M(j) = (b(j) - M(j - 1) * A.At(j, j - 1)) / (L(j - 1) * A.At(j, j - 1) + A.At(j, j));
		}
	}
	X(k) = M(k);
	k--;
	while (k >= 0) {
		X(k) = L(k) * X(k + 1) + M(k);
		k--;
	}
	clock_t finish = clock();
	time = (double)(finish - start) / CLOCKS_PER_SEC;
	//std::cout << "Время, затраченное на метод прогонки: " << time << std::endl;
	return X;
};

//структура данных "пятидиагональная матрица" - каждая диагональ хранится в виде отдельного вектора
template<class __DataType>
class FivediagonalMatrix {

	int mat_size;
	vector<__DataType> main_diag_vals;
	vector<__DataType> first_low_diag_vals;
	vector<__DataType> second_low_diag_vals;
	vector<__DataType> first_up_diag_vals;
	vector<__DataType> second_up_diag_vals;
public:
	FivediagonalMatrix() :mat_size(0), main_diag_vals(zero_vector<__DataType>(0)),
		first_low_diag_vals(zero_vector<__DataType>(0)),
		second_low_diag_vals(zero_vector<__DataType>(0)), 
		first_up_diag_vals(zero_vector<__DataType>(0)), 
		second_up_diag_vals(zero_vector<__DataType>(0)) {};
	FivediagonalMatrix(int p_mat_size, vector<__DataType> p_main_diag_vals, vector<__DataType> p_first_low_diag_vals, 
		vector<__DataType> p_first_up_diag_vals,
		vector<__DataType> p_second_low_diag_vals, vector<__DataType> p_second_up_diag_vals) :
		mat_size(p_mat_size),main_diag_vals(p_main_diag_vals), first_low_diag_vals(p_first_low_diag_vals),
		second_low_diag_vals(p_second_low_diag_vals), first_up_diag_vals(p_first_up_diag_vals), 
		second_up_diag_vals(p_second_up_diag_vals) {};
	~FivediagonalMatrix() {};
	FivediagonalMatrix<__DataType>& operator=(const FivediagonalMatrix& p_mat) {
		if (this == &p_mat) return (*this);
		mat_size = p_mat.mat_size;
		main_diag_vals = p_mat.main_diag_vals;
		first_low_diag_vals = p_mat.first_low_diag_vals;
		second_low_diag_vals = p_mat.second_low_diag_vals;
		first_up_diag_vals = p_mat.first_up_diag_vals;
		second_up_diag_vals = p_mat.second_up_diag_vals;
		return (*this);
	}
	// доступ к элементам
	__DataType At(int idx_i, int idx_j) {
		if ((idx_i >= mat_size) || (idx_j >= mat_size)) {
			throw std::range_error("Missing key");
		}
		else if (idx_i == idx_j) {
			return main_diag_vals(idx_i);
		}
		else if (idx_i == idx_j - 1) {
			return first_up_diag_vals(idx_i);
		}
		else if (idx_i == idx_j + 1) {
			return first_low_diag_vals(idx_i - 1);
		}
		else if (idx_i == idx_j - 2) {
			return second_up_diag_vals(idx_i);
		}
		else if (idx_i == idx_j + 2) {
			return second_low_diag_vals(idx_i - 2);
		}
		else {
			return 0;
		}
	}
	int size() {
		return mat_size;
	}
};


//  пятиточечная прогонка
template<class __DataType>
vector<__DataType> FivediagonalAlgorithm(FivediagonalMatrix<__DataType> A, vector<__DataType> f) {
	int N = A.size();
	matrix<__DataType> L = zero_matrix<__DataType>(2, N - 1);
	vector<__DataType> M = zero_vector<__DataType>(N);
	vector<__DataType> X = zero_vector<__DataType>(N);
	L(0, 0) = -A.At(0, 1) / A.At(0, 0);
	L(0, 1) = (A.At(0, 2) * A.At(1, 0) - A.At(1, 2) * A.At(0, 0)) / (A.At(1, 1) * A.At(0, 0) - A.At(0, 1) * A.At(1, 0));
	L(1, 0) = -A.At(0, 2) / A.At(0, 0);
	L(1, 1) = -A.At(1, 3) * A.At(0, 0) / (A.At(1, 1) * A.At(0, 0) - A.At(0, 1) * A.At(1, 0));
	M(0) = f(0) / A.At(0, 0);
	M(1) = (A.At(0, 0) * f(1) - A.At(1, 0) * f(0)) / (A.At(1, 1) * A.At(0, 0) - A.At(0, 1) * A.At(1, 0));
	int k = 0;
	double time;
	clock_t start = clock();
#pragma omp parallel for private(k) shared(L,M)
	for (k = 2; k < N - 2; k++)
	{
		L(0, k) = -(A.At(k, k + 1) + A.At(k, k - 1) * L(1, k - 1) + A.At(k, k - 2) * L(0, k - 2) * L(1, k - 1)) / (A.At(k, k) +
			A.At(k, k - 2) * L(0, k - 2) * L(0, k - 1) + A.At(k, k - 2) * L(1, k - 2) + A.At(k, k - 1) * L(1, k - 1));
		L(1, k) = -A.At(k, k + 2) / (A.At(k, k) +
			A.At(k, k - 2) * L(0, k - 2) * L(0, k - 1) + A.At(k, k - 2) * L(1, k - 2) + A.At(k, k - 1) * L(1, k - 1));
		M(k) = -(A.At(k, k - 2) * L(0, k - 2) * M(k - 1) * A.At(k, k - 2) * M(k - 2) + A.At(k, k - 1) * M(k - 1) - f(k)) / (A.At(k, k) +
			A.At(k, k - 2) * L(0, k - 2) * L(0, k - 1) + A.At(k, k - 2) * L(1, k - 2) + A.At(k, k - 1) * L(1, k - 1));
	}
	M(N - 2) = (f(N - 2) - A.At(N - 2, N - 4) * M(N - 4) - (f(N - 1) / A.At(N - 1, N - 3)) * (A.At(N - 2, N - 4) * L(0, N - 4) + A.At(N - 2, N - 3))) / 
		(A.At(N - 2, N - 4) *L(1, N - 4) + A.At(N - 2, N - 2) - (A.At(N - 1, N - 2) / A.At(N - 1, N - 3)) * (A.At(N - 2, N - 4) * L(0, N - 4) + A.At(N - 2, N - 3)));

	L(0, N - 2) = (-A.At(N - 2, N - 1) + (A.At(N - 1, N - 1) / A.At(N - 1, N - 2)) * (A.At(N - 2, N - 4) * L(0, N - 4) + A.At(N - 2, N - 3))) / (A.At(N - 2, N - 4) *
		L(1, N - 4) + A.At(N - 2, N - 2) - (A.At(N - 1, N - 2) / A.At(N - 1, N - 3)) * (A.At(N - 2, N - 4) * L(0, N - 4) + A.At(N - 2, N - 3)));
	M(N - 1) = (f(N - 1) - A.At(N - 1, N - 3) - M(N - 2) * (A.At(N - 1, N - 3) * L(0, N - 3) + A.At(N - 1, N - 2))) / 
		((A.At(N - 1, N - 3) * L(0, N - 3) + A.At(N - 1, N - 2)) * L(0, N - 2) + A.At(N - 1, N - 3) * L(1, N - 3) + A.At(N - 1, N - 1));

	X(N - 1) = M(N - 1);
	X(N - 2) = L(0, N - 2) * X(N - 1) + M(N - 2);
	k = N - 3;
	while (k >= 0) {
		X(k) = L(0, k) * X(k + 1) + L(1, k) * X(k + 2) + M(k);
		k--;
	}
	clock_t finish = clock();
	time = (double)(finish - start) / CLOCKS_PER_SEC;
	return X;

}

template<class __DataType>
vector<__DataType> calculateBeta1(int mesh_h_size, double h, vector<__DataType> BCvals) {
	
	vector<__DataType> result = zero_vector<__DataType>(mesh_h_size);
	double h4 = h * h * h * h;
	double time;
	clock_t start = clock();
	vector<__DataType> main_diag(mesh_h_size);
	vector<__DataType> fst_up_diag(mesh_h_size - 1);
	vector<__DataType> scnd_up_diag(mesh_h_size - 2);
	vector<__DataType> fst_low_diag(mesh_h_size - 1);
	vector<__DataType> scnd_low_diag(mesh_h_size - 2);

	// заполнение главной диагонали
	main_diag(0) = 1;
	main_diag(1) = 1 / h;
	for (size_t i = 2; i < mesh_h_size - 2; i++)
	{
		main_diag(i) = 6 / h4;
	}
	main_diag(mesh_h_size - 2) = -1 / h;
	main_diag(mesh_h_size - 1) = 1;

	//заполнение первой верхней диагонали
	fst_up_diag(0) = 0;
	fst_up_diag(0) = 0;
	for (size_t i = 2; i < mesh_h_size - 2; i++)
	{
		fst_up_diag(i) = -4 / h4;
	}
	fst_up_diag(mesh_h_size - 2) = 1 / h;
	
	// заполнение второй верхней диагонали
	scnd_up_diag(0) = 0;
	scnd_up_diag(1) = 0;
	for (size_t i = 2; i < mesh_h_size - 2; i++)
	{
		scnd_up_diag(i) = 1 / h4;
	}
	
	// заполнение первой нижней диагонали
	fst_low_diag(0) = -1 / h;
	for (size_t i = 1; i < mesh_h_size - 3; i++)
	{
		fst_low_diag(i) = -4 / h4;
	}
	fst_low_diag(mesh_h_size - 3) = 0;
	fst_low_diag(mesh_h_size - 2) = 0;
	
	// заполнение второй нижней диагонали
	for (size_t i = 0; i < mesh_h_size - 4; i++)
	{
		scnd_low_diag(i) = 1 / h4;
	}
	scnd_low_diag(mesh_h_size - 4) = 0;
	scnd_low_diag(mesh_h_size - 3) = 0;


	vector<__DataType> b = zero_vector<__DataType>(mesh_h_size);
	b(1) = BCvals(0);
	b(mesh_h_size-2) = BCvals(1);


	FivediagonalMatrix<__DataType> a(mesh_h_size, main_diag, fst_low_diag, fst_up_diag, scnd_low_diag, scnd_up_diag);
	std::cout << a.size();
	result = FivediagonalAlgorithm<__DataType>(a, b);
	clock_t finish = clock();
	time = (double)(finish - start) / CLOCKS_PER_SEC;
	std::cout << "Время, затраченное на расчет уравнения равновесия(beta1): " << time << std::endl;
	return result;
}



//                                               ***ЗАДАЧА ТЕПЛОПРОВОДНОСТИ***

// функция массовой теплоемкости
double CC(double x, double t) {
	return cos(x) + 2 + 0.001 * t;
};

// положительный тепловой поток
double qeplus(double x, double t) {
	return x * t * x * t;
};

// отрицательный тепловой поток
double qeminus(double x, double t) {
	return -x * t * x * t;
};

// переменный коэффициент теплопроводности
double lambda33(double x, double t) {
	return sin(7 * x / 4) + 1.5 + 0.001 * t;
};

// функция решения задачи теплопроводности
template<class __DataType>
vector<vector<__DataType>> SolveConductivityProblem(int mesh_h_size, int mesh_t_size, double h, double dt) {
	vector<vector<__DataType>> result(mesh_t_size);
	result(0) = zero_vector<__DataType>(mesh_t_size);
	double t = 0;
	double time;
	vector<__DataType> F(mesh_h_size);
	vector<__DataType> A(mesh_h_size);
	vector<__DataType> B(mesh_h_size - 1);
	vector<__DataType> C(mesh_h_size - 1);
	vector<__DataType> U;
	omp_set_num_threads(8);
	clock_t start = clock();
#pragma omp parallel 
	{
	#pragma omp for private(i) shared(result)
		for (size_t i = 1; i < mesh_t_size; ++i)
		{
			//составляем СЛАУ на i-ом временном слое и решаем ее методом прогонки
			double ksi = 0;
			double r = dt / (h * h);
			U = result(i - 1);
			
			F(0) = qeplus(ksi, t);
			A(0) = lambda33(ksi, t) / h;
			for (size_t k = 1; k < mesh_h_size - 1; k++)
			{
				A(k) = -(lambda33(ksi + (h / 2), t) + lambda33(ksi - (h / 2), t) + CC(ksi, t) / r);
				F(k) = -CC(ksi, t) * U(k) / r;
				ksi += h;
			}
			F(mesh_h_size - 1) = -qeminus(ksi, t);
			A(mesh_h_size - 1) = -lambda33(ksi, t);
			B(0) = -A(0);
			ksi = 0;
			for (size_t k = 1; k < mesh_h_size - 1; k++)
			{
				B(k) = lambda33(ksi + (h / 2), t);
				ksi += h;
			}
			ksi = 0;
			for (size_t k = 0; k < mesh_h_size - 2; k++)
			{
				C(k) = lambda33(ksi + (h / 2), t);
				ksi += h;
			}
			C(mesh_h_size - 2) = lambda33(ksi, t);
			TridiagonalMatrix<__DataType> H(mesh_h_size, A, C, B);
			result(i) = TridiagonalAlgorithm<__DataType>(H, F);
			t += dt;
		}
	}
	clock_t finish = clock();
	time = (double)(finish - start) / CLOCKS_PER_SEC;
	std::cout << "Время, затраченное на расчет задачи теплопроводности: " << time << std::endl;
	return result;
};


//                               ***ЗАДАЧА НАХОЖДЕНИЯ КОНЦЕНТРАЦИЙ ФАЗ***


// температурные функции 
const double R = 8.31; //универсальная газовая постоянная

double f_1(double theta) {
	return exp(-1 / (R * theta));
};

double f_2(double theta) {
	return exp(-2 / (R * theta));
};

double f_3(double theta) {
	return exp(-3 / (R * theta));
};

double f_h(double theta) {
	return exp(-9 / (R * theta));
};


// плотности

double r_1(double theta) {
	return theta;
};

double r_2(double theta) {
	return theta / 2;
};

double r_3(double theta) {
	return theta / 3;
};

double r_4(double theta) {
	return theta / 4;
};

double r_h(double theta) {
	return theta / 9;
};

// функция расчета концентраций фаз на одном временном слое
template<class __DataType>
vector<vector<__DataType>> ComputePhases_Layer(int mesh_size, vector<__DataType> Jiw0, vector<vector<__DataType>> fs, vector<vector<__DataType>> rhos, vector<__DataType> Gas_quots, vector<__DataType> ics) {

	// решаем задачу нахождения концентраций фаз на одном температурном слое
	vector<__DataType> f1 = fs(0), f2 = fs(1), f3 = fs(2), fh = fs(3);
	vector<__DataType> rho1 = rhos(0), rho2 = rhos(1), rho3 = rhos(2), rho4 = rhos(3), rhoh = rhos(4);
	vector<__DataType> phi1 = zero_vector<__DataType>(mesh_size), phi2 = zero_vector<__DataType>(mesh_size),
		phi3 = zero_vector<__DataType>(mesh_size), phi4 = zero_vector<__DataType>(mesh_size), phih = zero_vector<__DataType>(mesh_size);
	phi1(0) = ics(0), phi2(0) = ics(1), phi3(0) = ics(2), phi4(0) = ics(3), phih(0) = ics(4);
	__DataType G1 = Gas_quots(0), G4 = Gas_quots(1);
	__DataType delta_t;
	double time;
	clock_t start = clock();
	omp_set_num_threads(4);
#pragma omp parallel for private(i) shared(phi1,phi2,phi3,phi4)
	for (size_t i = 1; i < mesh_size; ++i)
	{
		delta_t = 1/(mesh_size-1);
		phi1(i) = phi1(i - 1) * (1 - (delta_t * Jiw0(0) * f1(i - 1) / rho1(i - 1)));
		phi2(i) = phi2(i - 1) * (1 - (delta_t * Jiw0(1) * f2(i - 1) / rho2(i - 1))) + delta_t * Jiw0(0) * (1 - G1) * f1(i - 1) * phi1(i - 1) / rho2(i - 1);
		phi3(i) = phi3(i - 1) * (1 - (delta_t * Jiw0(2) * f3(i - 1) / rho3(i - 1))) + delta_t * Jiw0(1) * f1(i - 1) * phi1(i - 1) / rho3(i - 1);
		phi4(i) = phi4(i - 1) - delta_t * Jiw0(2) * f3(i - 1) * (1 - G4) * phi3(i - 1) / rho4(i - 1);
		phih(i) = phih(i - 1) * (1 + (delta_t * Jiw0(3) * fh(i - 1) / rhoh(i - 1)));
	}
	clock_t finish = clock();
	time = (double)(finish - start) / CLOCKS_PER_SEC;
	//std::cout << "Время, затраченное на расчет фазовых концентраций на слое: " << time << std::endl;
	vector<vector<__DataType>> phases(5);
	phases(0) = phi1, phases(1) = phi2, phases(2) = phi3, phases(3) = phi4, phases(4) = phih;
	return phases;
};


// функция расчета всей задачи нахождения концентраций фаз
template<class __DataType>
vector<vector<vector<__DataType>>> ComputePhases_Macro(vector<vector<__DataType>> THETA, int mesh_size) {
	vector<vector<vector<__DataType>>> res(THETA.size());
	vector<__DataType> Jiw0(4); // предэкспоненциальные множители
	Jiw0(0) = 1.1, Jiw0(1) = 1.2, Jiw0(2) = 1.3, Jiw0(3) = 1.4;
	vector<__DataType> Gas_qouts(2);
	Gas_qouts(0) = 0.3, Gas_qouts(1) = 0.5;
	vector<__DataType> ics(5);
	ics(0) = 0.1, ics(1) = 0.1, ics(2) = 0.1, ics(3) = 0.1, ics(4) = 0.1;
	vector<vector<__DataType>> fs(4);
	vector<vector<__DataType>> rhos(5);
	vector<__DataType> th(THETA(0).size());
	vector<__DataType> f1(th.size());
	vector<__DataType> f2(th.size());
	vector<__DataType> f3(th.size());
	vector<__DataType> fh(th.size());
	vector<__DataType> r1(th.size());
	vector<__DataType> r2(th.size());
	vector<__DataType> r3(th.size());
	vector<__DataType> r4(th.size());
	vector<__DataType> rh(th.size());
	double time;
	clock_t start = clock();
	omp_set_num_threads(8);
#pragma omp parallel
	{
		#pragma omp for private(i) shared(res)
		for (size_t i = 0; i < THETA.size(); ++i)
		{
			th = THETA(i);
			for (size_t j = 0; j < th.size(); j++)
			{
				f1(j) = f_1(th(j));
				f2(j) = f_2(th(j));
				f3(j) = f_3(th(j));
				fh(j) = f_h(th(j));
				r1(j) = r_1(th(j));
				r2(j) = r_2(th(j));
				r3(j) = r_3(th(j));
				r4(j) = r_4(th(j));
				rh(j) = r_h(th(j));
			}
			
			fs(0) = f1, fs(1) = f2, fs(2) = f3, fs(3) = fh;
			rhos(0) = r1, rhos(1) = r2, rhos(2) = r3, rhos(3) = r4, rhos(4) = rh;
			res.insert_element(i,ComputePhases_Layer<__DataType>(mesh_size, Jiw0, fs, rhos, Gas_qouts, ics));
		}
	}
	clock_t finish = clock();
	time = (double)(finish - start) / CLOCKS_PER_SEC;
	std::cout << "Время, затраченное на расчет фазовых концентраций всего материала: " << time << std::endl;
	return res;
}


int main() {

	setlocale(LC_ALL, "Russian");
	vector<vector<double>> res = SolveConductivityProblem<double>(1001, 10001, 0.001, 0.0001);
	vector<vector<vector<double>>> res1 = ComputePhases_Macro<double>(res, res(1).size());
	vector<double> bcvals(2);
	bcvals(0) = 154;
	bcvals(1) = bcvals(0);
	vector<double> b1 = calculateBeta1<double>(1001, 0.001, bcvals);
	return 0;
}

