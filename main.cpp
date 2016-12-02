#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <chrono>
#include <omp.h>
#include <fstream>

#include "function.hpp"

using namespace std;

#define SIZE 512

double A_[SIZE*SIZE];
double B_[SIZE*SIZE];
double C_[SIZE*SIZE];

int main()
{
	srand(time(NULL));

	//-------------------Preprocessing--------------------//

	double** A = new double*[SIZE];
	double** B = new double*[SIZE];
	double** C = new double*[SIZE];
	int i, j, k;
	double trash;

	ifstream fin;
	fin.open("A.txt");

	for (i = 0; i < SIZE; i++) {
		A[i] = new double[SIZE];
		B[i] = new double[SIZE];
		C[i] = new double[SIZE];
	}

	fin >> trash;
	fin >> trash;
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			fin >> A[i][j];
		}
	}

	fin.close();

	fin.open("B.txt");


	fin >> trash;
	fin >> trash;
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			fin >> B[i][j];
		}
	}

	fin.close();


	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			A_[i*SIZE + j] = A[i][j];
		}
	}

	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			B_[i + j*SIZE] = B[i][j];
		}
	}



	ofstream fout;
	
	/*for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			cout << A_[i*SIZE + j] << " ";
		}
		cout << endl;
	}

	cout << endl;

	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			cout << B_[i + j * SIZE] << " ";
		}
		cout << endl;
	}*/

	chrono::system_clock::time_point StartTime = chrono::system_clock::now();
	//---------------Matrix Multiplication---------------//
	omp_set_num_threads(4);

#pragma omp parallel for private(k,j)
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			const int t1 = i*SIZE + j;
			for (k = 0; k < SIZE ; k++) {
				const int t2 = i*SIZE +  k * 8;
				const int t3 =  k * 8 + j*SIZE;
				C_[t1] += A_[t2] * B_[t3];
				C_[t1] += A_[t2 + 1] * B_[t3 + 1];
				C_[t1] += A_[t2 + 2] * B_[t3 + 2];
				C_[t1] += A_[t2 + 3] * B_[t3 + 3];
				C_[t1] += A_[t2 + 4] * B_[t3 + 4];
				C_[t1] += A_[t2 + 5] * B_[t3 + 5];
				C_[t1] += A_[t2 + 6] * B_[t3 + 6];
				C_[t1] += A_[t2 + 7] * B_[t3 + 7];
			}
		}
	}

	//-------------------Preprocessing--------------------//

	chrono::system_clock::time_point EndTime = chrono::system_clock::now();
	chrono::microseconds micro = chrono::duration_cast<chrono::microseconds>(EndTime - StartTime);
	cout << "Matrix Multiplication done" << endl;
	cout << "After Manipulation, Time : " << micro.count() << endl;

	fin.open("Cout.txt");

	/*for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			double temp;
			fin >> temp;
			cout << C_[i*SIZE + j] << " ";
		}
		cout << endl;
	}*/

	while (1) {
		// 006504207
		// 821283632
	}

	return 0;
}