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

#define SIZE 2048

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

	ifstream fina;

	fina.open("A_2048.txt");

	for (i = 0; i < SIZE; i++) {
		A[i] = new double[SIZE];
		B[i] = new double[SIZE];
		C[i] = new double[SIZE];
	}

	fina >> trash;
	fina >> trash;

	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			fina >> A[i][j];
		}
	}

	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			A_[i*SIZE + j] = A[i][j];
		}
	}

	fina.close();

	ifstream finb;
	finb.open("B_2048.txt");

	finb >> trash;
	finb >> trash;

	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			finb >> B[i][j];
		}
	}

	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			B_[i*SIZE + j] = B[j][i];
		}
	}

	finb.close();

	chrono::system_clock::time_point StartTime = chrono::system_clock::now();
	//---------------Matrix Multiplication---------------//
	omp_set_num_threads(4);

#pragma omp parallel for private(i,j,k)
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			const int t1 = i*SIZE + j;
			for (k = 0; k < SIZE; k += 16) {
				const int t2 = i*SIZE + k;
				const int t3 = k + j*SIZE;
				C_[t1] += A_[t2] * B_[t3];
				C_[t1] += A_[t2 + 1] * B_[t3 + 1];
				C_[t1] += A_[t2 + 2] * B_[t3 + 2];
				C_[t1] += A_[t2 + 3] * B_[t3 + 3];
				C_[t1] += A_[t2 + 4] * B_[t3 + 4];
				C_[t1] += A_[t2 + 5] * B_[t3 + 5];
				C_[t1] += A_[t2 + 6] * B_[t3 + 6];
				C_[t1] += A_[t2 + 7] * B_[t3 + 7];
				C_[t1] += A_[t2 + 8] * B_[t3 + 8];
				C_[t1] += A_[t2 + 9] * B_[t3 + 9];
				C_[t1] += A_[t2 + 10] * B_[t3 + 10];
				C_[t1] += A_[t2 + 11] * B_[t3 + 11];
				C_[t1] += A_[t2 + 12] * B_[t3 + 12];
				C_[t1] += A_[t2 + 13] * B_[t3 + 13];
				C_[t1] += A_[t2 + 14] * B_[t3 + 14];
				C_[t1] += A_[t2 + 15] * B_[t3 + 15];
			}
		}
	}

	//-------------------Preprocessing--------------------//

	chrono::system_clock::time_point EndTime = chrono::system_clock::now();
	chrono::microseconds micro = chrono::duration_cast<chrono::microseconds>(EndTime - StartTime);
	cout << "Matrix Multiplication done" << endl;
	cout << "After Manipulation, Time : " << micro.count() << endl;
	
	ofstream fout;
	fout.open("C_2048.txt");
	for (i = 0; i < SIZE*SIZE; i++) {
		fout << C_[i] << endl ;
		//if ((i + 1) % SIZE == 0)
		//	fout << endl;
	}
	while (1) {
		// not to end the program
		// 006504207
	}

	return 0;
}