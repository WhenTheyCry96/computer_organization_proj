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

double A_[SIZE*SIZE];                  // 2D to 1D array
double B_[SIZE*SIZE];
double C_[SIZE*SIZE];

int main()
{
	srand(time(NULL));

	//-------------------Preprocessing--------------------//

	double** A = new double*[SIZE];    // double pointer
	double** B = new double*[SIZE];
	double** C = new double*[SIZE];

	int i, j, k;
	double trash;                      // since the text file starts with matrix size.

	ifstream fina;

	fina.open("A_2048.txt");

	for (i = 0; i < SIZE; i++) {
		A[i] = new double[SIZE];
		B[i] = new double[SIZE];
		C[i] = new double[SIZE];
	}

	fina >> trash;                     // to take 2048 size text as trash data
	fina >> trash;                     // to take 2048 size text as trash data

	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			fina >> A[i][j];
		}
	}

	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			A_[i*SIZE + j] = A[i][j];  // 2D array -> 1D array 
		}
	}

	fina.close();

	ifstream finb;
	finb.open("B_2048.txt");

	finb >> trash;                     // to take 2048 size text as trash data
	finb >> trash;                     // to take 2048 size text as trash data

	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			finb >> B[i][j];
		}
	}

	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			B_[i*SIZE + j] = B[j][i];  // 2D array -> 1D array
		}                              // transpose of B to simply multiply matrix
	}

	finb.close();

	chrono::system_clock::time_point StartTime = chrono::system_clock::now();
	//---------------Matrix Multiplication---------------//
	omp_set_num_threads(4);

#pragma omp parallel for private(i,j,k)              // multi-threading for variable i,j,k
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			const int t1 = i*SIZE + j;               // not to make an variable t1 everytime to function called
			for (k = 0; k < SIZE; k += 16) {
				const int t2 = i*SIZE + k;           // not to make an variable t1 everytime to function called
				const int t3 = k + j*SIZE;           // not to make an variable t1 everytime to function called
				C_[t1] += A_[t2] * B_[t3];           // loop unrolling for 16 times. 
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
	
	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			C[i][j] = C_[i*SIZE + j];                // 1D result -> 2D result
		}
	}
	/* 
	~~~not necessary part~~~ 
	cout << C[0][0] << endl;                         // check if the result is correct
	cout << C[0][1] << endl;                         // check if the result is correct
	cout << C[SIZE-1][SIZE-1] << endl;               // check if the result is correct
	*/
	ofstream fout;
	fout.open("C_2048_another.txt");

	for (i = 0; i < SIZE; i++) {
		for (j = 0; j < SIZE; j++) {
			fout  << C[i][j] << " " ;                       // 1D result -> 2D result
		}
		fout << "\r\n";
	}

	cout << "END of making C out file" << endl;

	while (1) {
		// cout << "END of making C out file" << endl;
		// not to end the program
		// 006504207
	}

	return 0;
}