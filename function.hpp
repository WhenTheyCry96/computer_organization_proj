#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

int read_matrix(vector<vector<float>> &M, string filename)
{
	ifstream input_matrix;
	input_matrix.open(filename);
	if (!input_matrix.good())
	{
		cout << "Could not open " << filename << endl;
		return -1;
	}
	int rows = 0;
	int cols = 0;
	input_matrix >> rows;
	input_matrix >> cols;

	M.resize(rows);

	for (int i = 0; i < rows; i++)
	{
		M[i].resize(cols, 0);
		for (int j = 0; j < cols; j++)
		{
			input_matrix >> M[i][j];
		}
	}
	input_matrix.close();
	if (rows != M.size() || cols != M[0].size())
	{
		cout << "Input failed" << endl;
		return -1;
	}
	return 0;
}

