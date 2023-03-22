
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

using namespace std;

__global__ void findEachMin(double* v1, double* v2, double* vr, int sz)
{
	int i = threadIdx.x;
	vr[i] = v1[i] < v2[i] ? v1[i] : v2[i];
}

cudaError_t findEachMinCuda(double* h_v1, double* h_v2, double* h_vr, int sz)
{
	double* d_v1;
	double* d_v2;
	double* d_vr;
	cudaError_t error;
	if (!
		(
			(error = cudaMalloc((void**)&d_v1, sizeof(double) * sz))
			||
			(error = cudaMalloc((void**)&d_v2, sizeof(double) * sz))
			||
			(error = cudaMalloc((void**)&d_vr, sizeof(double) * sz))
			||
			(error = cudaMemcpy(d_v1, h_v1, sizeof(double) * sz, cudaMemcpyHostToDevice))
			||
			(error = cudaMemcpy(d_v2, h_v2, sizeof(double) * sz, cudaMemcpyHostToDevice))
			||
			(error = cudaMemcpy(d_v2, h_v2, sizeof(double) * sz, cudaMemcpyHostToDevice))
			)
		)
	{
		findEachMin << <1, sz >> > (d_v1, d_v2, d_vr, sz);
		if (!(error = cudaGetLastError()))
		{
			error = cudaMemcpy(h_vr, d_vr, sizeof(double) * sz, cudaMemcpyDeviceToHost);
		}
	}

	cudaFree(d_v1);
	cudaFree(d_v2);
	cudaFree(d_vr);
	return error;
}

int main()
{
	int res = 0;
	int sz;
	double* h_v1;
	double* h_v2;
	double* h_vr;
	cin >> sz;
	h_v1 = new double[sz];
	h_v2 = new double[sz];
	h_vr = new double[sz];
	int j;

	for (int i = 0; i < sz; i++)
	{
		cin >> h_v1[i];
	}

	for (int i = 0; i < sz; i++)
	{
		cin >> h_v2[i];
	}
	cudaError_t error;
	if ((error = findEachMinCuda(h_v1, h_v2, h_vr, sz)) != cudaSuccess)
	{
		cerr << "Execution CUDA error: " << error << endl;
		res = 1;
	}
	else
	{
		cout << scientific;
		cout.precision(10);
		for (int i = 0; i < sz; i++)
		{
			cout << h_vr[i];
			if (i != sz - 1)
			{
				cout << " ";
			}
		}
	}

	delete[] h_v1;
	delete[] h_v2;
	delete[] h_vr;
	return res;
}
