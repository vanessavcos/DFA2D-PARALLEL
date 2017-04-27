import numpy as np
import codecs
import scipy.linalg.decomp_lu as lu
import sys
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.tools
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from scipy import stats
import os
import time
import argparse

mod = SourceModule("""
    #include <stdio.h>

        __global__ void fit(float *l, float *u, float *MatDados, int s, int tam, float *MatOut)
    {
        extern __shared__ float Aux[ ];
        __shared__ float B[3];
		__shared__ float y[3];
		__shared__ float x[3];
		__shared__ float su[6];
		__shared__ float sl[3];
		__shared__ float F[1];
        int tx = threadIdx.x;
        int endAux = (tx * s);
        int end = (blockIdx.y * s * tam) + (blockIdx.x * s);
		double t = 0.0;
		F[0] = 0;


        for (int c = 0; c< s; c++)
        {
            Aux[endAux + c] = MatDados[end + (tx*tam) + c];
        }

        if (tx < 6)
        {
		    su[tx] = u[tx];
		}

        __syncthreads();

        if (tx == 0)
        {
			sl[tx] = l[tx];
            B[tx] = 0;
            for (int i=0; i< s; i++)
            {
                for (int j=0; j < s; j++)
                {
                    B[tx] = B[tx] + ((j+1) * Aux[(i*s)+ j]);
                }
            }
        }
        if (tx == 1)
        {
			sl[tx] = l[tx];
            B[tx] = 0;
            for (int i=0; i< s; i++)
            {
                for (int j=0; j < s; j++)
                {
                    B[tx] = B[tx] + ((i+1) * Aux[(i*s)+ j]);
                }
            }
        }
        if (tx == 2)
        {
			sl[tx] = l[tx];
            B[tx] = 0;
            for (int i=0; i< s; i++)
            {
                for (int j=0; j < s; j++)
                {
                    B[tx] = B[tx] + (Aux[(i*s)+ j]);
                }
            }
        }
        __syncthreads();

        if (tx == 0)
        {
            //printf("%d %d -> %f, %f, %f\t", blockIdx.x, blockIdx.y, B[0], B[1], B[2]);
            y[0] = B[0];
            y[1] = B[1] - (sl[0] * y[0]);
            y[2] = B[2] - ((sl[1] * y[0]) + (sl[2] * y[1]));
            x[2] = y[2]/su[5];
            x[1] = (y[1] - (su[4] * x[2]))/(su[3]);
            x[0] = (y[0] - ((su[1] * x[1]) + (su[2] * x[2])))/(su[0]);
        }

        __syncthreads();


        for (int c = 0; c< s; c++)
        {
            t = (((c+1) * x[0]) +  ((tx+1) * x[1]) + x[2]);
            t = Aux[endAux + c] - t;
            t = powf(t, 2);
            atomicAdd(&F[0], (float)t);
            //printf("%d, %d ->%f\t", blockIdx.x, blockIdx.y, F[0]);
        }
        __syncthreads();


        if (tx == 0)
        {
            //printf("%d -> %f\t", s, F[0]);
            //printf("%d -> %d\t", s, ((blockIdx.y * gridDim.x) + blockIdx.x));
            MatOut[(blockIdx.y * gridDim.x) + blockIdx.x] = (F[0] / (s*s));
        }
        __syncthreads();

    return;
    }
    __global__ void fitBig(float *l, float *u, float *MatDados, int s, int tam, float *MatOut)
    {
        __shared__ float B[3];
		__shared__ float y[3];
		__shared__ float x[3];
		__shared__ float su[6];
		__shared__ float sl[3];
		__shared__ float F[1];
        int tx = threadIdx.x;
        int end = (blockIdx.y * s * tam) + (blockIdx.x * s);
		double res;
		F[0] = 0;

		if (tx < 6)
        {
            su[tx] = u[tx];
        }

		if (tx < 3)
        {
            sl[tx] = l[tx];
			B[tx] = 0;
        }
        __syncthreads();

        if (tx == 0)
        {
            for (int i=0; i< s; i++)
             {
                for (int j=0; j < s; j++)
                {
                    B[tx] = B[tx] + ((j+1) * MatDados[end + (i*tam) + j]);
                }
            }
        }
		if (tx == 1)
		{
			for (int i=0; i< s; i++)
			{
				for (int j=0; j < s; j++)
				{
					B[tx] = B[tx] + ((i+1) * MatDados[end + (i*tam) + j]);
				}
			}
		}
		if (tx == 2)
		{
			for (int i=0; i< s; i++)
			{
				for (int j=0; j < s; j++)
				{
					B[tx] = B[tx] + (MatDados[end + (i*tam) + j]);
				}
			}
		}
		__syncthreads();
			
		if (tx == 0)
        {
            //printf("%d %d -> %f, %f, %f\t", blockIdx.x, blockIdx.y, B[0], B[1], B[2]);
            y[0] = B[0];
            y[1] = B[1] - (sl[0] * y[0]);
            y[2] = B[2] - ((sl[1] * y[0]) + (sl[2] * y[1]));
            x[2] = y[2]/su[5];
            x[1] = (y[1] - (su[4] * x[2]))/(su[3]);
            x[0] = (y[0] - ((su[1] * x[1]) + (su[2] * x[2])))/(su[0]);
		}
        __syncthreads();


        for (int c = 0; c< s; c++)
        {
            res = (((c+1) * x[0]) +  ((tx+1) * x[1]) + x[2]);
            res = MatDados[end + (tx*tam) + c] - res;
            res = powf(res, 2);
            atomicAdd(&F[0], (float)res);
            //printf("%d, %d ->%f\t", blockIdx.x, blockIdx.y, F[0]);
        }
        __syncthreads();


        if (tx == 0)
        {
            //printf("%d -> %f\t", s, F[0]);
            //printf("%d -> %d\t", s, ((blockIdx.y * gridDim.x) + blockIdx.x));
            MatOut[(blockIdx.y * gridDim.x) + blockIdx.x] = (F[0] / (s*s));
        }
        __syncthreads();

    return;
    }
""")



def readData(fileName):
        with codecs.open(fileName, encoding='utf-8-sig') as f:
                m = np.loadtxt(f, delimiter='	', usecols=range(256))
        return m

def generateModel(opt, s):
        if opt == 1:
            nx = np.arange(1, s + 1)
            ny = np.arange(1, s + 1)
            x, y = np.meshgrid(nx, ny)
            x_fl = x.flatten()
            y_fl = y.flatten()
            z_ones = np.ones([x.size, 1])
            A = np.hstack((np.reshape(x_fl, ([len(x_fl), 1])), np.reshape(y_fl, ([len(y_fl), 1])), z_ones))
        return(A)

def LUDecomposition(A):
    lr = np.zeros(shape=(1,3)).astype(np.float32);
    ur = np.zeros(shape=(1,6)).astype(np.float32);
    At = np.transpose(A)
    AtA = np.dot(At, A)
    AtA = AtA.astype(np.float32)
    P, L, U = lu.lu(AtA,False,True)
    P = P.astype(np.float32)
    lr = np.extract(L != 1, L)
    lr = np.extract(lr != 0, lr)
    ur = np.extract(U > 0, U)
    return (lr, ur)

def dfaCuda(h_mat, opt):
    inicio = time.time()
    #casting nos dados de entrada
    h_mat = h_mat.astype(np.float32)
    [l, c] = np.shape(h_mat)
    tam = np.minimum(l, c)
    escalas = int(tam / 4)
    F = np.zeros(shape=(escalas - 5, 2)) #64 - 6 + 1
    #realizar a soma acumulada
    #h_mat = np.reshape(np.cumsum(h_mat), newshape=(l, c));
    start = cuda.Event()
    end = cuda.Event()
    #alocar e transferir a matriz de dados para a GPU
    d_mat = gpuarray.to_gpu(h_mat)
    #loop para variar as escalas
    k = 0
    for s in np.arange(6, escalas + 1):
        h_A = generateModel(opt, s)
        [h_L, h_U] = LUDecomposition(h_A)
        d_L = gpuarray.to_gpu(h_L)
        d_U = gpuarray.to_gpu(h_U)
        tSaida = np.power(np.floor(tam/s),2).astype(np.int)
        h_vetG = np.zeros(shape=(tSaida,1)).astype(np.float32)
        d_vetF = gpuarray.to_gpu(h_vetG)

        #calc to kernel parameters
        blockSize = (s, 1, 1)
        gridSize = np.floor((tam/s)).astype(int)
        grid = (gridSize, gridSize, 1)
        if (s <= 110) :
            sizeofSharedMemoryinBytes = s * s * (np.dtype(np.float32).itemsize)
            #launch kernel
            #get the kernel function from the compiled module
            kernel = mod.get_function("fit")
            #call the kernel on the card
            kernel(
                #inputs
                d_L, d_U, d_mat,np.int32(s), np.int32(tam),
                #output
                d_vetF,
                # kernel parameters
                grid= grid, block=blockSize,
                #size of shared memory
                shared = sizeofSharedMemoryinBytes)
        else :
            #launch kernel
            #get the kernel function from the compiled module
            kernel = mod.get_function("fitBig")
            #call the kernel on the card
            kernel(
                #inputs
                d_L, d_U, d_mat,np.int32(s), np.int32(tam),
                #output
                d_vetF,
                # kernel parameters
                grid= grid, block=blockSize
            )
        fs = np.sqrt(np.mean(d_vetF.get()))
        #print(d_vetF.get())
        F[k][0] = s
        F[k][1] = fs
        k = k + 1
    #print(F)
    #calcular o valor do alfa
    vetoutput = np.log10(F)
    x = vetoutput[:, 0]
    y = vetoutput[:, 1]
    slope, _, _, _, _ = stats.linregress(x, y)
    fim = time.time()
    print(fim - inicio)
    print(slope)
    return (slope, fim - inicio)


def main(dir, delimiter, tamanho, arqSaida):
    for file in os.listdir(dir):
        print(file)
        with codecs.open(dir + file, encoding='utf-8-sig') as f:
            m = np.loadtxt(f, delimiter= delimiter, usecols=range(tamanho))
        alfa, tempo = dfaCuda(m, 1)
        dt = np.dtype(str, 10)
        b = np.array([file, alfa, tempo], dtype=dt)
        b = np.reshape(b, newshape=(1, 3))
        with open(arqSaida, 'ab') as f:
            np.savetxt(f, b, fmt='%10s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paralell DFA 2D')
    parser.add_argument('-d','--Directory', type=str, help='Dataset Directory', required=True)
    parser.add_argument('-dt','--Delimiter', type=str, help='String that delimiter the values in file', required=True)
    parser.add_argument('-s','--Size', type=int, help='Size of matrix', required=True)
    parser.add_argument('-f','--File', type=str, help='name of the output file', required=True)
    parser.parse_args()