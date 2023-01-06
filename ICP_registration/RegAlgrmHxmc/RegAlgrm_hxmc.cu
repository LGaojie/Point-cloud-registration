#include <stdio.h>
#include "RegAlgrm_hxmc.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define BLOCK_SIZE 512

char szErrMsg[512];
texture<float, 1, cudaReadModeElementType> texA1D;
texture<float, 1, cudaReadModeElementType> texB1D;

/*******************************************************************************
说明：
	获得cuda的错误信息，放在全局变量里，用GetErrorMsg得到错误信息。
参数：
	error： cuda的错误信息对象。
 ********************************************************************************/
void PrintErrorMessage(cudaError_t error)
{
	sprintf(szErrMsg, "cuda error: %s\n.", cudaGetErrorString(error));
}

/*******************************************************************************
说明：
	获得cuda的错误信息。
参数：
	无。
返回：
	错误信息的字符串。
 ********************************************************************************/
char* GetErrorMsg()
{
	return szErrMsg;
}

/*******************************************************************************
说明：
	核函数，运行于GPU中。
	基于纹理内存，计算点集A(参考)与点集B(查询)点之间各自的距离（B中每点到A中每一点）。
	需要16 * (nAPtNum%16+1) * 16 * (nBPtNum%16 + 1)个线程。
前提：
	点集A已经存储于一个创建好的GPU中纹理buffer：texA1D中；
	点集B已经存储于一个创建好的GPU中纹理buffer：texB1D中；

参数：
	nAPtNum：    点集A中的点数(reference)；
	nBPtNum：    点集B中的点数(query)；
	nDistPitch： 点集矩阵列（点数）的间距（字节）；
	pfDistBuf：  指向计算结果（nAPtNum * nBPtNum个浮点数）。
 ********************************************************************************/
__global__ void cuComputeDistanceTexture1D(int nAPtNum, int nBPtNum, int nDistPitch, float* pfDistBuf)
{
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if (xIndex < nAPtNum && yIndex < nBPtNum)
	{
		float fSqSum = 0;
		for (int i = 0; i < 3; i++)
		{
			float tmp = tex1Dfetch(texA1D, i * nAPtNum + xIndex) - tex1Dfetch(texB1D, i * nBPtNum + yIndex);
			fSqSum += tmp * tmp;
		}
		fSqSum = sqrt(fSqSum);
		pfDistBuf[yIndex * nDistPitch + xIndex] = fSqSum;
	}
}

/*******************************************************************************
说明：
	核函数，运行于GPU中。
	计算B点集中最邻近点。
前提：
	点集A已经存储于一个创建好的GPU中纹理buffer：texA1D中；
	点集B中每一点到点集A中每一点的距离已经计算好，放在GPU全局内存pfDist_dev中；

参数：
	nAPtNum：    点集A中的点数(reference)；
	nBPtNum：    点集B中的点数(query)；
	nDistPitch： 点集矩阵列（点数）的间距（字节）；
	pfDist_dev： 点集B中每一点到点集A中每一点的距离；
	pnDistIdx：  与A集中最近点的指数；
	pfNearDist： 与A集中最近点的距离。
 ********************************************************************************/
__global__ void cuGetNearestPt(int nAPtNum, int nBPtNum, int nDistPitch, float* pfDist_dev, int* pnDistIdx, float* pfNearDist)
{
	float  fDistT;
	float  fDis = 1.0E10;
	int    nIndex;
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (xIndex < nBPtNum)
	{

		for (int i = 0; i < nAPtNum; i++)
		{
			fDistT = pfDist_dev[xIndex * xIndex + i];
			if (fDistT < fDis)
			{
				fDis = fDistT;
				nIndex = i;
			}
		}

		pnDistIdx[xIndex] = nIndex;
		pfNearDist[xIndex] = fDis;
	}
}

/*******************************************************************************
说明：
	核函数，运行于GPU中。
	计算pfDist所指向浮点数集中，将其分块(BLOCK_SIZE大小的块)，求每块距离最小的点以及位置。

参数：
	输入：
		pfDist：      距离的device的Buffer指针；
		nDataNum：    点集中浮点数的数量；
	输出：
		pfOutBlock：  每块中最小的距离buffer；
		pnIndexBlock：每块中最小的距离对应在点集中的编号；
 ********************************************************************************/
__global__ void cuFindNearestPt(float* pfDist, int nDataNum, float* pfOutBlock, int* pnIndexBlock)
{
	__shared__ float sfData[BLOCK_SIZE];
	__shared__ int   snData[BLOCK_SIZE];

	unsigned int tid = threadIdx.x;
	unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

	// load shared mem
	sfData[tid] = (gid < nDataNum) ? pfDist[gid] : 0;
	snData[tid] = tid;

	// do reduction in shared mem, stride is divided by 2,
	for (unsigned int nHalfB = blockDim.x / 2; nHalfB > 0; nHalfB >>= 1)
	{
		__syncthreads();

		if (tid < nHalfB)
		{
			if (sfData[tid] > sfData[tid + nHalfB])
			{
				sfData[tid] = sfData[tid + nHalfB];
				snData[tid] = tid + nHalfB;
			}
		}
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		pfOutBlock[blockIdx.x] = sfData[0];
		pnIndexBlock[blockIdx.x] = snData[0];
	}
}

/*****************************************************************************************************************
说明：
	核函数，运行于GPU中。
	计算查询点云pfQueSet中每点到参考点云pfRefSet的最近距离，以及相应最近点在参考点集的编号。
参数：
	输入：
		pfRefSet： （空间3D）参考点集的HOST指针；
		nRefNum：   参考点的数量；
		pfQueSet：  查询点集的HOST指针；
		nQueNum：   查询点的数量；
	输出：
		fpDist：    与查询点集对应编号的最小距离；
		pnIndex：   对应参考点集的编号（从0开始）。
 ***************************************************************************************************************** */
__global__ void cuKnn1_hxmc(float* pfRefSet, int nRefNum, float* pfQueSet, int nQueNum, float* fpDist, int* pnIndex)
{
	__shared__ float sfData[BLOCK_SIZE + 1];
	__shared__ int   snData[BLOCK_SIZE + 1];
  
	unsigned int tid = threadIdx.x;
	unsigned int gid;
	int          nCircN = (nRefNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
	float tmp1 = tex1Dfetch(texB1D, blockIdx.x);
	float tmp2 = tex1Dfetch(texB1D, blockIdx.x + nQueNum);
	float tmp3 = tex1Dfetch(texB1D, blockIdx.x + 2 * nQueNum);

	if (tid == 0)
	{
		sfData[BLOCK_SIZE] = 1.0E10;
	}

	for (int i = 0; i < nCircN; i++)      //每个block处理一个查询点，需要循环nCircN才能计算所有与参考点集的距离
	{
		//1. 计算本次循环的查询点与BLOCK_SIZE个参考点的距离
		gid = BLOCK_SIZE * i + tid;       //参考点集的位置

		if (gid < nRefNum)
		{
			float tmp = (tmp1 - tex1Dfetch(texA1D, gid)) * (tmp1 - tex1Dfetch(texA1D, gid));
			tmp += (tmp2 - tex1Dfetch(texA1D, gid + nRefNum)) * (tmp2 - tex1Dfetch(texA1D, gid + nRefNum));
			tmp += (tmp3 - tex1Dfetch(texA1D, gid + 2 * nRefNum)) * (tmp3 - tex1Dfetch(texA1D, gid + 2 * nRefNum));
			sfData[tid] = sqrt(tmp);
			snData[tid] = gid;
		}
		else
		{
			sfData[tid] = 1.0E10;
			snData[tid] = -1;
		}
		__syncthreads();

		//2. 计算本块中的最小距离和参考点集位置
		for (unsigned int nHalfB = blockDim.x / 2; nHalfB > 0; nHalfB >>= 1)
		{
			__syncthreads();

			if (tid < nHalfB)
			{
				if (sfData[tid] > sfData[tid + nHalfB])
				{
					sfData[tid] = sfData[tid + nHalfB];
					snData[tid] = snData[tid + nHalfB];
				}
			}
		}

		//3. write result for this circle to shared mem.
		if (tid == 0)
		{
			if (sfData[BLOCK_SIZE] > sfData[0])
			{
				sfData[BLOCK_SIZE] = sfData[0];
				snData[BLOCK_SIZE] = snData[0];
			}
		}
	} //for (int i = 0; i < nCircN; i++)

	if (tid == 0)
	{
		fpDist[blockIdx.x] = sfData[BLOCK_SIZE];
		pnIndex[blockIdx.x] = snData[BLOCK_SIZE];
	}
}

/*****************************************************************************************************************
说明：
	实现nRefNum个点的源点云每个点到nQueNum个目标点云的距离计算，返回所有点到点的距离。
参数：
	pfRefSet： （空间3D）参考点集的HOST指针；
	nRefNum：   参考点的数量；
	pfQueSet：  查询点集的HOST指针；
	nQueNum：   查询点的数量；
	fpDistance：所求出的点到点距离的HOST缓存（共pfQueSet*pfRefSet个浮点数，pfRefSet行，pfQueSet列）。
返回：
	成功计算将返回距离个数，如果显卡内存不够，返回-1。
 ***************************************************************************************************************** */
int PointSetDist_hxmc(float* pfRefSet, int nRefNum, float* pfQueSet, int nQueNum, float* fpDistance)
{
	float* pfRefSet_dev;
	float* pfQueSet_dev;
	float* pfDist_dev;
	size_t       nDistPitch;
	size_t       nDistPitchByte;
	cudaError_t  cuErr;

	//分配内存给参考和查询点
	cuErr = cudaMalloc((void**)&pfRefSet_dev, nRefNum * sizeof(float) * 3);
	if (cuErr)
	{
		PrintErrorMessage(cuErr);
		return -1;
	}

	cuErr = cudaMalloc((void**)&pfQueSet_dev, nQueNum * sizeof(float) * 3);
	if (cuErr)
	{
		cudaFree(pfRefSet_dev);
		PrintErrorMessage(cuErr);
		return -2;
	}

	cudaMemcpy(pfRefSet_dev, pfRefSet, nRefNum * sizeof(float) * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(pfQueSet_dev, pfQueSet, nQueNum * sizeof(float) * 3, cudaMemcpyHostToDevice);

	cudaBindTexture(0, texA1D, pfRefSet_dev);
	cudaBindTexture(0, texB1D, pfQueSet_dev);

	//分配内存给计算结果
	// Allocation of global memory for distance buffer.	
	cuErr = cudaMallocPitch((void**)&pfDist_dev, &nDistPitchByte, nRefNum * sizeof(float), nQueNum);
	if (cuErr)
	{
		cudaFree(pfRefSet_dev);
		cudaFree(pfQueSet_dev);
		PrintErrorMessage(cuErr);

		return -3;
	}
	nDistPitch = nDistPitchByte / sizeof(float);

	// Grids ans Blocks
	dim3 dmGrad_16x16(nQueNum / 16, nRefNum / 16, 1);
	dim3 dmBlock_16x16(16, 16, 1);

	if (nQueNum % 16 != 0)
	{
		dmGrad_16x16.x += 1;
	}
	if (nRefNum % 16 != 0)
	{
		dmGrad_16x16.y += 1;
	}

	cuComputeDistanceTexture1D << <dmGrad_16x16, dmBlock_16x16 >> > (nRefNum, nQueNum, nDistPitch, pfDist_dev);

	//// Memory copy of output from device to host
	cudaMemcpy2D(fpDistance, nRefNum * sizeof(float), pfDist_dev, nDistPitchByte, nRefNum * sizeof(float), nQueNum, cudaMemcpyDeviceToHost);

	// Free memory
	cudaUnbindTexture(texA1D);
	cudaUnbindTexture(texB1D);
	cudaFree(pfRefSet_dev);
	cudaFree(pfQueSet_dev);
	cudaFree(pfDist_dev);

	return nRefNum * nQueNum;
}

/*****************************************************************************************************************
说明：
	找到点集pfQueSet中每点到参考点集pfRefSet的最近点，返回最近点的距离和编号（0基准）。
参数：
	输入：
	pfRefSet：参考点集的HOST指针；
	nRefNum： 参考点的数量；
	pfQueSet：查询点集的HOST指针；
	nQueNum： 查询点的数量；
	输出：
	pfDistBuf(out)：  pfQueSet每个点到pfRefSet点集最近点的距离；
	pnRstIdxBuf(out)：对应pfDistBuf的点在参考点云pfRefSet的索引（0基准）。
 ******************************************************************************************************************/
extern "C" __declspec(dllexport)  int GetNearestPt_hxmc(
	float* pfRefSet, int nRefNum,
	float* pfQueSet, int nQueNum,
	float* pfNearDist,
	int* pnRstIdxBuf)
{
	float* pfRefSet_dev;
	float* pfQueSet_dev;
	float* pfDist_dev;
	float* pfNearDist_dev;
	int*   pnDistIdx_dev;
	size_t       nDistPitch;
	size_t       nDistPitchByte;
	cudaError_t  cuErr;

	//分配内存给参考和查询点
	cuErr = cudaMalloc((void**)&pfRefSet_dev, nRefNum * sizeof(float) * 3);
	if (cuErr)
	{
		PrintErrorMessage(cuErr);
		return -1;
	}

	cuErr = cudaMalloc((void**)&pfQueSet_dev, nQueNum * sizeof(float) * 3);
	if (cuErr)
	{
		cudaFree(pfRefSet_dev);
		PrintErrorMessage(cuErr);
		return -2;
	}

	cudaMemcpy(pfRefSet_dev, pfRefSet, nRefNum * sizeof(float) * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(pfQueSet_dev, pfQueSet, nQueNum * sizeof(float) * 3, cudaMemcpyHostToDevice);

	cudaBindTexture(0, texA1D, pfRefSet_dev);
	cudaBindTexture(0, texB1D, pfQueSet_dev);

	//分配内存给计算结果
	// Allocation of global memory for distance buffer.	
	cuErr = cudaMallocPitch((void**)&pfDist_dev, &nDistPitchByte, nRefNum * sizeof(float), nQueNum);
	if (cuErr)
	{
		cudaFree(pfRefSet_dev);
		cudaFree(pfQueSet_dev);
		PrintErrorMessage(cuErr);

		return -3;
	}
	nDistPitch = nDistPitchByte / sizeof(float);

	// Grids ans Blocks
	dim3 dmGrad_16x16(nQueNum / 16, nRefNum / 16, 1);
	dim3 dmBlock_16x16(16, 16, 1);

	if (nQueNum % 16 != 0)
	{
		dmGrad_16x16.x += 1;
	}
	if (nRefNum % 16 != 0)
	{
		dmGrad_16x16.y += 1;
	}

	cuComputeDistanceTexture1D << <dmGrad_16x16, dmBlock_16x16 >> > (nRefNum, nQueNum, nDistPitch, pfDist_dev);

	// Free memory
	cudaUnbindTexture(texA1D);
	cudaUnbindTexture(texB1D);
	cudaFree(pfRefSet_dev);
	cudaFree(pfQueSet_dev);

	cuErr = cudaMalloc((void**)&pfNearDist_dev, nQueNum * sizeof(float));
	if (cuErr)
	{
		cudaFree(pfDist_dev);
		PrintErrorMessage(cuErr);
		return -4;
	}

	cuErr = cudaMalloc((void**)&pnDistIdx_dev, nQueNum * sizeof(int));
	if (cuErr)
	{
		cudaFree(pfDist_dev);
		cudaFree(pfNearDist_dev);
		PrintErrorMessage(cuErr);
		return -5;
	}

	dim3 dmGrad_512(nQueNum / 512, 1, 1);
	dim3 dmBlock_512(512, 1, 1);
	cuGetNearestPt << <dmGrad_512, dmBlock_512 >> > (nRefNum, nQueNum, nDistPitch, pfDist_dev, pnDistIdx_dev, pfNearDist_dev);

	cudaMemcpy(pnRstIdxBuf, pnDistIdx_dev, nQueNum * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(pfNearDist, pfNearDist_dev, nQueNum * sizeof(float), cudaMemcpyDeviceToHost);

	// Free memory
	cudaFree(pfDist_dev);
	cudaFree(pfNearDist_dev);
	cudaFree(pnDistIdx_dev);

	return 0;
}

/*******************************************************************************************
说明：
	实现nRefNum个点集中（维度nPtDim），nQueNum个点的最近临的1个点查找
参数：
	输入：
		pfRefSet：参考点集的HOST指针；
		nRefNum： 参考点的数量；
		pfQueSet：查询点集的HOST指针；
		nQueNum： 查询点的数量；
	输出：
		pfDistBuf(out)：  pfQueSet每个点最近k个点的距离的Buffer（k*nQueNum个浮点数）；
		pnRstIdxBuf(out)：对应pfDistBuf的点在参考点云的索引。
 *******************************************************************************************/
int Knn1_hxmc(float* pfRefSet, int nRefNum, float* pfQueSet, int nQueNum, float* pfDistBuf, int* pnRstIdxBuf)
{
	dim3   dimGrid(nQueNum, 1, 1);
	dim3   dimBlock(BLOCK_SIZE, 1, 1);

	float* pfRefSet_d;
	float* pfQueSet_d;
	float* pfDistBuf_d;
	int*   pnRstIdxBuf_d;
	cudaError_t statCuda;

	statCuda = cudaMalloc(reinterpret_cast<void**>(&pfRefSet_d), sizeof(float) * nRefNum *3);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -1;
	}

	statCuda = cudaMalloc(reinterpret_cast<void**>(&pfQueSet_d), sizeof(float) * nQueNum * 3);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -2;
	}

	statCuda = cudaMalloc(reinterpret_cast<void**>(&pfDistBuf_d), sizeof(float) * nQueNum);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -3;
	}

	statCuda = cudaMalloc(reinterpret_cast<void**>(&pnRstIdxBuf_d), sizeof(int) * nQueNum);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -4;
	}

	statCuda = cudaMemcpy(pfRefSet_d, pfRefSet, sizeof(float) * nRefNum * 3, cudaMemcpyHostToDevice);
	statCuda = cudaMemcpy(pfQueSet_d, pfQueSet, sizeof(float) * nQueNum * 3, cudaMemcpyHostToDevice);

	cudaBindTexture(0, texA1D, pfRefSet_d);
	cudaBindTexture(0, texB1D, pfQueSet_d);

	cuKnn1_hxmc<<<dimGrid , dimBlock >>>(pfRefSet_d, nRefNum, pfQueSet_d, nQueNum, pfDistBuf_d, pnRstIdxBuf_d);

	statCuda = cudaMemcpy(pfDistBuf, pfDistBuf_d, sizeof(float) * nQueNum, cudaMemcpyDeviceToHost);
	statCuda = cudaMemcpy(pnRstIdxBuf, pnRstIdxBuf_d, sizeof(float) * nQueNum, cudaMemcpyDeviceToHost);

	cudaBindTexture(0, texA1D, pfRefSet_d);
	cudaBindTexture(0, texB1D, pfQueSet_d);

	statCuda = cudaFree(pfRefSet_d);
	statCuda = cudaFree(pfQueSet_d);
	statCuda = cudaFree(pfDistBuf_d);
	statCuda = cudaFree(pnRstIdxBuf_d);

	return 0;
}

/*******************************************************************************
说明：
	计算pfDist所指向浮点数集中，距离最小的点以及位置。
参数：
	输入：
		pfDist：      浮点数集（距离）Buffer指针；
		nDataNum：    点集中浮点数的数量；
	输出：
		fNearestDist：其中最小的浮点数；
		pnIndexBlock：最小数对应在点集中的编号；
 ********************************************************************************/
int FindNearestPt(float* pfDist, unsigned int nDataNum, float& fNearestDist, int& nIndex)
{
	int    nBlockNum = (nDataNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3   dimGrid(nBlockNum, 1, 1);
	dim3   dimBlock(BLOCK_SIZE, 1, 1);
	float* pfDist_d;
	float* pfNearBlock_d;
	int*   pnNearInxBlk_d;
	cudaError_t statCuda;

	statCuda = cudaMalloc(reinterpret_cast<void**>(&pfDist_d), sizeof(float) * nDataNum);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -1;
	}

	statCuda = cudaMalloc(reinterpret_cast<void**>(&pfNearBlock_d), sizeof(float) * nBlockNum);
	if (statCuda != cudaSuccess)
	{
		statCuda = cudaFree(pfDist_d);
		PrintErrorMessage(statCuda);
		return -2;
	}

	statCuda = cudaMalloc(reinterpret_cast<void**>(&pnNearInxBlk_d), sizeof(int) * nBlockNum);
	if (statCuda != cudaSuccess)
	{
		statCuda = cudaFree(pfDist_d);
		statCuda = cudaFree(pfNearBlock_d);
		PrintErrorMessage(statCuda);
		return -3;
	}

	statCuda = cudaMemcpy(pfDist_d, pfDist, sizeof(float) * nDataNum, cudaMemcpyHostToDevice);
	if (statCuda != cudaSuccess)
	{
		statCuda = cudaFree(pfDist_d);
		statCuda = cudaFree(pfNearBlock_d);
		statCuda = cudaFree(pnNearInxBlk_d);

		PrintErrorMessage(statCuda);
		return -4;
	}

	//第一次查找分块的集合各自的最小数
	cuFindNearestPt << <dimGrid, dimBlock >> > (pfDist_d, nDataNum, pfNearBlock_d, pnNearInxBlk_d);

	if (nBlockNum == 1)
	{
		statCuda = cudaMemcpy(&fNearestDist, pfNearBlock_d, sizeof(float), cudaMemcpyDeviceToHost);
		statCuda = cudaMemcpy(&nIndex, pnNearInxBlk_d, sizeof(int), cudaMemcpyDeviceToHost);

		statCuda = cudaFree(pfDist_d);
		statCuda = cudaFree(pfNearBlock_d);
		statCuda = cudaFree(pnNearInxBlk_d);

		return 0;
	}

	//在第一次查找返回的浮点数集合中，第二次查找最小数
	int    nBlockNum2 = (nBlockNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
	float* pfNearBlock2_d;
	int* pnNearInxBlk2_d;
	int* pnNearInxBlk = new int[nBlockNum];

	dimGrid.x = nBlockNum2;
	statCuda = cudaMemcpy(pnNearInxBlk, pnNearInxBlk_d, sizeof(int) * nBlockNum, cudaMemcpyDeviceToHost);

	statCuda = cudaMalloc(reinterpret_cast<void**>(&pfNearBlock2_d), sizeof(float) * nBlockNum2);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -1;
	}

	statCuda = cudaMalloc(reinterpret_cast<void**>(&pnNearInxBlk2_d), sizeof(int) * nBlockNum2);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -2;
	}

	cuFindNearestPt << <dimGrid, dimBlock >> > (pfNearBlock_d, nBlockNum, pfNearBlock2_d, pnNearInxBlk2_d);

	if (nBlockNum2 == 1)
	{
		int  nIndex2;

		statCuda = cudaMemcpy(&fNearestDist, pfNearBlock_d, sizeof(float), cudaMemcpyDeviceToHost);
		statCuda = cudaMemcpy(&nIndex2, pnNearInxBlk_d, sizeof(int), cudaMemcpyDeviceToHost);

		nIndex = pnNearInxBlk[nIndex2];

		statCuda = cudaFree(pfDist_d);
		statCuda = cudaFree(pfNearBlock_d);
		statCuda = cudaFree(pnNearInxBlk_d);
		statCuda = cudaFree(pfNearBlock2_d);
		statCuda = cudaFree(pnNearInxBlk2_d);

		delete[] pnNearInxBlk;

		return 0;
	}
	else
	{
		statCuda = cudaFree(pfDist_d);
		statCuda = cudaFree(pfNearBlock_d);
		statCuda = cudaFree(pnNearInxBlk_d);
		statCuda = cudaFree(pfNearBlock2_d);
		statCuda = cudaFree(pnNearInxBlk2_d);

		delete[] pnNearInxBlk;

		return -6;   //支持最多512*512个，超出返回-6
	}

	//在第二次查找返回的浮点数集合中，第三次查找最小数（因此最多可以查找：BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE个浮点数中最小数）
	//下次再做！

	return 0;
}

/*************************************************************************
说明：
	对3x3的双精度矩阵进行SVD分解。这个算法运用cuda的cuSolver库实现。

	注意：cuda的数据是列优先的，因此，传递的参数是：
	(1,1), (2,1),(3,1);(1,2), (2,2),(3,2);(1,3), (2,3),(3,3)的顺序！
参数：
	输入：
		pdA：待分解的3x3双精度实数矩阵；
	输出：
		pdU： 3x3双精度实数矩阵U；
		pdS： 对角矩阵的对角线，从大到小排列；
		pdVT：3x3双精度实数矩阵V的转置。
返回：
	成功返回0，失败返回一个负数。可调用GetErrorMsg函数得到出错的信息。
 **************************************************************************/
int SVD3x3_hxmc(double* pdA, double* pdU, double* pdS, double* pdVT)
{
	cusolverDnHandle_t cusolverH = NULL;
	//cublasHandle_t     cublasH = NULL;
	cudaStream_t       stream = NULL;
	cusolverStatus_t   statCusolver;
	//cublasStatus_t     statCublas;
	cudaError_t        statCuda;

	const int m = 3;       // 3*3 Matrix
	const int n = 3;       // 3*3 Matrix
	const int lda = m;     // lda = m：输出所有行和列

	int info_gpu = 0;      // host copy of error info

	double* d_A = nullptr;
	double* d_S = nullptr;  // singular values
	double* d_U = nullptr;  // left singular vectors
	double* d_VT = nullptr; // right singular vectors
	int*    devInfo = nullptr;
	int     lwork = 0;          // size of workspace
	double* d_work = nullptr;
	double* d_rwork = nullptr;

	// step 1: create cusolver handle and cublas handle
	statCusolver = cusolverDnCreate(&cusolverH);
	if (statCusolver != CUSOLVER_STATUS_SUCCESS)
	{
		sprintf(szErrMsg, "cuSolver error: Create Handle.\n.");
		return -1;
	}

	/*statCublas = cublasCreate(&cublasH);
	if (statCublas != CUBLAS_STATUS_SUCCESS)
	{
		sprintf(szErrMsg, "cuBlas error: Create Handle.\n.");
		return -2;
	}*/

	statCuda = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	statCusolver = cusolverDnSetStream(cusolverH, stream);

	// step 2: copy A to device
	statCuda = cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(double) * 9);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -3;
	}

	statCuda = cudaMalloc(reinterpret_cast<void**>(&d_S), sizeof(double) * 3);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -4;
	}

	statCuda = cudaMalloc(reinterpret_cast<void**>(&d_U), sizeof(double) * 9);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -5;
	}

	statCuda = cudaMalloc(reinterpret_cast<void**>(&d_VT), sizeof(double) * 9);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -6;
	}

	statCuda = cudaMalloc(reinterpret_cast<void**>(&devInfo), sizeof(int));
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -7;
	}

	statCuda = cudaMemcpyAsync(d_A, pdA, sizeof(double) * 9, cudaMemcpyHostToDevice, stream);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -8;
	}

	// step 3: query working space of SVD
	statCusolver = cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork);
	if (statCusolver != CUSOLVER_STATUS_SUCCESS)
	{
		sprintf(szErrMsg, "cuSolver error: Query workspace.\n.");
		return -9;
	}

	statCuda = cudaMalloc(reinterpret_cast<void**>(&d_work), sizeof(double) * lwork);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -10;
	}

	// step 4: compute SVD
	signed char jobu = 'A';  // all m columns of U
	signed char jobvt = 'A'; // all n columns of VT
	statCusolver = cusolverDnDgesvd(cusolverH, jobu, jobvt, m, n, d_A, lda, d_S, d_U,
		lda, // ldu
		d_VT,
		lda, // ldvt,
		d_work, lwork, d_rwork, devInfo);
	if (statCusolver != CUSOLVER_STATUS_SUCCESS)
	{
		sprintf(szErrMsg, "cuSolver error: Solver matrix (svd).\n.");
		return -11;
	}

	//step5: copy data to host
	statCuda = cudaMemcpyAsync(pdU, d_U, sizeof(double) * 9, cudaMemcpyDeviceToHost, stream);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -12;
	}

	statCuda = cudaMemcpyAsync(pdVT, d_VT, sizeof(double) * 9, cudaMemcpyDeviceToHost,	stream);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -13;
	}

	statCuda = cudaMemcpyAsync(pdS, d_S, sizeof(double) * 3, cudaMemcpyDeviceToHost, stream);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -14;
	}

	statCuda = cudaMemcpyAsync(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost, stream);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -15;
	}

	statCuda = cudaStreamSynchronize(stream);

	//step6: free resources
	statCuda =cudaFree(d_A);
	statCuda =cudaFree(d_U);
	statCuda =cudaFree(d_VT);
	statCuda =cudaFree(d_S);
	statCuda =cudaFree(devInfo);
	statCuda =cudaFree(d_work);
	statCuda =cudaFree(d_rwork);

	statCusolver = cusolverDnDestroy(cusolverH);

	statCuda = cudaStreamDestroy(stream);
	statCuda = cudaDeviceReset();

	return 0;
}

/*************************************************************************
说明：
	求解6x6线性方程组：Ax = b，双精度运算。这个算法运用cuda的cuSolver库LU分解实现。

	注意：cuda的数据是列优先的，因此，A矩阵传递参数的顺序是：
	(1,1), (2,1),(3,1)，(4,1), (5,1),(6,1)；
	(1,2), (2,2),(3,2)，(4,2), (5,2),(6,2)；
	(1,3), (2,3),(3,3)，(4,3), (5,3),(6,3)；
	(1,4), (2,4),(3,4)，(4,4), (5,4),(6,4)；
	(1,5), (2,5),(3,5)，(4,5), (5,5),(6,5)；
	(1,6), (2,6),(3,6)，(4,6), (5,6),(6,6)；
参数：
	输入：
		pdA：方程的系数矩阵；
		pdB：方程的常数矢量。
	输出：
		pdX： 解矢量。
返回：
	成功返回0，失败返回一个负数。可调用GetErrorMsg函数得到出错的信息。
 **************************************************************************/
int Solver6x6Equ_hxmc(double* pdA, double* pdB, double* pdX)
{
	cusolverDnHandle_t cusolverH = NULL;
	cudaStream_t       stream = NULL;
	cudaError_t        statCuda;
	cusolverStatus_t   statCusolver;

	//host
	const int m = 6;
	const int lda = m;
	const int ldb = m;
	double* pdLU = new double[lda * m];
	int*    pnIpiv = new int[m];
	int     nInfo = 0;
	//device
	double* pdA_d = nullptr;        // device copy of A
	double* pdB_d = nullptr;        // device copy of B
	int*    pnIpiv_d = nullptr;     // pivoting sequence
	int*    pnInfo_d = nullptr;     // error info
	int     lwork = 0;              // size of workspace
	double* pdWork_d = nullptr;     // device workspace for getrf

	memset((void*)pnIpiv, 0, m * sizeof(int));
	memset((void*)pdLU, 0, lda * m * sizeof(double));
	memset((void*)pdX, 0, m * sizeof(double));

	// step 1: create cusolver handle, bind a stream
	statCusolver = cusolverDnCreate(&cusolverH);
	if (statCusolver != CUSOLVER_STATUS_SUCCESS)
	{
		sprintf(szErrMsg, "cuSolver error: Create Handle.\n.");
		return -1;
	}

	statCuda = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	if (statCuda != cudaSuccess)
	{
		PrintErrorMessage(statCuda);
		return -2;
	}

	statCusolver = cusolverDnSetStream(cusolverH, stream);
	if (statCusolver != CUSOLVER_STATUS_SUCCESS)
	{
		sprintf(szErrMsg, "cuSolver error: Set cudaStream.\n.");
		return -3;
	}

	/* step 2: copy A to device */
	statCuda = cudaMalloc(reinterpret_cast<void**>(&pdA_d), sizeof(double) * m *m );
	statCuda = cudaMalloc(reinterpret_cast<void**>(&pdB_d), sizeof(double) * m);
	statCuda = cudaMalloc(reinterpret_cast<void**>(&pnIpiv_d), sizeof(int) * m);
	statCuda = cudaMalloc(reinterpret_cast<void**>(&pnInfo_d), sizeof(int));

	statCuda = cudaMemcpyAsync(pdA_d, pdA, sizeof(double) * m * m, cudaMemcpyHostToDevice, stream);
	statCuda = cudaMemcpyAsync(pdB_d, pdB, sizeof(double) * m, cudaMemcpyHostToDevice, stream);

	// step 3: query working space of getrf
	statCusolver = cusolverDnDgetrf_bufferSize(cusolverH, m, m, pdA_d, lda, &lwork);
	if (statCusolver != CUSOLVER_STATUS_SUCCESS)
	{
		sprintf(szErrMsg, "cuSolver error: Query workspace.\n.");

		delete[] pdLU;
		delete[] pnIpiv;

		// free resources
		statCuda = cudaFree(pdA_d);
		statCuda = cudaFree(pdB_d);
		statCuda = cudaFree(pnIpiv_d);
		statCuda = cudaFree(pnInfo_d);

		return -4;
	}

	statCuda = cudaMalloc(reinterpret_cast<void**>(&pdWork_d), sizeof(double) * lwork);

	// step 4: LU factorization
	statCusolver = cusolverDnDgetrf(cusolverH, m, m, pdA_d, lda, pdWork_d, pnIpiv_d, pnInfo_d);

	// 取回LU结果 Test code
	statCuda = cudaMemcpyAsync(pnIpiv, pnIpiv_d, sizeof(int) * m, cudaMemcpyDeviceToHost, stream);        //主元的位置
	statCuda = cudaMemcpyAsync(pdLU, pdA_d, sizeof(double) * m * m, cudaMemcpyDeviceToHost, stream);      //LU三角矩阵;
	statCuda = cudaMemcpyAsync(&nInfo, pnInfo_d, sizeof(int), cudaMemcpyDeviceToHost, stream);            //求解信息
	statCuda = cudaStreamSynchronize(stream);

	if (0 > nInfo)
	{
		sprintf(szErrMsg, "cuSolver error: LU factorization %d-th Parameter is wrong.\n.", -nInfo);

		delete[] pdLU;
		delete[] pnIpiv;

		// free resources
		statCuda = cudaFree(pdA_d);
		statCuda = cudaFree(pdB_d);
		statCuda = cudaFree(pnIpiv_d);
		statCuda = cudaFree(pnInfo_d);
		statCuda = cudaFree(pdWork_d);
		return -5;
	}

	// step 5: solve A*X = B
	statCusolver = cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, m, 1, pdA_d, lda, pnIpiv_d, pdB_d, ldb, pnInfo_d);
	if (statCusolver != CUSOLVER_STATUS_SUCCESS)
	{
		sprintf(szErrMsg, "cuSolver error: Solve equation.\n.");

		delete[] pdLU;
		delete[] pnIpiv;

		// free resources
		statCuda = cudaFree(pdA_d);
		statCuda = cudaFree(pdB_d);
		statCuda = cudaFree(pnIpiv_d);
		statCuda = cudaFree(pnInfo_d);
		statCuda = cudaFree(pdWork_d);

		return -6;
	}

	statCuda = cudaMemcpyAsync(pdX, pdB_d, sizeof(double) * m, cudaMemcpyDeviceToHost, stream);
	statCuda = cudaStreamSynchronize(stream);

	delete[] pdLU;
	delete[] pnIpiv;

	// free resources
	statCuda = cudaFree(pdA_d);
	statCuda = cudaFree(pdB_d);
	statCuda = cudaFree(pnIpiv_d);
	statCuda = cudaFree(pnInfo_d);
	statCuda = cudaFree(pdWork_d);

	statCusolver = cusolverDnDestroy(cusolverH);
	statCuda = cudaStreamDestroy(stream);
	statCuda = cudaDeviceReset();

	return 0;
}
