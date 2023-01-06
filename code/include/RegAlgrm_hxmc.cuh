#pragma once

/*********************************************************************************************************
这是RegAlgrmHxmc.dll的接口文件，实现快速配准的相关算法：
	1. PointSetDist_hxmc：两个点云间所有点到另一个点云所有点的距离快速计算；
	2. Knn_hxmc：查找给定的Query点云中每点到Reference点云的最近k个点距离和索引号；
	3：SVD3x3_hxmc：3*3矩阵的SVD分解。
*********************************************************************************************************/

/************************************************************************************************
说明：
	实现nRefNum个点的源点云每个点到nQueNum个目标点云的距离计算，返回所有点到点的距离。
参数：
	pfRefSet：（空间3D）参考点集的HOST指针；
	nRefNum： 参考点的数量；
	pfQueSet：查询点集的HOST指针；
	nQueNum： 查询点的数量；
	fpDistance：所求出的点到点距离的HOST缓存（共pfQueSet*pfRefSet个浮点数，pfRefSet行，pfQueSet列）。
返回：
	成功计算将返回距离个数，如果显卡内存不够，返回-1。
 **************************************************************************************************/
extern "C" __declspec(dllexport) int PointSetDist_hxmc(
	float* pfPtSetSrc, int nSrcNum,
	float* pfPtSetTarget, int nTarNum,
	float* fpDistance);

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
extern "C" __declspec(dllexport)  int Knn1_hxmc(
	float* pfRefSet, int nRefNum,
	float* pfQueSet, int nQueNum,
	float* pfDistBuf,
	int* pnRstIdxBuf);

/*************************************************************************************
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
 *************************************************************************************/
extern "C" __declspec(dllexport)  int GetNearestPt_hxmc(
	float* pfRefSet, int nRefNum,
	float* pfQueSet, int nQueNum,
	float* pfDistBuf,
	int*   pnRstIdxBuf);

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
extern "C" __declspec(dllexport)  int SVD3x3_hxmc(double* pdA, double* pdU, double* pdS, double* pdVT);

/*************************************************************************
说明：
	求解6x6线性方程组：Ax = b，双精度运算。这个算法运用cuda的cuSolver库LU分解实现。

	注意：cuda的数据是列优先的，因此，传递的参数是：
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
extern "C" __declspec(dllexport)  int Solver6x6Equ_hxmc(double* pdA, double* pdB, double* pdX);

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
extern "C" __declspec(dllexport) int FindNearestPt(float* pfDist, unsigned int nDataNum, float& fNearestDist, int& nIndex);