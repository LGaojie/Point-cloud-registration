/*****************************************************************************
RegICPHxmc.cpp : 采用开源库Flann和Eigen实现5特征点的初配准和ICP配准。

Author:   Zhang Xiaochun
Date:     2022.11.21
*****************************************************************************/

#include "pch.h"
#include "RegICPHxmc.h"
#include <Eigen\Dense>
#include <flann\flann.hpp>
#include <omp.h>
#include "Matrix_hxmc.h"

#define JACOBI_MAX_ROTATIONS   20
#define DOUBLE_MAX_HX          1.0E38
#define TOLERANCE_HX           1.0E-5
#define RESERVED_QUE_NUM       60000
#define MIN_REGIST_NUM         600                                //相机点云过滤后的点少于这个值将返回很大的平均距离:MAX_AVE_DIST
#define MAX_AVE_DIST           100.0                              //单位：mm。配准允许最大的平均距离，返回这个值说明没有配准

flann::KDTreeIndexParams            g_IndexParams(4);               //KNN查询的参数（采用KT-TREE）
flann::Index<flann::L2<double> >    g_flannIndexObj(g_IndexParams); //KNN查询对象
flann::Matrix<double>               g_flmatRef;                     //参考数据（空间点，对应配准的目标）

CRegICPHxmc::CRegICPHxmc()
{
	m_dMinDist = 0.88;       //希望达到的点集间平局距离
	m_pdRefPtSet = new double[3 * RESERVED_QUE_NUM];
	m_pdQuePtSet = new double[3 * RESERVED_QUE_NUM];
	m_pdDistance = new double[RESERVED_QUE_NUM];
	m_pnIndexSet = new int[RESERVED_QUE_NUM];

	m_nReservPtNum = RESERVED_QUE_NUM;

	m_stObbox.corner[0] = m_stBaseObbox.corner[0] = -10000.0;
	m_stObbox.corner[1] = m_stBaseObbox.corner[1] = -10000.0;
	m_stObbox.corner[2] = m_stBaseObbox.corner[2] = -10000.0;
	m_stObbox.max[0] = m_stBaseObbox.max[0] = 20000.0;
	m_stObbox.max[1] = m_stBaseObbox.max[1] = 0.0;
	m_stObbox.max[2] = m_stBaseObbox.max[2] = 0.0;
	m_stObbox.mid[0] = m_stBaseObbox.mid[0] = 0.0;
	m_stObbox.mid[1] = m_stBaseObbox.mid[1] = 20000.0;
	m_stObbox.mid[2] = m_stBaseObbox.mid[2] = 0.0;
	m_stObbox.min[0] = m_stBaseObbox.min[0] = 0.0;
	m_stObbox.min[1] = m_stBaseObbox.min[1] = 0.0;
	m_stObbox.min[2] = m_stBaseObbox.min[2] = 20000.0;
	

	for (int i = 0; i < 3; i++)
	{
		m_stObbox.corner[i] = m_stObbox.max[i] = m_stObbox.mid[i] = m_stObbox.min[i] = 0.0;
	}

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			m_dTransMat[i][j] = 0.0;
			m_dFinalTransMat[i][j] = 0.0;
		}
		m_dTransMat[i][i] = 1.0;
		m_dFinalTransMat[i][i] = 1.0;
	}

	//Test code
	//MatrixHxmc<double> matA;
}

CRegICPHxmc::~CRegICPHxmc()
{
	if (m_pdRefPtSet != NULL)
	{
		delete[] m_pdRefPtSet;
	}

	if (m_pdQuePtSet != NULL)
	{
		delete[] m_pdQuePtSet;
	}

	if (m_pdDistance != NULL)
	{
		delete[] m_pdDistance;
	}

	if (m_pnIndexSet != NULL)
	{
		delete[] m_pnIndexSet;
	}
}

/*************************************************************************
函数说明：
	获取已将计算好的配准矩阵。注意返回的是对象内的矩阵指针。
参数:
	无。
返回值：
	指向变换矩阵：排列{(1,1),(1,2),(1,3),(1,4);(2,1),......,(4,3),(4,4)}。
**************************************************************************/
double* CRegICPHxmc::GetRegMatrix()
{
	int k = 0;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			m_dMatRegiste[k++] = m_dTransMat[i][j];
		}
	}

	return m_dMatRegiste;
}

/*************************************************************************
函数说明：
	获取中间变换矩阵：例如5特征点的SVD法求出的中间结果等。
参数:
	dTransMat(out)：齐次变换矩阵。
返回值：
	无。
**************************************************************************/
void CRegICPHxmc::GetTransMatrix(double dTransMat[4][4])
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			dTransMat[i][j] = m_dTransMat[i][j];
		}
	}
}

/*************************************************************************
函数说明：
	获取ICP迭代后的最终变换矩阵。
参数:
	dTransMat(out)：齐次变换矩阵。
返回值：
	无。
**************************************************************************/
void  CRegICPHxmc::GetFinalTransMatrix(double dTransMat[4][4])
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			dTransMat[i][j] = m_dFinalTransMat[i][j];
		}
	}
}

/*************************************************************************
函数说明：
	获取ICP迭代后的最终变换矩阵的逆矩阵。
参数:
	dTransMat(out)：齐次变换矩阵。
返回值：
	无。
**************************************************************************/
void  CRegICPHxmc::GetFinalTransMatrixInv(double dTransMat[4][4])
{
	GetInverseMatHomo();

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			dTransMat[i][j] = m_dFinalInversMat[i][j];
		}
	}
}

/******************************************************************
函数说明：
	求本次待配准源点云的包围盒：利用上次变换得到的矩阵以及基准
	包围盒，我们认为人头基本不动或运动很少(<10mm)，因而得到的
	包围盒仍然基本上包围5特征点形成的包围部分。
******************************************************************/
void CRegICPHxmc::GetCurrentObBox()
{
	double corner[3], maxPt[3], midPt[3], minPt[3];
	double ptCorner[3], ptMax[3], ptMid[3], ptMin[3];

	corner[0] = m_stBaseObbox.corner[0];
	corner[1] = m_stBaseObbox.corner[1];
	corner[2] = m_stBaseObbox.corner[2];
	maxPt[0] = corner[0] + m_stBaseObbox.max[0];
	maxPt[1] = corner[1] + m_stBaseObbox.max[1];
	maxPt[2] = corner[2] + m_stBaseObbox.max[2];
	midPt[0] = corner[0] + m_stBaseObbox.mid[0];
	midPt[1] = corner[1] + m_stBaseObbox.mid[1];
	midPt[2] = corner[2] + m_stBaseObbox.mid[2];
	minPt[0] = corner[0] + m_stBaseObbox.min[0];
	minPt[1] = corner[1] + m_stBaseObbox.min[1];
	minPt[2] = corner[2] + m_stBaseObbox.min[2];

	for (int i = 0; i < 3; i++)
	{
		ptCorner[i] = 0.0;
		ptMax[i] = 0.0;
		ptMid[i] = 0.0;
		ptMin[i] = 0.0;
		for (int j = 0; j < 3; j++)
		{
			ptCorner[i] += m_dFinalInversMat[i][j] * corner[j];
			ptMax[i] += m_dFinalInversMat[i][j] * maxPt[j];
			ptMid[i] += m_dFinalInversMat[i][j] * midPt[j];
			ptMin[i] += m_dFinalInversMat[i][j] * minPt[j];
		}
		ptCorner[i] += m_dFinalInversMat[i][3];
		ptMax[i] += m_dFinalInversMat[i][3];
		ptMid[i] += m_dFinalInversMat[i][3];
		ptMin[i] += m_dFinalInversMat[i][3];
	}

	for (int i = 0; i < 3; i++)
	{
		m_stObbox.corner[i] = ptCorner[i];
		m_stObbox.max[i] = ptMax[i] - ptCorner[i];
		m_stObbox.mid[i] = ptMid[i] - ptCorner[i];
		m_stObbox.min[i] = ptMin[i] - ptCorner[i];
	}
}

/******************************************************************
函数说明：
	通过源点云上的包围盒以及首次得到的变换矩阵，求基准包围盒：
	即首次传入的源上5特征点的最小包围盒，变换后得到对应在MRI上
	的对应包围盒作为基准的包围盒。
******************************************************************/
void CRegICPHxmc::GetBaseObBox()
{
	double corner[3], maxPt[3], midPt[3], minPt[3];
	double ptCorner[3], ptMax[3], ptMid[3], ptMin[3];

	corner[0] = m_stObbox.corner[0];
	corner[1] = m_stObbox.corner[1];
	corner[2] = m_stObbox.corner[2];
	maxPt[0] = corner[0] + m_stObbox.max[0];
	maxPt[1] = corner[1] + m_stObbox.max[1];
	maxPt[2] = corner[2] + m_stObbox.max[2];
	midPt[0] = corner[0] + m_stObbox.mid[0];
	midPt[1] = corner[1] + m_stObbox.mid[1];
	midPt[2] = corner[2] + m_stObbox.mid[2];
	minPt[0] = corner[0] + m_stObbox.min[0];
	minPt[1] = corner[1] + m_stObbox.min[1];
	minPt[2] = corner[2] + m_stObbox.min[2];

	for (int i = 0; i < 3; i++)
	{
		ptCorner[i] = 0.0;
		ptMax[i] = 0.0;
		ptMid[i] = 0.0;
		ptMin[i] = 0.0;
		for (int j = 0; j < 3; j++)
		{
			ptCorner[i] += m_dFinalTransMat[i][j] * corner[j];
			ptMax[i] += m_dFinalTransMat[i][j] * maxPt[j];
			ptMid[i] += m_dFinalTransMat[i][j] * midPt[j];
			ptMin[i] += m_dFinalTransMat[i][j] * minPt[j];
		}
		ptCorner[i] += m_dFinalTransMat[i][3];
		ptMax[i] += m_dFinalTransMat[i][3];
		ptMid[i] += m_dFinalTransMat[i][3];
		ptMin[i] += m_dFinalTransMat[i][3];
	}

	for (int i = 0; i < 3; i++)
	{
		m_stBaseObbox.corner[i] = ptCorner[i];
		m_stBaseObbox.max[i] = ptMax[i] - ptCorner[i];
		m_stBaseObbox.mid[i] = ptMid[i] - ptCorner[i];
		m_stBaseObbox.min[i] = ptMin[i] - ptCorner[i];
	}
}

/********************************************************************
函数说明：
	销毁已经分配内存的实例。
输入参数:
输出参数:
	无。
返回值：
	无。
*********************************************************************/
void CRegICPHxmc::SetTargetPtCloud(float* pPtCloud, int nPtNum)
{

}

/***********************************************************************
函数说明：
	将五个特征点的坐标传入本类的对象。
参数:
	pdFeatCoords：坐标参数，排列方式{x1,y1,z1,x2,y2,z2,......x5,y5,z5}；
	isTarget：是否为目标点云的坐标，否则传入源点云特征点的坐标。
返回值：
	无。
************************************************************************/
void CRegICPHxmc::Set5FeaturePtsCoord(double* pdFeatCoords, bool isTarget)
{
	double* pdTemp = pdFeatCoords;
	if (isTarget)
	{
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				m_d5FeatMri[i][j] = *pdTemp++;
			}
		}
	}
	else
	{
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				m_d5FeatCam[i][j] = *pdTemp++;
			}
		}
	}
}

/***********************************************************************
函数说明：
	将五个特征点的坐标传入本类的对象。
参数:
	dFeatCoords：坐标参数。排列方式为C++的二维数组：
	    {{x1,y1,z1}
		 {x2,y2,z2}
		  ......
		 {x5,y5,z5}}；
	isTarget：是否为目标点云的坐标，否则传入源点云特征点的坐标。
返回值：
	无。
************************************************************************/
void CRegICPHxmc::Set5FeaturePtsCoord(double  dFeatCoords[][3], bool isTarget)
{
	if (isTarget)
	{
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				m_d5FeatMri[i][j] = dFeatCoords[i][j];
			}
		}
	}
	else
	{
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				m_d5FeatCam[i][j] = dFeatCoords[i][j];
			}
		}
	}
}

/***********************************************************************
函数说明：
	获得五个特征点的坐标。
参数:
	dFeatCoords(out)：坐标参数。排列方式为C++的二维数组：
		{{x1,y1,z1}
		 {x2,y2,z2}
		  ......
		 {x5,y5,z5}}；
	isTarget：取回的是目标点云的坐标，或是源点云特征点的坐标。
返回值：
	无。
************************************************************************/
void CRegICPHxmc::Get5FeaturePtsCoord(double  dFeatCoords[][3], bool isTarget)
{
	if (isTarget)
	{
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				dFeatCoords[i][j] = m_d5FeatMri[i][j];
			}
		}
	}
	else
	{
		for (int i = 0; i < 5; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				dFeatCoords[i][j] = m_d5FeatCam[i][j];
			}
		}
	}
}

/*************************************************************************
说明：
	用其次坐标的对称性，求最终变换矩阵m_dFinalTransMat的逆矩阵：
	m_dFinalInversMat。
**************************************************************************/
void CRegICPHxmc::GetInverseMatHomo()
{
	m_dFinalInversMat[0][3] = m_dFinalInversMat[1][3] = m_dFinalInversMat[2][3] = 0.0;
	m_dFinalInversMat[3][0] = m_dFinalInversMat[3][1] = m_dFinalInversMat[3][2] = 0.0;
	m_dFinalInversMat[3][3] = 1.0;

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			m_dFinalInversMat[j][i] = m_dFinalTransMat[i][j];
			m_dFinalInversMat[i][3] -= m_dFinalTransMat[j][i] * m_dFinalTransMat[j][3];
		}
	}
}

/*************************************************************************
说明：
	用Eigen的inverse函数，求最终变换矩阵m_dFinalTransMat的逆矩阵：
	m_dFinalInversMat。
**************************************************************************/
void CRegICPHxmc::GetInverseMatEigen()
{
	Eigen::Matrix4d mat, matInv;

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			mat(i, j) = m_dFinalTransMat[i][j];
		}
	}

	matInv = mat.inverse();

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			m_dFinalInversMat[i][j] = matInv(i, j);
		}
	}
}

/************************************************************************
说明：
	SVD方法求解对应点的刚性变换矩阵。至少输入对应4个点的坐标。
	注意：输出的变换矩阵在成员 m_dTransMat保存，用GetTransMatrix得到结果。
参数：
	nDataNum：对应点的个数；
	dSource： 待配准的数据（3*nDataNum个浮点数）；
	dTarget： 配准数据。输出的变换矩阵乘dSource，映射dTarget的对应点；
	dpError(out)： 每个点配准的误差；
返回：
	整体配准的标准误差。
*************************************************************************/
double CRegICPHxmc::SolveTransfMatrix(int nDataNum, double* pdSource, double* dpTarget)
{
	int    i, j;
	double dStdErr = -1.0;
	Eigen::MatrixXd  matSource, matTarget;
	Eigen::Vector3d  vecScrCenter(0, 0, 0), vecTarCenter(0, 0, 0);

	

	if (nDataNum <= 3)
	{
		return dStdErr;
	}

	//给源和目标(3*nDataNum)矩阵赋值，并计算几何中心
	matSource = Eigen::Map<Eigen::MatrixXd>(pdSource, 3, nDataNum);
	matTarget = Eigen::Map<Eigen::MatrixXd>(dpTarget, 3, nDataNum);
	
	vecScrCenter = matSource.rowwise().sum()/ nDataNum;
	vecTarCenter = matTarget.rowwise().sum()/ nDataNum;

	//中心平移到各自坐标系的原点
	Eigen::MatrixXd  matUcSrc, matUcTar;
	matUcSrc.resize(3, nDataNum);
	matUcTar.resize(3, nDataNum);

#pragma omp parallel for
	for (i = 0; i < nDataNum; i++)
	{
		matUcSrc.col(i) = matSource.col(i) - vecScrCenter;
		matUcTar.col(i) = matTarget.col(i) - vecTarCenter;
	}

	Eigen::Matrix<double, 3, 3> matTemp3x3 = matUcSrc * matUcTar.transpose();
	Eigen::JacobiSVD<Eigen::MatrixXd> svd_holder(matTemp3x3, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXd matU = svd_holder.matrixU();
	Eigen::MatrixXd matV = svd_holder.matrixV();
	Eigen::MatrixXd matD = svd_holder.singularValues();
	Eigen::MatrixXd   matRot = matV * (matU.transpose());
	Eigen::VectorXd   vecT = vecTarCenter - matRot * vecScrCenter;

	for (j = 0; j < 3; j++)
	{
		for (i = 0; i < 3; i++)
		{
			m_dTransMat[j][i] = matRot(j, i);
		}
		m_dTransMat[j][3] = vecT[j];
	}
	m_dTransMat[3][0] = m_dTransMat[3][1] = m_dTransMat[3][2] = 0.0;
	m_dTransMat[3][3] = 1.0;

	return 0.0;
}

/************************************************************************
说明：
	SVD方法求解对应点的刚性变换矩阵。至少输入对应4个点的坐标。
	注意：输出的变换矩阵在成员 m_dTransMat保存，用GetTransMatrix得到结果。
参数：
	nDataNum：对应点的个数；
	dSource： 待配准的数据（3*nDataNum个浮点数）；
	dTarget： 配准数据。输出的变换矩阵乘dSource，映射dTarget的对应点；
	dpError(out)： 每个点配准的误差；
返回：
	整体配准的标准误差。
*************************************************************************/
double CRegICPHxmc::SolveTransfMatrix(int nDataNum, double dSource[][3], double dTarget[][3], double dError[])
{
	int    i, j;
	double dStdErr = -1.0;
	Eigen::MatrixXd  matSource, matTarget;
	Eigen::Vector3d  vecScrCenter(0, 0, 0), vecTarCenter(0, 0, 0);

	if (nDataNum <= 3)
	{
		return dStdErr;
	}

	//给源和目标(3*nDataNum)矩阵赋值，并计算几何中心
	matSource.resize(3, nDataNum);
	matTarget.resize(3, nDataNum);
	for (i = 0; i < 3; i++)
	{
		double dSrcSum = 0.0;
		double dTarSum = 0.0;
#pragma omp parallel for reduction (+:dSrcSum,dTarSum)
		for (j = 0; j < nDataNum; j++)
		{
			matSource(i, j) = dSource[j][i];
			matTarget(i, j) = dTarget[j][i];
			dSrcSum += matSource(i, j);
			dTarSum += matTarget(i, j);
		}
		vecScrCenter(i) = dSrcSum/nDataNum;
		vecTarCenter(i) = dTarSum/nDataNum;
	}

	//中心平移到各自坐标系的原点
	Eigen::MatrixXd  matUcSrc, matUcTar;
	matUcSrc.resize(3, nDataNum);
	matUcTar.resize(3, nDataNum);
#pragma omp parallel for
	for (i = 0; i < nDataNum; i++)
	{
		matUcSrc.col(i) = matSource.col(i) - vecScrCenter;
		matUcTar.col(i) = matTarget.col(i) - vecTarCenter;
	}

	Eigen::Matrix<double, 3, 3> matTemp3x3 = matUcSrc * matUcTar.transpose();
	Eigen::JacobiSVD<Eigen::MatrixXd> svd_holder(matTemp3x3, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXd matU = svd_holder.matrixU();
	Eigen::MatrixXd matV = svd_holder.matrixV();
	Eigen::MatrixXd matD = svd_holder.singularValues();
	Eigen::MatrixXd   matRot = matV * (matU.transpose());
	Eigen::VectorXd   vecT = vecTarCenter - matRot * vecScrCenter;

	for (j = 0; j < 3; j++)
	{
		for (i = 0; i < 3; i++)
		{
			m_dTransMat[j][i] = matRot(j, i);
		}
		m_dTransMat[j][3] = vecT[j];
	}
	m_dTransMat[3][0] = m_dTransMat[3][1] = m_dTransMat[3][2] = 0.0;
	m_dTransMat[3][3] = 1.0;

	Eigen::MatrixXd  matError = matRot * matSource - matTarget;

	dStdErr = 0.0;
	for (j = 0; j < nDataNum; j++)
	{
		dError[j] = 0.0;
		matError.col(j) += vecT;
		for (i = 0; i < 3; i++)
		{
			dError[j] += matError(i, j) * matError(i, j);
		}
		dError[j] = sqrt(dError[j]);
		dStdErr += dError[j] * dError[j];
	}
	dStdErr = sqrt(dStdErr / (nDataNum - 1.0));

	return dStdErr;
}

/************************************************************************
说明：
	SVD方法求解5特征点的刚性变换矩阵。前提是本类对象已经得到了两个对应
	特征点的值了。
	注意：输出的变换矩阵在成员 m_dTransMat保存，用GetTransMatrix得到结果。
参数：
	dError(out)：      每个点配准的标准误差；
返回：
	整体配准的标准误差。
*************************************************************************/
double CRegICPHxmc::SolveTransfMatrix(double dError[])
{
	int    i, j;
	double dStdErr = -1.0;
	Eigen::MatrixXd  matSource, matTarget;
	Eigen::Vector3d  vecScrCenter(0, 0, 0), vecTarCenter(0, 0, 0);

	//给源和目标(3*FEAT_NUM_HX)矩阵赋值，并计算几何中心
	matSource.resize(3, FEAT_NUM_HX);
	matTarget.resize(3, FEAT_NUM_HX);
	for (i = 0; i < 3; i++)
	{
		double dSrcSum = 0.0;
		double dTarSum = 0.0;
#pragma omp parallel for reduction (+:dSrcSum,dTarSum)
		for (j = 0; j < FEAT_NUM_HX; j++)
		{
			matSource(i, j) = m_d5FeatCam[j][i];
			matTarget(i, j) = m_d5FeatMri[j][i];
			dSrcSum += matSource(i, j);
			dTarSum += matTarget(i, j);
		}
		vecScrCenter(i) = dSrcSum / FEAT_NUM_HX;
		vecTarCenter(i) = dTarSum / FEAT_NUM_HX;
	}

	//中心平移到各自坐标系的原点
	Eigen::MatrixXd  matUcSrc, matUcTar;
	matUcSrc.resize(3, FEAT_NUM_HX);
	matUcTar.resize(3, FEAT_NUM_HX);
#pragma omp parallel for
	for (i = 0; i < FEAT_NUM_HX; i++)
	{
		matUcSrc.col(i) = matSource.col(i) - vecScrCenter;
		matUcTar.col(i) = matTarget.col(i) - vecTarCenter;
	}

	Eigen::Matrix<double, 3, 3> matTemp3x3 = matUcSrc * matUcTar.transpose();
	Eigen::JacobiSVD<Eigen::MatrixXd> svd_holder(matTemp3x3, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXd matU = svd_holder.matrixU();
	Eigen::MatrixXd matV = svd_holder.matrixV();
	Eigen::MatrixXd matD = svd_holder.singularValues();
	Eigen::MatrixXd   matRot = matV * (matU.transpose());
	Eigen::VectorXd   vecT = vecTarCenter - matRot * vecScrCenter;

	for (j = 0; j < 3; j++)
	{
		for (i = 0; i < 3; i++)
		{
			m_dTransMat[j][i] = matRot(j, i);
		}
		m_dTransMat[j][3] = vecT[j];
	}
	m_dTransMat[3][0] = m_dTransMat[3][1] = m_dTransMat[3][2] = 0.0;
	m_dTransMat[3][3] = 1.0;

	Eigen::MatrixXd  matError = matRot * matSource - matTarget;

	dStdErr = 0.0;
	for (j = 0; j < FEAT_NUM_HX; j++)
	{
		dError[j] = 0.0;
		matError.col(j) += vecT;
		for (i = 0; i < 3; i++)
		{
			dError[j] += matError(i, j) * matError(i, j);
		}
		dError[j] = sqrt(dError[j]);
		dStdErr += dError[j] * dError[j];
	}
	dStdErr = sqrt(dStdErr / (FEAT_NUM_HX - 1.0));

	return dStdErr;
}

/************************************************************************
说明：
	Jacobi旋转法迭代求解n*n对称方阵的特征根与特征矢量。
	特征值是降序排列的，特征矢量是归一化的。
参数：
	a：输入的n*n对称矩阵；
	n：输入矩阵的维度；
	w：输出的n个特征值；
	v：输出的n*n特征矢量矩阵（列矢量）；
	
返回：
	超过迭代次数返回0，否则返回1。
*************************************************************************/
template<class T>
int CRegICPHxmc::JacobiN(T** a, int n, T* w, T** v)
{
	int i, j, k, iq, ip, numPos;
	T tresh, theta, tau, t, sm, s, h, g, c, tmp;
	T bspace[4], zspace[4];
	T* b = bspace;
	T* z = zspace;

	// only allocate memory if the matrix is large
	if (n > 4)
	{
		b = new T[n];
		z = new T[n];
	}

	// initialize
	for (ip = 0; ip < n; ip++)
	{
		for (iq = 0; iq < n; iq++)
		{
			v[ip][iq] = 0.0;
		}
		v[ip][ip] = 1.0;
	}
	for (ip = 0; ip < n; ip++)
	{
		b[ip] = w[ip] = a[ip][ip];
		z[ip] = 0.0;
	}

	// begin rotation sequence
	for (i = 0; i < JACOBI_MAX_ROTATIONS; i++)
	{
		sm = 0.0;
		for (ip = 0; ip < n - 1; ip++)
		{
			for (iq = ip + 1; iq < n; iq++)
			{
				sm += fabs(a[ip][iq]);
			}
		}
		if (sm == 0.0)
		{
			break;
		}

		if (i < 3)                                // first 3 sweeps
		{
			tresh = 0.2 * sm / (n * n);
		}
		else
		{
			tresh = 0.0;
		}

		for (ip = 0; ip < n - 1; ip++)
		{
			for (iq = ip + 1; iq < n; iq++)
			{
				g = 100.0 * fabs(a[ip][iq]);

				// after 4 sweeps
				if (i > 3 && (fabs(w[ip]) + g) == fabs(w[ip])
					&& (fabs(w[iq]) + g) == fabs(w[iq]))
				{
					a[ip][iq] = 0.0;
				}
				else if (fabs(a[ip][iq]) > tresh)
				{
					h = w[iq] - w[ip];
					if ((fabs(h) + g) == fabs(h))
					{
						t = (a[ip][iq]) / h;
					}
					else
					{
						theta = 0.5 * h / (a[ip][iq]);
						t = 1.0 / (fabs(theta) + sqrt(1.0 + theta * theta));
						if (theta < 0.0)
						{
							t = -t;
						}
					}
					c = 1.0 / sqrt(1 + t * t);
					s = t * c;
					tau = s / (1.0 + c);
					h = t * a[ip][iq];
					z[ip] -= h;
					z[iq] += h;
					w[ip] -= h;
					w[iq] += h;
					a[ip][iq] = 0.0;

					// ip already shifted left by 1 unit
					for (j = 0; j <= ip - 1; j++)
					{
						g = a[j][ip];
						h = a[j][iq];
						a[j][ip] = g - s * (h + g * tau);
						a[j][iq] = h + s * (g - h * tau);
					}
					// ip and iq already shifted left by 1 unit
					for (j = ip + 1; j <= iq - 1; j++)
					{
						g = a[ip][j];
						h = a[j][iq];
						a[ip][j] = g - s * (h + g * tau);
						a[j][iq] = h + s * (g - h * tau);
					}
					// iq already shifted left by 1 unit
					for (j = iq + 1; j < n; j++)
					{
						g = a[ip][j];
						h = a[iq][j];
						a[ip][j] = g - s * (h + g * tau);
						a[iq][j] = h + s * (g - h * tau);
					}
					for (j = 0; j < n; j++)
					{
						g = v[j][ip];
						h = v[j][iq];
						v[j][ip] = g - s * (h + g * tau);
						v[j][iq] = h + s * (g - h * tau);
					}
				}
			}
		}

		for (ip = 0; ip < n; ip++)
		{
			b[ip] += z[ip];
			w[ip] = b[ip];
			z[ip] = 0.0;
		}
	}

	//// this is NEVER called
	if (i >= JACOBI_MAX_ROTATIONS)
	{
		return 0;
	}

	// sort eigenfunctions                 these changes do not affect accuracy
	for (j = 0; j < n - 1; j++)                  // boundary incorrect
	{
		k = j;
		tmp = w[k];
		for (i = j + 1; i < n; i++)                // boundary incorrect, shifted already
		{
			if (w[i] >= tmp)                   // why exchange if same?
			{
				k = i;
				tmp = w[k];
			}
		}
		if (k != j)
		{
			w[k] = w[j];
			w[j] = tmp;
			for (i = 0; i < n; i++)
			{
				tmp = v[i][j];
				v[i][j] = v[i][k];
				v[i][k] = tmp;
			}
		}
	}
	// insure eigenvector consistency (i.e., Jacobi can compute vectors that
	// are negative of one another (.707,.707,0) and (-.707,-.707,0). This can
	// reek havoc in hyperstreamline/other stuff. We will select the most
	// positive eigenvector.
	int ceil_half_n = (n >> 1) + (n & 1);
	for (j = 0; j < n; j++)
	{
		for (numPos = 0, i = 0; i < n; i++)
		{
			if (v[i][j] >= 0.0)
			{
				numPos++;
			}
		}
		//    if ( numPos < ceil(double(n)/double(2.0)) )
		if (numPos < ceil_half_n)
		{
			for (i = 0; i < n; i++)
			{
				v[i][j] *= -1.0;
			}
		}
	}

	if (n > 4)
	{
		delete[] b;
		delete[] z;
	}

	return 1;
}

/************************************************************************
说明：
	计算3D空间点集（点云）的最小包围盒（OBB）。包围盒由参数：
	角点corner，和三个相互垂直的矢量构成盒子的三个轴。
参数：
	dPtCoords：   输入空间点坐标；
	nPtNum：      空点点的数量；
	corner(out)： 包围盒角点的坐标；
	max(out)：    盒子最长轴的矢量（长度和方向）；
	mid(out)：    盒子中长轴的矢量（长度和方向）；
	min(out)：    盒子最短轴的矢量（长度和方向）；
	size(out)：   点集协方差矩阵的特征值。
返回：
	无。
*************************************************************************/
void CRegICPHxmc::ComputeOBB(double dPtCoords[][3], int nPtNum, double corner[3], double max[3],
	double mid[3], double min[3], double size[3])
{
	int    i, j;
	double  mean[3], xp[3], * v[3], v0[3], v1[3], v2[3];
	double* a[3], a0[3], a1[3], a2[3];
	double tMin[3], tMax[3], closest[3], t;

	// Compute mean
	mean[0] = mean[1] = mean[2] = 0.0;
	for (j = 0; j < nPtNum; j++)
	{
		for (i = 0; i < 3; i++)
		{
			mean[i] += dPtCoords[j][i];
		}
	}
	for (i = 0; i < 3; i++)
	{
		mean[i] /= nPtNum;
	}

	// Compute covariance matrix
	a[0] = a0; a[1] = a1; a[2] = a2;
	for (i = 0; i < 3; i++)
	{
		a0[i] = a1[i] = a2[i] = 0.0;
	}

	for (j = 0; j < nPtNum; j++)
	{
		xp[0] = dPtCoords[j][0] - mean[0]; xp[1] = dPtCoords[j][1] - mean[1]; xp[2] = dPtCoords[j][2] - mean[2];
		for (i = 0; i < 3; i++)
		{
			a0[i] += xp[0] * xp[i];
			a1[i] += xp[1] * xp[i];
			a2[i] += xp[2] * xp[i];
		}
	}//for all points

	for (i = 0; i < 3; i++)
	{
		a0[i] /= nPtNum;
		a1[i] /= nPtNum;
		a2[i] /= nPtNum;
	}

	// Extract axes (i.e., eigenvectors) from covariance matrix.
	v[0] = v0; v[1] = v1; v[2] = v2;
	JacobiN<double>(a, 3, size, v);
	max[0] = v[0][0]; max[1] = v[1][0]; max[2] = v[2][0];
	mid[0] = v[0][1]; mid[1] = v[1][1]; mid[2] = v[2][1];
	min[0] = v[0][2]; min[1] = v[1][2]; min[2] = v[2][2];

	for (i = 0; i < 3; i++)
	{
		a[0][i] = mean[i] + max[i];
		a[1][i] = mean[i] + mid[i];
		a[2][i] = mean[i] + min[i];
	}

	// Create oriented bounding box by projecting points onto eigenvectors.
	tMin[0] = tMin[1] = tMin[2] = DOUBLE_MAX_HX;
	tMax[0] = tMax[1] = tMax[2] = -DOUBLE_MAX_HX;

	for (j = 0; j < nPtNum; j++)
	{
		for (i = 0; i < 3; i++)
		{
			DistanceToLine(dPtCoords[j], mean, a[i], t, closest);
			if (t < tMin[i])
			{
				tMin[i] = t;
			}
			if (t > tMax[i])
			{
				tMax[i] = t;
			}
		}
	}//for all points

	for (i = 0; i < 3; i++)
	{
		corner[i] = mean[i] + tMin[0] * max[i] + tMin[1] * mid[i] + tMin[2] * min[i];

		max[i] = (tMax[0] - tMin[0]) * max[i];
		mid[i] = (tMax[1] - tMin[1]) * mid[i];
		min[i] = (tMax[2] - tMin[2]) * min[i];
	}
}

/************************************************************************
说明：
	计算3D空间点集（点云）的最小包围盒（OBB）。包围盒由参数：
	角点corner，和三个相互垂直的矢量构成盒子的三个轴。
参数：
	corner(out)： 包围盒角点的坐标；
	max(out)：    盒子最长轴的矢量（长度和方向）；
	mid(out)：    盒子中长轴的矢量（长度和方向）；
	min(out)：    盒子最短轴的矢量（长度和方向）；
	size(out)：   点集协方差矩阵的特征值。
返回：
	无。
*************************************************************************/
void CRegICPHxmc::ComputeOBB(double corner[3], double max[3], double mid[3], double min[3], double size[3])
{
	ComputeOBB(m_d5FeatCam, FEAT_NUM_HX, corner, max, mid, min, size);
}

double CRegICPHxmc::PtLineProjF(float fPt[3], double dP1[3], double dP2[3])
{
	double dPt[3];

	dPt[0] = (double)fPt[0];
	dPt[1] = (double)fPt[1];
	dPt[2] = (double)fPt[2];

	double dDenom = (dP2[0] - dP1[0]) * (dP2[0] - dP1[0])
		+ (dP2[1] - dP1[1]) * (dP2[1] - dP1[1])
		+ (dP2[2] - dP1[2]) * (dP2[2] - dP1[2]);
	double dNumer = (dPt[0] - dP1[0]) * (dP2[0] - dP1[0])
		+ (dPt[1] - dP1[1]) * (dP2[1] - dP1[1])
		+ (dPt[2] - dP1[2]) * (dP2[2] - dP1[2]);

	return dNumer / dDenom;
}
/************************************************************************
说明：
	计算3D空间点到线段的投影点位置：返回的是直线参数方程的t值。
参数：
	dPt：空间待求点坐标；
	dP1：线段起点坐标；
	dP2：线段终点坐标。
返回：
	t值：参数方程：P = P1 + t*(P2 - P1)；P直线上的3D点，坐标(Px,Py,Pz)。
	0<t<1时，投影点在线段上，否则在线段外。
*************************************************************************/
double CRegICPHxmc::PtLineProj(double dPt[3], double dP1[3], double dP2[3])
{
	double dDenom = (dP2[0] - dP1[0]) * (dP2[0] - dP1[0])
		+ (dP2[1] - dP1[1]) * (dP2[1] - dP1[1])
		+ (dP2[2] - dP1[2]) * (dP2[2] - dP1[2]);
	double dNumer = (dPt[0] - dP1[0]) * (dP2[0] - dP1[0])
		+ (dPt[1] - dP1[1]) * (dP2[1] - dP1[1])
		+ (dPt[2] - dP1[2]) * (dP2[2] - dP1[2]);

	return dNumer / dDenom;
}

/************************************************************************
说明：
	计算3D空间点到线段的投影点位置：返回的是直线参数方程的t值。
参数：
	x：      空间待求点坐标；
	p1：     线段起点坐标；
	p2：     线段终点坐标；
	t(out)： 参数方程：P = P1 + t*(P2 - P1)；
	closestPoint(out)：x到直线的最近点（不一定是投影点，当0<t<1时为
	投影点，否则为起点或终点）。
返回：
	x到closestPoint的距离的平方。
*************************************************************************/
double CRegICPHxmc::DistanceToLine(double x[3], double p1[3], double p2[3],
	double& t, double* closestPoint)
{
	double p21[3], denom, num;
	double* closest;
	double tolerance;
	
	//   Determine appropriate vectors
	p21[0] = p2[0] - p1[0];
	p21[1] = p2[1] - p1[1];
	p21[2] = p2[2] - p1[2];

	//   Get parametric location
	num = p21[0] * (x[0] - p1[0]) + p21[1] * (x[1] - p1[1]) + p21[2] * (x[2] - p1[2]);
	denom = p21[0]*p21[0] + p21[1] * p21[1] + p21[2] * p21[2];

	// trying to avoid an expensive fabs
	tolerance = TOLERANCE_HX * num;
	if (tolerance < 0.0)
	{
		tolerance = -tolerance;
	}
	if (-tolerance < denom && denom < tolerance) //numerically bad!
	{
		closest = p1; //arbitrary, point is (numerically) far away
	}
	
	// If parametric coordinate is within 0<=p<=1, then the point is closest to
	// the line.  Otherwise, it's closest to a point at the end of the line.
	else if (denom <= 0.0 || (t = num / denom) < 0.0)
	{
		closest = p1;
	}
	else if (t > 1.0)
	{
		closest = p2;
	}
	else
	{
		closest = p21;
		p21[0] = p1[0] + t * p21[0];
		p21[1] = p1[1] + t * p21[1];
		p21[2] = p1[2] + t * p21[2];
	}

	if (closestPoint)
	{
		closestPoint[0] = closest[0];
		closestPoint[1] = closest[1];
		closestPoint[2] = closest[2];
	}

	double dDist = (closest[0] - x[0]) * (closest[0] - x[0])
		+ (closest[1] - x[1]) * (closest[1] - x[1])
		+ (closest[2] - x[2]) * (closest[2] - x[2]);

	return  dDist;
}

/************************************************************
说明：
	计算给定包围盒内的点云：包围盒为一长方形的空间盒子。
参数：
	pdPtSetIn：      待滤波的点集；
	pdPtSetOut(out)：滤波后的点集；
	corner：包围盒的角点；
	max：   包围盒的最长轴；
	mid：   包围盒的中长轴；
	min：   包围盒的短轴；
返回：
	在盒子内即滤波后的点数量。
**************************************************************/
int CRegICPHxmc::BoundBoxFilter(double* pdPtSetIn, int nPtNum, double** ppdPtSetOut,
	double corner[3], double max[3], double mid[3], double min[3])
{
	//取出盒子内的点
	double* pdPtSetOut;
	double  dP1Long[3], dP2Long[3], dP1Mid[3], dP2Mid[3], dP1Min[3], dP2Min[3];

	dP1Long[0] = dP1Mid[0] = dP1Min[0] = corner[0];
	dP1Long[1] = dP1Mid[1] = dP1Min[1] = corner[1];
	dP1Long[2] = dP1Mid[2] = dP1Min[2] = corner[2];
	dP2Long[0] = corner[0] + max[0];
	dP2Long[1] = corner[1] + max[1];
	dP2Long[2] = corner[2] + max[2];
	dP2Mid[0] = corner[0] + mid[0];
	dP2Mid[1] = corner[1] + mid[1];
	dP2Mid[2] = corner[2] + mid[2];
	dP2Min[0] = corner[0] + min[0];
	dP2Min[1] = corner[1] + min[1];
	dP2Min[2] = corner[2] + min[2];

	int* pnIndex = new int[nPtNum];
	//memset(pnIndex, nPtNum * sizeof(int), 0);

	int nOut = 0;

#pragma omp parallel for reduction(+:nOut) firstprivate(dP1Long,dP2Long,dP1Mid,dP2Mid)
	for (int i = 0; i < nPtNum; i++)
	{
		double t1, t2, t3;
		double dNowPt[3];
		dNowPt[0] = pdPtSetIn[i * 3];
		dNowPt[1] = pdPtSetIn[i * 3 + 1];
		dNowPt[2] = pdPtSetIn[i * 3 + 2];
		t1 = PtLineProj(dNowPt, dP1Long, dP2Long);
		t2 = PtLineProj(dNowPt, dP1Mid, dP2Mid);
		//t3 = PtLineProj(dNowPt, dP1Min, dP2Min);

		if ((t1 > 0 && t1 < 1) && (t2 > 0 && t2 < 1)) //&& (t3 > 0 && t3 < 1))
		{
			pnIndex[i] = 1;
			nOut++;
		}
		else
		{
			pnIndex[i] = 0;
		}
	}

	pdPtSetOut = new double[3* nOut *sizeof(double)];

	int nC = 0;
	for (int i = 0; i < nPtNum; i++)
	{
		if (pnIndex[i] == 1)
		{
			pdPtSetOut[nC * 3] = pdPtSetIn[i * 3];
			pdPtSetOut[nC * 3 + 1] = pdPtSetIn[i * 3 + 1];
			pdPtSetOut[nC * 3 + 2] = pdPtSetIn[i * 3 + 2];
			nC++;
		}
	}

	delete[] pnIndex;

	*ppdPtSetOut = pdPtSetOut;

	return nC;
}

/************************************************************
说明：
	计算给定包围盒内的点云：包围盒为一长方形的空间盒子。
参数：
	vecSrcPtIn：待滤波的点集；
	vecPtOut(out)：滤波后的点集；
	corner：包围盒的角点；
	max： 包围盒的最长轴；
	mid： 包围盒的中长轴；
	min： 包围盒的短轴；
返回：
	在盒子内的点数量。
**************************************************************/
int CRegICPHxmc::BoundBoxFilter(std::vector<fPoint_Hxmx>vecSrcPtIn, std::vector<fPoint_Hxmx>& vecPtOut,
	double corner[3], double max[3], double mid[3], double min[3])
{
	//取出盒子内的点
	double dP1Long[3], dP2Long[3], dP1Mid[3], dP2Mid[3], dP1Min[3], dP2Min[3];
	double t1, t2, t3;

	dP1Long[0] = dP1Mid[0] = dP1Min[0] = corner[0];
	dP1Long[1] = dP1Mid[1] = dP1Min[1] = corner[1];
	dP1Long[2] = dP1Mid[2] = dP1Min[2] = corner[2];
	dP2Long[0] = corner[0] + max[0];
	dP2Long[1] = corner[1] + max[1];
	dP2Long[2] = corner[2] + max[2];
	dP2Mid[0] = corner[0] + mid[0];
	dP2Mid[1] = corner[1] + mid[1];
	dP2Mid[2] = corner[2] + mid[2];
	dP2Min[0] = corner[0] + min[0];
	dP2Min[1] = corner[1] + min[1];
	dP2Min[2] = corner[2] + min[2];

	vecPtOut.clear();

	int nPtNum = vecSrcPtIn.size();
	int* pnIndex = new int[nPtNum];
	memset(pnIndex, nPtNum * sizeof(int), 0);

#pragma omp parallel for firstprivate(dP1Long,dP2Long,dP1Mid,dP2Mid,dP1Min,dP2Min)
	for (int i = 0; i < nPtNum; i++)
	{
		t1 = PtLineProjF(vecSrcPtIn[i].fPt, dP1Long, dP2Long);
		t2 = PtLineProjF(vecSrcPtIn[i].fPt, dP1Mid, dP2Mid);
		t3 = PtLineProjF(vecSrcPtIn[i].fPt, dP1Min, dP2Min);

		if ((t1 > 0 && t1 < 1) && (t2 > 0 && t2 < 1) && (t3 > 0 && t3 < 1))
		{
			pnIndex[i] = 1;
		}
	}

	for (int i = 0; i < nPtNum; i++)
	{
		if (pnIndex[i] == 1)
		{
			vecPtOut.push_back(vecSrcPtIn[i]);
		}
	}

	delete[] pnIndex;

	return vecPtOut.size();
}

/************************************************************
说明：
	计算给定包围盒内的点云：包围盒为一长方形的空间盒子。
参数：
	ptCorner：包围盒的角点；
	axisMax： 包围盒的最长轴； 
	axisMid： 包围盒的中长轴；
	axisMin： 包围盒的短轴；
	dPtCoords： 输入的点云；
	nPtNum：    输入点云的点数量；
	pnIndex(out)：滤波后的点云索引（对输入点云：
	              0-不在盒子内；1：在盒子内）。
返回：
	在盒子内的点数量。
**************************************************************/
int CRegICPHxmc::BoundBoxFilter(double ptCorner[3], double axisMax[3], 
	double axisMid[3], double axisMin[3],double dPtCoords[][3], int nPtNum, int* pnIndex)
{
	int nInNum = 0;

	memset(pnIndex, nPtNum * sizeof(int), 0);

	//取出盒子内的点
	double dP1Long[3], dP2Long[3], dP1Mid[3], dP2Mid[3], dP1Min[3], dP2Min[3];
	double t1, t2, t3;
	dP1Long[0] = dP1Mid[0] = dP1Min[0] = ptCorner[0];
	dP1Long[1] = dP1Mid[1] = dP1Min[1] = ptCorner[1];
	dP1Long[2] = dP1Mid[2] = dP1Min[2] = ptCorner[2];
	dP2Long[0] = ptCorner[0] + axisMax[0];
	dP2Long[1] = ptCorner[1] + axisMax[1];
	dP2Long[2] = ptCorner[2] + axisMax[2];
	dP2Mid[0] = ptCorner[0] + axisMid[0];
	dP2Mid[1] = ptCorner[1] + axisMid[1];
	dP2Mid[2] = ptCorner[2] + axisMid[2];
	dP2Min[0] = ptCorner[0] + axisMin[0];
	dP2Min[1] = ptCorner[1] + axisMin[1];
	dP2Min[2] = ptCorner[2] + axisMin[2];

	//通过计算投影点确定改点是否在盒子内
#pragma omp parallel for reduction(+:nInNum) firstprivate(dP1Long,dP2Long,dP1Mid,dP2Mid,dP1Min,dP2Min)
	for (int i = 0; i < nPtNum; i++)
	{
		t1 = PtLineProj(dPtCoords[i], dP1Long, dP2Long);
		t2 = PtLineProj(dPtCoords[i], dP1Mid, dP2Mid);
		t3 = PtLineProj(dPtCoords[i], dP1Min, dP2Min);

		if ((t1 > 0 && t1 < 1) && (t2 > 0 && t2 < 1) && (t3 > 0 && t3 < 1))
		{
			pnIndex[i] = 1;
			nInNum++;
		}
	}

	return nInNum;
}

/************************************************************
说明：
	构建Knn的查询结构。
参数：
	pfRefPtSet：参考点集的坐标；
	nPtNum： 参考点集的点数量。
返回：
	无。
**************************************************************/
void CRegICPHxmc::BuildIndex(double* pdRefPtSet, int nPtNum)
{
	g_flmatRef = flann::Matrix<double>(pdRefPtSet, nPtNum, 3);
	g_flannIndexObj.buildIndex(g_flmatRef);
}

/*************************************************************************
说明：
	查询结构输入点集与已经构建好的树最近点对，求出参考点集
	中对应点索引和相应的平方距离。
	注意：查询对应点索引存放在成员 m_pnIndexSet 指向的存储空间里
	      对应点之间的距离的平方存放在成员 m_pfDistance 指向的存储空间里。
参数：
	pfQuePtSet：查询点集的3D空间坐标；
	nPtNum： 查询点集的点数量。
返回：
	对应点之间的平均距离。
**************************************************************************/
double CRegICPHxmc::CalcNearestPointPairs(double* pdQuePtSet, int nPtNum)
{
	int    nKnn = 1;
	double dAveDist = 0.0;

	flann::Matrix<double> flmatQuery = flann::Matrix<double>(pdQuePtSet, nPtNum, 3);

	if (nPtNum > RESERVED_QUE_NUM)
	{
		delete[] m_pnIndexSet;
		delete[] m_pdDistance;

		m_pdDistance = new double[nPtNum];
		m_pnIndexSet = new int[nPtNum];
	}

	flann::Matrix<int>   flmatIndices(m_pnIndexSet, nPtNum, nKnn);
	flann::Matrix<double> flmatDists(m_pdDistance, nPtNum, nKnn);
	g_flannIndexObj.knnSearch(flmatQuery, flmatIndices, flmatDists, nKnn, flann::SearchParams(128));

#pragma omp parallel for reduction(+:dAveDist)        //采用openmmp加速
	for (int i = 0; i < nPtNum; ++i)
	{
		double dD1 = sqrt(flmatDists.ptr()[i]);
		dAveDist += dD1;
	}

	dAveDist /= nPtNum;

	return dAveDist;
}

/*************************************************************************
说明：
	点对点的ICP算法实现。如果点云之间很接近，建议迭代次数不要大于10。
	前提条件：目标点云已经输入本类对象，且用BuildIndex建立了查询索引。
参数：
	pdSrcPtSet：源点集（点云）；
	nPtNum：    源点集的点数量；
	nIterNum：  最大迭代次数，如果迭代后近邻点之间平均距离小于设定值
	            也将推出迭代；
	dMatrix(out)：输出的变换矩阵。
返回：
	对应点之间的平均距离。
**************************************************************************/
double CRegICPHxmc::RegistByICP(double* pdSrcPtSet, int nPtNum, int nIterNum, double dMatrix[4][4])
{
	int    nKnn = 1, nIter = 0;
	double dAveDist = 100.0;
	
	double* pdRefData = g_flmatRef.ptr();

	//如果预留的查询和对应点空间不够，重新申请内存
	if (nPtNum > m_nReservPtNum)
	{
		delete[] m_pdRefPtSet;
		m_pdRefPtSet = new double[3 * nPtNum];

		delete[] m_pdQuePtSet;
		m_pdQuePtSet = new double[3 * nPtNum];

		delete[] m_pdDistance;
		m_pdDistance = new double[nPtNum];
	
		delete[] m_pnIndexSet;
		m_pnIndexSet = new int[nPtNum];

		m_nReservPtNum = nPtNum;
	}

	memcpy((void*)m_pdQuePtSet, pdSrcPtSet, 3 * nPtNum * sizeof(double));

	//建立查询点集的矩阵和查询结构矩阵
	flann::Matrix<double> flmatQuery = flann::Matrix<double>(m_pdQuePtSet, nPtNum, 3);
	flann::Matrix<int>    flmatIndices(m_pnIndexSet, nPtNum, nKnn);
	flann::Matrix<double> flmatDists(m_pdDistance, nPtNum, nKnn);

	//矩阵初始化
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			dMatrix[i][j] = 0.0;
			m_dFinalTransMat[i][j] = 0.0;
		}
		dMatrix[i][i] = 1.0;
		m_dFinalTransMat[i][i] = 1.0;
	}

	//开始迭代
	while (dAveDist > m_dMinDist && nIter < nIterNum)
	{
		nIter++;
		g_flannIndexObj.knnSearch(flmatQuery, flmatIndices, flmatDists, nKnn, flann::SearchParams(128));

		double dSum = 0.0;
		//取出参考点集中对应点，并计算平均距离
#pragma omp parallel for reduction(+:dSum)            //采用openmmp加速
		for (int i = 0; i < nPtNum; i++)
		{
			m_pdRefPtSet[3 * i] = pdRefData[3 * m_pnIndexSet[i]];
			m_pdRefPtSet[3 * i + 1] = pdRefData[3 * m_pnIndexSet[i] + 1];
			m_pdRefPtSet[3 * i + 2] = pdRefData[3 * m_pnIndexSet[i] + 2];

			double dD1 = sqrt(flmatDists.ptr()[i]);
			dSum += dD1;
		}
		
		dAveDist = dSum / nPtNum;

		SolveTransfMatrix(nPtNum, m_pdQuePtSet, m_pdRefPtSet);

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				dMatrix[i][j] = 0.0;
				for (int k = 0; k < 4; k++)
				{
					dMatrix[i][j] += m_dTransMat[i][k] * m_dFinalTransMat[k][j];
				}
			}
		}

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				m_dFinalTransMat[i][j] = dMatrix[i][j];
			}
		}

		double a11 = m_dTransMat[0][0];
		double a12 = m_dTransMat[0][1];
		double a13 = m_dTransMat[0][2];
		double a14 = m_dTransMat[0][3];
		double a21 = m_dTransMat[1][0];
		double a22 = m_dTransMat[1][1];
		double a23 = m_dTransMat[1][2];
		double a24 = m_dTransMat[1][3];
		double a31 = m_dTransMat[2][0];
		double a32 = m_dTransMat[2][1];
		double a33 = m_dTransMat[2][2];
		double a34 = m_dTransMat[2][3];

#pragma omp parallel for firstprivate(a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34)
		for (int i = 0; i < nPtNum; i++)
		{
			double x, y, z;
			x = m_pdQuePtSet[3 * i] * a11 + m_pdQuePtSet[3 * i + 1] * a12 + m_pdQuePtSet[3 * i + 2] * a13 + a14;
			y = m_pdQuePtSet[3 * i] * a21 + m_pdQuePtSet[3 * i + 1] * a22 + m_pdQuePtSet[3 * i + 2] * a23 + a24;
			z = m_pdQuePtSet[3 * i] * a31	+ m_pdQuePtSet[3 * i + 1] * a32 + m_pdQuePtSet[3 * i + 2] * a33	+ a34;

			m_pdQuePtSet[3 * i] = x;
			m_pdQuePtSet[3 * i + 1] = y;
			m_pdQuePtSet[3 * i + 2] = z;
		}
	}
	
	g_flannIndexObj.knnSearch(flmatQuery, flmatIndices, flmatDists, nKnn, flann::SearchParams(128));
	double dSum = 0.0;
	//并计算平均距离
#pragma omp parallel for reduction(+:dSum)            //采用openmmp加速
	for (int i = 0; i < nPtNum; i++)
	{
		double dD1 = sqrt(flmatDists.ptr()[i]);
		dSum += dD1;
	}

	dAveDist = dSum / nPtNum;

	return dAveDist;
}

/*************************************************************************
说明：
	先采用SVD法求刚性对应点的坐标变换得到初始的配准效果，然后用
	点对点的ICP算法实现精配准。
	建议迭代次数不要大于10。
	前提条件：目标点云已经输入本类对象，且用BuildIndex建立了查询索引。
参数：
	dSrcFea：   源特征点坐标的数组；
	dTarFea：   目标特征点坐标的数组（须对应）；
	nFNum：     特征点的数量；
	pdSrcPtSet：源点集（直接输入从相机中得到的点云）；
	nSrcPtNum： 源点集的点数量；
	nIterNum：  最大迭代次数，如果迭代后近邻点之间平均距离小于设定值
				也将推出迭代；
	dMatrix(out)：输出的变换矩阵。
返回：
	源点云与目标点云对应点之间的平均距离。
**************************************************************************/
double CRegICPHxmc::RegBy5FPlusICP(double dSrcFea[][3], double dTarFea[][3], int nFNum,
	double* pdSrcPtSet, int nSrcPtNum, double dMatrix[4][4])
{
	double dAveDist{-1.0};
	double size[3];
	double dP1Long[3], dP2Long[3], dP1Mid[3], dP2Mid[3], dP1Min[3], dP2Min[3];

	//1. 计算包围盒
	ComputeOBB(dSrcFea, nFNum, m_stObbox.corner, m_stObbox.max, m_stObbox.mid, m_stObbox.min, size);

	//Test code
	//ComputeOBB(dTarFea, nFNum, m_stBaseObbox.corner, m_stBaseObbox.max, m_stBaseObbox.mid, m_stBaseObbox.min, size);

	//2. 用包围盒对源数据进行滤波
	dP1Long[0] = dP1Mid[0] = dP1Min[0] = m_stObbox.corner[0];
	dP1Long[1] = dP1Mid[1] = dP1Min[1] = m_stObbox.corner[1];
	dP1Long[2] = dP1Mid[2] = dP1Min[2] = m_stObbox.corner[2];
	dP2Long[0] = m_stObbox.corner[0] + m_stObbox.max[0];
	dP2Long[1] = m_stObbox.corner[1] + m_stObbox.max[1];
	dP2Long[2] = m_stObbox.corner[2] + m_stObbox.max[2];
	dP2Mid[0] = m_stObbox.corner[0] + m_stObbox.mid[0];
	dP2Mid[1] = m_stObbox.corner[1] + m_stObbox.mid[1];
	dP2Mid[2] = m_stObbox.corner[2] + m_stObbox.mid[2];
	dP2Min[0] = m_stObbox.corner[0] + m_stObbox.min[0];
	dP2Min[1] = m_stObbox.corner[1] + m_stObbox.min[1];
	dP2Min[2] = m_stObbox.corner[2] + m_stObbox.min[2];

	int* pnIndex = new int[nSrcPtNum];
	int  nOut = 0;

#pragma omp parallel for reduction(+:nOut) firstprivate(dP1Long,dP2Long,dP1Mid,dP2Mid,dP1Min,dP2Min)
	for (int i = 0; i < nSrcPtNum; i++)
	{
		double t1, t2, t3;
		double dNowPt[3];
		dNowPt[0] = pdSrcPtSet[i * 3];
		dNowPt[1] = pdSrcPtSet[i * 3 + 1];
		dNowPt[2] = pdSrcPtSet[i * 3 + 2];
		t1 = PtLineProj(dNowPt, dP1Long, dP2Long);
		t2 = PtLineProj(dNowPt, dP1Mid, dP2Mid);
		t3 = PtLineProj(dNowPt, dP1Min, dP2Min);
	
		if ((t1 > 0 && t1 < 1) && (t2 > 0 && t2 < 1) && (t3 > 0 && t3 < 1))
		{
			pnIndex[i] = 1;
			nOut++;
		}
		else
		{
			pnIndex[i] = 0;
		}
	}

	if (m_nReservPtNum < nOut)
	{
		delete[] m_pdRefPtSet;
		delete[] m_pdQuePtSet;
		delete[] m_pnIndexSet;
		delete[] m_pdDistance;

		m_pdRefPtSet = new double[3 * nOut];
		m_pdQuePtSet = new double[3 * nOut];
		m_pnIndexSet = new int[nOut];
		m_pdDistance = new double[nOut];

		m_nReservPtNum = nOut;
	}

	m_nQuePtNum = 0;
	for (int i = 0; i < nSrcPtNum; i++)
	{
		if (pnIndex[i] == 1)
		{
			m_pdQuePtSet[m_nQuePtNum * 3] = pdSrcPtSet[i * 3];
			m_pdQuePtSet[m_nQuePtNum * 3 + 1] = pdSrcPtSet[i * 3 + 1];
			m_pdQuePtSet[m_nQuePtNum * 3 + 2] = pdSrcPtSet[i * 3 + 2];
			m_nQuePtNum++;
		}
	}
	delete[] pnIndex;

	//3. 求初始坐标变换矩阵，结果在成员m_dTransMat里。
	double dStdErr, dErr[32];
	dStdErr = SolveTransfMatrix(nFNum, dSrcFea, dTarFea, dErr);
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			m_dFinalTransMat[i][j] = m_dTransMat[i][j];
		}
	}

	//4. 将滤波源点云用矩阵m_dTransMat变换到指定位置
	double a11 = m_dTransMat[0][0];
	double a12 = m_dTransMat[0][1];
	double a13 = m_dTransMat[0][2];
	double a14 = m_dTransMat[0][3];
	double a21 = m_dTransMat[1][0];
	double a22 = m_dTransMat[1][1];
	double a23 = m_dTransMat[1][2];
	double a24 = m_dTransMat[1][3];
	double a31 = m_dTransMat[2][0];
	double a32 = m_dTransMat[2][1];
	double a33 = m_dTransMat[2][2];
	double a34 = m_dTransMat[2][3];

#pragma omp parallel for firstprivate(a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34)
	for (int i = 0; i < m_nQuePtNum; i++)
	{
		double x, y, z;
		x = m_pdQuePtSet[3 * i] * a11 + m_pdQuePtSet[3 * i + 1] * a12 + m_pdQuePtSet[3 * i + 2] * a13 + a14;
		y = m_pdQuePtSet[3 * i] * a21 + m_pdQuePtSet[3 * i + 1] * a22 + m_pdQuePtSet[3 * i + 2] * a23 + a24;
		z = m_pdQuePtSet[3 * i] * a31 + m_pdQuePtSet[3 * i + 1] * a32 + m_pdQuePtSet[3 * i + 2] * a33 + a34;

		m_pdQuePtSet[3 * i] = x;
		m_pdQuePtSet[3 * i + 1] = y;
		m_pdQuePtSet[3 * i + 2] = z;
	}

	//5. 求对应点以及对应点之间平局距离
	int nKnn = 1;
	flann::Matrix<double> flmatQuery = flann::Matrix<double>(m_pdQuePtSet, m_nQuePtNum, 3);
	flann::Matrix<int>    flmatIndices(m_pnIndexSet, m_nQuePtNum, nKnn);
	flann::Matrix<double> flmatDists(m_pdDistance, m_nQuePtNum, nKnn);
	g_flannIndexObj.knnSearch(flmatQuery, flmatIndices, flmatDists, nKnn, flann::SearchParams(128));
	//取出参考点集中对应点，并计算平均距离
	double* pdRefData = g_flmatRef.ptr();
	double dSum = 0.0;
#pragma omp parallel for reduction(+:dSum)            //采用openmmp加速
	for (int i = 0; i < m_nQuePtNum; i++)
	{
		m_pdRefPtSet[3 * i] = pdRefData[3 * m_pnIndexSet[i]];
		m_pdRefPtSet[3 * i + 1] = pdRefData[3 * m_pnIndexSet[i] + 1];
		m_pdRefPtSet[3 * i + 2] = pdRefData[3 * m_pnIndexSet[i] + 2];

		double dD1 = sqrt(flmatDists.ptr()[i]);
		dSum += dD1;
	}
	dAveDist = dSum / m_nQuePtNum;

	//6. 求变换矩阵，完成第一次迭代
	SolveTransfMatrix(m_nQuePtNum, m_pdQuePtSet, m_pdRefPtSet);
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			dMatrix[i][j] = 0.0;
			for (int k = 0; k < 4; k++)
			{
				dMatrix[i][j] += m_dTransMat[i][k] * m_dFinalTransMat[k][j];
			}
		}
	}

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			m_dFinalTransMat[i][j] = dMatrix[i][j];
		}
	}

	//7. 求平均距离
	a11 = m_dTransMat[0][0];
	a12 = m_dTransMat[0][1];
	a13 = m_dTransMat[0][2];
	a14 = m_dTransMat[0][3];
	a21 = m_dTransMat[1][0];
	a22 = m_dTransMat[1][1];
	a23 = m_dTransMat[1][2];
	a24 = m_dTransMat[1][3];
	a31 = m_dTransMat[2][0];
	a32 = m_dTransMat[2][1];
	a33 = m_dTransMat[2][2];
	a34 = m_dTransMat[2][3];

#pragma omp parallel for firstprivate(a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34)
	for (int i = 0; i < m_nQuePtNum; i++)
	{
		double x, y, z;
		x = m_pdQuePtSet[3 * i] * a11 + m_pdQuePtSet[3 * i + 1] * a12 + m_pdQuePtSet[3 * i + 2] * a13 + a14;
		y = m_pdQuePtSet[3 * i] * a21 + m_pdQuePtSet[3 * i + 1] * a22 + m_pdQuePtSet[3 * i + 2] * a23 + a24;
		z = m_pdQuePtSet[3 * i] * a31 + m_pdQuePtSet[3 * i + 1] * a32 + m_pdQuePtSet[3 * i + 2] * a33 + a34;

		m_pdQuePtSet[3 * i] = x;
		m_pdQuePtSet[3 * i + 1] = y;
		m_pdQuePtSet[3 * i + 2] = z;
	}

	g_flannIndexObj.knnSearch(flmatQuery, flmatIndices, flmatDists, nKnn, flann::SearchParams(128));
	dSum = 0.0;
#pragma omp parallel for reduction(+:dSum)            //采用openmmp加速
	for (int i = 0; i < m_nQuePtNum; i++)
	{
		double dD1 = sqrt(flmatDists.ptr()[i]);
		dSum += dD1;
	}
	dAveDist = dSum / m_nQuePtNum;

	GetInverseMatHomo();
	//GetInverseMatEigen();

	//Test code
	/*double dTempMat[4][4], dTempMat1[4][4];
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			dTempMat[i][j] = 0.0;
			dTempMat1[i][j] = 0.0;
			for (int k = 0; k < 4; k++)
			{
				dTempMat[i][j] += m_dFinalTransMat[i][k] * m_dFinalInversMat[k][j];
				dTempMat1[i][j] += m_dFinalInversMat[i][k] * m_dFinalTransMat[k][j];
			}
		}
	}*/
	//Test end

	GetBaseObBox();
	//GetCurrentObBox();
	
	return dAveDist;
}

/*************************************************************************
说明：
	点对点的ICP算法实现。建议迭代次数不要大于10。
	前提条件：1.目标点云已经输入本类对象，且用BuildIndex建立了查询索引；
	          2.之前的帧已经输入了5个特征点；
			  3.上一帧的配准精度较高（平均距离<1.5mm）。
参数：
	pdSrcPtSet：源点集（直接输入从相机中得到的点云）；
	nPtNum：    源点集的点数量；
	nIterNum：  最大迭代次数，如果迭代后近邻点之间平均距离小于设定值
				也将推出迭代；
	dMatrix(out)：输出的变换矩阵。这个矩阵乘源点云即与目标点云重合。
返回：
	源点云与目标点云对应点之间的平均距离。
**************************************************************************/
double CRegICPHxmc::RegByIcp(double* pdSrcPtSet, int nSrcPtNum, int nIterNum, double dMatrix[4][4])
{
	double dAveDist{ 100.0 };
	double dP1Long[3], dP2Long[3], dP1Mid[3], dP2Mid[3], dP1Min[3], dP2Min[3];

	//根据上一帧结果，得到近似包围盒
	GetCurrentObBox();

	//2. 用包围盒对源数据进行滤波
	dP1Long[0] = dP1Mid[0] = dP1Min[0] = m_stObbox.corner[0];
	dP1Long[1] = dP1Mid[1] = dP1Min[1] = m_stObbox.corner[1];
	dP1Long[2] = dP1Mid[2] = dP1Min[2] = m_stObbox.corner[2];
	dP2Long[0] = m_stObbox.corner[0] + m_stObbox.max[0];
	dP2Long[1] = m_stObbox.corner[1] + m_stObbox.max[1];
	dP2Long[2] = m_stObbox.corner[2] + m_stObbox.max[2];
	dP2Mid[0] = m_stObbox.corner[0] + m_stObbox.mid[0];
	dP2Mid[1] = m_stObbox.corner[1] + m_stObbox.mid[1];
	dP2Mid[2] = m_stObbox.corner[2] + m_stObbox.mid[2];
	dP2Min[0] = m_stObbox.corner[0] + m_stObbox.min[0];
	dP2Min[1] = m_stObbox.corner[1] + m_stObbox.min[1];
	dP2Min[2] = m_stObbox.corner[2] + m_stObbox.min[2];

	int* pnIndex = new int[nSrcPtNum];
	int  nOut = 0;

#pragma omp parallel for reduction(+:nOut) firstprivate(dP1Long,dP2Long,dP1Mid,dP2Mid,dP1Min,dP2Min)
	for (int i = 0; i < nSrcPtNum; i++)
	{
		double t1, t2, t3;
		double dNowPt[3];
		dNowPt[0] = pdSrcPtSet[i * 3];
		dNowPt[1] = pdSrcPtSet[i * 3 + 1];
		dNowPt[2] = pdSrcPtSet[i * 3 + 2];
		t1 = PtLineProj(dNowPt, dP1Long, dP2Long);
		t2 = PtLineProj(dNowPt, dP1Mid, dP2Mid);
		t3 = PtLineProj(dNowPt, dP1Min, dP2Min);

		if ((t1 > 0 && t1 < 1) && (t2 > 0 && t2 < 1) && (t3 > 0 && t3 < 1))
		{
			pnIndex[i] = 1;
			nOut++;
		}
		else
		{
			pnIndex[i] = 0;
		}
	}

	if (MIN_REGIST_NUM > nOut)    //相机点云过滤后的点少于这个值将返回很大的平均距离:MAX_AVE_DIST
	{
		return MAX_AVE_DIST;
	}

	if (m_nReservPtNum < nOut)
	{
		delete[] m_pdRefPtSet;
		delete[] m_pdQuePtSet;
		delete[] m_pnIndexSet;
		delete[] m_pdDistance;

		m_pdRefPtSet = new double[3 * nOut];
		m_pdQuePtSet = new double[3 * nOut];
		m_pnIndexSet = new int[nOut];
		m_pdDistance = new double[nOut];

		m_nReservPtNum = nOut;
	}

	//得到过滤数据
	m_nQuePtNum = 0;
	for (int i = 0; i < nSrcPtNum; i++)
	{
		if (pnIndex[i] == 1)
		{
			m_pdQuePtSet[m_nQuePtNum * 3] = pdSrcPtSet[i * 3];
			m_pdQuePtSet[m_nQuePtNum * 3 + 1] = pdSrcPtSet[i * 3 + 1];
			m_pdQuePtSet[m_nQuePtNum * 3 + 2] = pdSrcPtSet[i * 3 + 2];
			m_nQuePtNum++;
		}
	}
	delete[] pnIndex;

	//3. 将滤波源点云用矩阵m_dFinalTransMat变换到指定位置
	double a11 = m_dFinalTransMat[0][0];
	double a12 = m_dFinalTransMat[0][1];
	double a13 = m_dFinalTransMat[0][2];
	double a14 = m_dFinalTransMat[0][3];
	double a21 = m_dFinalTransMat[1][0];
	double a22 = m_dFinalTransMat[1][1];
	double a23 = m_dFinalTransMat[1][2];
	double a24 = m_dFinalTransMat[1][3];
	double a31 = m_dFinalTransMat[2][0];
	double a32 = m_dFinalTransMat[2][1];
	double a33 = m_dFinalTransMat[2][2];
	double a34 = m_dFinalTransMat[2][3];

#pragma omp parallel for firstprivate(a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34)
	for (int i = 0; i < m_nQuePtNum; i++)
	{
		double x, y, z;
		x = m_pdQuePtSet[3 * i] * a11 + m_pdQuePtSet[3 * i + 1] * a12 + m_pdQuePtSet[3 * i + 2] * a13 + a14;
		y = m_pdQuePtSet[3 * i] * a21 + m_pdQuePtSet[3 * i + 1] * a22 + m_pdQuePtSet[3 * i + 2] * a23 + a24;
		z = m_pdQuePtSet[3 * i] * a31 + m_pdQuePtSet[3 * i + 1] * a32 + m_pdQuePtSet[3 * i + 2] * a33 + a34;

		m_pdQuePtSet[3 * i] = x;
		m_pdQuePtSet[3 * i + 1] = y;
		m_pdQuePtSet[3 * i + 2] = z;
	}

	//4. 构造近邻查询的矩阵
	int nKnn = 1;
	flann::Matrix<double> flmatQuery = flann::Matrix<double>(m_pdQuePtSet, m_nQuePtNum, 3);
	flann::Matrix<int>    flmatIndices(m_pnIndexSet, m_nQuePtNum, nKnn);
	flann::Matrix<double> flmatDists(m_pdDistance, m_nQuePtNum, nKnn);
	double* pdRefData = g_flmatRef.ptr();
	
	//5. 计算用上次变换得到的位置的平均距离，可以知道这一帧比上一帧运动的平均距离
	g_flannIndexObj.knnSearch(flmatQuery, flmatIndices, flmatDists, nKnn, flann::SearchParams(64));
	double dSum = 0.0;
#pragma omp parallel for reduction(+:dSum)            //采用openmmp加速
	for (int i = 0; i < m_nQuePtNum; i++)
	{
		m_pdRefPtSet[3 * i] = pdRefData[3 * m_pnIndexSet[i]];
		m_pdRefPtSet[3 * i + 1] = pdRefData[3 * m_pnIndexSet[i] + 1];
		m_pdRefPtSet[3 * i + 2] = pdRefData[3 * m_pnIndexSet[i] + 2];

		double dD1 = sqrt(flmatDists.ptr()[i]);
		dSum += dD1;
	}
	dAveDist = dSum / m_nQuePtNum;

	if (dAveDist > MAX_AVE_DIST / 10.0)          //运动平局距离超过了10mm (当MAX_AVE_DIST设为100mm)，就返回。
	{
		return MAX_AVE_DIST;
	}

	//开始迭代
	int nItems = 0;
	while (dAveDist > m_dMinDist && nItems < nIterNum)
	{
		nItems++;

		//6. 求变换矩阵，完成一次迭代（本次变换存在m_dTransMat）
		SolveTransfMatrix(m_nQuePtNum, m_pdQuePtSet, m_pdRefPtSet);
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				dMatrix[i][j] = 0.0;
				for (int k = 0; k < 4; k++)
				{
					dMatrix[i][j] += m_dTransMat[i][k] * m_dFinalTransMat[k][j];
				}
			}
		}

		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				m_dFinalTransMat[i][j] = dMatrix[i][j];
			}
		}

		//7. 将目标（查询）点云变换到新的位置（按照配准结果）
		a11 = m_dTransMat[0][0];
		a12 = m_dTransMat[0][1];
		a13 = m_dTransMat[0][2];
		a14 = m_dTransMat[0][3];
		a21 = m_dTransMat[1][0];
		a22 = m_dTransMat[1][1];
		a23 = m_dTransMat[1][2];
		a24 = m_dTransMat[1][3];
		a31 = m_dTransMat[2][0];
		a32 = m_dTransMat[2][1];
		a33 = m_dTransMat[2][2];
		a34 = m_dTransMat[2][3];

#pragma omp parallel for firstprivate(a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34)
		for (int i = 0; i < m_nQuePtNum; i++)
		{
			double x, y, z;
			x = m_pdQuePtSet[3 * i] * a11 + m_pdQuePtSet[3 * i + 1] * a12 + m_pdQuePtSet[3 * i + 2] * a13 + a14;
			y = m_pdQuePtSet[3 * i] * a21 + m_pdQuePtSet[3 * i + 1] * a22 + m_pdQuePtSet[3 * i + 2] * a23 + a24;
			z = m_pdQuePtSet[3 * i] * a31 + m_pdQuePtSet[3 * i + 1] * a32 + m_pdQuePtSet[3 * i + 2] * a33 + a34;

			m_pdQuePtSet[3 * i] = x;
			m_pdQuePtSet[3 * i + 1] = y;
			m_pdQuePtSet[3 * i + 2] = z;
		}

		//8. 取出参考点集中对应点，并计算平均距离
		g_flannIndexObj.knnSearch(flmatQuery, flmatIndices, flmatDists, nKnn, flann::SearchParams(64));
		double dSum = 0.0;
#pragma omp parallel for reduction(+:dSum)            //采用openmmp加速
		for (int i = 0; i < m_nQuePtNum; i++)
		{
			m_pdRefPtSet[3 * i] = pdRefData[3 * m_pnIndexSet[i]];
			m_pdRefPtSet[3 * i + 1] = pdRefData[3 * m_pnIndexSet[i] + 1];
			m_pdRefPtSet[3 * i + 2] = pdRefData[3 * m_pnIndexSet[i] + 2];

			double dD1 = sqrt(flmatDists.ptr()[i]);
			dSum += dD1;
		}
		dAveDist = dSum / m_nQuePtNum;

	}   //while (dAveDist < m_dMinDist && nItems < nIterNum)
	
//	g_flannIndexObj.knnSearch(flmatQuery, flmatIndices, flmatDists, nKnn, flann::SearchParams(128));
//	double dSum = 0.0;
//#pragma omp parallel for reduction(+:dSum)            //采用openmmp加速
//	for (int i = 0; i < m_nQuePtNum; i++)
//	{
//		double dD1 = sqrt(flmatDists.ptr()[i]);
//		dSum += dD1;
//	}
//	dAveDist = dSum / m_nQuePtNum;

	GetInverseMatHomo();

	return dAveDist;
}

//Test
double CRegICPHxmc::AddHxmc(double a, double b)
{
	return a + b;
}