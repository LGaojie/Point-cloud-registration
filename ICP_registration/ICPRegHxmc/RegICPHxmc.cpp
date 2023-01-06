/*****************************************************************************
RegICPHxmc.cpp : ���ÿ�Դ��Flann��Eigenʵ��5������ĳ���׼��ICP��׼��

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
#define MIN_REGIST_NUM         600                                //������ƹ��˺�ĵ��������ֵ�����غܴ��ƽ������:MAX_AVE_DIST
#define MAX_AVE_DIST           100.0                              //��λ��mm����׼��������ƽ�����룬�������ֵ˵��û����׼

flann::KDTreeIndexParams            g_IndexParams(4);               //KNN��ѯ�Ĳ���������KT-TREE��
flann::Index<flann::L2<double> >    g_flannIndexObj(g_IndexParams); //KNN��ѯ����
flann::Matrix<double>               g_flmatRef;                     //�ο����ݣ��ռ�㣬��Ӧ��׼��Ŀ�꣩

CRegICPHxmc::CRegICPHxmc()
{
	m_dMinDist = 0.88;       //ϣ���ﵽ�ĵ㼯��ƽ�־���
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
����˵����
	��ȡ�ѽ�����õ���׼����ע�ⷵ�ص��Ƕ����ڵľ���ָ�롣
����:
	�ޡ�
����ֵ��
	ָ��任��������{(1,1),(1,2),(1,3),(1,4);(2,1),......,(4,3),(4,4)}��
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
����˵����
	��ȡ�м�任��������5�������SVD��������м����ȡ�
����:
	dTransMat(out)����α任����
����ֵ��
	�ޡ�
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
����˵����
	��ȡICP����������ձ任����
����:
	dTransMat(out)����α任����
����ֵ��
	�ޡ�
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
����˵����
	��ȡICP����������ձ任����������
����:
	dTransMat(out)����α任����
����ֵ��
	�ޡ�
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
����˵����
	�󱾴δ���׼Դ���Ƶİ�Χ�У������ϴα任�õ��ľ����Լ���׼
	��Χ�У�������Ϊ��ͷ�����������˶�����(<10mm)������õ���
	��Χ����Ȼ�����ϰ�Χ5�������γɵİ�Χ���֡�
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
����˵����
	ͨ��Դ�����ϵİ�Χ���Լ��״εõ��ı任�������׼��Χ�У�
	���״δ����Դ��5���������С��Χ�У��任��õ���Ӧ��MRI��
	�Ķ�Ӧ��Χ����Ϊ��׼�İ�Χ�С�
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
����˵����
	�����Ѿ������ڴ��ʵ����
�������:
�������:
	�ޡ�
����ֵ��
	�ޡ�
*********************************************************************/
void CRegICPHxmc::SetTargetPtCloud(float* pPtCloud, int nPtNum)
{

}

/***********************************************************************
����˵����
	���������������괫�뱾��Ķ���
����:
	pdFeatCoords��������������з�ʽ{x1,y1,z1,x2,y2,z2,......x5,y5,z5}��
	isTarget���Ƿ�ΪĿ����Ƶ����꣬������Դ��������������ꡣ
����ֵ��
	�ޡ�
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
����˵����
	���������������괫�뱾��Ķ���
����:
	dFeatCoords��������������з�ʽΪC++�Ķ�ά���飺
	    {{x1,y1,z1}
		 {x2,y2,z2}
		  ......
		 {x5,y5,z5}}��
	isTarget���Ƿ�ΪĿ����Ƶ����꣬������Դ��������������ꡣ
����ֵ��
	�ޡ�
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
����˵����
	����������������ꡣ
����:
	dFeatCoords(out)��������������з�ʽΪC++�Ķ�ά���飺
		{{x1,y1,z1}
		 {x2,y2,z2}
		  ......
		 {x5,y5,z5}}��
	isTarget��ȡ�ص���Ŀ����Ƶ����꣬����Դ��������������ꡣ
����ֵ��
	�ޡ�
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
˵����
	���������ĶԳ��ԣ������ձ任����m_dFinalTransMat�������
	m_dFinalInversMat��
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
˵����
	��Eigen��inverse�����������ձ任����m_dFinalTransMat�������
	m_dFinalInversMat��
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
˵����
	SVD��������Ӧ��ĸ��Ա任�������������Ӧ4��������ꡣ
	ע�⣺����ı任�����ڳ�Ա m_dTransMat���棬��GetTransMatrix�õ������
������
	nDataNum����Ӧ��ĸ�����
	dSource�� ����׼�����ݣ�3*nDataNum������������
	dTarget�� ��׼���ݡ�����ı任�����dSource��ӳ��dTarget�Ķ�Ӧ�㣻
	dpError(out)�� ÿ������׼����
���أ�
	������׼�ı�׼��
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

	//��Դ��Ŀ��(3*nDataNum)����ֵ�������㼸������
	matSource = Eigen::Map<Eigen::MatrixXd>(pdSource, 3, nDataNum);
	matTarget = Eigen::Map<Eigen::MatrixXd>(dpTarget, 3, nDataNum);
	
	vecScrCenter = matSource.rowwise().sum()/ nDataNum;
	vecTarCenter = matTarget.rowwise().sum()/ nDataNum;

	//����ƽ�Ƶ���������ϵ��ԭ��
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
˵����
	SVD��������Ӧ��ĸ��Ա任�������������Ӧ4��������ꡣ
	ע�⣺����ı任�����ڳ�Ա m_dTransMat���棬��GetTransMatrix�õ������
������
	nDataNum����Ӧ��ĸ�����
	dSource�� ����׼�����ݣ�3*nDataNum������������
	dTarget�� ��׼���ݡ�����ı任�����dSource��ӳ��dTarget�Ķ�Ӧ�㣻
	dpError(out)�� ÿ������׼����
���أ�
	������׼�ı�׼��
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

	//��Դ��Ŀ��(3*nDataNum)����ֵ�������㼸������
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

	//����ƽ�Ƶ���������ϵ��ԭ��
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
˵����
	SVD�������5������ĸ��Ա任����ǰ���Ǳ�������Ѿ��õ���������Ӧ
	�������ֵ�ˡ�
	ע�⣺����ı任�����ڳ�Ա m_dTransMat���棬��GetTransMatrix�õ������
������
	dError(out)��      ÿ������׼�ı�׼��
���أ�
	������׼�ı�׼��
*************************************************************************/
double CRegICPHxmc::SolveTransfMatrix(double dError[])
{
	int    i, j;
	double dStdErr = -1.0;
	Eigen::MatrixXd  matSource, matTarget;
	Eigen::Vector3d  vecScrCenter(0, 0, 0), vecTarCenter(0, 0, 0);

	//��Դ��Ŀ��(3*FEAT_NUM_HX)����ֵ�������㼸������
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

	//����ƽ�Ƶ���������ϵ��ԭ��
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
˵����
	Jacobi��ת���������n*n�ԳƷ����������������ʸ����
	����ֵ�ǽ������еģ�����ʸ���ǹ�һ���ġ�
������
	a�������n*n�Գƾ���
	n����������ά�ȣ�
	w�������n������ֵ��
	v�������n*n����ʸ��������ʸ������
	
���أ�
	����������������0�����򷵻�1��
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
˵����
	����3D�ռ�㼯�����ƣ�����С��Χ�У�OBB������Χ���ɲ�����
	�ǵ�corner���������໥��ֱ��ʸ�����ɺ��ӵ������ᡣ
������
	dPtCoords��   ����ռ�����ꣻ
	nPtNum��      �յ���������
	corner(out)�� ��Χ�нǵ�����ꣻ
	max(out)��    ��������ʸ�������Ⱥͷ��򣩣�
	mid(out)��    �����г����ʸ�������Ⱥͷ��򣩣�
	min(out)��    ����������ʸ�������Ⱥͷ��򣩣�
	size(out)��   �㼯Э������������ֵ��
���أ�
	�ޡ�
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
˵����
	����3D�ռ�㼯�����ƣ�����С��Χ�У�OBB������Χ���ɲ�����
	�ǵ�corner���������໥��ֱ��ʸ�����ɺ��ӵ������ᡣ
������
	corner(out)�� ��Χ�нǵ�����ꣻ
	max(out)��    ��������ʸ�������Ⱥͷ��򣩣�
	mid(out)��    �����г����ʸ�������Ⱥͷ��򣩣�
	min(out)��    ����������ʸ�������Ⱥͷ��򣩣�
	size(out)��   �㼯Э������������ֵ��
���أ�
	�ޡ�
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
˵����
	����3D�ռ�㵽�߶ε�ͶӰ��λ�ã����ص���ֱ�߲������̵�tֵ��
������
	dPt���ռ��������ꣻ
	dP1���߶�������ꣻ
	dP2���߶��յ����ꡣ
���أ�
	tֵ���������̣�P = P1 + t*(P2 - P1)��Pֱ���ϵ�3D�㣬����(Px,Py,Pz)��
	0<t<1ʱ��ͶӰ�����߶��ϣ��������߶��⡣
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
˵����
	����3D�ռ�㵽�߶ε�ͶӰ��λ�ã����ص���ֱ�߲������̵�tֵ��
������
	x��      �ռ��������ꣻ
	p1��     �߶�������ꣻ
	p2��     �߶��յ����ꣻ
	t(out)�� �������̣�P = P1 + t*(P2 - P1)��
	closestPoint(out)��x��ֱ�ߵ�����㣨��һ����ͶӰ�㣬��0<t<1ʱΪ
	ͶӰ�㣬����Ϊ�����յ㣩��
���أ�
	x��closestPoint�ľ����ƽ����
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
˵����
	���������Χ���ڵĵ��ƣ���Χ��Ϊһ�����εĿռ���ӡ�
������
	pdPtSetIn��      ���˲��ĵ㼯��
	pdPtSetOut(out)���˲���ĵ㼯��
	corner����Χ�еĽǵ㣻
	max��   ��Χ�е���᣻
	mid��   ��Χ�е��г��᣻
	min��   ��Χ�еĶ��᣻
���أ�
	�ں����ڼ��˲���ĵ�������
**************************************************************/
int CRegICPHxmc::BoundBoxFilter(double* pdPtSetIn, int nPtNum, double** ppdPtSetOut,
	double corner[3], double max[3], double mid[3], double min[3])
{
	//ȡ�������ڵĵ�
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
˵����
	���������Χ���ڵĵ��ƣ���Χ��Ϊһ�����εĿռ���ӡ�
������
	vecSrcPtIn�����˲��ĵ㼯��
	vecPtOut(out)���˲���ĵ㼯��
	corner����Χ�еĽǵ㣻
	max�� ��Χ�е���᣻
	mid�� ��Χ�е��г��᣻
	min�� ��Χ�еĶ��᣻
���أ�
	�ں����ڵĵ�������
**************************************************************/
int CRegICPHxmc::BoundBoxFilter(std::vector<fPoint_Hxmx>vecSrcPtIn, std::vector<fPoint_Hxmx>& vecPtOut,
	double corner[3], double max[3], double mid[3], double min[3])
{
	//ȡ�������ڵĵ�
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
˵����
	���������Χ���ڵĵ��ƣ���Χ��Ϊһ�����εĿռ���ӡ�
������
	ptCorner����Χ�еĽǵ㣻
	axisMax�� ��Χ�е���᣻ 
	axisMid�� ��Χ�е��г��᣻
	axisMin�� ��Χ�еĶ��᣻
	dPtCoords�� ����ĵ��ƣ�
	nPtNum��    ������Ƶĵ�������
	pnIndex(out)���˲���ĵ�����������������ƣ�
	              0-���ں����ڣ�1���ں����ڣ���
���أ�
	�ں����ڵĵ�������
**************************************************************/
int CRegICPHxmc::BoundBoxFilter(double ptCorner[3], double axisMax[3], 
	double axisMid[3], double axisMin[3],double dPtCoords[][3], int nPtNum, int* pnIndex)
{
	int nInNum = 0;

	memset(pnIndex, nPtNum * sizeof(int), 0);

	//ȡ�������ڵĵ�
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

	//ͨ������ͶӰ��ȷ���ĵ��Ƿ��ں�����
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
˵����
	����Knn�Ĳ�ѯ�ṹ��
������
	pfRefPtSet���ο��㼯�����ꣻ
	nPtNum�� �ο��㼯�ĵ�������
���أ�
	�ޡ�
**************************************************************/
void CRegICPHxmc::BuildIndex(double* pdRefPtSet, int nPtNum)
{
	g_flmatRef = flann::Matrix<double>(pdRefPtSet, nPtNum, 3);
	g_flannIndexObj.buildIndex(g_flmatRef);
}

/*************************************************************************
˵����
	��ѯ�ṹ����㼯���Ѿ������õ��������ԣ�����ο��㼯
	�ж�Ӧ����������Ӧ��ƽ�����롣
	ע�⣺��ѯ��Ӧ����������ڳ�Ա m_pnIndexSet ָ��Ĵ洢�ռ���
	      ��Ӧ��֮��ľ����ƽ������ڳ�Ա m_pfDistance ָ��Ĵ洢�ռ��
������
	pfQuePtSet����ѯ�㼯��3D�ռ����ꣻ
	nPtNum�� ��ѯ�㼯�ĵ�������
���أ�
	��Ӧ��֮���ƽ�����롣
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

#pragma omp parallel for reduction(+:dAveDist)        //����openmmp����
	for (int i = 0; i < nPtNum; ++i)
	{
		double dD1 = sqrt(flmatDists.ptr()[i]);
		dAveDist += dD1;
	}

	dAveDist /= nPtNum;

	return dAveDist;
}

/*************************************************************************
˵����
	��Ե��ICP�㷨ʵ�֡��������֮��ܽӽ����������������Ҫ����10��
	ǰ��������Ŀ������Ѿ����뱾���������BuildIndex�����˲�ѯ������
������
	pdSrcPtSet��Դ�㼯�����ƣ���
	nPtNum��    Դ�㼯�ĵ�������
	nIterNum��  �����������������������ڵ�֮��ƽ������С���趨ֵ
	            Ҳ���Ƴ�������
	dMatrix(out)������ı任����
���أ�
	��Ӧ��֮���ƽ�����롣
**************************************************************************/
double CRegICPHxmc::RegistByICP(double* pdSrcPtSet, int nPtNum, int nIterNum, double dMatrix[4][4])
{
	int    nKnn = 1, nIter = 0;
	double dAveDist = 100.0;
	
	double* pdRefData = g_flmatRef.ptr();

	//���Ԥ���Ĳ�ѯ�Ͷ�Ӧ��ռ䲻�������������ڴ�
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

	//������ѯ�㼯�ľ���Ͳ�ѯ�ṹ����
	flann::Matrix<double> flmatQuery = flann::Matrix<double>(m_pdQuePtSet, nPtNum, 3);
	flann::Matrix<int>    flmatIndices(m_pnIndexSet, nPtNum, nKnn);
	flann::Matrix<double> flmatDists(m_pdDistance, nPtNum, nKnn);

	//�����ʼ��
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

	//��ʼ����
	while (dAveDist > m_dMinDist && nIter < nIterNum)
	{
		nIter++;
		g_flannIndexObj.knnSearch(flmatQuery, flmatIndices, flmatDists, nKnn, flann::SearchParams(128));

		double dSum = 0.0;
		//ȡ���ο��㼯�ж�Ӧ�㣬������ƽ������
#pragma omp parallel for reduction(+:dSum)            //����openmmp����
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
	//������ƽ������
#pragma omp parallel for reduction(+:dSum)            //����openmmp����
	for (int i = 0; i < nPtNum; i++)
	{
		double dD1 = sqrt(flmatDists.ptr()[i]);
		dSum += dD1;
	}

	dAveDist = dSum / nPtNum;

	return dAveDist;
}

/*************************************************************************
˵����
	�Ȳ���SVD������Զ�Ӧ�������任�õ���ʼ����׼Ч����Ȼ����
	��Ե��ICP�㷨ʵ�־���׼��
	�������������Ҫ����10��
	ǰ��������Ŀ������Ѿ����뱾���������BuildIndex�����˲�ѯ������
������
	dSrcFea��   Դ��������������飻
	dTarFea��   Ŀ����������������飨���Ӧ����
	nFNum��     �������������
	pdSrcPtSet��Դ�㼯��ֱ�����������еõ��ĵ��ƣ���
	nSrcPtNum�� Դ�㼯�ĵ�������
	nIterNum��  �����������������������ڵ�֮��ƽ������С���趨ֵ
				Ҳ���Ƴ�������
	dMatrix(out)������ı任����
���أ�
	Դ������Ŀ����ƶ�Ӧ��֮���ƽ�����롣
**************************************************************************/
double CRegICPHxmc::RegBy5FPlusICP(double dSrcFea[][3], double dTarFea[][3], int nFNum,
	double* pdSrcPtSet, int nSrcPtNum, double dMatrix[4][4])
{
	double dAveDist{-1.0};
	double size[3];
	double dP1Long[3], dP2Long[3], dP1Mid[3], dP2Mid[3], dP1Min[3], dP2Min[3];

	//1. �����Χ��
	ComputeOBB(dSrcFea, nFNum, m_stObbox.corner, m_stObbox.max, m_stObbox.mid, m_stObbox.min, size);

	//Test code
	//ComputeOBB(dTarFea, nFNum, m_stBaseObbox.corner, m_stBaseObbox.max, m_stBaseObbox.mid, m_stBaseObbox.min, size);

	//2. �ð�Χ�ж�Դ���ݽ����˲�
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

	//3. ���ʼ����任���󣬽���ڳ�Աm_dTransMat�
	double dStdErr, dErr[32];
	dStdErr = SolveTransfMatrix(nFNum, dSrcFea, dTarFea, dErr);
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			m_dFinalTransMat[i][j] = m_dTransMat[i][j];
		}
	}

	//4. ���˲�Դ�����þ���m_dTransMat�任��ָ��λ��
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

	//5. ���Ӧ���Լ���Ӧ��֮��ƽ�־���
	int nKnn = 1;
	flann::Matrix<double> flmatQuery = flann::Matrix<double>(m_pdQuePtSet, m_nQuePtNum, 3);
	flann::Matrix<int>    flmatIndices(m_pnIndexSet, m_nQuePtNum, nKnn);
	flann::Matrix<double> flmatDists(m_pdDistance, m_nQuePtNum, nKnn);
	g_flannIndexObj.knnSearch(flmatQuery, flmatIndices, flmatDists, nKnn, flann::SearchParams(128));
	//ȡ���ο��㼯�ж�Ӧ�㣬������ƽ������
	double* pdRefData = g_flmatRef.ptr();
	double dSum = 0.0;
#pragma omp parallel for reduction(+:dSum)            //����openmmp����
	for (int i = 0; i < m_nQuePtNum; i++)
	{
		m_pdRefPtSet[3 * i] = pdRefData[3 * m_pnIndexSet[i]];
		m_pdRefPtSet[3 * i + 1] = pdRefData[3 * m_pnIndexSet[i] + 1];
		m_pdRefPtSet[3 * i + 2] = pdRefData[3 * m_pnIndexSet[i] + 2];

		double dD1 = sqrt(flmatDists.ptr()[i]);
		dSum += dD1;
	}
	dAveDist = dSum / m_nQuePtNum;

	//6. ��任������ɵ�һ�ε���
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

	//7. ��ƽ������
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
#pragma omp parallel for reduction(+:dSum)            //����openmmp����
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
˵����
	��Ե��ICP�㷨ʵ�֡��������������Ҫ����10��
	ǰ��������1.Ŀ������Ѿ����뱾���������BuildIndex�����˲�ѯ������
	          2.֮ǰ��֡�Ѿ�������5�������㣻
			  3.��һ֡����׼���Ƚϸߣ�ƽ������<1.5mm����
������
	pdSrcPtSet��Դ�㼯��ֱ�����������еõ��ĵ��ƣ���
	nPtNum��    Դ�㼯�ĵ�������
	nIterNum��  �����������������������ڵ�֮��ƽ������С���趨ֵ
				Ҳ���Ƴ�������
	dMatrix(out)������ı任������������Դ���Ƽ���Ŀ������غϡ�
���أ�
	Դ������Ŀ����ƶ�Ӧ��֮���ƽ�����롣
**************************************************************************/
double CRegICPHxmc::RegByIcp(double* pdSrcPtSet, int nSrcPtNum, int nIterNum, double dMatrix[4][4])
{
	double dAveDist{ 100.0 };
	double dP1Long[3], dP2Long[3], dP1Mid[3], dP2Mid[3], dP1Min[3], dP2Min[3];

	//������һ֡������õ����ư�Χ��
	GetCurrentObBox();

	//2. �ð�Χ�ж�Դ���ݽ����˲�
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

	if (MIN_REGIST_NUM > nOut)    //������ƹ��˺�ĵ��������ֵ�����غܴ��ƽ������:MAX_AVE_DIST
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

	//�õ���������
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

	//3. ���˲�Դ�����þ���m_dFinalTransMat�任��ָ��λ��
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

	//4. ������ڲ�ѯ�ľ���
	int nKnn = 1;
	flann::Matrix<double> flmatQuery = flann::Matrix<double>(m_pdQuePtSet, m_nQuePtNum, 3);
	flann::Matrix<int>    flmatIndices(m_pnIndexSet, m_nQuePtNum, nKnn);
	flann::Matrix<double> flmatDists(m_pdDistance, m_nQuePtNum, nKnn);
	double* pdRefData = g_flmatRef.ptr();
	
	//5. �������ϴα任�õ���λ�õ�ƽ�����룬����֪����һ֡����һ֡�˶���ƽ������
	g_flannIndexObj.knnSearch(flmatQuery, flmatIndices, flmatDists, nKnn, flann::SearchParams(64));
	double dSum = 0.0;
#pragma omp parallel for reduction(+:dSum)            //����openmmp����
	for (int i = 0; i < m_nQuePtNum; i++)
	{
		m_pdRefPtSet[3 * i] = pdRefData[3 * m_pnIndexSet[i]];
		m_pdRefPtSet[3 * i + 1] = pdRefData[3 * m_pnIndexSet[i] + 1];
		m_pdRefPtSet[3 * i + 2] = pdRefData[3 * m_pnIndexSet[i] + 2];

		double dD1 = sqrt(flmatDists.ptr()[i]);
		dSum += dD1;
	}
	dAveDist = dSum / m_nQuePtNum;

	if (dAveDist > MAX_AVE_DIST / 10.0)          //�˶�ƽ�־��볬����10mm (��MAX_AVE_DIST��Ϊ100mm)���ͷ��ء�
	{
		return MAX_AVE_DIST;
	}

	//��ʼ����
	int nItems = 0;
	while (dAveDist > m_dMinDist && nItems < nIterNum)
	{
		nItems++;

		//6. ��任�������һ�ε��������α任����m_dTransMat��
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

		//7. ��Ŀ�꣨��ѯ�����Ʊ任���µ�λ�ã�������׼�����
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

		//8. ȡ���ο��㼯�ж�Ӧ�㣬������ƽ������
		g_flannIndexObj.knnSearch(flmatQuery, flmatIndices, flmatDists, nKnn, flann::SearchParams(64));
		double dSum = 0.0;
#pragma omp parallel for reduction(+:dSum)            //����openmmp����
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
//#pragma omp parallel for reduction(+:dSum)            //����openmmp����
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