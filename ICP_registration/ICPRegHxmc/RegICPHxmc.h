#pragma once

#ifdef ICPREGHXMC_EXPORTS
#define REGICPHXMC_API		__declspec(dllexport)
#else
#define REGICPHXMC_API		__declspec(dllimport)
#endif

#define FEAT_NUM_HX            5

#include <vector>

struct FeatObb
{
	double corner[3];
	double max[3];
	double mid[3];
	double min[3];
};

struct fPoint_Hxmx
{
	float fPt[3];
};

struct dPoint_Hxmx
{
	double dPt[3];
};

class REGICPHXMC_API CRegICPHxmc
{
public:
	CRegICPHxmc();
	~CRegICPHxmc();
	
	//Test
	double   AddHxmc(double a, double b);
	//Interface
	double*  GetRegMatrix();
	void     GetTransMatrix(double dTransMat[4][4]);
	void     GetFinalTransMatrix(double dTransMat[4][4]);
	void     GetFinalTransMatrixInv(double dTransMat[4][4]);
	void     Set5FeaturePtsCoord(double* pdFeatCoords, bool isTarget);
	void     Set5FeaturePtsCoord(double  dFeatCoords[][3], bool isTarget);
	void     Get5FeaturePtsCoord(double  dFeatCoords[][3], bool isTarget);
	void     SetTargetPtCloud(float* pPtCloud, int nPtNum);
	void     SetMinDist2PtSets(double dMinDist) { m_dMinDist = dMinDist; };
	double   GetMinDist2PtSets() { return m_dMinDist; };
	void     GetInverseMatHomo();
	void     GetInverseMatEigen();
	void     GetBaseObBox();
	void     GetCurrentObBox();

	void     ComputeOBB(double dPtCoords[][3], int nPtNum, 
		                double corner[3], double max[3],
		                double mid[3], double min[3], double size[3]);
	void     ComputeOBB(double corner[3], double max[3],double mid[3], double min[3], double size[3]);
	double   PtLineProj(double dPt[3], double dP1[3], double dP2[3]);
	double   PtLineProjF(float fPt[3], double dP1[3], double dP2[3]);
	double   DistanceToLine(double x[3], double p1[3], double p2[3],
		                    double& t, double* closestPoint);
	double   SolveTransfMatrix(int nDataNum, double* pdSource, double* dpTarget);
	double   SolveTransfMatrix(int nDataNum, double dSource[][3], double dTarget[][3], double dError[]);
	double   SolveTransfMatrix(double dError[]);
	int      BoundBoxFilter(double* pdPtSetIn, int nPtNum, double** ppdPtSetOut,
		double corner[3], double max[3], double mid[3], double min[3]);
	int      BoundBoxFilter(double ptCorner[3], double axisMax[3], double axisMid[3], double axisMin[3],
		double dPtCoords[][3], int nPtNum, int* pnIndex);
	int      BoundBoxFilter(std::vector<fPoint_Hxmx>vecSrcPtIn, std::vector<fPoint_Hxmx>& vecPtOut, 
		double corner[3], double max[3], double mid[3], double min[3]);
	double   CalcNearestPointPairs(double* pdQuePtSet, int nPtNum);
	void     BuildIndex(double* pdRefPtSet, int nPtNum);
	double   RegistByICP(double* pdSrcPtSet, int nPtNum, int nIterNum, double dMatrix[4][4]);

	double   RegBy5FPlusICP(double dSrcFea[][3], double dTarFea[][3], int nFNum,
		double* pdSrcPtSet, int nSrcPtNum, double dMatrix[4][4]);
	double   RegByIcp(double* pdSrcPtSet, int nSrcPtNum, int nIterNum, double dMatrix[4][4]);
	

private:
	double m_dMatRegiste[16];
	double m_dTransMat[4][4];
	double m_dFinalTransMat[4][4];
	double m_dFinalInversMat[4][4];
	double m_d5FeatCam[FEAT_NUM_HX][3];
	double m_d5FeatMri[FEAT_NUM_HX][3];

	double* m_pdRefPtSet;
	double* m_pdQuePtSet;
	double* m_pdDistance;
	int*    m_pnIndexSet;
	double  m_dMinDist;       //希望配准达到的点集间平均距离
	struct  FeatObb   m_stObbox;
	struct  FeatObb   m_stBaseObbox;
	int     m_nQuePtNum;
	int     m_nReservPtNum;   //预留的查询和ICP算法内存空间的点数量。

protected:
	template<class T>
	int JacobiN(T** a, int n, T* w, T** v);
};

