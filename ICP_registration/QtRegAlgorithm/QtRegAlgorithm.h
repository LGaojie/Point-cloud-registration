#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QtRegAlgorithm.h"

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkSTLReader.h>
#include <vtkOBJReader.h>
#include <vtkPLYReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>

#include "RegICPHxmc.h"

#define FEATURE_PT_NUM            5

class QtRegAlgorithm : public QMainWindow
{
	Q_OBJECT

public slots:
	
	//Buttons responses.
	int OnOpenSrcMeshClicked(bool checked);
	int OnSaveSrcMeshClicked(bool checked);
	int OnOpenTarMeshClicked(bool checked);
	int OnSaveTarMeshClicked(bool checked);
	int OnICPRegistClicked(bool checked);
	int OnReg5FPlusICPClicked(bool checked);
	int OnICPRegICP2Clicked(bool checked);
	int OnFiveFeatureClicked(bool checked);
	int OnOpenCam5FeaturesClicked(bool checked);
	int OnOpenMri5FeaturesClicked(bool checked);
	int OnBoxFilterClicked(bool checked);
	int OnTransFilterClicked(bool checked); 
	int OnAlgorithmTestClicked(bool checked);
	//CheckBoxes responses
	int OnIfShowSource(int nState);
	int OnIfDisplaySrcMesh(int nState);
	int OnIfShowFilter(int nState);
	int OnIfDisplayFilterMesh(int nState);
	int OnIfShowTarget(int nState);
	int OnIfDisplayTarMesh(int nState);
	int OnIfShowOBBox(int nState);

public:
	QtRegAlgorithm(QWidget *parent = Q_NULLPTR);

protected:
	virtual void resizeEvent(QResizeEvent* event) override;
	void closeEvent(QCloseEvent* event);

private:
	Ui::QtRegAlgorithmClass  ui;
	CRegICPHxmc              m_objRegIcp;
	double*                  m_pdSrcPtSet = NULL;
	double*                  m_pdFilterPtSet = NULL;
	double*                  m_pdFltCpyPtSet = NULL;
	double*                  m_pdTarPtSet = NULL;
	double                   m_dSrcFea[FEAT_NUM_HX][3];
	double                   m_dTarFea[FEAT_NUM_HX][3];
	double                   m_dMatTrans[4][4];
	int                      m_nFltPtNum{ 0 };
	int                      m_nSrcPtNum{ 0 };

	bool  m_bSrcMeshOpened;
	bool  m_bTarMeshOpened;
	
	vtkSmartPointer<vtkPolyData>          m_pvtkPolysSrc;
	vtkSmartPointer<vtkPolyData>          m_pvtkPolysSrcVert;
	vtkSmartPointer<vtkPolyData>          m_pvtkPolysFlt;
	vtkSmartPointer<vtkPolyData>          m_pvtkPolysFltVert;
	vtkSmartPointer<vtkPolyData>          m_pvtkPolysTar;
	vtkSmartPointer<vtkPolyData>          m_pvtkPolysTarVert;
	vtkSmartPointer<vtkPolyData>          m_pvtk5PtsOBB;

	vtkSmartPointer<vtkPolyDataMapper>    m_pvtkMapperSrc;
	vtkSmartPointer<vtkPolyDataMapper>    m_pvtkMapperSrcVert;
	vtkSmartPointer<vtkPolyDataMapper>    m_pvtkMapperFlt;
	vtkSmartPointer<vtkPolyDataMapper>    m_pvtkMapperFltVert;
	vtkSmartPointer<vtkPolyDataMapper>    m_pvtkMapperTar;
	vtkSmartPointer<vtkPolyDataMapper>    m_pvtkMapperTarVert;
	vtkSmartPointer<vtkPolyDataMapper>    m_pvtkMapper5PtsOBB;

	vtkSmartPointer<vtkActor>             m_pvtkActorSrc;
	vtkSmartPointer<vtkActor>             m_pvtkActorSrcVert;
	vtkSmartPointer<vtkActor>             m_pvtkActorFlt;
	vtkSmartPointer<vtkActor>             m_pvtkActorFltVert;
	vtkSmartPointer<vtkActor>             m_pvtkActorTar;
	vtkSmartPointer<vtkActor>             m_pvtkActorTarVert;
	vtkSmartPointer<vtkActor>             m_pvtkActor5PtsOBB;

	vtkSmartPointer<vtkRenderer>          m_pvtkRenderer;

	//vtkSmartPointer<vtkOBJReader>         m_pvtkObjReader;
	//vtkSmartPointer<vtkSTLReader>         m_pvtkStlReader;
	//vtkSmartPointer<vtkPLYReader>         m_pvtkPlyReader;

	void   Display(bool bSrcShow,bool bSrcMesh, bool bFilterShow, bool bFltMesh, bool bTarShow, bool bTarMesh, bool bOBBShow);
	void   DisplaySrcMeshes(bool bMesh);
	void   DisplayTarMeshes(bool bMesh);
	void   Display5PtBox(bool bShow); 
	int    fromMesh2Points(vtkSmartPointer<vtkPolyData>pvtkVert, vtkSmartPointer<vtkPolyData>pvtkMesh);
	int    fromMesh2PtSet(vtkSmartPointer<vtkPolyData>pvtkMesh, double** ppdPtSet);
	void   fromPtSet2Vertex(double* pdPtSet, int nPtNum, vtkSmartPointer<vtkPolyData>pvtkMesh);
	void   fromPtSet2VertexByTrans(double* pdPtSet, int nPtNum, vtkSmartPointer<vtkPolyData>pvtkMesh, double matTrans[4][4]);
	int    fromMesh2Vector(vtkSmartPointer<vtkPolyData>pvtkMesh, std::vector<dPoint_Hxmx>& vecPtSet);
	int    readFeaturePts(QString qstrFileName, double pdPtsSet[FEATURE_PT_NUM][3]);
	int    findAllDirectory(QString qstrPathName);
	void   CreateVtkBox(vtkSmartPointer<vtkPolyData> pvtkPolysBox,
		double dCorner[3], double dMax[3], double dMid[3], double dMin[3]);
	void   TransfPointSet(double** ppdPtSet, int nPtNum, vtkSmartPointer<vtkPolyData> pvtkPolysVert);
};
