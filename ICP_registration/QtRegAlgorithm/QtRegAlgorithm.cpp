#include "QtRegAlgorithm.h"

#include <stdlib.h>
#include <time.h>

#include <QMessageBox>
#include <QFileDialog>
#include <QDirIterator>
#include <qtextstream.h>
#include <qdebug.h>

#include <vtkProperty.h>
#include <vtkSTLWriter.h>
#include <vtkOBJWriter.h>
#include <vtkPLYWriter.h>
#include <vtkMatrix4x4.h>
#include <vtkTransform.h>
#include <vtkTriangle.h>

#define QT_GROUPBOX_WIDTH     311

QtRegAlgorithm::QtRegAlgorithm(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	m_bSrcMeshOpened = false;
	m_bTarMeshOpened = false;

	m_pvtkPolysSrc = vtkSmartPointer<vtkPolyData>::New();
	m_pvtkPolysSrcVert = vtkSmartPointer<vtkPolyData>::New();
	m_pvtkPolysFlt = vtkSmartPointer<vtkPolyData>::New();
	m_pvtkPolysFltVert = vtkSmartPointer<vtkPolyData>::New();
	m_pvtkPolysTar = vtkSmartPointer<vtkPolyData>::New();
	m_pvtkPolysTarVert = vtkSmartPointer<vtkPolyData>::New();
	m_pvtk5PtsOBB = vtkSmartPointer<vtkPolyData>::New();

	m_pvtkMapperSrc = vtkSmartPointer<vtkPolyDataMapper>::New();
	m_pvtkMapperSrcVert = vtkSmartPointer<vtkPolyDataMapper>::New();
	m_pvtkMapperFlt = vtkSmartPointer<vtkPolyDataMapper>::New();
	m_pvtkMapperFltVert = vtkSmartPointer<vtkPolyDataMapper>::New();
	m_pvtkMapperTar = vtkSmartPointer<vtkPolyDataMapper>::New();
	m_pvtkMapperTarVert = vtkSmartPointer<vtkPolyDataMapper>::New();
	m_pvtkMapper5PtsOBB = vtkSmartPointer<vtkPolyDataMapper>::New();

	m_pvtkActorSrc = vtkSmartPointer<vtkActor>::New();
	m_pvtkActorSrcVert = vtkSmartPointer<vtkActor>::New();
	m_pvtkActorFlt = vtkSmartPointer<vtkActor>::New();
	m_pvtkActorFltVert = vtkSmartPointer<vtkActor>::New();
	m_pvtkActorTar = vtkSmartPointer<vtkActor>::New();
	m_pvtkActorTarVert = vtkSmartPointer<vtkActor>::New();
	m_pvtkActor5PtsOBB = vtkSmartPointer<vtkActor>::New();

	m_pvtkRenderer = vtkSmartPointer<vtkRenderer>::New();

	//Connect signal slots
	//Push "OpenMesh" button for open source mesh file.
	connect(
		ui.btnOpenSrcMesh,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnOpenSrcMeshClicked(bool)));

	//Push "SaveMesh" button for save source mesh to a file.
	connect(
		ui.btnSaveSrcMesh,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnSaveSrcMeshClicked(bool)));

	//Push "OpenMesh" button for open target mesh file.
	connect(
		ui.btnOpenTarMesh,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnOpenTarMeshClicked(bool)));

	//Push "SaveMesh" button for save target mesh to a file.
	connect(
		ui.btnSaveTarMesh,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnSaveTarMeshClicked(bool)));

	//按下 "ICP1" 按钮时调用，用于调试ICP算法
	connect(
		ui.btnICPReg,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnICPRegistClicked(bool)));

	//按下 "5F+ICP" 按钮时调用，用于调试5Feature + ICP算法
	connect(
		ui.btn5FPlusICP,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnReg5FPlusICPClicked(bool)));

	//按下 "ICP2" 按钮时调用，用于调试knn算法
	connect(
		ui.btnICP2,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnICPRegICP2Clicked(bool)));

	//按下 "5Feats" 按钮时调用，用于调试5特征点配准算法
	connect(
		ui.btnFiveFeature,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnFiveFeatureClicked(bool)));

	//按下 "OpenCam" 按钮时调用，用于打开相机的5特征点坐标并读出
	connect(
		ui.btnOpenCam5Feat,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnOpenCam5FeaturesClicked(bool)));

	//按下 "OpenMri" 按钮时调用，用于打开MRI的5特征点坐标并读出 btnBoxFilter
	connect(
		ui.btnOpenMri5Feat,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnOpenMri5FeaturesClicked(bool)));

	//按下 "BoxFilter" 按钮时调用，过滤出5特征点包围的源点云区域的点集 
	connect(
		ui.btnBoxFilter,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnBoxFilterClicked(bool)));

	//按下 "TransFlt" 按钮时调用，将滤波的点集用类成员m_dMatTrans变换后得到新点集坐标和vtk顶点对象
	connect(
		ui.btnTransFilter,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnTransFilterClicked(bool)));

	//按下 "AlgoTest" 按钮时调用，用于测试算法
	connect(
		ui.btnTestAlgo,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnAlgorithmTestClicked(bool)));

	//选择Source的“Show”复选框，确定是否显示源物体或源点云
	connect(
		ui.chkSrcShow,
		SIGNAL(stateChanged(int)),
		this,
		SLOT(OnIfShowSource(int)));

	//选择Source的“Mesh”复选框，确定是显示源mesh物体，还是显示点云
	connect(
		ui.chkSrcMesh,
		SIGNAL(stateChanged(int)),
		this,
		SLOT(OnIfDisplaySrcMesh(int)));

	//选择Filtered的“Show”复选框，确定是否显示源经过滤波、变换后的mesh物体或相关点云
	connect(
		ui.chkFltShow,
		SIGNAL(stateChanged(int)),
		this,
		SLOT(OnIfShowFilter(int)));

	//选择Filtered的“Mesh”复选框，确定是显示滤波、变换后的源mesh物体，还是显示相关点云
	connect(
		ui.chkFltMesh,
		SIGNAL(stateChanged(int)),
		this,
		SLOT(OnIfDisplayFilterMesh(int)));

	//选择Target的“Show”复选框，确定是否显示目标mesh物体或点云
	connect(
		ui.chkTarShow,
		SIGNAL(stateChanged(int)),
		this,
		SLOT(OnIfShowTarget(int)));

	//选择Target的“Mesh”复选框，确定是显示目标mesh物体，还是显示点云
	connect(
		ui.chkTarMesh,
		SIGNAL(stateChanged(int)),
		this,
		SLOT(OnIfDisplayTarMesh(int)));

	//选择OBBox的“Show”复选框，确定是否显示OBB
	connect(
		ui.chkObbShow,
		SIGNAL(stateChanged(int)),
		this,
		SLOT(OnIfShowOBBox(int)));

	findAllDirectory("F:/ModeData/FiveSurface/");
	double dCorner[] = { 0.0,0.0,0.0 };
	double dMax[] = { 10.0, 0.0, 0.0 };
	double dMid[] = { 0.0, 10.0, 0.0 };
	double dMin[] = { 0.0, 0.0, 10.0 };
	CreateVtkBox(m_pvtk5PtsOBB, dCorner, dMax, dMid, dMin);
	m_pvtkMapper5PtsOBB->SetInputData(m_pvtk5PtsOBB);
	m_pvtkActor5PtsOBB->GetProperty()->SetOpacity(0.5);
	m_pvtkActor5PtsOBB->SetMapper(m_pvtkMapper5PtsOBB);
	m_pvtkRenderer->AddActor(m_pvtkActor5PtsOBB);
	
	ui.qvtkWidget->GetRenderWindow()->AddRenderer(m_pvtkRenderer);
}

/*********************************************************************
说明：
	当窗口尺寸发生变化时调用该方法。然后，根据窗口大小更新vtk窗口大小。
	同时设置几个控件的特性。
输入：
	event：窗口尺寸变换事件的指针。
返回：
	无。
**********************************************************************/
void QtRegAlgorithm::resizeEvent(QResizeEvent* event)
{
	if (nullptr != ui.qvtkWidget)
	{
		QSize szWnd = geometry().size();

		szWnd.setWidth(szWnd.width() - QT_GROUPBOX_WIDTH);
		ui.qvtkWidget->resize(szWnd);
	}
}

void QtRegAlgorithm::closeEvent(QCloseEvent* event)
{
	if (m_pdSrcPtSet != NULL)
	{
		delete[] m_pdSrcPtSet;
	}
	if (m_pdFilterPtSet != NULL)
	{
		delete[] m_pdFilterPtSet;
	}
	if (m_pdFltCpyPtSet != NULL)
	{
		delete[] m_pdFltCpyPtSet;
	}
	if (m_pdTarPtSet != NULL)
	{
		delete[] m_pdTarPtSet;
	}
}

//Respons slot method
//Buttons responses.
/*****************************************************************
说明：
	点击“Source File Operator”分组框里“OpenMesh”，打开模型
	mesh文件并显示在模型窗口里。同时提取中间的点云到类成员变量
	里。支持ply、obj和stl格式。
输入：
	checked：（不起作用）。
返回：
	正确打开文件返回0；否则返回负数。
******************************************************************/
int QtRegAlgorithm::OnOpenSrcMeshClicked(bool checked)
{
	bool    bShowMesh;
	int     nPtNum, nFaceNum;
	QString qstrPath = "F:/ModeData/FiveSurface/";
	qstrPath += ui.cbBoxSubjectName->currentText();
	qstrPath += "/camera/";
	QString fileName = QFileDialog::getOpenFileName(
		this, tr("open source mesh file"), qstrPath,
		tr("ply file(*.ply);; stl file(*.stl);; PointCloud file(*.pcd);; obj file(*.obj)"));

	if (fileName.isEmpty())
	{
		return -1;
	}

	if (fileName.endsWith("obj"))
	{
		auto pvtkObjReader = vtkSmartPointer<vtkOBJReader>::New();
		pvtkObjReader->SetFileName(fileName.toLocal8Bit().constData());
		pvtkObjReader->Update();
		m_pvtkPolysSrc = pvtkObjReader->GetOutput();
	}
	else if (fileName.endsWith("ply"))
	{
		auto pvtkPlyReader = vtkSmartPointer<vtkPLYReader>::New();
		pvtkPlyReader->SetFileName(fileName.toLocal8Bit().constData());
		pvtkPlyReader->Update();
		m_pvtkPolysSrc = pvtkPlyReader->GetOutput();
	}
	else if (fileName.endsWith("stl"))
	{
		auto pvtkStlReader = vtkSmartPointer<vtkSTLReader>::New();
		pvtkStlReader->SetFileName(fileName.toLocal8Bit().constData());
		pvtkStlReader->Update();
		m_pvtkPolysSrc = pvtkStlReader->GetOutput();
	}
	else
	{
		QMessageBox::warning(this, "Warning", "不支持的mesh文件格式！");
		return -6;
	}

	m_bSrcMeshOpened = true;

	m_nSrcPtNum = fromMesh2PtSet(m_pvtkPolysSrc, &m_pdSrcPtSet);
	fromMesh2Points(m_pvtkPolysSrcVert, m_pvtkPolysSrc);

	m_pvtkMapperSrc->SetInputData(m_pvtkPolysSrc);
	m_pvtkActorSrc->SetMapper(m_pvtkMapperSrc);
	m_pvtkRenderer->AddActor(m_pvtkActorSrc);
	m_pvtkActorSrc->GetProperty()->SetColor(1.0, 0.0, 0.0);

	m_pvtkMapperSrcVert->SetInputData(m_pvtkPolysSrcVert);
	m_pvtkActorSrcVert->SetMapper(m_pvtkMapperSrcVert);
	m_pvtkRenderer->AddActor(m_pvtkActorSrcVert);
	m_pvtkActorSrcVert->GetProperty()->SetPointSize(2.0);
	m_pvtkActorSrcVert->GetProperty()->SetColor(1.0, 0.0, 0.0);

	m_pvtkMapperFlt->SetInputData(m_pvtkPolysFlt);
	m_pvtkActorFlt->SetMapper(m_pvtkMapperFlt);
	m_pvtkRenderer->AddActor(m_pvtkActorFlt);
	m_pvtkActorFlt->GetProperty()->SetColor(1.0, 1.0, 0.0);

	m_pvtkMapperFltVert->SetInputData(m_pvtkPolysFltVert);
	m_pvtkActorFltVert->SetMapper(m_pvtkMapperFltVert);
	m_pvtkRenderer->AddActor(m_pvtkActorFltVert);
	m_pvtkActorFltVert->GetProperty()->SetPointSize(2.0);
	m_pvtkActorFltVert->GetProperty()->SetColor(1.0, 1.0, 0.0);

	ui.chkSrcShow->setChecked(true);
	ui.chkSrcMesh->setChecked(false);

	Display(ui.chkSrcShow->isChecked(), ui.chkSrcMesh->isChecked(),
		ui.chkFltShow->isChecked(), ui.chkFltMesh->isChecked(),
		ui.chkTarShow->isChecked(), ui.chkTarMesh->isChecked(),
		ui.chkObbShow->isChecked());

	nFaceNum = m_pvtkPolysSrc->GetNumberOfPolys();
	nPtNum = m_pvtkPolysSrc->GetNumberOfPoints();
	
	ui.lEditSrcPtNum->setText(QString::number(nPtNum));
	ui.lEditSrcFcNum->setText(QString::number(nFaceNum));

	/*m_pvtkRenderer->ResetCamera();
	ui.qvtkWidget->GetRenderWindow()->AddRenderer(m_pvtkRenderer);
	ui.qvtkWidget->GetRenderWindow()->Render();*/

	return 0;
}

/*****************************************************************
说明：
	点击“Source File Operator”分组框里“SaveMesh”按钮,保存经过
	滤波等处理后的mesh数据，支持obj、stl和ply三种文件格式。
输入：
	checked：（不起作用）。
返回：
	保存成功返回0，否则返回一个负数。
******************************************************************/
int QtRegAlgorithm::OnSaveSrcMeshClicked(bool checked)
{
	if (m_pvtkPolysSrc->GetPoints() == NULL)
	{
		QMessageBox::warning(this, "Warning", "没有保存对象物体！");
		return -1;
	}

	//获取保存路径
	QString qstrPath = "F:/ModeData/FiveSurface/";
	qstrPath += ui.cbBoxSubjectName->currentText();
	qstrPath += "/camera/";
	QString fileName = QFileDialog::getSaveFileName(this, tr("Save mesh file"), qstrPath,
		tr("ply file(*.ply);; stl file(*.stl);; obj file(*.obj)"));

	if (fileName.isEmpty())
	{
		return -1;
	}

	if (fileName.endsWith("ply"))
	{
		vtkSmartPointer<vtkPLYWriter> pvtkPlyWriter = vtkSmartPointer<vtkPLYWriter>::New();
		pvtkPlyWriter->SetFileName(fileName.toLocal8Bit().constData());
		pvtkPlyWriter->SetInputData(m_pvtkPolysSrc);
		pvtkPlyWriter->Update();
	}
	else if (fileName.endsWith("obj"))
	{
		vtkSmartPointer<vtkOBJWriter> pvtkObjWriter = vtkSmartPointer<vtkOBJWriter>::New();
		pvtkObjWriter->SetFileName(fileName.toLocal8Bit().constData());
		pvtkObjWriter->SetInputData(m_pvtkPolysSrc);
		pvtkObjWriter->Update();
	}
	else if (fileName.endsWith("stl"))
	{
		vtkSmartPointer<vtkSTLWriter> pvtkStlWriter = vtkSmartPointer<vtkSTLWriter>::New();
		pvtkStlWriter->SetFileName(fileName.toLocal8Bit().constData());
		pvtkStlWriter->SetInputData(m_pvtkPolysSrc);
		pvtkStlWriter->Update();
	}
	else
	{
		QMessageBox::warning(this, "Warning", "mesh文件格式错误！");
		return -2;
	}

	return 0;
}

/*****************************************************************
说明：
	点击“Target File Operator”“OpenMesh”,打开模型mesh文件并
	显示在模型窗口里。同时提取中间的点云到类成员变量里。
	支持ply、obj和stl格式。
输入：
	checked：（不起作用）。
返回：
	正确打开文件返回0；否则返回负数。
******************************************************************/
int QtRegAlgorithm::OnOpenTarMeshClicked(bool checked)
{
	bool    bShowMesh;
	int     nPtNum, nFaceNum;
	QString qstrPath = "F:/ModeData/FiveSurface/";
	qstrPath += ui.cbBoxSubjectName->currentText();
	qstrPath += "/Mri/";
	QString fileName = QFileDialog::getOpenFileName(
		this, tr("open target mesh file"), qstrPath,
		tr("stl file(*.stl);;PointCloud file(*.pcd);; obj file(*.obj);; ply file(*.ply)"));

	if (fileName.isEmpty())
	{
		return -1;
	}

	if (fileName.endsWith("obj"))
	{
		auto pvtkObjReader = vtkSmartPointer<vtkOBJReader>::New();
		pvtkObjReader->SetFileName(fileName.toLocal8Bit().constData());
		pvtkObjReader->Update();
		m_pvtkPolysTar = pvtkObjReader->GetOutput();
	}
	else if (fileName.endsWith("ply"))
	{
		auto pvtkPlyReader = vtkSmartPointer<vtkPLYReader>::New();
		pvtkPlyReader->SetFileName(fileName.toLocal8Bit().constData());
		pvtkPlyReader->Update();
		m_pvtkPolysTar = pvtkPlyReader->GetOutput();
	}
	else if (fileName.endsWith("stl"))
	{
		auto pvtkStlReader = vtkSmartPointer<vtkSTLReader>::New();
		pvtkStlReader->SetFileName(fileName.toLocal8Bit().constData());
		pvtkStlReader->Update();
		m_pvtkPolysTar = pvtkStlReader->GetOutput();
	}
	else
	{
		QMessageBox::warning(this, "Warning", "不支持的mesh文件格式！");
		return -6;
	}

	m_bTarMeshOpened = true;

	fromMesh2PtSet(m_pvtkPolysTar, &m_pdTarPtSet);
	fromMesh2Points(m_pvtkPolysTarVert, m_pvtkPolysTar);

	nPtNum = m_pvtkPolysTar->GetNumberOfPoints();
	m_objRegIcp.BuildIndex(m_pdTarPtSet, nPtNum);

	m_pvtkMapperTar->SetInputData(m_pvtkPolysTar);
	m_pvtkActorTar->SetMapper(m_pvtkMapperTar);
	m_pvtkRenderer->AddActor(m_pvtkActorTar);
	m_pvtkActorTar->GetProperty()->SetColor(0.0, 1.0, 0.16);

	m_pvtkMapperTarVert->SetInputData(m_pvtkPolysTarVert);
	m_pvtkActorTarVert->SetMapper(m_pvtkMapperTarVert);
	m_pvtkRenderer->AddActor(m_pvtkActorTarVert);
	m_pvtkActorTarVert->GetProperty()->SetPointSize(1.0);
	m_pvtkActorTarVert->GetProperty()->SetColor(0.0, 1.0, 0.16);

	ui.chkTarShow->setChecked(true);
	ui.chkTarMesh->setChecked(true);

	Display(ui.chkSrcShow->isChecked(), ui.chkSrcMesh->isChecked(),
		ui.chkFltShow->isChecked(), ui.chkFltMesh->isChecked(),
		ui.chkTarShow->isChecked(), ui.chkTarMesh->isChecked(),
		ui.chkObbShow->isChecked());

	nFaceNum = m_pvtkPolysTar->GetNumberOfPolys();
	nPtNum = m_pvtkPolysTar->GetNumberOfPoints();

	ui.lEditTarPtNum->setText(QString::number(nPtNum));
	ui.lEditTarFcNum->setText(QString::number(nFaceNum));

	//m_pvtkRenderer->ResetCamera();
	//ui.qvtkWidget->GetRenderWindow()->AddRenderer(m_pvtkRenderer);
	//ui.qvtkWidget->GetRenderWindow()->Render();

	return 0;
}

/*****************************************************************
说明：
	点击“Target File Operator”分组框里“SaveMesh”,保存经过
	滤波等处理后的mesh数据，支持obj、stl和ply三种文件格式。
输入：
	checked：（不起作用）。
返回：
	保存成功返回0，否则返回一个负数。
******************************************************************/
int QtRegAlgorithm::OnSaveTarMeshClicked(bool checked)
{
	if (m_pvtkPolysTar->GetPoints() == NULL)
	{
		QMessageBox::warning(this, "Warning", "No target mesh is opened!");
		return -1;
	}

	//获取保存路径
	QString qstrPath = "F:/ModeData/FiveSurface/";
	qstrPath += ui.cbBoxSubjectName->currentText();
	qstrPath += "/Mri/";
	QString fileName = QFileDialog::getSaveFileName(this, tr("Save mesh file"),	qstrPath,
		tr("stl file(*.stl);; ply file(*.ply);; obj file(*.obj)"));

	if (fileName.isEmpty())
	{
		return -1;
	}

	if (fileName.endsWith("ply"))
	{
		vtkSmartPointer<vtkPLYWriter> pvtkPlyWriter = vtkSmartPointer<vtkPLYWriter>::New();
		pvtkPlyWriter->SetFileName(fileName.toLocal8Bit().constData());
		pvtkPlyWriter->SetInputData(m_pvtkPolysTar);
		pvtkPlyWriter->Update();
	}
	else if (fileName.endsWith("obj"))
	{
		vtkSmartPointer<vtkOBJWriter> pvtkObjWriter = vtkSmartPointer<vtkOBJWriter>::New();
		pvtkObjWriter->SetFileName(fileName.toLocal8Bit().constData());
		pvtkObjWriter->SetInputData(m_pvtkPolysTar);
		pvtkObjWriter->Update();
	}
	else if (fileName.endsWith("stl"))
	{
		vtkSmartPointer<vtkSTLWriter> pvtkStlWriter = vtkSmartPointer<vtkSTLWriter>::New();
		pvtkStlWriter->SetFileName(fileName.toLocal8Bit().constData());
		pvtkStlWriter->SetInputData(m_pvtkPolysTar);
		pvtkStlWriter->Update();
	}
	else
	{
		QMessageBox::warning(this, "Warning", "mesh文件格式错误！");
		return -2;
	}

	return 0;
}

/*****************************************************************
说明：
	保存经过滤波等处理后的mesh数据，支持obj、stl和ply三种文件格式。
输入：
	checked：（不起作用）。
返回：
	保存成功返回0，否则返回一个负数。
******************************************************************/
int QtRegAlgorithm::OnICPRegistClicked(bool checked)
{
	double  dMatrix[4][4];

	double  dAveDist = m_objRegIcp.RegistByICP(m_pdFilterPtSet, m_nFltPtNum, 5, dMatrix);

	double a11 = dMatrix[0][0];
	double a12 = dMatrix[0][1];
	double a13 = dMatrix[0][2];
	double a14 = dMatrix[0][3];
	double a21 = dMatrix[1][0];
	double a22 = dMatrix[1][1];
	double a23 = dMatrix[1][2];
	double a24 = dMatrix[1][3];
	double a31 = dMatrix[2][0];
	double a32 = dMatrix[2][1];
	double a33 = dMatrix[2][2];
	double a34 = dMatrix[2][3];

#pragma omp parallel for firstprivate(a11,a12,a13,a14,a21,a22,a23,a24,a31,a32,a33,a34)
	for (int i = 0; i < m_nFltPtNum; i++)
	{
		double x, y, z;
		x = m_pdFilterPtSet[3 * i] * a11 + m_pdFilterPtSet[3 * i + 1] * a12 + m_pdFilterPtSet[3 * i + 2] * a13 + a14;
		y = m_pdFilterPtSet[3 * i] * a21 + m_pdFilterPtSet[3 * i + 1] * a22 + m_pdFilterPtSet[3 * i + 2] * a23 + a24;
		z = m_pdFilterPtSet[3 * i] * a31 + m_pdFilterPtSet[3 * i + 1] * a32 + m_pdFilterPtSet[3 * i + 2] * a33 + a34;

		m_pdFilterPtSet[3 * i] = x;
		m_pdFilterPtSet[3 * i + 1] = y;
		m_pdFilterPtSet[3 * i + 2] = z;
	}

	fromPtSet2Vertex(m_pdFilterPtSet, m_nFltPtNum, m_pvtkPolysFltVert);

	ui.chkTarShow->setChecked(true);
	ui.chkTarMesh->setChecked(true);

	ui.chkFltShow->setChecked(true);
	ui.chkFltMesh->setChecked(false);

	ui.chkSrcShow->setChecked(false);
	ui.chkObbShow->setChecked(false);

	//显示平均距离
	ui.lEditAveDist->setText(QString::number(dAveDist, 'f', 4));
	//显示变换矩阵
	QString qstrMatrix, qstrTemp;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			qstrTemp = QString("%1 ").arg(dMatrix[i][j], 0, 'g', 7);
			qstrMatrix.append(qstrTemp);
		}
		qstrMatrix.append("\r\n");
	}
	ui.txtEditTranMat->setPlainText(qstrMatrix);

	Display(ui.chkSrcShow->isChecked(), ui.chkSrcMesh->isChecked(),
		ui.chkFltShow->isChecked(), ui.chkFltMesh->isChecked(),
		ui.chkTarShow->isChecked(), ui.chkTarMesh->isChecked(),
		ui.chkObbShow->isChecked());

	//ui.lEditAveDist->setText(QString::number(dAveDist, 'f', 4));

	return 0;
}

/*****************************************************************
说明：
	点击按钮“5Feat”,调用配准控件m_objRegIcp的相关方法计算变换
	矩阵。并显示误差和总方差。
输入：
	checked：（不起作用）。
返回：
	返回0。
******************************************************************/
int QtRegAlgorithm::OnReg5FPlusICPClicked(bool checked)
{
	double dMatrix[4][4];

	double dAveDist = m_objRegIcp.RegBy5FPlusICP(m_dSrcFea, m_dTarFea, FEAT_NUM_HX, m_pdSrcPtSet, m_nSrcPtNum, dMatrix);
	fromPtSet2VertexByTrans(m_pdSrcPtSet, m_nSrcPtNum, m_pvtkPolysFltVert, dMatrix);

	ui.lEditAveDist->setText(QString::number(dAveDist, 'f', 4));
	//显示变换矩阵
	QString qstrMatrix, qstrTemp;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			qstrTemp = QString("%1 ").arg(dMatrix[i][j], 0, 'g', 7);
			qstrMatrix.append(qstrTemp);
		}
		qstrMatrix.append("\r\n");
	}
	ui.txtEditTranMat->setPlainText(qstrMatrix);

	return 0;
}

/*****************************************************************
说明：
	点击按钮“ICP2”,调用配准控件m_objRegIcp的相关方法计算变换
	矩阵。并显示误差和总方差。
输入：
	checked：（不起作用）。
返回：
	返回0。
******************************************************************/
int QtRegAlgorithm::OnICPRegICP2Clicked(bool checked)
{
	double dMatrix[4][4];

	double dAveDist = m_objRegIcp.RegByIcp(m_pdSrcPtSet, m_nSrcPtNum, 3, dMatrix);

	fromPtSet2VertexByTrans(m_pdSrcPtSet, m_nSrcPtNum, m_pvtkPolysFltVert, dMatrix);

	Display(ui.chkSrcShow->isChecked(), ui.chkSrcMesh->isChecked(),
		ui.chkFltShow->isChecked(), ui.chkFltMesh->isChecked(),
		ui.chkTarShow->isChecked(), ui.chkTarMesh->isChecked(),
		ui.chkObbShow->isChecked());

	//显示变换矩阵
	QString qstrMatrix, qstrTemp;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			qstrTemp = QString("%1 ").arg(dMatrix[i][j], 0, 'g', 7);
			qstrMatrix.append(qstrTemp);
		}
		qstrMatrix.append("\r\n");
	}
	ui.txtEditTranMat->setPlainText(qstrMatrix);
	ui.lEditAveDist->setText(QString::number(dAveDist, 'f', 4));

	return 0;
}

/*****************************************************************
说明：
	点击按钮“5Feat”,调用配准控件m_objRegIcp的相关方法计算变换
	矩阵。并显示误差和总方差。
输入：
	checked：（不起作用）。
返回：
	返回0。
******************************************************************/
int QtRegAlgorithm::OnFiveFeatureClicked(bool checked)
{
	double  dStdErr = -1;
	double  dError[FEAT_NUM_HX];
	double  dFeatureCam[FEAT_NUM_HX][3];
	//double  dMatrix[4][4];
	QString qstrMatrix, qstrTemp, qstrFeatures;
	//auto pvtkMat = vtkSmartPointer<vtkMatrix4x4>::New();
	//auto pvtkTransf = vtkSmartPointer<vtkTransform>::New();

	dStdErr = m_objRegIcp.SolveTransfMatrix(dError);
	m_objRegIcp.GetTransMatrix(m_dMatTrans);

	//显示变换矩阵
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			qstrTemp = QString("%1 ").arg(m_dMatTrans[i][j], 10, 'g', 5, ' ');
			qstrMatrix.append(qstrTemp);
			//pvtkMat->SetElement(i, j, m_dMatTrans[i][j]);
		}
		qstrMatrix.append("\r\n");
	}
	ui.txtEditTranMat->setPlainText(qstrMatrix);

	/*pvtkTransf->SetMatrix(pvtkMat);
	m_pvtkActorSrc->SetUserTransform(pvtkTransf);
	DisplaySrcMeshes(true);*/

	m_objRegIcp.Get5FeaturePtsCoord(dFeatureCam, false);
	//显示误差
	for (int i = 0; i < FEATURE_PT_NUM; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			qstrTemp = QString("%1 ").arg(dFeatureCam[i][j], 9, 'g', 5, ' ');
			qstrFeatures.append(qstrTemp);
		}

		qstrTemp = QString("%1\r\n").arg(dError[i], 9, 'g', 5, ' ');
		qstrFeatures.append(qstrTemp);
	}

	ui.lEditStdErr->setText(QString::number(dStdErr, 'f', 4));
	ui.txtEditCamFeaturePts->setPlainText(qstrFeatures);

	return 0;
}

/*****************************************************************
说明：
	按下“BoxFilter”按钮，执行用5特征点的最小包围盒对源点云滤波：
	将OBB盒子长轴和中长轴包围的开顶长方盒内的点去出来，其它点均
	过滤掉。
输入：
	checked：（不起作用）。
返回：
	返回0。
******************************************************************/
int QtRegAlgorithm::OnBoxFilterClicked(bool checked)
{
	double corner[3], max[3], mid[3], min[3], size[3];

	m_objRegIcp.ComputeOBB(corner, max, mid, min, size);

	if (m_pdFilterPtSet != NULL)
	{
		delete[] m_pdFilterPtSet;
		m_pdFilterPtSet = NULL;
	}

	m_nFltPtNum = m_objRegIcp.BoundBoxFilter(m_pdSrcPtSet, m_pvtkPolysSrc->GetNumberOfPoints(), &m_pdFilterPtSet, corner, max, mid, min);

	if (m_pdFltCpyPtSet != NULL)
	{
		delete[] m_pdFltCpyPtSet;
		
	}

	m_pdFltCpyPtSet = new double[3 * m_nFltPtNum * sizeof(double)];

	memcpy(m_pdFltCpyPtSet, m_pdFilterPtSet, 3 * m_nFltPtNum * sizeof(double));

	fromPtSet2Vertex(m_pdFilterPtSet, m_nFltPtNum, m_pvtkPolysFltVert);

	CreateVtkBox(m_pvtk5PtsOBB, corner, max, mid, min);

	ui.chkObbShow->setChecked(true);
	ui.chkFltShow->setChecked(true);
	ui.chkFltMesh->setChecked(false);

	Display(ui.chkSrcShow->isChecked(), ui.chkSrcMesh->isChecked(),
		ui.chkFltShow->isChecked(), ui.chkFltMesh->isChecked(),
		ui.chkTarShow->isChecked(), ui.chkTarMesh->isChecked(),
		ui.chkObbShow->isChecked());

	ui.lEditFltPtNum->setText(QString::number(m_nFltPtNum));

	return 0;
}

/*****************************************************************
说明：
	按下“TransFlt”按钮，用类成员m_dMatTrans对滤波的点云进行
	坐标变换，并显示变换后的位置。
输入：
	checked：（不起作用）。
返回：
	返回0。
******************************************************************/
int QtRegAlgorithm::OnTransFilterClicked(bool checked)
{
	int nFltNum = m_pvtkPolysFltVert->GetNumberOfPoints();

	memcpy(m_pdFilterPtSet, m_pdFltCpyPtSet, 3 * nFltNum * sizeof(double));

	TransfPointSet(&m_pdFilterPtSet, nFltNum, m_pvtkPolysFltVert);

	double dAveDist = m_objRegIcp.CalcNearestPointPairs(m_pdFilterPtSet, nFltNum);

	ui.chkTarShow->setChecked(true);
	ui.chkTarMesh->setChecked(true);

	ui.chkFltShow->setChecked(true);
	ui.chkFltMesh->setChecked(false);

	ui.chkSrcShow->setChecked(false);
	ui.chkObbShow->setChecked(false);

	ui.lEditAveDist->setText(QString::number(dAveDist, 'f', 4));
	Display(ui.chkSrcShow->isChecked(), ui.chkSrcMesh->isChecked(),
		ui.chkFltShow->isChecked(), ui.chkFltMesh->isChecked(),
		ui.chkTarShow->isChecked(), ui.chkTarMesh->isChecked(),
		ui.chkObbShow->isChecked());

	return 0;
}

/*****************************************************************
说明：
	按下“AlgoTest”按钮，用于测试和调试各种算法。
输入：
	checked：（不起作用）。
返回：
	返回0。
******************************************************************/
int QtRegAlgorithm::OnAlgorithmTestClicked(bool checked)
{
	//CRegICPHxmc  objRegIcp;
	//double       dMatrix[4][4];
	//double*      pdTarPtSet = NULL;
	//double*      pdSrcPtSet = NULL;

	//int nSrcPtNum = fromMesh2PtSet(m_pvtkPolysSrc, &pdSrcPtSet);
	//int nTarPtNum = fromMesh2PtSet(m_pvtkPolysTar, &pdTarPtSet);

	//objRegIcp.BuildIndex(pdTarPtSet, nTarPtNum);

	//
	////double dAveDist = objRegIcp.RegistByICP(pdSrcPtSet, nSrcPtNum, 1, dMatrix);
	//double dAveDist = objRegIcp.RegByIcp(pdSrcPtSet, nSrcPtNum, 1, dMatrix);

	//ui.lEditAveDist->setText(QString::number(dAveDist, 'f', 4));
	////显示变换矩阵
	//QString qstrMatrix, qstrTemp;
	//for (int i = 0; i < 4; i++)
	//{
	//	for (int j = 0; j < 4; j++)
	//	{
	//		qstrTemp = QString("%1 ").arg(dMatrix[i][j], 0, 'g', 7);
	//		qstrMatrix.append(qstrTemp);
	//	}
	//	qstrMatrix.append("\r\n");
	//}
	//ui.txtEditTranMat->setPlainText(qstrMatrix);

	CRegICPHxmc  m_objRegIcp;
	qDebug() << "starting";
	/////////////////////////////1 OnOpenTarMesh 
	vtkSmartPointer<vtkPolyData> m_pvtkPolysTar = vtkSmartPointer<vtkPolyData>::New();
	QString qstrPath = "F:\\ModeData\\FiveSurface\\wangrenjie\\mri\\mriSurface.stl";
	auto pvtkStlReader = vtkSmartPointer<vtkSTLReader>::New();
	pvtkStlReader->SetFileName(qstrPath.toLocal8Bit().constData());
	pvtkStlReader->Update();
	m_pvtkPolysTar = pvtkStlReader->GetOutput();
	double* m_pdTarPtSet = NULL;
	fromMesh2PtSet(m_pvtkPolysTar, &m_pdTarPtSet);
	m_objRegIcp.BuildIndex(m_pdTarPtSet, m_pvtkPolysTar->GetNumberOfPoints());
	qDebug() << "111";
	/////////////////////2 OnOpenSrcMesh para4
	vtkSmartPointer<vtkPolyData> m_pvtkPolysSrc = vtkSmartPointer<vtkPolyData>::New();
	auto pvtkPlyReader = vtkSmartPointer<vtkPLYReader>::New();
	QString qstrPathmesh = "F:\\ModeData\\FiveSurface\\wangrenjie\\Camera\\fullface.ply";
	pvtkPlyReader->SetFileName(qstrPathmesh.toLocal8Bit().constData());
	pvtkPlyReader->Update();
	double* m_pdSrcPtSet = NULL;
	m_pvtkPolysSrc = pvtkPlyReader->GetOutput();
	int m_nSrcPtNum = fromMesh2PtSet(m_pvtkPolysSrc, &m_pdSrcPtSet);
	qDebug() << "222";

	double dMatrix[4][4];
	double dAveDist = m_objRegIcp.RegByIcp(m_pdSrcPtSet, m_nSrcPtNum, 1, dMatrix);

	qDebug() << "out : " << dAveDist;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			qDebug() << dMatrix[i][j];
		}
	}


	return 0;
}

/*****************************************************************
说明：
	点击“OpenCam”按钮，打开5特征点的相机文件，读取5点坐标，
	并传递到配准对象m_objRegIcp里。
输入：
	checked：（不起作用）。
返回：
	读取数据的数量，应该为15。
******************************************************************/
int QtRegAlgorithm::OnOpenCam5FeaturesClicked(bool checked)
{
	int     i, j, nDataNum;
	double  dFeatureCam[5][3];
	QString strFeatures, strTemp;
	QString qstrPath = "F:/ModeData/FiveSurface/";
	qstrPath += ui.cbBoxSubjectName->currentText();
	qstrPath += "/camera/";
	QString fileName = QFileDialog::getOpenFileName(
		this, tr("open point cloud or mesh file"),
		qstrPath,
		tr("FeaturePts file(*.txt);; all file(*.*)"));

	if (fileName.isEmpty())
	{
		return -1;
	}

	nDataNum = readFeaturePts(fileName, dFeatureCam);

	if (nDataNum == FEATURE_PT_NUM * 3)
	{
		//显示特征点
		for (i = 0; i < FEATURE_PT_NUM; i++)
		{
			for (j = 0; j < 3; j++)
			{
				m_dSrcFea[i][j] = dFeatureCam[i][j];
				strTemp = QString("%1 ").arg(dFeatureCam[i][j], 10, 'g', 4, ' ');
				strFeatures.append(strTemp);
			}
			strFeatures.append("\r\n");
		}

		ui.txtEditCamFeaturePts->setPlainText(strFeatures);
	}
	else
	{
		QMessageBox::warning(this, "Warning", "读取特征点文件出错！");
		return -2;
	}

	m_objRegIcp.Set5FeaturePtsCoord(dFeatureCam, false);

	return nDataNum;
}

/*****************************************************************
说明：
	点击“OpenMri”按钮，打开5特征点的MRI文件，读取5点坐标，
	并传递到配准对象m_objRegIcp里。
输入：
	checked：（不起作用）。
返回：
	读取数据的数量，应该为15。
******************************************************************/
int QtRegAlgorithm::OnOpenMri5FeaturesClicked(bool checked)
{
	int     i, j, nDataNum;
	double  dFeatureMri[5][3];
	QString strFeatures, strTemp;
	QString qstrPath = "F:/ModeData/FiveSurface/";
	qstrPath += ui.cbBoxSubjectName->currentText();
	qstrPath += "/mri/";
	QString fileName = QFileDialog::getOpenFileName(
		this, tr("open point cloud or mesh file"),
		qstrPath,
		tr("FeaturePts file(*.txt);; all file(*.*)"));

	if (fileName.isEmpty())
	{
		return -1;
	}

	nDataNum = readFeaturePts(fileName, dFeatureMri);

	if (nDataNum == FEATURE_PT_NUM * 3)
	{
		//显示特征点
		for (i = 0; i < FEATURE_PT_NUM; i++)
		{
			for (j = 0; j < 3; j++)
			{
				m_dTarFea[i][j] = dFeatureMri[i][j];
				strTemp = QString("%1  ").arg(dFeatureMri[i][j], 10, 'g', 4, ' ');
				strFeatures.append(strTemp);
			}
			strFeatures.append("\r\n");
		}

		ui.txtEditMriFeaturePts->setPlainText(strFeatures);
	}
	else
	{
		QMessageBox::warning(this, "Warning", "读取特征点文件出错！");
		return -2;
	}

	m_objRegIcp.Set5FeaturePtsCoord(dFeatureMri, true);

	return nDataNum;
}

//CheckBoxes responses
/************************************************************
说明：
	选择Source的“Show”复选框，以确定是否显示源mesh或点云。
参数：
	nState：复选框的状态。
返回：
	0。
**************************************************************/
int QtRegAlgorithm::OnIfShowSource(int nState)
{
	bool bShowSrc;

	if (nState == Qt::Checked)
	{
		bShowSrc = true;
	}
	else
	{
		bShowSrc = false;
	}

	Display(bShowSrc, ui.chkSrcMesh->isChecked(),
		ui.chkFltShow->isChecked(), ui.chkFltMesh->isChecked(),
		ui.chkTarShow->isChecked(), ui.chkTarMesh->isChecked(),
		ui.chkObbShow->isChecked());

	return 0;
}

/*******************************************************************
说明：
	选择Source的“Mesh”复选框，以确定是显示源mesh，还是显示源点云。
参数：
	nState：复选框的状态。
返回：
	0。
*********************************************************************/
int QtRegAlgorithm::OnIfDisplaySrcMesh(int nState)
{
	bool bDispSrcMesh;

	if (nState == Qt::Checked)
	{
		bDispSrcMesh = true;
	}
	else
	{
		bDispSrcMesh = false;
	}

	Display(ui.chkSrcShow->isChecked(), bDispSrcMesh,
		ui.chkFltShow->isChecked(), ui.chkFltMesh->isChecked(),
		ui.chkTarShow->isChecked(), ui.chkTarMesh->isChecked(),
		ui.chkObbShow->isChecked());

	return 0;
}

int QtRegAlgorithm::OnIfShowFilter(int nState)
{
	bool bShowFlt;

	if (nState == Qt::Checked)
	{
		bShowFlt = true;
	}
	else
	{
		bShowFlt = false;
	}

	Display(ui.chkSrcShow->isChecked(), ui.chkSrcMesh->isChecked(),
		bShowFlt, ui.chkFltMesh->isChecked(),
		ui.chkTarShow->isChecked(), ui.chkTarMesh->isChecked(),
		ui.chkObbShow->isChecked());

	return 0;
}

int QtRegAlgorithm::OnIfDisplayFilterMesh(int nState)
{
	bool bDispFltMesh;

	if (nState == Qt::Checked)
	{
		bDispFltMesh = true;
	}
	else
	{
		bDispFltMesh = false;
	}

	Display(ui.chkSrcShow->isChecked(), ui.chkSrcMesh->isChecked(),
		ui.chkFltShow->isChecked(), bDispFltMesh,
		ui.chkTarShow->isChecked(), ui.chkTarMesh->isChecked(),
		ui.chkObbShow->isChecked());

	return 0;
}

int QtRegAlgorithm::OnIfShowTarget(int nState)
{
	bool bShowTar;

	if (nState == Qt::Checked)
	{
		bShowTar = true;
	}
	else
	{
		bShowTar = false;
	}

	Display(ui.chkSrcShow->isChecked(), ui.chkSrcMesh->isChecked(),
		ui.chkFltShow->isChecked(), ui.chkFltMesh->isChecked(),
		bShowTar, ui.chkTarMesh->isChecked(),
		ui.chkObbShow->isChecked());

	return 0;
}

int QtRegAlgorithm::OnIfDisplayTarMesh(int nState)
{
	bool bDispTarMesh;

	if (nState == Qt::Checked)
	{
		bDispTarMesh = true;
	}
	else
	{
		bDispTarMesh = false;
	}

	Display(ui.chkSrcShow->isChecked(), ui.chkSrcMesh->isChecked(),
		ui.chkFltShow->isChecked(), ui.chkFltMesh->isChecked(),
		ui.chkTarShow->isChecked(), bDispTarMesh,
		ui.chkObbShow->isChecked());

	return 0;
}

int QtRegAlgorithm::OnIfShowOBBox(int nState)
{
	bool bShowOBB;

	if (nState == Qt::Checked)
	{
		bShowOBB = true;
	}
	else
	{
		bShowOBB = false;
	}

	Display(ui.chkSrcShow->isChecked(), ui.chkSrcMesh->isChecked(), 
		ui.chkFltShow->isChecked(), ui.chkFltMesh->isChecked(),
		ui.chkTarShow->isChecked(), ui.chkTarMesh->isChecked(), 
		bShowOBB);

	
	return 0;
}

//Methods
/************************************************************
说明：
	根据Display Control组的四组复选框，确定显示的对象。
参数：
	bSrcShow：   是否显示源mesh或cloud；
	bSrcMesh：   是显示源的mesh，还是显示Cloud；
	bFilterShow：是否显示源滤波、变换的mesh或cloud；
	bFltMesh：   是显示滤波变换源的mesh，还是显示相应Cloud；
	bTarShow：   是否显示目标mesh或cloud；
	bTarMesh：   是显示目标的mesh，还是显示Cloud；
	bOBBShow：   是否显示5特征点的最小包围盒。
返回：
	无。
**************************************************************/
void QtRegAlgorithm::Display(bool bSrcShow, bool bSrcMesh, bool bFilterShow, bool bFltMesh, bool bTarShow, bool bTarMesh, bool bOBBShow)
{
	vtkActor* pvtkActor;
	vtkActorCollection* pvtkActorCollection = m_pvtkRenderer->GetActors();
	pvtkActorCollection->InitTraversal();
	int nNumActor = pvtkActorCollection->GetNumberOfItems();

	for (int i = 0; i < nNumActor; i++)
	{
		pvtkActor = pvtkActorCollection->GetNextActor();
		m_pvtkRenderer->RemoveActor(pvtkActor);
	}

	if (bSrcShow)
	{
		if (bSrcMesh)
		{
			m_pvtkRenderer->AddActor(m_pvtkActorSrc);
		}
		else
		{
			m_pvtkRenderer->AddActor(m_pvtkActorSrcVert);
		}
	}

	if (bFilterShow)
	{
		if (bFltMesh)
		{
			m_pvtkRenderer->AddActor(m_pvtkActorFlt);
		}
		else
		{
			m_pvtkRenderer->AddActor(m_pvtkActorFltVert);
		}
	}

	if (bTarShow)
	{
		if (bTarMesh)
		{
			m_pvtkRenderer->AddActor(m_pvtkActorTar);
		}
		else
		{
			m_pvtkRenderer->AddActor(m_pvtkActorTarVert);
		}
	}

	if (bOBBShow)
	{
		m_pvtkRenderer->AddActor(m_pvtkActor5PtsOBB);
	}

	m_pvtkRenderer->ResetCamera();
	ui.qvtkWidget->GetRenderWindow()->Render();
}

/************************************************************
说明：
	在显示窗口上显示源mesh，或者显示源点云。
参数：
	bMesh：显示mesh或点云。
返回：
	无。
**************************************************************/
void QtRegAlgorithm::DisplaySrcMeshes(bool bMesh)
{
	vtkActor* pvtkActor;
	vtkActorCollection* pvtkActorCollection = m_pvtkRenderer->GetActors();
	pvtkActorCollection->InitTraversal();
	int nNumActor = pvtkActorCollection->GetNumberOfItems();

	if (bMesh)
	{
		for (int i = 0; i < nNumActor; i++)
		{
			pvtkActor = pvtkActorCollection->GetNextActor();
			if (pvtkActor == m_pvtkActorSrcVert)
			{
				m_pvtkRenderer->RemoveActor(pvtkActor);
				m_pvtkRenderer->AddActor(m_pvtkActorSrc);

				break;
			}
		}
	}
	else
	{
		for (int i = 0; i < nNumActor; i++)
		{
			pvtkActor = pvtkActorCollection->GetNextActor();
			if (pvtkActor == m_pvtkActorSrc)
			{
				m_pvtkRenderer->RemoveActor(pvtkActor);
				m_pvtkRenderer->AddActor(m_pvtkActorSrcVert);

				break;
			}
		}
	}

	m_pvtkRenderer->ResetCamera();
	ui.qvtkWidget->GetRenderWindow()->Render();
}

/************************************************************
说明：
	在显示窗口上显示目标mesh，或者显示目标点云。
参数：
	bMesh：显示mesh或点云。
返回：
	无。
**************************************************************/
void QtRegAlgorithm::DisplayTarMeshes(bool bMesh)
{
	vtkActor* pvtkActor;
	vtkActorCollection* pvtkActorCollection = m_pvtkRenderer->GetActors();
	pvtkActorCollection->InitTraversal();
	int nNumActor = pvtkActorCollection->GetNumberOfItems();

	if (bMesh)
	{
		for (int i = 0; i < nNumActor; i++)
		{
			pvtkActor = pvtkActorCollection->GetNextActor();
			if (pvtkActor == m_pvtkActorTarVert)
			{
				m_pvtkRenderer->RemoveActor(pvtkActor);
				m_pvtkRenderer->AddActor(m_pvtkActorTar);

				break;
			}
		}
	}
	else
	{
		for (int i = 0; i < nNumActor; i++)
		{
			pvtkActor = pvtkActorCollection->GetNextActor();
			if (pvtkActor == m_pvtkActorTar)
			{
				m_pvtkRenderer->RemoveActor(pvtkActor);
				m_pvtkRenderer->AddActor(m_pvtkActorTarVert);

				break;
			}
		}
	}

	m_pvtkRenderer->ResetCamera();
	ui.qvtkWidget->GetRenderWindow()->Render();
}

/************************************************************
说明：
	是否显示5特征点的最小包围盒。
参数：
	bShow：是否显示包围盒。
返回：
	无。
**************************************************************/
void QtRegAlgorithm::Display5PtBox(bool bShow)
{
	if (bShow)
	{
		m_pvtkRenderer->AddActor(m_pvtkActor5PtsOBB);
	}
	else
	{
		m_pvtkRenderer->RemoveActor(m_pvtkActor5PtsOBB);
	}

	m_pvtkRenderer->ResetCamera();
	ui.qvtkWidget->GetRenderWindow()->Render();
}

/************************************************************
说明：
	用vtkPolyData对象中的点，构造成仅仅含顶点的vtkPolyData对象。
参数：
	pvtkMesh(in)：mesh对象，可能含独立的顶点数据；
	pvtkVert(out)：待构造的vtk点云。
返回：
	构造对象所含的顶点数量。
**************************************************************/
int QtRegAlgorithm::fromMesh2Points(vtkSmartPointer<vtkPolyData>pvtkVert, vtkSmartPointer<vtkPolyData>pvtkMesh)
{
	int nPtNum = pvtkMesh->GetNumberOfPoints();

	vtkSmartPointer<vtkCellArray> pvtkVertices = vtkSmartPointer<vtkCellArray>::New();
	vtkSmartPointer<vtkPoints> pvtkPoints = vtkSmartPointer<vtkPoints>::New();

	pvtkVert->Initialize();

	for (int i = 0; i < nPtNum; i++)
	{
		double* pdPtCoord;
		pdPtCoord = pvtkMesh->GetPoint(i);
		pvtkPoints->InsertPoint(i, pdPtCoord[0], pdPtCoord[1], pdPtCoord[2]);
		pvtkVertices->InsertNextCell(1);
		pvtkVertices->InsertCellPoint(i);
	}

	pvtkVert->SetPoints(pvtkPoints);
	pvtkVert->SetVerts(pvtkVertices);

	return nPtNum;
}

/************************************************************
说明：
	将vtkPolyData对象中的点拷贝到ppdPtSet指针的指针所指向的
	内存缓冲区中。
参数：
	pvtkMesh(in)：mesh对象，可能含独立的顶点数据；
	ppdPtSet(out)：缓存点云的点坐标。
返回：
	点集中点的数量。
**************************************************************/
int QtRegAlgorithm::fromMesh2PtSet(vtkSmartPointer<vtkPolyData>pvtkMesh, double** ppdPtSet)
{
	int nPtNum = pvtkMesh->GetNumberOfPoints();
	double* pdPtSet = *ppdPtSet;

	if (NULL != pdPtSet)
	{
		delete[] pdPtSet;
		pdPtSet = NULL;
	}

	pdPtSet = new double[3* nPtNum*sizeof(double)];

	for (int i = 0; i < nPtNum; i++)
	{
		double* pdPtCoord;
		pdPtCoord = pvtkMesh->GetPoint(i);
		pdPtSet[i*3] = pdPtCoord[0];
		pdPtSet[i*3+1] = pdPtCoord[1];
		pdPtSet[i*3+2] = pdPtCoord[2];
	}

	*ppdPtSet = pdPtSet;

	return nPtNum;
}

/************************************************************
说明：
	用pdPtSet指向的点集构造vtkPolyData对象，对象仅仅含顶点。
参数：
	pdPtSet：指向输入的点集；
	nPtNum：点集中点数；
	pvtkMesh(out)：vtk的顶点对象。
返回：
	无。
**************************************************************/
void QtRegAlgorithm::fromPtSet2Vertex(double* pdPtSet, int nPtNum, vtkSmartPointer<vtkPolyData>pvtkMesh)
{
	vtkSmartPointer<vtkCellArray> pvtkVertices = vtkSmartPointer<vtkCellArray>::New();
	vtkSmartPointer<vtkPoints> pvtkPoints = vtkSmartPointer<vtkPoints>::New();

	pvtkMesh->Initialize();

	for (int i = 0; i < nPtNum; i++)
	{
		pvtkPoints->InsertPoint(i, pdPtSet[3 * i], pdPtSet[3 * i + 1], pdPtSet[3 * i + 2]);
		pvtkVertices->InsertNextCell(1);
		pvtkVertices->InsertCellPoint(i);
	}

	pvtkMesh->SetPoints(pvtkPoints);
	pvtkMesh->SetVerts(pvtkVertices);
}

/************************************************************
说明：
	用pdPtSet指向的点集构造vtkPolyData对象，对象仅仅含顶点。
参数：
	pdPtSet：指向输入的点集；
	nPtNum：点集中点数；
	pvtkMesh(out)：vtk的顶点对象。
返回：
	无。
**************************************************************/
void QtRegAlgorithm::fromPtSet2VertexByTrans(double* pdPtSet, int nPtNum, vtkSmartPointer<vtkPolyData>pvtkMesh, double matTrans[4][4])
{
	double dPt[3];

	vtkSmartPointer<vtkCellArray> pvtkVertices = vtkSmartPointer<vtkCellArray>::New();
	vtkSmartPointer<vtkPoints> pvtkPoints = vtkSmartPointer<vtkPoints>::New();

	pvtkMesh->Initialize();

	for (int i = 0; i < nPtNum; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			dPt[j] = 0.0;
			for (int k = 0; k < 3; k++)
			{
				dPt[j] += matTrans[j][k] * pdPtSet[3 * i + k];
			}
			dPt[j] += matTrans[j][3];
		}
		pvtkPoints->InsertPoint(i, dPt[0], dPt[1], dPt[2]);
		pvtkVertices->InsertNextCell(1);
		pvtkVertices->InsertCellPoint(i);
	}

	pvtkMesh->SetPoints(pvtkPoints);
	pvtkMesh->SetVerts(pvtkVertices);
}

/************************************************************
说明：
	将vtkPolyData对象中的点拷贝到std::vector对象中。
参数：
	pvtkMesh(in)：mesh对象，可能含独立的顶点数据；
	vecPtSet(out)：缓存点云的点坐标。
返回：
	构造对象所含的顶点数量。
**************************************************************/
int QtRegAlgorithm::fromMesh2Vector(vtkSmartPointer<vtkPolyData>pvtkMesh, std::vector<dPoint_Hxmx>& vecPtSet)
{
	int nPtNum = pvtkMesh->GetNumberOfPoints();
	struct dPoint_Hxmx  stPt;

	vecPtSet.clear();
	
	for (int i = 0; i < nPtNum; i++)
	{
		double* pdPtCoord;
		pdPtCoord = pvtkMesh->GetPoint(i);
		stPt.dPt[0] = pdPtCoord[0];
		stPt.dPt[1] = pdPtCoord[1];
		stPt.dPt[2] = pdPtCoord[2];
		vecPtSet.push_back(stPt);
	}

	return nPtNum;
}

/************************************************************************
说明：
	读取txt文件中的特征点坐标。本方法仅仅支持读取5个特征点，多余点被丢弃。
参数：
	qstrFileName：文件名；
	pdPtsSet (out)： 读回的点坐标；
返回：
	读出的浮点数个数。
*************************************************************************/
int QtRegAlgorithm::readFeaturePts(QString qstrFileName, double pdPtsSet[FEATURE_PT_NUM][3])
{
	int nPos, nC = 0, nCLn = 0;

	if (qstrFileName.isEmpty())
	{
		return -1;
	}

	QFile  fileFeature(qstrFileName);

	if (fileFeature.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		QString     qstrLine, qstrData;
		QTextStream inFeature(&fileFeature);         //用文件构造流

		qstrLine = inFeature.readLine();                //读取一行放到字符串里
		while (!qstrLine.isNull())                      //字符串有内容
		{
			//decode
			nPos = qstrLine.indexOf(",");
			while (nPos > 0)
			{
				qstrData = qstrLine.left(nPos);
				qstrLine = qstrLine.right(qstrLine.length() - nPos - 1);
				pdPtsSet[nCLn][nC] = qstrData.toDouble();
				nC++;

				if (nC == 2)
				{
					//qstrData = qstrLine.right(qstrLine.length() - nPos - 1);
					pdPtsSet[nCLn][nC] = qstrLine.toDouble();
					nPos = -1;
				}
				else
				{
					nPos = qstrLine.indexOf(",");
				}
			}

			qstrLine = inFeature.readLine();
			nC = 0;
			nCLn++;

			if (nCLn > 4)
			{
				break;
			}
		}

		fileFeature.close();
	}
	else
	{
		return -2;
	}

	return 3 * nCLn;
}

/************************************************************************
说明：
	找到给定路径下所有的目录，并插入到QtComboBox里。
参数：
	qstrPathName：给定的路径名。
返回：
	查找到的目录个数。
*************************************************************************/
int QtRegAlgorithm::findAllDirectory(QString qstrPathName)
{
	QDir dir(qstrPathName);

	dir.setFilter(QDir::Dirs);

	foreach(QFileInfo fullDir, dir.entryInfoList())
	{

		if (fullDir.fileName() == "." || fullDir.fileName() == "..")
		{
			continue;
		}

		//QString fileName = fullDir.absoluteFilePath();
		ui.cbBoxSubjectName->addItem(fullDir.fileName());
	}

	return 0;
}

/************************************************************
说明：
	利用ComputeOBB求出的最小包围盒数据，构造vtk盒子对象。
参数：
	dCorner(in)：角点；
	dMax(in)：长轴的长度和方向；
	dMid(in)：中长轴的长度和方向；
	dMin(in)：短轴的长度和方向；
	pvtkPolysBox(out)：输出盒子的mesh。
返回：
	无。
**************************************************************/
void QtRegAlgorithm::CreateVtkBox(vtkSmartPointer<vtkPolyData> pvtkPolysBox,
	double dCorner[3], double dMax[3], double dMid[3], double dMin[3])
{
	double dX[3], dY[3], dZ[3], dT[3], dLen;
	//Long axis as x
	dLen = sqrt(dMax[0] * dMax[0] + dMax[1] * dMax[1] + dMax[2] * dMax[2]);
	dX[0] = dMax[0] / dLen, dX[1] = dMax[1] / dLen, dX[2] = dMax[2] / dLen;

	dLen = sqrt(dMid[0] * dMid[0] + dMid[1] * dMid[1] + dMid[2] * dMid[2]);
	dY[0] = dMid[0] / dLen, dY[1] = dMid[1] / dLen, dY[2] = dMid[2] / dLen;
	//Short axis as z
	dLen = sqrt(dMin[0] * dMin[0] + dMin[1] * dMin[1] + dMin[2] * dMin[2]);
	dZ[0] = dMin[0] / dLen, dZ[1] = dMin[1] / dLen, dZ[2] = dMin[2] / dLen;

	dT[0] = dX[1] * dY[2] - dX[2] * dY[1];
	dT[1] = dX[2] * dY[0] - dX[0] * dY[2];
	dT[2] = dX[0] * dY[1] - dX[1] * dY[0];

	dLen = (dT[0] - dZ[0]) * (dT[0] - dZ[0])
		+ (dT[1] - dZ[1]) * (dT[1] - dZ[1])
		+ (dT[2] - dZ[2]) * (dT[2] - dZ[2]);

	double v1[3], v2[3], v3[3], v4[3], v5[3], v6[3], v7[3], v8[3];

	for (int i = 0; i < 3; i++)
	{
		v1[i] = dCorner[i];
		v2[i] = dCorner[i] + dMax[i];
		v3[i] = dCorner[i] + dMax[i] + dMid[i];
		v4[i] = dCorner[i] + dMid[i];

		v5[i] = dCorner[i] + dMin[i];
		v6[i] = dCorner[i] + dMax[i] + dMin[i];
		v7[i] = dCorner[i] + dMax[i] + dMid[i] + dMin[i];
		v8[i] = dCorner[i] + dMid[i] + dMin[i];
	}

	pvtkPolysBox->Initialize();

	auto pvtkPtset = vtkSmartPointer<vtkPoints>::New();
	auto pvtkTriangle = vtkSmartPointer<vtkTriangle>::New();
	auto pvtkTriangleArr = vtkSmartPointer<vtkCellArray>::New();
	pvtkPtset->InsertNextPoint(v1);
	pvtkPtset->InsertNextPoint(v2);
	pvtkPtset->InsertNextPoint(v3);
	pvtkPtset->InsertNextPoint(v4);
	pvtkPtset->InsertNextPoint(v5);
	pvtkPtset->InsertNextPoint(v6);
	pvtkPtset->InsertNextPoint(v7);
	pvtkPtset->InsertNextPoint(v8);

	if (dLen < 1.0E-5)    //3轴符合右手系
	{
		//1st triangle
		pvtkTriangle->GetPointIds()->SetId(0, 0);
		pvtkTriangle->GetPointIds()->SetId(1, 2);
		pvtkTriangle->GetPointIds()->SetId(2, 1);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//2dn
		pvtkTriangle->GetPointIds()->SetId(0, 0);
		pvtkTriangle->GetPointIds()->SetId(1, 3);
		pvtkTriangle->GetPointIds()->SetId(2, 2);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//3rd
		pvtkTriangle->GetPointIds()->SetId(0, 0);
		pvtkTriangle->GetPointIds()->SetId(1, 1);
		pvtkTriangle->GetPointIds()->SetId(2, 5);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//4th
		pvtkTriangle->GetPointIds()->SetId(0, 0);
		pvtkTriangle->GetPointIds()->SetId(1, 5);
		pvtkTriangle->GetPointIds()->SetId(2, 4);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//5th
		pvtkTriangle->GetPointIds()->SetId(0, 1);
		pvtkTriangle->GetPointIds()->SetId(1, 2);
		pvtkTriangle->GetPointIds()->SetId(2, 6);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//6th
		pvtkTriangle->GetPointIds()->SetId(0, 1);
		pvtkTriangle->GetPointIds()->SetId(1, 6);
		pvtkTriangle->GetPointIds()->SetId(2, 5);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//7th
		pvtkTriangle->GetPointIds()->SetId(0, 2);
		pvtkTriangle->GetPointIds()->SetId(1, 3);
		pvtkTriangle->GetPointIds()->SetId(2, 7);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//8th
		pvtkTriangle->GetPointIds()->SetId(0, 2);
		pvtkTriangle->GetPointIds()->SetId(1, 7);
		pvtkTriangle->GetPointIds()->SetId(2, 6);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//9th
		pvtkTriangle->GetPointIds()->SetId(0, 0);
		pvtkTriangle->GetPointIds()->SetId(1, 7);
		pvtkTriangle->GetPointIds()->SetId(2, 3);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//10th
		pvtkTriangle->GetPointIds()->SetId(0, 0);
		pvtkTriangle->GetPointIds()->SetId(1, 4);
		pvtkTriangle->GetPointIds()->SetId(2, 7);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//11th
		pvtkTriangle->GetPointIds()->SetId(0, 4);
		pvtkTriangle->GetPointIds()->SetId(1, 5);
		pvtkTriangle->GetPointIds()->SetId(2, 6);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//12th
		pvtkTriangle->GetPointIds()->SetId(0, 4);
		pvtkTriangle->GetPointIds()->SetId(1, 6);
		pvtkTriangle->GetPointIds()->SetId(2, 7);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
	}
	else                  //xy轴交换，符合右手系
	{
		//1st triangle
		pvtkTriangle->GetPointIds()->SetId(0, 0);
		pvtkTriangle->GetPointIds()->SetId(1, 1);
		pvtkTriangle->GetPointIds()->SetId(2, 2);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//2dn
		pvtkTriangle->GetPointIds()->SetId(0, 0);
		pvtkTriangle->GetPointIds()->SetId(1, 2);
		pvtkTriangle->GetPointIds()->SetId(2, 3);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//3rd
		pvtkTriangle->GetPointIds()->SetId(0, 0);
		pvtkTriangle->GetPointIds()->SetId(1, 5);
		pvtkTriangle->GetPointIds()->SetId(2, 1);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//4th
		pvtkTriangle->GetPointIds()->SetId(0, 0);
		pvtkTriangle->GetPointIds()->SetId(1, 4);
		pvtkTriangle->GetPointIds()->SetId(2, 5);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//5th
		pvtkTriangle->GetPointIds()->SetId(0, 1);
		pvtkTriangle->GetPointIds()->SetId(1, 6);
		pvtkTriangle->GetPointIds()->SetId(2, 2);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//6th
		pvtkTriangle->GetPointIds()->SetId(0, 1);
		pvtkTriangle->GetPointIds()->SetId(1, 5);
		pvtkTriangle->GetPointIds()->SetId(2, 6);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//7th
		pvtkTriangle->GetPointIds()->SetId(0, 2);
		pvtkTriangle->GetPointIds()->SetId(1, 7);
		pvtkTriangle->GetPointIds()->SetId(2, 3);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//8th
		pvtkTriangle->GetPointIds()->SetId(0, 2);
		pvtkTriangle->GetPointIds()->SetId(1, 6);
		pvtkTriangle->GetPointIds()->SetId(2, 7);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//9th
		pvtkTriangle->GetPointIds()->SetId(0, 0);
		pvtkTriangle->GetPointIds()->SetId(1, 3);
		pvtkTriangle->GetPointIds()->SetId(2, 7);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//10th
		pvtkTriangle->GetPointIds()->SetId(0, 0);
		pvtkTriangle->GetPointIds()->SetId(1, 7);
		pvtkTriangle->GetPointIds()->SetId(2, 4);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//11th
		pvtkTriangle->GetPointIds()->SetId(0, 4);
		pvtkTriangle->GetPointIds()->SetId(1, 6);
		pvtkTriangle->GetPointIds()->SetId(2, 5);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
		//12th
		pvtkTriangle->GetPointIds()->SetId(0, 4);
		pvtkTriangle->GetPointIds()->SetId(1, 7);
		pvtkTriangle->GetPointIds()->SetId(2, 6);
		pvtkTriangleArr->InsertNextCell(pvtkTriangle);
	}

	pvtkPolysBox->SetPoints(pvtkPtset);
	pvtkPolysBox->SetPolys(pvtkTriangleArr);
}

/**********************************************************************************
说明：
	利用类成员m_dMatTrans变换点集坐标，并且清除pvtkPolysVert中点重新生成顶点对象。
参数：
	ppdPtSet(in,out)：    指针的指针指向点集的坐标(x1,y1,z1,x2,y2,z2,......xn,yn,zn), 
	                      n=nPtNum；
	nPtNum：              点集中点的数量；
	pvtkPolysVert(out)：  vtk的顶点对象（没有mesh）。
返回：
	无。
***********************************************************************************/
void QtRegAlgorithm::TransfPointSet(double** ppdPtSet, int nPtNum, vtkSmartPointer<vtkPolyData> pvtkPolysVert)
{
	double* pdPtSet = *ppdPtSet;
	double  x, y, z;

	for (int i = 0; i < nPtNum; i++)
	{
		x = pdPtSet[3 * i] * m_dMatTrans[0][0]
			+ pdPtSet[3 * i + 1] * m_dMatTrans[0][1]
			+ pdPtSet[3 * i + 2] * m_dMatTrans[0][2]
			+ m_dMatTrans[0][3];
		y = pdPtSet[3 * i] * m_dMatTrans[1][0]
			+ pdPtSet[3 * i + 1] * m_dMatTrans[1][1]
			+ pdPtSet[3 * i + 2] * m_dMatTrans[1][2]
			+ m_dMatTrans[1][3];
		z = pdPtSet[3 * i] * m_dMatTrans[2][0]
			+ pdPtSet[3 * i + 1] * m_dMatTrans[2][1]
			+ pdPtSet[3 * i + 2] * m_dMatTrans[2][2]
			+ m_dMatTrans[2][3];

		pdPtSet[3 * i] = x;
		pdPtSet[3 * i + 1] = y;
		pdPtSet[3 * i + 2] = z;
	}

	fromPtSet2Vertex(pdPtSet, nPtNum, pvtkPolysVert);
}