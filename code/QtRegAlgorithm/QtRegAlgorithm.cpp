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

	//���� "ICP1" ��ťʱ���ã����ڵ���ICP�㷨
	connect(
		ui.btnICPReg,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnICPRegistClicked(bool)));

	//���� "5F+ICP" ��ťʱ���ã����ڵ���5Feature + ICP�㷨
	connect(
		ui.btn5FPlusICP,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnReg5FPlusICPClicked(bool)));

	//���� "ICP2" ��ťʱ���ã����ڵ���knn�㷨
	connect(
		ui.btnICP2,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnICPRegICP2Clicked(bool)));

	//���� "5Feats" ��ťʱ���ã����ڵ���5��������׼�㷨
	connect(
		ui.btnFiveFeature,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnFiveFeatureClicked(bool)));

	//���� "OpenCam" ��ťʱ���ã����ڴ������5���������겢����
	connect(
		ui.btnOpenCam5Feat,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnOpenCam5FeaturesClicked(bool)));

	//���� "OpenMri" ��ťʱ���ã����ڴ�MRI��5���������겢���� btnBoxFilter
	connect(
		ui.btnOpenMri5Feat,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnOpenMri5FeaturesClicked(bool)));

	//���� "BoxFilter" ��ťʱ���ã����˳�5�������Χ��Դ��������ĵ㼯 
	connect(
		ui.btnBoxFilter,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnBoxFilterClicked(bool)));

	//���� "TransFlt" ��ťʱ���ã����˲��ĵ㼯�����Աm_dMatTrans�任��õ��µ㼯�����vtk�������
	connect(
		ui.btnTransFilter,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnTransFilterClicked(bool)));

	//���� "AlgoTest" ��ťʱ���ã����ڲ����㷨
	connect(
		ui.btnTestAlgo,
		SIGNAL(clicked(bool)),
		this,
		SLOT(OnAlgorithmTestClicked(bool)));

	//ѡ��Source�ġ�Show����ѡ��ȷ���Ƿ���ʾԴ�����Դ����
	connect(
		ui.chkSrcShow,
		SIGNAL(stateChanged(int)),
		this,
		SLOT(OnIfShowSource(int)));

	//ѡ��Source�ġ�Mesh����ѡ��ȷ������ʾԴmesh���壬������ʾ����
	connect(
		ui.chkSrcMesh,
		SIGNAL(stateChanged(int)),
		this,
		SLOT(OnIfDisplaySrcMesh(int)));

	//ѡ��Filtered�ġ�Show����ѡ��ȷ���Ƿ���ʾԴ�����˲����任���mesh�������ص���
	connect(
		ui.chkFltShow,
		SIGNAL(stateChanged(int)),
		this,
		SLOT(OnIfShowFilter(int)));

	//ѡ��Filtered�ġ�Mesh����ѡ��ȷ������ʾ�˲����任���Դmesh���壬������ʾ��ص���
	connect(
		ui.chkFltMesh,
		SIGNAL(stateChanged(int)),
		this,
		SLOT(OnIfDisplayFilterMesh(int)));

	//ѡ��Target�ġ�Show����ѡ��ȷ���Ƿ���ʾĿ��mesh��������
	connect(
		ui.chkTarShow,
		SIGNAL(stateChanged(int)),
		this,
		SLOT(OnIfShowTarget(int)));

	//ѡ��Target�ġ�Mesh����ѡ��ȷ������ʾĿ��mesh���壬������ʾ����
	connect(
		ui.chkTarMesh,
		SIGNAL(stateChanged(int)),
		this,
		SLOT(OnIfDisplayTarMesh(int)));

	//ѡ��OBBox�ġ�Show����ѡ��ȷ���Ƿ���ʾOBB
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
˵����
	�����ڳߴ緢���仯ʱ���ø÷�����Ȼ�󣬸��ݴ��ڴ�С����vtk���ڴ�С��
	ͬʱ���ü����ؼ������ԡ�
���룺
	event�����ڳߴ�任�¼���ָ�롣
���أ�
	�ޡ�
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
˵����
	�����Source File Operator��������OpenMesh������ģ��
	mesh�ļ�����ʾ��ģ�ʹ����ͬʱ��ȡ�м�ĵ��Ƶ����Ա����
	�֧��ply��obj��stl��ʽ��
���룺
	checked�����������ã���
���أ�
	��ȷ���ļ�����0�����򷵻ظ�����
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
		QMessageBox::warning(this, "Warning", "��֧�ֵ�mesh�ļ���ʽ��");
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
˵����
	�����Source File Operator��������SaveMesh����ť,���澭��
	�˲��ȴ�����mesh���ݣ�֧��obj��stl��ply�����ļ���ʽ��
���룺
	checked�����������ã���
���أ�
	����ɹ�����0�����򷵻�һ��������
******************************************************************/
int QtRegAlgorithm::OnSaveSrcMeshClicked(bool checked)
{
	if (m_pvtkPolysSrc->GetPoints() == NULL)
	{
		QMessageBox::warning(this, "Warning", "û�б���������壡");
		return -1;
	}

	//��ȡ����·��
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
		QMessageBox::warning(this, "Warning", "mesh�ļ���ʽ����");
		return -2;
	}

	return 0;
}

/*****************************************************************
˵����
	�����Target File Operator����OpenMesh��,��ģ��mesh�ļ���
	��ʾ��ģ�ʹ����ͬʱ��ȡ�м�ĵ��Ƶ����Ա�����
	֧��ply��obj��stl��ʽ��
���룺
	checked�����������ã���
���أ�
	��ȷ���ļ�����0�����򷵻ظ�����
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
		QMessageBox::warning(this, "Warning", "��֧�ֵ�mesh�ļ���ʽ��");
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
˵����
	�����Target File Operator��������SaveMesh��,���澭��
	�˲��ȴ�����mesh���ݣ�֧��obj��stl��ply�����ļ���ʽ��
���룺
	checked�����������ã���
���أ�
	����ɹ�����0�����򷵻�һ��������
******************************************************************/
int QtRegAlgorithm::OnSaveTarMeshClicked(bool checked)
{
	if (m_pvtkPolysTar->GetPoints() == NULL)
	{
		QMessageBox::warning(this, "Warning", "No target mesh is opened!");
		return -1;
	}

	//��ȡ����·��
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
		QMessageBox::warning(this, "Warning", "mesh�ļ���ʽ����");
		return -2;
	}

	return 0;
}

/*****************************************************************
˵����
	���澭���˲��ȴ�����mesh���ݣ�֧��obj��stl��ply�����ļ���ʽ��
���룺
	checked�����������ã���
���أ�
	����ɹ�����0�����򷵻�һ��������
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

	//��ʾƽ������
	ui.lEditAveDist->setText(QString::number(dAveDist, 'f', 4));
	//��ʾ�任����
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
˵����
	�����ť��5Feat��,������׼�ؼ�m_objRegIcp����ط�������任
	���󡣲���ʾ�����ܷ��
���룺
	checked�����������ã���
���أ�
	����0��
******************************************************************/
int QtRegAlgorithm::OnReg5FPlusICPClicked(bool checked)
{
	double dMatrix[4][4];

	double dAveDist = m_objRegIcp.RegBy5FPlusICP(m_dSrcFea, m_dTarFea, FEAT_NUM_HX, m_pdSrcPtSet, m_nSrcPtNum, dMatrix);
	fromPtSet2VertexByTrans(m_pdSrcPtSet, m_nSrcPtNum, m_pvtkPolysFltVert, dMatrix);

	ui.lEditAveDist->setText(QString::number(dAveDist, 'f', 4));
	//��ʾ�任����
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
˵����
	�����ť��ICP2��,������׼�ؼ�m_objRegIcp����ط�������任
	���󡣲���ʾ�����ܷ��
���룺
	checked�����������ã���
���أ�
	����0��
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

	//��ʾ�任����
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
˵����
	�����ť��5Feat��,������׼�ؼ�m_objRegIcp����ط�������任
	���󡣲���ʾ�����ܷ��
���룺
	checked�����������ã���
���أ�
	����0��
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

	//��ʾ�任����
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
	//��ʾ���
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
˵����
	���¡�BoxFilter����ť��ִ����5���������С��Χ�ж�Դ�����˲���
	��OBB���ӳ�����г����Χ�Ŀ����������ڵĵ�ȥ�������������
	���˵���
���룺
	checked�����������ã���
���أ�
	����0��
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
˵����
	���¡�TransFlt����ť�������Աm_dMatTrans���˲��ĵ��ƽ���
	����任������ʾ�任���λ�á�
���룺
	checked�����������ã���
���أ�
	����0��
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
˵����
	���¡�AlgoTest����ť�����ڲ��Ժ͵��Ը����㷨��
���룺
	checked�����������ã���
���أ�
	����0��
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
	////��ʾ�任����
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
˵����
	�����OpenCam����ť����5�����������ļ�����ȡ5�����꣬
	�����ݵ���׼����m_objRegIcp�
���룺
	checked�����������ã���
���أ�
	��ȡ���ݵ�������Ӧ��Ϊ15��
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
		//��ʾ������
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
		QMessageBox::warning(this, "Warning", "��ȡ�������ļ�����");
		return -2;
	}

	m_objRegIcp.Set5FeaturePtsCoord(dFeatureCam, false);

	return nDataNum;
}

/*****************************************************************
˵����
	�����OpenMri����ť����5�������MRI�ļ�����ȡ5�����꣬
	�����ݵ���׼����m_objRegIcp�
���룺
	checked�����������ã���
���أ�
	��ȡ���ݵ�������Ӧ��Ϊ15��
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
		//��ʾ������
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
		QMessageBox::warning(this, "Warning", "��ȡ�������ļ�����");
		return -2;
	}

	m_objRegIcp.Set5FeaturePtsCoord(dFeatureMri, true);

	return nDataNum;
}

//CheckBoxes responses
/************************************************************
˵����
	ѡ��Source�ġ�Show����ѡ����ȷ���Ƿ���ʾԴmesh����ơ�
������
	nState����ѡ���״̬��
���أ�
	0��
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
˵����
	ѡ��Source�ġ�Mesh����ѡ����ȷ������ʾԴmesh��������ʾԴ���ơ�
������
	nState����ѡ���״̬��
���أ�
	0��
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
˵����
	����Display Control������鸴ѡ��ȷ����ʾ�Ķ���
������
	bSrcShow��   �Ƿ���ʾԴmesh��cloud��
	bSrcMesh��   ����ʾԴ��mesh��������ʾCloud��
	bFilterShow���Ƿ���ʾԴ�˲����任��mesh��cloud��
	bFltMesh��   ����ʾ�˲��任Դ��mesh��������ʾ��ӦCloud��
	bTarShow��   �Ƿ���ʾĿ��mesh��cloud��
	bTarMesh��   ����ʾĿ���mesh��������ʾCloud��
	bOBBShow��   �Ƿ���ʾ5���������С��Χ�С�
���أ�
	�ޡ�
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
˵����
	����ʾ��������ʾԴmesh��������ʾԴ���ơ�
������
	bMesh����ʾmesh����ơ�
���أ�
	�ޡ�
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
˵����
	����ʾ��������ʾĿ��mesh��������ʾĿ����ơ�
������
	bMesh����ʾmesh����ơ�
���أ�
	�ޡ�
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
˵����
	�Ƿ���ʾ5���������С��Χ�С�
������
	bShow���Ƿ���ʾ��Χ�С�
���أ�
	�ޡ�
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
˵����
	��vtkPolyData�����еĵ㣬����ɽ����������vtkPolyData����
������
	pvtkMesh(in)��mesh���󣬿��ܺ������Ķ������ݣ�
	pvtkVert(out)���������vtk���ơ�
���أ�
	������������Ķ���������
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
˵����
	��vtkPolyData�����еĵ㿽����ppdPtSetָ���ָ����ָ���
	�ڴ滺�����С�
������
	pvtkMesh(in)��mesh���󣬿��ܺ������Ķ������ݣ�
	ppdPtSet(out)��������Ƶĵ����ꡣ
���أ�
	�㼯�е��������
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
˵����
	��pdPtSetָ��ĵ㼯����vtkPolyData���󣬶�����������㡣
������
	pdPtSet��ָ������ĵ㼯��
	nPtNum���㼯�е�����
	pvtkMesh(out)��vtk�Ķ������
���أ�
	�ޡ�
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
˵����
	��pdPtSetָ��ĵ㼯����vtkPolyData���󣬶�����������㡣
������
	pdPtSet��ָ������ĵ㼯��
	nPtNum���㼯�е�����
	pvtkMesh(out)��vtk�Ķ������
���أ�
	�ޡ�
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
˵����
	��vtkPolyData�����еĵ㿽����std::vector�����С�
������
	pvtkMesh(in)��mesh���󣬿��ܺ������Ķ������ݣ�
	vecPtSet(out)��������Ƶĵ����ꡣ
���أ�
	������������Ķ���������
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
˵����
	��ȡtxt�ļ��е����������ꡣ����������֧�ֶ�ȡ5�������㣬����㱻������
������
	qstrFileName���ļ�����
	pdPtsSet (out)�� ���صĵ����ꣻ
���أ�
	�����ĸ�����������
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
		QTextStream inFeature(&fileFeature);         //���ļ�������

		qstrLine = inFeature.readLine();                //��ȡһ�зŵ��ַ�����
		while (!qstrLine.isNull())                      //�ַ���������
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
˵����
	�ҵ�����·�������е�Ŀ¼�������뵽QtComboBox�
������
	qstrPathName��������·������
���أ�
	���ҵ���Ŀ¼������
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
˵����
	����ComputeOBB�������С��Χ�����ݣ�����vtk���Ӷ���
������
	dCorner(in)���ǵ㣻
	dMax(in)������ĳ��Ⱥͷ���
	dMid(in)���г���ĳ��Ⱥͷ���
	dMin(in)������ĳ��Ⱥͷ���
	pvtkPolysBox(out)��������ӵ�mesh��
���أ�
	�ޡ�
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

	if (dLen < 1.0E-5)    //3���������ϵ
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
	else                  //xy�ύ������������ϵ
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
˵����
	�������Աm_dMatTrans�任�㼯���꣬�������pvtkPolysVert�е��������ɶ������
������
	ppdPtSet(in,out)��    ָ���ָ��ָ��㼯������(x1,y1,z1,x2,y2,z2,......xn,yn,zn), 
	                      n=nPtNum��
	nPtNum��              �㼯�е��������
	pvtkPolysVert(out)��  vtk�Ķ������û��mesh����
���أ�
	�ޡ�
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