#include "QtRegAlgorithm.h"
#include <QtWidgets/QApplication>
#include <vtkAutoInit.h>

VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);
VTK_MODULE_INIT(vtkRenderingFreeType);

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	QtRegAlgorithm w;
	w.show();
	w.setWindowState(Qt::WindowMaximized);
	return a.exec();
}
