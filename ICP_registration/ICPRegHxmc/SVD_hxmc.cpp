
#include "pch.h"
#include "matrix_hxmc.h"
#include <cmath>
#include <map>

using namespace std;

const double THRESHOLD = 1E-8;
const int ITERATION = 30;   //��������������

/***********************************************
˵����
    ���ź��������ڵ���0��������1��С��0�򷵻�-1.
������
    number����������
���أ�
    �������ķ��ŷ���1��-1.
************************************************/
inline int sign(double number)
{
    if (number < 0)
    {
        return -1;
    }
    else
    {
        return 1;
    }
}

/***********************************************
˵����
    Ӧ���ڶԳƷ����i,j���е�Jacobi��ת�任.
������
    matrix(in, out)������ת�任�ľ���
    i,j��            ���任���к��У�
    pass(in,out)��   �ǶԽ�Ϊ0������Ϊtrue��
    J(out)��         �ۻ�����ת����
���أ�
    �������ķ��ŷ���1��-1.
************************************************/
void rotate(Matrix < double >& matrix, int i, int j, bool& pass, Matrix < double >& J)
{
    double ele = matrix.get(i, j);

    if (fabs(ele) < THRESHOLD) 
    {
        return;
    }

    pass = false;

    double ele1 = matrix.get(i, i);
    double ele2 = matrix.get(j, j);
    int    size = matrix.getRows();
    double tao = (ele1 - ele2) / (2 * ele);
    double tan = sign(tao) / (fabs(tao) + sqrt(1 + pow(tao, 2)));
    double cos = 1 / sqrt(1 + pow(tan, 2));
    double sin = cos * tan;

    Matrix < double >G(IdentityMatrix < double >(size, size));
    G.put(i, i, cos);
    G.put(i, j, -1 * sin);
    G.put(j, i, sin);
    G.put(j, j, cos);
    matrix = G.getTranspose() * matrix * G;
    J *= G;
}

/*************************************************************
˵����
    ˫��Jacobi��ԳƷ����������������ʸ������
������
    matrix(in, out)�����任�ľ���
    size��           �ԳƷ����ά�ȣ�
    E(out)��         ��������ɵ�ʸ����
    J(out)��         ����ʸ����ɵķ���
���أ�
    ��.
**************************************************************/
void jacobi(Matrix < double >& matrix, int size, vector < double >& E, Matrix < double >& J)
{

    int iteration = ITERATION;

    while (iteration-- > 0) 
    {
        bool pass = true;
        for (int i = 0; i < size; ++i) 
        {
            for (int j = i + 1; j < size; ++j) 
            {
                rotate(matrix, i, j, pass, J);
            }
        }
        if (pass)   //���ǶԽ�Ԫ��ȫ����Ϊ0ʱ�����˳�
        {
            break;
        }
    }

    cout << "����������" << ITERATION - iteration << endl;

    for (int i = 0; i < size; ++i) 
    {
        E[i] = matrix.get(i, i);

        if (E[i] < THRESHOLD)
        {
            E[i] = 0.0;
        }
    }
}

/*************************************************************
˵����
    ˫��Jacobi�����������(MxN����M<=N)������ֵ�ֽ�SVD��
    ע�⣺�������������С�ڵ���������
������
    A(in, out)�����任�ľ���
    U(out)��    �пռ����������
    V(out)��    �пռ����������
    E(out)��    �Խ�СԪ��ʸ����
���أ�
    ��.
**************************************************************/
void svd(Matrix < double >& A, Matrix < double >& U, Matrix < double >& V, vector < double >& E)
{
    int rows = A.getRows();
    int columns = A.getColumns();

    assert(rows <= columns);
    assert(U.getRows() == rows);
    assert(U.getColumns() == rows);
    assert(V.getRows() == columns);
    assert(V.getColumns() == columns);
    assert(E.size() == columns);

    Matrix < double >B = A.getTranspose() * A;                        //A��ת�ó���A���õ�һ���Գƾ���B
    Matrix < double >J(IdentityMatrix < double >(columns, columns));
    vector < double >S(columns);

    jacobi(B, columns, S, J);  //��B������ֵ����������
    for (int i = 0; i < S.size(); ++i)
    {
        S[i] = sqrt(S[i]);   //B������ֵ������õ�A������ֵ
    }
        
    /*����ֵ���ݼ����򣬶�Ӧ��V�е���������ҲҪ������ */
    multimap < double, int >eigen;

    for (int i = 0; i < S.size(); ++i)   //��multimap�ڲ��Զ���key��������
    {
        eigen.insert(make_pair(S[i], i));
    }
        
    multimap < double, int >::const_iterator iter = --eigen.end();

    int num_eig = 0;    //��¼��0����ֵ�ĸ���
    for (int i = 0; i < columns; ++i, iter--)  //�������multimap,ʹ����ֵ�Ӵ�С����
    {  
        int index = iter->second;
        E[i] = S[index];
        if (E[i] > THRESHOLD) 
        {
            num_eig++;
        }
        for (int row = 0; row < columns; ++row)
        {
            V.put(row, i, J.get(row, index));
        }  
    }

    assert(num_eig <= rows);

    for (int i = 0; i < num_eig; ++i) 
    {
        Matrix < double > vi = V.getColumn(i); //��ȡV�ĵ�i��
        double sigma = E[i];
        Matrix < double > ui(rows, 1);

        ui = A * vi;
        for (int j = 0; j < rows; ++j) 
        {
            U.put(j, i, ui.get(j, 0) / sigma);
        }
    }
    //U����ĺ�(rows-none_zero)�оͲ������ˣ�����Ĭ��ֵ0����Ϊ����ж�Ӧ������ֵΪ0,��������ѹ��ʱ�ò����⼸��   
}

/*************************************************************
˵����
    ����Jacobi�����������(MxN����M<=N)������ֵ�ֽ�SVD��
    ע�⣺�������������С�ڵ���������
������
    A(in, out)�����任�ľ���
    U(out)��    �пռ����������
    V(out)��    �пռ����������
    E(out)��    �Խ�СԪ��ʸ����
���أ�
    ��.
**************************************************************/
void orthogonal(Matrix < double >& matrix, int i, int j, bool& pass, Matrix < double >& V)
{
    assert(i < j);

    Matrix<double> Ci = matrix.getColumn(i);
    Matrix<double> Cj = matrix.getColumn(j);

    double ele = ((Ci.getTranspose()) * Cj).get(0, 0);

    if (fabs(ele) < THRESHOLD)          //i,j�����Ѿ�����
    {
        return;
    }
        
    int rows = matrix.getRows();
    int columns = matrix.getColumns();

    pass = false;
    double ele1 = ((Ci.getTranspose()) * Ci).get(0, 0);
    double ele2 = ((Cj.getTranspose()) * Cj).get(0, 0);

    /*ֻҪÿ����תǰ���ѷ�������з���ǰ�棬�Ϳ��Ա�֤���������ֵ�ǵݼ������*/
    if (ele1 < ele2)            //���matrix��i�еķ���С�ڵ�j�У��򽻻�����.ͬʱV����Ҳ����Ӧ�ĵ���
    {           
        for (int row = 0; row < rows; ++row) 
        {
            matrix.put(row, i, Cj.get(row, 0));
            matrix.put(row, j, Ci.get(row, 0));
        }
        for (int row = 0; row < columns; ++row) 
        {
            double tmp = V.get(row, i);
            V.put(row, i, V.get(row, j));
            V.put(row, j, tmp);
        }
    }

    double tao = (ele1 - ele2) / (2 * ele);
    double tan = sign(tao) / (fabs(tao) + sqrt(1 + pow(tao, 2)));
    double cos = 1 / sqrt(1 + pow(tan, 2));
    double sin = cos * tan;

    for (int row = 0; row < rows; ++row) 
    {
        double var1 = matrix.get(row, i) * cos + matrix.get(row, j) * sin;
        double var2 = matrix.get(row, j) * cos - matrix.get(row, i) * sin;
        matrix.put(row, i, var1);
        matrix.put(row, j, var2);
    }

    for (int col = 0; col < columns; ++col) 
    {
        double var1 = V.get(col, i) * cos + V.get(col, j) * sin;
        double var2 = V.get(col, j) * cos - V.get(col, i) * sin;
        V.put(col, i, var1);
        V.put(col, j, var2);
    }
}

/*************************************************************
˵����
    ˫��Jacobi�����������(MxN����M<=N)������ֵ�ֽ�SVD��
    ע�⣺�������������С�ڵ���������
������
    A(in, out)�����任�ľ���
    U(out)��    �пռ����������
    V(out)��    �пռ����������
    E(out)��    �Խ�СԪ��ʸ����
���أ�
    ��.
**************************************************************/
void hestens_jacobi(Matrix < double >& matrix, Matrix < double >& V)
{
    int rows = matrix.getRows();
    int columns = matrix.getColumns();

    int iteration = ITERATION;

    while (iteration-- > 0) 
    {
        bool pass = true;
        for (int i = 0; i < columns; ++i) 
        {
            for (int j = i + 1; j < columns; ++j) 
            {
                orthogonal(matrix, i, j, pass, V);      //������εĵ���������V���������
            }
        }
        if (pass)   //���������ж�����ʱ�˳�����
        {
            break;
        }
    }
    cout << "����������" << ITERATION - iteration << endl;
}

/*************************************************************
˵����
    ˫��Jacobi�����������(MxN����M<=N)������ֵ�ֽ�SVD��
    ע�⣺�������������С�ڵ���������
������
    matrix(in, out)�����任�ľ���
    U(out)��    �пռ����������
    V(out)��    �пռ����������
    S(out)��    �ԽǷ���
���أ�
    ��.
**************************************************************/
int svn(Matrix < double >& matrix, Matrix < double >& S, Matrix < double >& U, Matrix < double >& V)
{
    int rows = matrix.getRows();
    int columns = matrix.getColumns();

    assert(rows <= columns);

    hestens_jacobi(matrix, V);

    vector<double> E(columns);        //E�д������ֵ
    int none_zero = 0;                //��¼��0����ֵ�ĸ���

    for (int i = 0; i < columns; ++i) 
    {
        double norm = sqrt((matrix.getColumn(i).getTranspose() * (matrix.getColumn(i))).get(0, 0));

        if (norm > THRESHOLD) 
        {
            none_zero++;
        }

        E[i] = norm;
    }

    /**
    * U����ĺ�(rows-none_zero)���Լ�V�ĺ�(columns-none_zero)�оͲ������ˣ�����Ĭ��ֵ0��
    * ��������ֵ�ֽ�A=U*Sigma*V^T������ֻ��ҪU��ǰr�У�V^T��ǰr�У���V��ǰr�У����Ϳ��Իָ�A�ˡ�r��A����
    */
    for (int row = 0; row < rows; ++row)
    {
        S.put(row, row, E[row]);

        for (int col = 0; col < none_zero; ++col) 
        {
            U.put(row, col, matrix.get(row, col) / E[col]);
        }
    }

    return none_zero;   //������ֵ�ĸ����༴�������
}