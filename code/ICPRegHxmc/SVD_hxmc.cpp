
#include "pch.h"
#include "matrix_hxmc.h"
#include <cmath>
#include <map>

using namespace std;

const double THRESHOLD = 1E-8;
const int ITERATION = 30;   //迭代次数的上限

/***********************************************
说明：
    符号函数，大于等于0返回整数1，小于0则返回-1.
参数：
    number：浮点数。
返回：
    根据数的符号返回1或-1.
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
说明：
    应用于对称方阵的i,j行列的Jacobi旋转变换.
参数：
    matrix(in, out)：待旋转变换的矩阵；
    i,j：            待变换的行和列；
    pass(in,out)：   非对角为0否，输入为true；
    J(out)：         累积的旋转矩阵。
返回：
    根据数的符号返回1或-1.
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
说明：
    双边Jacobi求对称方阵的特征根与特征矢量方阵。
参数：
    matrix(in, out)：待变换的矩阵；
    size：           对称方阵的维度；
    E(out)：         特征根组成的矢量；
    J(out)：         特征矢量组成的方阵。
返回：
    无.
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
        if (pass)   //当非对角元素全部变为0时迭代退出
        {
            break;
        }
    }

    cout << "迭代次数：" << ITERATION - iteration << endl;

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
说明：
    双边Jacobi法求任意矩阵(MxN矩阵，M<=N)的奇异值分解SVD。
    注意：矩阵的行数必须小于等于列数。
参数：
    A(in, out)：待变换的矩阵；
    U(out)：    行空间的正交方阵；
    V(out)：    列空间的正交方阵；
    E(out)：    对角小元的矢量。
返回：
    无.
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

    Matrix < double >B = A.getTranspose() * A;                        //A的转置乘以A，得到一个对称矩阵B
    Matrix < double >J(IdentityMatrix < double >(columns, columns));
    vector < double >S(columns);

    jacobi(B, columns, S, J);  //求B的特征值和特征向量
    for (int i = 0; i < S.size(); ++i)
    {
        S[i] = sqrt(S[i]);   //B的特征值开方后得到A的奇异值
    }
        
    /*奇异值按递减排序，对应的V中的特征向量也要重排序 */
    multimap < double, int >eigen;

    for (int i = 0; i < S.size(); ++i)   //在multimap内部自动按key进行排序
    {
        eigen.insert(make_pair(S[i], i));
    }
        
    multimap < double, int >::const_iterator iter = --eigen.end();

    int num_eig = 0;    //记录非0奇异值的个数
    for (int i = 0; i < columns; ++i, iter--)  //反向遍历multimap,使奇异值从大到小排序
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
        Matrix < double > vi = V.getColumn(i); //获取V的第i列
        double sigma = E[i];
        Matrix < double > ui(rows, 1);

        ui = A * vi;
        for (int j = 0; j < rows; ++j) 
        {
            U.put(j, i, ui.get(j, 0) / sigma);
        }
    }
    //U矩阵的后(rows-none_zero)列就不计算了，采用默认值0。因为这后几列对应的奇异值为0,在做数据压缩时用不到这几列   
}

/*************************************************************
说明：
    单边Jacobi法求任意矩阵(MxN矩阵，M<=N)的奇异值分解SVD。
    注意：矩阵的行数必须小于等于列数。
参数：
    A(in, out)：待变换的矩阵；
    U(out)：    行空间的正交方阵；
    V(out)：    列空间的正交方阵；
    E(out)：    对角小元的矢量。
返回：
    无.
**************************************************************/
void orthogonal(Matrix < double >& matrix, int i, int j, bool& pass, Matrix < double >& V)
{
    assert(i < j);

    Matrix<double> Ci = matrix.getColumn(i);
    Matrix<double> Cj = matrix.getColumn(j);

    double ele = ((Ci.getTranspose()) * Cj).get(0, 0);

    if (fabs(ele) < THRESHOLD)          //i,j两列已经正交
    {
        return;
    }
        
    int rows = matrix.getRows();
    int columns = matrix.getColumns();

    pass = false;
    double ele1 = ((Ci.getTranspose()) * Ci).get(0, 0);
    double ele2 = ((Cj.getTranspose()) * Cj).get(0, 0);

    /*只要每次旋转前都把范数大的列放在前面，就可以保证求出的奇异值是递减排序的*/
    if (ele1 < ele2)            //如果matrix第i列的范数小于第j列，则交换两列.同时V矩阵也作相应的调换
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
说明：
    双边Jacobi法求任意矩阵(MxN矩阵，M<=N)的奇异值分解SVD。
    注意：矩阵的行数必须小于等于列数。
参数：
    A(in, out)：待变换的矩阵；
    U(out)：    行空间的正交方阵；
    V(out)：    列空间的正交方阵；
    E(out)：    对角小元的矢量。
返回：
    无.
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
                orthogonal(matrix, i, j, pass, V);      //经过多次的迭代正交后，V就求出来了
            }
        }
        if (pass)   //当任意两列都正交时退出迭代
        {
            break;
        }
    }
    cout << "迭代次数：" << ITERATION - iteration << endl;
}

/*************************************************************
说明：
    双边Jacobi法求任意矩阵(MxN矩阵，M<=N)的奇异值分解SVD。
    注意：矩阵的行数必须小于等于列数。
参数：
    matrix(in, out)：待变换的矩阵；
    U(out)：    行空间的正交方阵；
    V(out)：    列空间的正交方阵；
    S(out)：    对角方阵。
返回：
    无.
**************************************************************/
int svn(Matrix < double >& matrix, Matrix < double >& S, Matrix < double >& U, Matrix < double >& V)
{
    int rows = matrix.getRows();
    int columns = matrix.getColumns();

    assert(rows <= columns);

    hestens_jacobi(matrix, V);

    vector<double> E(columns);        //E中存放奇异值
    int none_zero = 0;                //记录非0奇异值的个数

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
    * U矩阵的后(rows-none_zero)列以及V的后(columns-none_zero)列就不计算了，采用默认值0。
    * 对于奇异值分解A=U*Sigma*V^T，我们只需要U的前r列，V^T的前r行（即V的前r列），就可以恢复A了。r是A的秩
    */
    for (int row = 0; row < rows; ++row)
    {
        S.put(row, row, E[row]);

        for (int col = 0; col < none_zero; ++col) 
        {
            U.put(row, col, matrix.get(row, col) / E[col]);
        }
    }

    return none_zero;   //非奇异值的个数亦即矩阵的秩
}