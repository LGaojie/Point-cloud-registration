#include "pch.h"
#include "MatrixHx.h"

//构造函数
/*********************************************************
函数说明：
    缺省构造函数，将成员数据置0，指针置为NULL。
参数:
    无。
返回值：
    无。
**********************************************************/
template<class Type> MatrixHx<Type>::MatrixHx()	: m_nRow(0), m_nCol(0)
{
    m_ppData = NULL;
    m_pData = NULL;
}

/*********************************************************
函数说明：
    构造函数。构造一个nRow*nCol的矩阵。数据列表的nRow
    个指针对应地指向每行的第一个数据。
参数:
    nRow：构造矩阵的行数；
    nCol：构造矩阵的列数。
返回值：
    无。
**********************************************************/
template<class Type> MatrixHx<Type>::MatrixHx(unsigned int nRow, unsigned int nCol)
    : m_nRow(nRow), m_nCol(nCol), m_ppData(NULL), m_pData(NULL)
{
    if (0 == m_nRow * m_nCol)
    {
        return;
    }
        
    m_ppData = new Type* [m_nRow];

    m_pData = new Type[m_nRow * m_nCol];

    for (unsigned int i = 0; i < nRow; i++)
    {
        m_ppData[i] = &m_pData[nCol * i];
    }
}

/*********************************************************
函数说明：
    拷贝构造函数。构造一个与rhs一样的矩阵。
参数:
    rhs：待拷贝的矩阵；
返回值：
    无。
**********************************************************/
template<class Type> MatrixHx<Type>::MatrixHx(const MatrixHx<Type>& rhs)
    : m_nRow(0), m_nCol(0), m_ppData(NULL), m_pData(NULL)
{
    unsigned int nTotal = rhs.m_nRow * rhs.m_nCol;

    if (0 == nTotal)
    {
        return;
    }
        

    m_nRow = rhs.m_nRow;
    m_nCol = rhs.m_nCol;

    m_pData = new Type[nTotal];
    m_ppData = new Type* [m_nRow];


    for (unsigned int i = 0; i < m_nRow; i++)
    {
        m_ppData[i] = &m_pData[m_nCol * i];

        for (unsigned int j = 0; j < m_nCol; j++)
        {
            m_pData[i * m_nCol + j] = rhs.m_pData[i * m_nCol + j];
        }
    }
}

//析构
/*********************************************************
函数说明：
    析构函数：释放两个指针指向内存，将行列数置为0。
参数:
    无。
返回值：
    无。
**********************************************************/
template<class Type> MatrixHx<Type>::~MatrixHx()
{
    if (m_ppData != NULL)
    {
        delete[] m_ppData;
        m_ppData = NULL;
    }

    if (m_pData != NULL)
    {
        delete[] m_pData;
        m_pData = NULL;
    }

    m_nRow = 0;
    m_nCol = 0;
}

//重载算符
/*********************************************************
函数说明：
    拷贝算符：重构this矩阵与rhs的数据相同。
参数:
    rhs：待拷贝的矩阵。
返回值：
    this指向的矩阵。
**********************************************************/
template<class Type> MatrixHx<Type>& MatrixHx<Type>::operator = (const MatrixHx<Type>& rhs)
{
    if ((m_nRow != rhs.m_nRow) || (m_nCol != rhs.m_nCol))
    {
        if (m_pData != NULL)
        {
            delete[] m_pData;
            m_pData = NULL;
        }

        if (m_ppData)
        {
            delete[] m_ppData;
            m_ppData = NULL;
        }

        m_nRow = 0;
        m_nCol = 0;

        if (rhs.m_nRow * rhs.m_nCol > 0)
        {
            m_nRow = rhs.m_nRow;
            m_nCol = rhs.m_nCol;
            m_ppData = new Type* [m_nRow];
            m_pData = new Type[m_nRow * m_nCol];
        }
    }

    for (unsigned int i = 0; i < m_nRow; i++)
    {
        m_ppData[i] = &m_pData[m_nCol * i];

        for (unsigned int j = 0; j < m_nCol; j++)
        {
            m_pData[i * m_nCol + j] = rhs.m_data[i * m_nCol + j];
        }
    }

    return *this;
}

/*********************************************************
函数说明：
    拷贝算符：重构this矩阵与rhs的数据相同。
参数:
    rhs：待拷贝的矩阵。
返回值：
    this指向的矩阵。
**********************************************************/
template<class Type> MatrixHx<Type> MatrixHx<Type>::operator * (const MatrixHx<Type>& B) const
{
    MatrixHx<Type> C;
    MUL(*this, B, C);
    return C;
}

/*********************************************************
函数说明：
    []运算符：获得指定行数据的指针。
参数:
    row：指定的行。
返回值：
    指定行数据的指针。
**********************************************************/
template<class Type> Type*& MatrixHx<Type>::operator[](unsigned int row)
{
    return m_ppData[row];
}

/*********************************************************
函数说明：
    []运算符：获得指定行数据的指针（const指针）。
参数:
    row：指定的行。
返回值：
    指定行数据的指针。
**********************************************************/
template<class Type> const Type* MatrixHx<Type>::operator[](unsigned int row) const
{
    return m_ppData[row];
}

//基本操作
/*********************************************************
函数说明：
    获取矩阵指定的行矢量（1*m_nCol矩阵）。
参数:
    row：指定的行。
返回值：
    行矢量矩阵。
**********************************************************/
template<class Type> MatrixHx<Type> MatrixHx<Type>::GetRow(unsigned int row) const
{
    MatrixHx<Type> vecRow(1, m_nCol);

    for (unsigned int i = 0; i < m_nCol; i++)
    {
        vecRow.m_ppData[0][i] = m_ppData[row][i];
    }

    return vecRow;
}

/*********************************************************
函数说明：
    获取矩阵指定的列矢量（m_nRow*1矩阵）。
参数:
    col：指定的列。
返回值：
    列矢量矩阵。
**********************************************************/
template<class Type> MatrixHx<Type> MatrixHx<Type>::GetColumn(unsigned int col) const
{
    MatrixHx<Type> vecCol(m_nRow, 1);
    for (unsigned int i = 0; i < m_nRow; i++)
    {
        vecCol.m_ppData[i][0] = m_ppData[i][col];
    }

    return vecCol;
}

/*********************************************************
函数说明：
    矩阵相乘：C = A*B。
参数:
    A：待乘矩阵；
    B：待乘矩阵；
    C：结果矩阵。
返回值：
    如果A矩阵的列数与B矩阵的行数不等返回false，否则返回true。
**********************************************************/
template<class Type>
bool MatrixHx<Type>::MUL(const MatrixHx<Type>& A, const MatrixHx<Type>& B, MatrixHx<Type>& C)
{
    if (A.m_nCol != B.m_nRow)
    {
        return false;
    }
        
    C.Reset(A.m_nRow, B.m_nCol);

    for (unsigned int i = 0; i < C.m_nRow; i++)
    {
        for (unsigned int j = 0; j < C.m_nCol; j++)
        {
            C.m_ppData[i][j] = A.m_ppData[i][0] * B.m_ppData[0][j];
            for (unsigned int k = 1; k < A.m_nCol; k++)
            {
                C.m_ppData[i][j] += A.m_ppData[i][k] * B.m_ppData[k][j];
            }
        }
    }

    return true;
}

/*********************************************************
函数说明：
    重置矩阵为指定的行数和列数。重置后数据全部清除（不确定值）。
参数:
    row：重置后的行数；
    col：重置后的列数；
返回值：
    无。
**********************************************************/
template<class Type> void MatrixHx<Type>::Reset(unsigned int row, unsigned int col)
{
    if ((m_nRow != row) || m_nCol != col)
    {
        if (m_ppData != NULL)
        {
            delete[] m_ppData;
            m_ppData = NULL;
        }

        if (m_pData != NULL)
        {
            delete[] m_pData;
            m_pData = NULL;
        }

        m_nRow = row;
        m_nCol = col;

        if (row * col > 0)
        {
            m_ppData = new Type* [m_nRow];
            m_pData = new Type[m_nRow * m_nCol];

            for (unsigned int i = 0; i < m_nRow; i++)
            {
                m_ppData[i] = &m_pData[m_nCol * i];
            }
        }
    }
}

/*********************************************************
函数说明：
    矩阵的点对点的乘法：C[i,j] = A[i,j] * B[i,j]。
参数:
    A：被乘的矩阵；
    B：除数矩阵；
    C：结果矩阵（行数和列数与A,B相同）。
返回值：
    当A,B矩阵的行列数不一样时，返回false，一般返回true。
**********************************************************/
template<class Type> bool MatrixHx<Type>::DOTMUL(const MatrixHx<Type>& A, const MatrixHx<Type>& B, MatrixHx<Type>& C)
{
    if ((A.m_nRow != B.m_nRow) || (A.m_nCol != B.m_nCol))
    {
        return false;
    }

    C.Reset(A.m_nRow, A.m_nCol);

    for (unsigned int i = 0; i < C.m_nRow; i++)
    {
        for (unsigned int j = 0; j < C.m_nCol; j++)
        {
            C.m_ppData[i][j] = A.m_ppData[i][j] * B.m_ppData[i][j];
        }
    }

    return true;
}

/************************************************************
函数说明：
    矩阵的点对点的乘法：C[i,j] = A[i,j] * B[i,j]。
参数:
    A：被乘的矩阵；
    B：除数矩阵；
返回值：
    返回点对点乘的结果矩阵。
    注意：当A,B的行数或列数不等时，会返回一个行列为0的空矩阵。
*************************************************************/
template<class Type> MatrixHx<Type> MatrixHx<Type>::DOTMUL(const MatrixHx<Type>& A, const MatrixHx<Type>& B)
{
    MatrixHx<Type> C;
    DOTMUL(A, B, C);

    return C;
}

/*********************************************************
函数说明：
    矩阵的点对点的除法：C[i,j] = A[i,j] / B[i,j]。
参数:
    A：被除的矩阵；
    B：除数矩阵；
    C：结果矩阵（行数和列数与A,B相同）。
返回值：
    当A,B矩阵的行列数不一样时，返回false，一般返回true。
**********************************************************/
template<class Type> bool MatrixHx<Type>::DOTDIV(const MatrixHx<Type>& A, const MatrixHx<Type>& B, MatrixHx<Type>& C)
{
    if ((A.m_nRow != B.m_nRow) || (A.m_nCol != B.m_nCol))
    {
        return false;
    }
        
    C.Reset(A.m_nRow, A.m_nCol);

    for (unsigned int i = 0; i < C.m_nRow; i++)
    {
        for (unsigned int j = 0; j < C.m_nCol; j++)
        {
            C.m_ppData[i][j] = A.m_ppData[i][j] / B.m_ppData[i][j];
        }
    }

    return true;
}

/***********************************************************
函数说明：
    静态成员函数。矩阵的转置：B = A~。A为m*n矩阵，B为n*m矩阵。
参数:
    A：被转置的矩阵；
    B：A的转置矩阵。
返回值：
    返回true。
************************************************************/
template<class Type> bool MatrixHx<Type>::Transpose(const MatrixHx<Type>& A, MatrixHx<Type>& B)
{
    B.Reset(A.m_nCol, A.m_nRow);

    for (unsigned int i = 0; i < A.m_nRow; i++)
    {
        for (unsigned int j = 0; j < A.m_nCol; j++)
            B.m_ppData[j][i] = A.m_ppData[i][j];
    }

    return true;
}

/*********************************************************
函数说明：
    矩阵的转置：B = A~。A为m*n矩阵，B为n*m矩阵。
参数:
    无。
返回值：
    返回被转置的矩阵（返回矩阵被置为n*m矩阵）。
**********************************************************/
template<class Type> MatrixHx<Type> MatrixHx<Type>::Transpose() const
{
    MatrixHx<Type> B;
    Transpose(*this, B);

    return B;
}