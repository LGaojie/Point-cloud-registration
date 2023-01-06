/*************************************************************************
说明：
    简单的线性代数库，实现了矩阵的求特征矢量、特征根；SVD分解；LU分解；
    以及矩阵的基本运算。
    用C++的类模板实现。
作者：
    Zhang X.C              2022.10.1
版权：
    HXMC
**************************************************************************/

#pragma once

#include <iostream>
#include <cassert>
#include <vector>

// 提前声明单位化矩阵的类
template< class TYPE > class IdentityMatrix;

// 矩阵类的定义
template< class TYPE = float > class MatrixHxmc
{

public:
    // Used for matrix concatenation.
    typedef enum
    {
        TO_RIGHT,
        TO_BOTTOM
    } Position;

    // 构造和析构
    MatrixHxmc();
    MatrixHxmc(unsigned rowsParameter, unsigned columnsParameter);
    MatrixHxmc(MatrixHxmc const& copyMatrix, unsigned omittedRow = INT_MAX, unsigned omittedColumn = INT_MAX);
    MatrixHxmc(MatrixHxmc const& copyMatrixA, MatrixHxmc const& copyMatrixB, Position position = TO_RIGHT);
    ~MatrixHxmc();
    // 接口函数
    unsigned GetRows() const;
    unsigned GetColumns() const;
    TYPE     Get(unsigned row, unsigned column) const;
    void     LU_Decomposition(MatrixHxmc& upper, MatrixHxmc& lower) const;
    void     Put(unsigned row, unsigned column, TYPE const& value);
    MatrixHxmc GetSubMatrix(unsigned startRow, unsigned endRow, unsigned startColumn,
        unsigned endColumn, unsigned const* newOrder = NULL);
    MatrixHxmc GetColumn(unsigned column);
    MatrixHxmc GetRow(unsigned row);
    void       ReducedRowEcholon();
    TYPE       Determinant() const;
    TYPE       DotProduct(MatrixHxmc const& otherMatrix) const;
    MatrixHxmc const GetTranspose() const;
    void       Transpose();
    MatrixHxmc const  GetInverse() const;
    void       Invert();
    void       Jacobi(MatrixHxmc<TYPE>& matrix, int size, std::vector<TYPE>& E, MatrixHxmc<TYPE>& J);
    int        JacobiN(TYPE** a, int n, TYPE* w, TYPE** v);
    void       SVD(MatrixHxmc<TYPE>& A, MatrixHxmc<TYPE>& U, MatrixHxmc<TYPE>& V, std::vector<TYPE>& E);
    void       Orthogonal(MatrixHxmc<TYPE>& matrix, int i, int j, bool& pass, MatrixHxmc<TYPE>& V);
    void       Hestens_Jacobi(MatrixHxmc<TYPE>& matrix, MatrixHxmc<TYPE>& V);
    int        SVN(MatrixHxmc<TYPE>& matrix, MatrixHxmc<TYPE>& S, MatrixHxmc<TYPE>& U, MatrixHxmc<TYPE>& V);

    // Operators
    MatrixHxmc const operator + (MatrixHxmc const& otherMatrix) const;
    MatrixHxmc const& operator += (MatrixHxmc const& otherMatrix);
    MatrixHxmc const operator - (MatrixHxmc const& otherMatrix) const;
    MatrixHxmc const& operator -= (MatrixHxmc const& otherMatrix);
    MatrixHxmc const operator * (MatrixHxmc const& otherMatrix) const;
    MatrixHxmc const& operator *= (MatrixHxmc const& otherMatrix);
    MatrixHxmc const operator * (TYPE const& scalar) const;
    MatrixHxmc const& operator *= (TYPE const& scalar);
    // Copy matrix.
    MatrixHxmc& operator = (MatrixHxmc const& otherMatrix);
    MatrixHxmc& operator = (TYPE const* data);
    bool operator == (MatrixHxmc const& value) const;
    bool operator != (MatrixHxmc const& value) const;

protected:
    unsigned  m_sRow;           //矩阵的行数；
    unsigned  m_sCol;           //矩阵的列数；
    unsigned* m_sRowOrder;      //存储矩阵的行顺序；

    std::vector< std::vector< TYPE > > m_data; // 矩阵的数据，用STL的二维矢量存储

protected:
    unsigned GetLeadingZeros(unsigned row) const;
    void     Reorder();
    void     DivideRow(unsigned row, TYPE const& divisor);
    void     RowOperation(unsigned row, unsigned addRow, TYPE const& scale);
    void     Allocate(unsigned rowNumber, unsigned columnNumber);
    void     Deallocate(unsigned rowNumber, unsigned columnNumber);
    //For  SVD and Eigen values and vector
    int     Sign(TYPE number);
    void    Rotate(MatrixHxmc < TYPE >& matrix, int i, int j, bool& pass, MatrixHxmc < TYPE >& J);
};


/*****************************************************************
类说明：
    衍生一个产生单位矩阵的类。
******************************************************************/
template< class TYPE >
class IdentityMatrix : public MatrixHxmc< TYPE >
{
public:
    IdentityMatrix(unsigned rowsParameter, unsigned columnsParameter);
};

/*****************************************************************
函数说明：
    从标准模板衍生一个输出矩阵为字符串的函数。
输入：
    matrix：输入矩阵。
输出：
    stream：输出字符串流。
******************************************************************/
template< class TYPE >
std::ostream& operator<< (std::ostream& stream, MatrixHxmc< TYPE > const& matrix)
{
    for (unsigned row = 0; row < matrix.GetRows(); ++row)
    {
        for (unsigned column = 0; column < matrix.GetColumns(); ++column)
        {
            stream << "\t" << matrix.Get(row, column);
        }
            
        stream << std::endl;
    }

    return stream;
}

/*************************************************************************
说明：
    简单的线性代数库，实现了矩阵的求特征矢量、特征根；SVD分解；LU分解；
    以及矩阵的基本运算。
    用C++的类模板实现。
作者：
    Zhang Xiaochun              2022.10.1
版权：
    HXMC
**************************************************************************/

#include "pch.h"
#include "matrix_hxmc.h"
#include <cmath>
#include <map>

using namespace std;

const double THRESHOLD = 1E-8;
const int    ITERATION = 30;   //迭代次数的上限

/*********************************************************
函数说明：
    构造函数：构造一个空的矩阵。
**********************************************************/
template< class TYPE >
MatrixHxmc<TYPE>::MatrixHxmc() : m_sRow(0), m_sCol(0)
{
    Allocate(0, 0);
}


//-------------------------------------------------------------
// Constructor using m_sRow and m_sCol.
//-------------------------------------------------------------
/*********************************************************
函数说明：
    构造函数：构造一个rows行、cols列的矩阵，每个矩阵元为0。
参数：
    rows：矩阵的行数；
    cols：矩阵的列数。
**********************************************************/
template< class TYPE >
MatrixHxmc<TYPE>::MatrixHxmc(unsigned rows, unsigned cols)
    :m_sRow(rows), m_sCol(cols)
{
    TYPE const ZERO = static_cast<TYPE>(0);

    // Allocate memory for new matrix.
    Allocate(m_sRow, m_sCol);

    // Fill matrix with zero.
    for (unsigned row = 0; row < m_sRow; ++row)
    {
        m_sRowOrder[row] = row;

        for (unsigned column = 0; column < m_sCol; ++column)
        {
            m_data[row][column] = ZERO;
        }
    }
}

/***********************************************************************************
函数说明：
    构造函数：当行参数和/或列参数被省略时，构造一个与被Copy矩阵一样的矩阵；
              当没有省略时，构造比被Copy矩阵少一行和一列的矩阵。
参数：
    copyMatrix：待复制的矩阵；
    sOmitRow：待去掉行的编号（0开始）；
    sOmitCol：待去掉列的编号（0开始）；
************************************************************************************/
template< class TYPE >
MatrixHxmc<TYPE>::MatrixHxmc(MatrixHxmc const& copyMatrix, unsigned sOmitRow, unsigned sOmitCol)
{
    // Start with the number of m_sRow/m_sCol from matrix to be copied.
    m_sRow = copyMatrix.GetRows();
    m_sCol = copyMatrix.GetColumns();

    // If a row is omitted, then there is one less row.
    if (INT_MAX != sOmitRow)
    {
        m_sRow--;
    }

    // If a column is omitted, then there is one less column.
    if (INT_MAX != sOmitCol)
    {
        m_sCol--;
    }

    // Allocate memory for new matrix.
    Allocate(m_sRow, m_sCol);

    unsigned rowindex = 0;
    for (unsigned row = 0; row < m_sRow; ++row)
    {
        // if this row is to be skipped...
        if (rowindex == sOmitRow)
        {
            rowindex++;
        }

        // set default m_sroworder.
        m_sRowOrder[row] = row;

        unsigned columnindex = 0;
        for (unsigned column = 0; column < m_sCol; ++column)
        {
            // if this column is to be skipped...
            if (columnindex == sOmitCol)
            {
                columnindex++;
            }

            m_data[row][column] = copyMatrix.m_data[rowindex][columnindex];

            columnindex++;
        }

        ++rowindex;
    }
}

/****************************************************************************
函数说明：
    构造函数：用A和B两个矩阵构造一个拼接矩阵：当pos参数是RIGHT时，新构造矩阵
    将B矩阵置于A的右边；当pos参数是BOTTOM时，拼接在底部，。
参数：
    copyMatrixA：矩阵A；
    copyMatrixB：矩阵B；
    pos：B拼接矩阵的位置。
****************************************************************************/
template< class TYPE >
MatrixHxmc<TYPE>::MatrixHxmc(MatrixHxmc const& copyMatrixA, MatrixHxmc const& copyMatrixB, Position pos)
{
    unsigned rowOffset = 0;
    unsigned columnOffset = 0;

    if (TO_RIGHT == pos)
    {
        columnOffset = copyMatrixA.m_sCol;
    }
    else
    {
        rowOffset = copyMatrixA.m_sRow;
    }

    m_sRow = copyMatrixA.m_sRow + rowOffset;
    m_sCol = copyMatrixA.m_sCol + columnOffset;

    // Allocate memory for new matrix.
    Allocate(m_sRow, m_sCol);

    for (unsigned row = 0; row < copyMatrixA.m_sRow; ++row)
    {
        for (unsigned column = 0; column < copyMatrixA.m_sCol; ++column)
        {
            m_data[row][column] = copyMatrixA.m_data[row][column];
        }
    }


    for (unsigned row = 0; row < copyMatrixB.m_sRow; ++row)
    {
        for (unsigned column = 0; column < copyMatrixB.m_sCol; ++column)
        {
            m_data[row + rowOffset][column + columnOffset] = copyMatrixB.m_data[row][column];
        }
    }
}

/***********************************************
函数说明：
    析构函数。
参数：
    无。
************************************************/
template< class TYPE > MatrixHxmc<TYPE>::~MatrixHxmc()
{
    // Release memory.
    Deallocate(m_sRow, m_sCol);
}

/***********************************************
函数说明：
    获得矩阵的行数。
参数：
    无。
返回：
    矩阵的行数。
************************************************/
template< class TYPE > unsigned MatrixHxmc<TYPE>::GetRows() const
{
    return m_sRow;
}

/***********************************************
函数说明：
    获得矩阵的列数。
参数：
    无。
返回：
    矩阵的列数。
************************************************/
template< class TYPE > unsigned MatrixHxmc<TYPE>::GetColumns() const
{
    return m_sCol;
}

/***********************************************
函数说明：
    获得给定行、列矩阵元。
参数：
    row：给定行编号（0开始）；
    column：给定列编号（0开始）。
返回：
    该行该列矩阵元的值。
************************************************/
template< class TYPE > TYPE MatrixHxmc<TYPE>::Get(unsigned row, unsigned column) const
{
    assert(row < m_sRow);
    assert(column < m_sCol);

    return m_data[row][column];
}

/*****************************************************
函数说明：
    方阵的LU分解：This will create matrices L and U
    such that A=LxU。
参数：
    row：给定行编号（0开始）；
    column：给定列编号（0开始）。
返回：
    该行该列矩阵元的值。
******************************************************/
template< class TYPE >
void MatrixHxmc<TYPE>::LU_Decomposition(MatrixHxmc& upper, MatrixHxmc& lower) const
{
    assert(m_sRow == m_sCol);

    TYPE const ZERO = static_cast<TYPE>(0);

    upper = *this;
    lower = *this;

    for (unsigned row = 0; row < m_sRow; ++row)
    {
        for (unsigned column = 0; column < m_sCol; ++column)
        {
            lower.m_data[row][column] = ZERO;
        }
    }


    for (unsigned row = 0; row < m_sRow; ++row)
    {
        TYPE value = upper.m_data[row][row];
        if (ZERO != value)
        {
            upper.DivideRow(row, value);
            lower.m_data[row][row] = value;
        }

        for (unsigned subRow = row + 1; subRow < m_sRow; ++subRow)
        {
            TYPE value = upper.m_data[subRow][row];
            upper.RowOperation(subRow, row, -value);
            lower.m_data[subRow][row] = value;
        }
    }
}

/*****************************************************
函数说明：
    设置矩阵指定行、列的值。
参数：
    row：给定行编号（0开始）；
    column：给定列编号（0开始）；
    value：设置的值。
返回：
    无。
******************************************************/
template< class TYPE >void MatrixHxmc<TYPE>::Put(unsigned row, unsigned column, TYPE const& value)
{
    assert(row < m_sRow);
    assert(column < m_sCol);

    m_data[row][column] = value;
}

/********************************************************************
函数说明：
    获取矩阵指定位置（行列）的子矩阵。
    注意：startRow = endRow（或startColumn = endColumn）表示仅仅获取
          一行的部分（或一列的部分）。
参数：
    startRow：子矩阵在本矩阵中开始的行；
    endRow：子矩阵在本矩阵中结束的行；
    startColumn：子矩阵在本矩阵中开始的列；
    endColumn：子矩阵在本矩阵中结束的列；
    newOrder：如为空，不会重新排序；否则按照newOrder指定的顺序排列行。
返回：
    子矩阵。
********************************************************************/
template< class TYPE >
MatrixHxmc<TYPE> MatrixHxmc<TYPE>::GetSubMatrix(unsigned startRow, unsigned endRow, unsigned startColumn,
    unsigned endColumn, unsigned const* newOrder)
{
    MatrixHxmc subMatrix(endRow - startRow + 1, endColumn - startColumn + 1);

    for (unsigned row = startRow; row <= endRow; ++row)
    {
        unsigned subRow;
        if (NULL == newOrder)
        {
            subRow = row;
        }
        else
        {
            subRow = newOrder[row];
        }

        for (unsigned column = startColumn; column <= endColumn; ++column)
        {
            subMatrix.m_data[row - startRow][column - startColumn] = m_data[subRow][column];
        }

    }

    return subMatrix;
}

/*********************************************************
函数说明：
    获取矩阵第column列(m_sCol*1矩阵，即列矢量)。
参数:
    column：列指标，从0开始。
返回值：
    子矩阵（第column列的列矢量）。
**********************************************************/
template< class TYPE >MatrixHxmc<TYPE> MatrixHxmc<TYPE>::GetColumn(unsigned column)
{
    return GetSubMatrix(0, m_sRow - 1, column, column);
}

/*********************************************************
函数说明：
    获取矩阵第row行(1*m_sRow矩阵，即行矢量)。
参数:
    row：行指标，从0开始。
返回值：
    子矩阵（第row行的行矢量）。
**********************************************************/
template< class TYPE >MatrixHxmc<TYPE> MatrixHxmc<TYPE>::GetRow(unsigned row)
{
    return GetSubMatrix(row, row, 0, m_sCol - 1);
}

//-------------------------------------------------------------
    // Place matrix in reduced row echelon form.
    //-------------------------------------------------------------
/*********************************************************
函数说明：
    获取矩阵第row行(1*m_sRow矩阵，即行矢量)。
参数:
    row：行指标，从0开始。
返回值：
    子矩阵（第row行的行矢量）。
**********************************************************/
template< class TYPE >void MatrixHxmc<TYPE>::ReducedRowEcholon()
{
    TYPE const ZERO = static_cast<TYPE>(0);

    // For each row...
    for (unsigned rowIndex = 0; rowIndex < m_sRow; ++rowIndex)
    {
        // Reorder the m_sRow.
        Reorder();

        unsigned row = m_sRowOrder[rowIndex];

        // Divide row down so first term is 1.
        unsigned column = GetLeadingZeros(row);
        TYPE divisor = m_data[row][column];

        if (ZERO != divisor)
        {
            DivideRow(row, divisor);

            // Subtract this row from all subsequent m_sRow.
            for (unsigned subRowIndex = (rowIndex + 1); subRowIndex < m_sRow; ++subRowIndex)
            {
                unsigned subRow = m_sRowOrder[subRowIndex];
                if (ZERO != m_data[subRow][column])
                {
                    RowOperation(subRow, row, -m_data[subRow][column]);
                }
            }
        }
    }

    // Back substitute all lower m_sRow.
    for (unsigned rowIndex = (m_sRow - 1); rowIndex > 0; --rowIndex)
    {
        unsigned row = m_sRowOrder[rowIndex];
        unsigned column = GetLeadingZeros(row);
        for (unsigned subRowIndex = 0; subRowIndex < rowIndex; ++subRowIndex)
        {
            unsigned subRow = m_sRowOrder[subRowIndex];
            RowOperation(subRow, row, -m_data[subRow][column]);
        }
    }

} // ReducedRowEcholon


/*********************************************************
函数说明：
    获得矩阵的行列式值（递归方法实现）。
参数:
    无。
返回值：
    行列式值。
**********************************************************/
template< class TYPE >TYPE MatrixHxmc<TYPE>::Determinant() const
{
    TYPE result = static_cast<TYPE>(0);

    // Must have a square matrix to even bother.
    assert(m_sRow == m_sCol);

    if (m_sRow > 2)
    {
        int sign = 1;
        for (unsigned column = 0; column < m_sCol; ++column)
        {
            TYPE subDeterminant;

            MatrixHxmc subMatrix = MatrixHxmc(*this, 0, column);

            subDeterminant = subMatrix.Determinant();
            subDeterminant *= m_data[0][column];

            if (sign > 0)
            {
                result += subDeterminant;
            }
            else
            {
                result -= subDeterminant;
            }

            sign = -sign;
        }
    }
    else
    {
        result = (m_data[0][0] * m_data[1][1]) - (m_data[0][1] * m_data[1][0]);
    }

    return result;

} // Determinant

/*********************************************************
函数说明：
    计算两矩阵对应行列元之积的和。
参数:
    otherMatrix：与本矩阵进行点积运算的矩阵。
返回值：
    点积（矢量点积的扩展）。
**********************************************************/
template< class TYPE >TYPE MatrixHxmc<TYPE>::DotProduct(MatrixHxmc const& otherMatrix) const
{
    // Dimentions of each matrix must be the same.
    assert(m_sRow == otherMatrix.m_sRow);
    assert(m_sCol == otherMatrix.m_sCol);

    TYPE result = static_cast<TYPE>(0);
    for (unsigned row = 0; row < m_sRow; ++row)
    {
        for (unsigned column = 0; column < m_sCol; ++column)
        {
            result += m_data[row][column] * otherMatrix.m_data[row][column];
        }
    }

    return result;

} // DotProduct

/*********************************************************
函数说明：
    生成一个本矩阵的转置矩阵。
参数:
    无。
返回值：
    转置矩阵。
**********************************************************/
template< class TYPE > MatrixHxmc<TYPE> const MatrixHxmc<TYPE>::GetTranspose() const
{
    MatrixHxmc result(m_sCol, m_sRow);

    // Transpose the matrix by filling the result's m_sRow will
    // these m_sCol, and vica versa.
    for (unsigned row = 0; row < m_sRow; ++row)
    {
        for (unsigned column = 0; column < m_sCol; ++column)
        {
            result.m_data[column][row] = m_data[row][column];
        }

    }

    return result;

} // Transpose

/***************************************
说明：
    转置本矩阵。
参数:
    无。
返回值：
    无。
*****************************************/
template< class TYPE > void MatrixHxmc<TYPE>::Transpose()
{
    *this = GetTranspose();
}

/***************************************
说明：
    产生一本矩阵的逆矩阵。
参数:
    无。
返回值：
    逆矩阵。
*****************************************/
template< class TYPE > MatrixHxmc<TYPE> const  MatrixHxmc<TYPE>::GetInverse() const
{
    // Concatenate the identity matrix onto this matrix.
    MatrixHxmc inverseMatrix(*this, IdentityMatrix< TYPE >(m_sRow, m_sCol), TO_RIGHT);

    // Row reduce this matrix.  This will result in the identity
    // matrix on the left, and the inverse matrix on the right.
    inverseMatrix.ReducedRowEcholon();

    // Copy the inverse matrix data back to this matrix.
    MatrixHxmc result(inverseMatrix.GetSubMatrix(0, m_sRow - 1, m_sCol, m_sCol + m_sCol - 1, inverseMatrix.m_sRowOrder));

    return result;
} // Invert


/***************************************
说明：
   将本矩阵变换成其逆矩阵。
参数:
    无。
返回值：
    无。
*****************************************/
template< class TYPE > void  MatrixHxmc<TYPE>::Invert()
{
    *this = GetInverse();
} // Invert


// Operators.
/***************************************
说明：
    矩阵的加：R = this + otherMatrix。
参数:
    otherMatrix：被加的矩阵。
返回值：
    两个矩阵的和。
*****************************************/
template< class TYPE > MatrixHxmc<TYPE> const MatrixHxmc<TYPE>::operator + (MatrixHxmc const& otherMatrix) const
{
    assert(otherMatrix.m_sRow == m_sRow);
    assert(otherMatrix.m_sCol == m_sCol);

    MatrixHxmc result(m_sRow, m_sCol);

    for (unsigned row = 0; row < m_sRow; ++row)
    {
        for (unsigned column = 0; column < m_sCol; ++column)
        {
            result.m_data[row][column] = m_data[row][column] + otherMatrix.m_data[row][column];
        }
    }

    return result;
}

/***************************************
说明：
    矩阵的加：R = *this + otherMatrix。
参数:
    otherMatrix：被加的矩阵。
返回值：
    两个矩阵的和。
*****************************************/
template< class TYPE > MatrixHxmc<TYPE> const& MatrixHxmc<TYPE>::operator += (MatrixHxmc const& otherMatrix)
{
    *this = *this + otherMatrix;
    return *this;
}

/***************************************
说明：
    矩阵的加：R = *this - otherMatrix。
参数:
    otherMatrix：被减的矩阵。
返回值：
    两个矩阵的差。
*****************************************/
template< class TYPE > MatrixHxmc<TYPE> const MatrixHxmc<TYPE>::operator - (MatrixHxmc const& otherMatrix) const
{
    assert(otherMatrix.m_sRow == m_sRow);
    assert(otherMatrix.m_sCol == m_sCol);

    MatrixHxmc result(m_sRow, m_sCol);

    for (unsigned row = 0; row < m_sRow; ++row)
    {
        for (unsigned column = 0; column < m_sCol; ++column)
        {
            result.m_data[row][column] =
                m_data[row][column]
                - otherMatrix.m_data[row][column];
        }
    }

    return result;
}

/*************************************************
说明：
    自减的矩阵运算：*this = *this - otherMatrix。
参数:
    otherMatrix：被减的矩阵。
返回值：
    自己与另外矩阵的差。
**************************************************/
template< class TYPE > MatrixHxmc<TYPE> const& MatrixHxmc<TYPE>::operator -= (MatrixHxmc const& otherMatrix)
{
    *this = *this - otherMatrix;
    return *this;
}

/*******************************************
说明：
    矩阵的乘法：R = *this * otherMatrix。
参数:
    otherMatrix：被乘的矩阵。
返回值：
    两个矩阵的乘。
*********************************************/
template< class TYPE > MatrixHxmc<TYPE> const MatrixHxmc<TYPE>::operator * (MatrixHxmc const& otherMatrix) const
{
    TYPE const ZERO = static_cast<TYPE>(0);

    assert(otherMatrix.m_sRow == m_sCol);

    MatrixHxmc result(m_sRow, otherMatrix.m_sCol);

    for (unsigned row = 0; row < m_sRow; ++row)
    {
        for (unsigned column = 0; column < otherMatrix.m_sCol; ++column)
        {
            result.m_data[row][column] = ZERO;

            for (unsigned index = 0; index < m_sCol; ++index)
            {
                result.m_data[row][column] += m_data[row][index] * otherMatrix.m_data[index][column];
            }

        }
    }

    return result;
}

/*************************************************
说明：
    自乘法的矩阵运算：*this = *this * otherMatrix。
参数:
    otherMatrix：被乘的矩阵。
返回值：
    自己与另外矩阵的乘。
**************************************************/
template< class TYPE >MatrixHxmc<TYPE> const& MatrixHxmc<TYPE>::operator *= (MatrixHxmc const& otherMatrix)
{
    *this = *this * otherMatrix;
    return *this;
}

/*******************************************
说明：
    矩阵与数的乘法：R = *this * scalar。
参数:
    scalar：被乘的数。
返回值：
    每个矩阵元乘上该数。
*********************************************/
template< class TYPE >MatrixHxmc<TYPE> const MatrixHxmc<TYPE>::operator * (TYPE const& scalar) const
{
    MatrixHxmc result(m_sRow, m_sCol);

    for (unsigned row = 0; row < m_sRow; ++row)
    {
        for (unsigned column = 0; column < m_sCol; ++column)
        {
            result.m_data[row][column] = m_data[row][column] * scalar;
        }
    }
        
    return result;
}

/*******************************************
说明：
    本矩阵与标量的乘法：*this = *this * scalar。
参数:
    scalar：被乘的数（标量）。
返回值：
    每个矩阵元乘上该数。
*********************************************/
template< class TYPE > MatrixHxmc<TYPE> const& MatrixHxmc<TYPE>::operator *= (TYPE const& scalar)
{
    *this = *this * scalar;
    return *this;
}

/*************************************************
说明：
    运算符=：将otherMatrix的数据拷贝到*this。
参数:
    otherMatrix：待拷贝的矩阵。
返回值：
    *this。
***************************************************/
template< class TYPE > MatrixHxmc<TYPE>& MatrixHxmc<TYPE>::operator = (MatrixHxmc const& otherMatrix)
{
    if (this == &otherMatrix)
    {
        return *this;
    }
        
    // Release memory currently in use.
    Deallocate(m_sRow, m_sCol);

    m_sRow = otherMatrix.m_sRow;
    m_sCol = otherMatrix.m_sCol;
    Allocate(m_sRow, m_sCol);

    for (unsigned row = 0; row < m_sRow; ++row)
    {
        for (unsigned column = 0; column < m_sCol; ++column)
        {
            m_data[row][column] = otherMatrix.m_data[row][column];
        }

    }

    return *this;
}

/*********************************************************
说明：
    运算符=：从线性数组中拷贝数据到矩阵元。
    注意：矩阵是两维的。待拷贝数据是一维的，假定数据是
    线性存储的，并且至少m_sRow*m_sCol个。

    Example for 3x2 matrix:
    int const data[ 3 * 2 ] =
    {
       1, 2, 3,
       4, 5, 6
    };
    MatrixHxmc< int > matrix( 3, 2 )。
参数:
    data：指向线性一维数组。
返回值：
    每个矩阵元乘上该数。
**********************************************************/
template< class TYPE > MatrixHxmc<TYPE>& MatrixHxmc<TYPE>::operator = (TYPE const* data)
{
    unsigned index = 0;

    for (unsigned row = 0; row < m_sRow; ++row)
    {
        for (unsigned column = 0; column < m_sCol; ++column)
        {
            m_data[row][column] = data[index++];
        }
    }

    return *this;
}

/****************************************************
说明：
    判断两个矩阵是否全等。
参数:
    value：矩阵。
返回值：
    如果矩阵的每个元均相等，返回true，否则返回false。
*****************************************************/
template< class TYPE > bool MatrixHxmc<TYPE>::operator == (MatrixHxmc const& value) const
{
    bool isEqual = true;

    for (unsigned row = 0; row < m_sRow; ++row)
    {
        for (unsigned column = 0; column < m_sCol; ++column)
        {
            if (m_data[row][column] != value.m_data[row][column])
            {
                isEqual = false;
            }
        }
    }

    return isEqual;
}

/****************************************************
说明：
    判断两个矩阵是否不等：只要一个元素不等则成立。
参数:
    value：矩阵。
返回值：
    如果矩阵只要一个元不相等，返回true，否则返回false。
*****************************************************/
template< class TYPE > bool MatrixHxmc<TYPE>::operator != (MatrixHxmc const& value) const
{
    return !(*this == value);
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
template< class TYPE >
void MatrixHxmc<TYPE>::Jacobi(MatrixHxmc<TYPE>& matrix, int size, std::vector<TYPE>& E, MatrixHxmc<TYPE>& J)
{

    int iteration = ITERATION;

    while (iteration-- > 0)
    {
        bool pass = true;
        for (int i = 0; i < size; ++i)
        {
            for (int j = i + 1; j < size; ++j)
            {
                Rotate(matrix, i, j, pass, J);
            }
        }
        if (pass)   //当非对角元素全部变为0时迭代退出
        {
            break;
        }
    }

    //cout << "迭代次数：" << ITERATION - iteration << endl;

    for (int i = 0; i < size; ++i)
    {
        E[i] = matrix.get(i, i);

        if (E[i] < THRESHOLD)
        {
            E[i] = 0.0;
        }
    }
}

/************************************************************************
说明：
    双边Jacobi旋转法迭代求解n*n对称方阵的特征根与特征矢量。
    特征值是降序排列的，特征矢量是归一化的。
参数：
    a：输入的n*n对称矩阵；
    n：输入矩阵的维度；
    w：输出的n个特征值；
    v：输出的n*n特征矢量矩阵（列矢量）；

返回：
    超过迭代次数返回0，否则返回1。
*************************************************************************/
template< class TYPE >
int MatrixHxmc<TYPE>::JacobiN(TYPE** a, int n, TYPE* w, TYPE** v)
{
    int i, j, k, iq, ip, numPos;
    TYPE tresh, theta, tau, t, sm, s, h, g, c, tmp;
    TYPE bspace[4], zspace[4];
    TYPE* b = bspace;
    TYPE* z = zspace;

    // only allocate memory if the matrix is large
    if (n > 4)
    {
        b = new TYPE[n];
        z = new TYPE[n];
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
    for (i = 0; i < ITERATION; i++)
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
    if (i >= ITERATION)
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
template< class TYPE >
void MatrixHxmc<TYPE>::SVD(MatrixHxmc<TYPE>& A, MatrixHxmc<TYPE>& U, MatrixHxmc<TYPE>& V, std::vector<TYPE>& E)
{
    int rows = A.GetRows();
    int columns = A.GetColumns();

    assert(rows <= columns);
    assert(U.GetRows() == rows);
    assert(U.GetColumns() == rows);
    assert(V.GetRows() == columns);
    assert(V.GetColumns() == columns);
    assert(E.size() == columns);

    MatrixHxmc<TYPE> B = A.GetTranspose() * A;                        //A的转置乘以A，得到一个对称矩阵B
    MatrixHxmc<TYPE> J(IdentityMatrix<TYPE>(columns, columns));
    std::vector <TYPE>S(columns);

    Jacobi(B, columns, S, J);  //求B的特征值和特征向量
    for (int i = 0; i < S.size(); ++i)
    {
        S[i] = sqrt(S[i]);   //B的特征值开方后得到A的奇异值
    }

    /*奇异值按递减排序，对应的V中的特征向量也要重排序 */
    multimap < TYPE, int > eigen;

    for (int i = 0; i < S.size(); ++i)   //在multimap内部自动按key进行排序
    {
        eigen.insert(make_pair(S[i], i));
    }

    auto iter = --eigen.end();
    //multimap < TYPE, int >::const_iterator iter = --eigen.end();

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
            V.Put(row, i, J.Get(row, index));
        }
    }

    assert(num_eig <= rows);

    for (int i = 0; i < num_eig; ++i)
    {
        MatrixHxmc<TYPE> vi = V.GetColumn(i); //获取V的第i列
        TYPE sigma = E[i];
        MatrixHxmc<TYPE> ui(rows, 1);

        ui = A * vi;
        for (int j = 0; j < rows; ++j)
        {
            U.Put(j, i, ui.Get(j, 0) / sigma);
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
template< class TYPE >
void MatrixHxmc<TYPE>::Orthogonal(MatrixHxmc<TYPE>& matrix, int i, int j, bool& pass, MatrixHxmc<TYPE>& V)
{
    assert(i < j);

    MatrixHxmc<TYPE> Ci = matrix.GetColumn(i);
    MatrixHxmc<TYPE> Cj = matrix.GetColumn(j);

    TYPE ele = ((Ci.GetTranspose()) * Cj).Get(0, 0);

    if (fabs(ele) < THRESHOLD)          //i,j两列已经正交
    {
        return;
    }

    int rows = matrix.GetRows();
    int columns = matrix.GetColumns();

    pass = false;
    TYPE ele1 = ((Ci.GetTranspose()) * Ci).Get(0, 0);
    TYPE ele2 = ((Cj.GetTranspose()) * Cj).Get(0, 0);

    /*只要每次旋转前都把范数大的列放在前面，就可以保证求出的奇异值是递减排序的*/
    if (ele1 < ele2)            //如果matrix第i列的范数小于第j列，则交换两列.同时V矩阵也作相应的调换
    {
        for (int row = 0; row < rows; ++row)
        {
            matrix.Put(row, i, Cj.Get(row, 0));
            matrix.Put(row, j, Ci.Get(row, 0));
        }
        for (int row = 0; row < columns; ++row)
        {
            TYPE tmp = V.Get(row, i);
            V.Put(row, i, V.Get(row, j));
            V.Put(row, j, tmp);
        }
    }

    TYPE tao = (ele1 - ele2) / (2 * ele);
    TYPE tan = sign(tao) / (fabs(tao) + sqrt(1 + pow(tao, 2)));
    TYPE cos = 1 / sqrt(1 + pow(tan, 2));
    TYPE sin = cos * tan;

    for (int row = 0; row < rows; ++row)
    {
        TYPE var1 = matrix.Get(row, i) * cos + matrix.Get(row, j) * sin;
        TYPE var2 = matrix.Get(row, j) * cos - matrix.Get(row, i) * sin;
        matrix.Put(row, i, var1);
        matrix.Put(row, j, var2);
    }

    for (int col = 0; col < columns; ++col)
    {
        TYPE var1 = V.Get(col, i) * cos + V.Get(col, j) * sin;
        TYPE var2 = V.Get(col, j) * cos - V.Get(col, i) * sin;
        V.Put(col, i, var1);
        V.Put(col, j, var2);
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
template< class TYPE >
void MatrixHxmc<TYPE>::Hestens_Jacobi(MatrixHxmc<TYPE>& matrix, MatrixHxmc<TYPE>& V)
{
    int rows = matrix.GetRows();
    int columns = matrix.GetColumns();

    int iteration = ITERATION;

    while (iteration-- > 0)
    {
        bool pass = true;
        for (int i = 0; i < columns; ++i)
        {
            for (int j = i + 1; j < columns; ++j)
            {
                Orthogonal(matrix, i, j, pass, V);      //经过多次的迭代正交后，V就求出来了
            }
        }
        if (pass)   //当任意两列都正交时退出迭代
        {
            break;
        }
    }
    //cout << "迭代次数：" << ITERATION - iteration << endl;
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
template< class TYPE >
int MatrixHxmc<TYPE>::SVN(MatrixHxmc<TYPE>& matrix, MatrixHxmc<TYPE>& S, MatrixHxmc<TYPE>& U, MatrixHxmc<TYPE>& V)
{
    int rows = matrix.GetRows();
    int columns = matrix.GetColumns();

    assert(rows <= columns);

    Hestens_Jacobi(matrix, V);

    vector<TYPE> E(columns);          //E中存放奇异值
    int none_zero = 0;                //记录非0奇异值的个数

    for (int i = 0; i < columns; ++i)
    {
        TYPE norm = sqrt((matrix.GetColumn(i).GetTranspose() * (matrix.GetColumn(i))).Get(0, 0));

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
        S.Put(row, row, E[row]);

        for (int col = 0; col < none_zero; ++col)
        {
            U.Put(row, col, matrix.Get(row, col) / E[col]);
        }
    }

    return none_zero;   //非奇异值的个数亦即矩阵的秩
}

//Protected
/*********************************************************
函数说明：
    获取矩阵第row行中第一个不为0的列编号（行的前导0个数）。
参数:
    row：列指标，从0开始。
返回值：
    开始到第一个不为零的指标，指标从0开始。
**********************************************************/
template< class TYPE > unsigned MatrixHxmc<TYPE>::GetLeadingZeros(unsigned row) const
{
    TYPE const ZERO = static_cast<TYPE>(0);
    unsigned column = 0;

    while (ZERO == m_data[row][column])
    {
        ++column;
    }

    return column;
}

/*********************************************************
函数说明：
    矩阵的行按照前导0个数的多少重新排序：前导0少的行
    排在前面，最多的排最后。
    注意：数据没有重排，重排结果放在成员m_sRowOrder。
参数:
    无。
返回值：
    无。
**********************************************************/
template< class TYPE > void MatrixHxmc<TYPE>::Reorder()
{
    //保存每行的前导0个数
    unsigned* zeros = new unsigned[m_sRow];

    for (unsigned row = 0; row < m_sRow; ++row)
    {
        m_sRowOrder[row] = row;
        zeros[row] = GetLeadingZeros(row);
    }

    //进行排序，结果放在m_sRowOrder
    for (unsigned row = 0; row < (m_sRow - 1); ++row)
    {
        unsigned swapRow = row;
        for (unsigned subRow = row + 1; subRow < m_sRow; ++subRow)
        {
            if (zeros[m_sRowOrder[subRow]] < zeros[m_sRowOrder[swapRow]])
            {
                swapRow = subRow;
            }
        }
        unsigned hold = m_sRowOrder[row];
        m_sRowOrder[row] = m_sRowOrder[swapRow];
        m_sRowOrder[swapRow] = hold;
    }

    delete zeros;
}

/*********************************************************
函数说明：
    用给定的值做被除数，给定行的每个元素除以这个数。
参数:
    row：    给定行编号，0开始；
    divisor：被除数。
返回值：
    无。
**********************************************************/
template< class TYPE > void MatrixHxmc<TYPE>::DivideRow(unsigned row, TYPE const& divisor)
{
    for (unsigned column = 0; column < m_sCol; ++column)
    {
        m_data[row][column] /= divisor;
    }

}

/*****************************************************************
函数说明：
    第addRow行（0开始）乘上scale，加到第row行（0开始），
    更新第row行（0开始）。
参数:
    row：    待更新行编号，0开始；
    addRow： 被加的另一行编号，0开始；
    scale：  addRow行的缩放因子。
返回值：
    无。
******************************************************************/
template< class TYPE >
void MatrixHxmc<TYPE>::RowOperation(unsigned row, unsigned addRow, TYPE const& scale)
{
    for (unsigned column = 0; column < m_sCol; ++column)
    {
        m_data[row][column] += m_data[addRow][column] * scale;
    }
}

/******************************************************
函数说明：
    预申请rowNumber*columnNumber矩阵的存储空间。
参数:
    rowNumber：预留空间满足矩阵的行数；
    columnNumber： 预留空间满足矩阵的列数。
返回值：
    无。
********************************************************/
template< class TYPE > void MatrixHxmc<TYPE>::Allocate(unsigned rowNumber, unsigned columnNumber)
{
    // Allocate m_sRowOrder integers.
    m_sRowOrder = new unsigned[rowNumber];

    // Setup matrix sizes.
    m_data.resize(rowNumber);
    for (unsigned row = 0; row < rowNumber; ++row)
    {
        m_data[row].resize(columnNumber);
    }
}

/******************************************************
函数说明：
    释放矩阵所占有的存储空间（主要是行排列的存储空间）；
    矩阵元采用2维std::vector会自动释放存储空间。
参数:
    rowNumber：预留空间满足矩阵的行数；
    columnNumber： 预留空间满足矩阵的列数。
返回值：
    无。
********************************************************/
template< class TYPE > void MatrixHxmc<TYPE>::Deallocate(unsigned rowNumber, unsigned columnNumber)
{
    // Free memory used for storing m_sRowOrder (if there is any).
    if (0 != rowNumber)
    {
        delete[] m_sRowOrder;
    }
}

/***********************************************
说明：
    符号函数，大于等于0返回整数1，小于0则返回-1.
参数：
    number：浮点数。
返回：
    根据数的符号返回1或-1.
************************************************/
template< class TYPE >
inline int MatrixHxmc<TYPE>::Sign(TYPE number)
{
    TYPE const ZERO = static_cast<TYPE>(0);
    if (number < ZERO)
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
template< class TYPE >
void MatrixHxmc<TYPE>::Rotate(MatrixHxmc < TYPE >& matrix, int i, int j, bool& pass, MatrixHxmc < TYPE >& J)
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
    double tan = Sign(tao) / (fabs(tao) + sqrt(1 + pow(tao, 2)));
    double cos = 1 / sqrt(1 + pow(tan, 2));
    double sin = cos * tan;

    MatrixHxmc < TYPE > G(IdentityMatrix < TYPE >(size, size));
    G.put(i, i, cos);
    G.put(i, j, -1 * sin);
    G.put(j, i, sin);
    G.put(j, j, cos);
    matrix = G.getTranspose() * matrix * G;
    J *= G;
}

/*****************************************************************
函数说明：
    构造函数：构造一个rowsParameter行和columnsParameter列的单位矩阵。
参数:
    rowsParameter：    构造矩阵的行数；
    columnsParameter： 构造矩阵的行数；
返回值：
    无。
******************************************************************/
template< class TYPE >
IdentityMatrix<TYPE>::IdentityMatrix(unsigned rowsParameter, unsigned columnsParameter) : MatrixHxmc< TYPE >(rowsParameter, columnsParameter)
{
    TYPE const ZERO = static_cast<TYPE>(0);
    TYPE const ONE = static_cast<TYPE>(1);

    for (unsigned row = 0; row < MatrixHxmc< TYPE >::m_sRow; ++row)
    {
        {
            for (unsigned column = 0; column < MatrixHxmc< TYPE >::m_sCol; ++column)
            {
                if (row == column)
                {
                    MatrixHxmc< TYPE >::m_data[row][column] = ONE;
                }
                else
                {
                    MatrixHxmc< TYPE >::m_data[row][column] = ZERO;
                }
            }

        }
    }
}
