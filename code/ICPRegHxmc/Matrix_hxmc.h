/*************************************************************************
˵����
    �򵥵����Դ����⣬ʵ���˾����������ʸ������������SVD�ֽ⣻LU�ֽ⣻
    �Լ�����Ļ������㡣
    ��C++����ģ��ʵ�֡�
���ߣ�
    Zhang X.C              2022.10.1
��Ȩ��
    HXMC
**************************************************************************/

#pragma once

#include <iostream>
#include <cassert>
#include <vector>

// ��ǰ������λ���������
template< class TYPE > class IdentityMatrix;

// ������Ķ���
template< class TYPE = float > class MatrixHxmc
{

public:
    // Used for matrix concatenation.
    typedef enum
    {
        TO_RIGHT,
        TO_BOTTOM
    } Position;

    // ���������
    MatrixHxmc();
    MatrixHxmc(unsigned rowsParameter, unsigned columnsParameter);
    MatrixHxmc(MatrixHxmc const& copyMatrix, unsigned omittedRow = INT_MAX, unsigned omittedColumn = INT_MAX);
    MatrixHxmc(MatrixHxmc const& copyMatrixA, MatrixHxmc const& copyMatrixB, Position position = TO_RIGHT);
    ~MatrixHxmc();
    // �ӿں���
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
    unsigned  m_sRow;           //�����������
    unsigned  m_sCol;           //�����������
    unsigned* m_sRowOrder;      //�洢�������˳��

    std::vector< std::vector< TYPE > > m_data; // ��������ݣ���STL�Ķ�άʸ���洢

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
��˵����
    ����һ��������λ������ࡣ
******************************************************************/
template< class TYPE >
class IdentityMatrix : public MatrixHxmc< TYPE >
{
public:
    IdentityMatrix(unsigned rowsParameter, unsigned columnsParameter);
};

/*****************************************************************
����˵����
    �ӱ�׼ģ������һ���������Ϊ�ַ����ĺ�����
���룺
    matrix���������
�����
    stream������ַ�������
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
˵����
    �򵥵����Դ����⣬ʵ���˾����������ʸ������������SVD�ֽ⣻LU�ֽ⣻
    �Լ�����Ļ������㡣
    ��C++����ģ��ʵ�֡�
���ߣ�
    Zhang Xiaochun              2022.10.1
��Ȩ��
    HXMC
**************************************************************************/

#include "pch.h"
#include "matrix_hxmc.h"
#include <cmath>
#include <map>

using namespace std;

const double THRESHOLD = 1E-8;
const int    ITERATION = 30;   //��������������

/*********************************************************
����˵����
    ���캯��������һ���յľ���
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
����˵����
    ���캯��������һ��rows�С�cols�еľ���ÿ������ԪΪ0��
������
    rows�������������
    cols�������������
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
����˵����
    ���캯�������в�����/���в�����ʡ��ʱ������һ���뱻Copy����һ���ľ���
              ��û��ʡ��ʱ������ȱ�Copy������һ�к�һ�еľ���
������
    copyMatrix�������Ƶľ���
    sOmitRow����ȥ���еı�ţ�0��ʼ����
    sOmitCol����ȥ���еı�ţ�0��ʼ����
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
����˵����
    ���캯������A��B����������һ��ƴ�Ӿ��󣺵�pos������RIGHTʱ���¹������
    ��B��������A���ұߣ���pos������BOTTOMʱ��ƴ���ڵײ�����
������
    copyMatrixA������A��
    copyMatrixB������B��
    pos��Bƴ�Ӿ����λ�á�
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
����˵����
    ����������
������
    �ޡ�
************************************************/
template< class TYPE > MatrixHxmc<TYPE>::~MatrixHxmc()
{
    // Release memory.
    Deallocate(m_sRow, m_sCol);
}

/***********************************************
����˵����
    ��þ����������
������
    �ޡ�
���أ�
    �����������
************************************************/
template< class TYPE > unsigned MatrixHxmc<TYPE>::GetRows() const
{
    return m_sRow;
}

/***********************************************
����˵����
    ��þ����������
������
    �ޡ�
���أ�
    �����������
************************************************/
template< class TYPE > unsigned MatrixHxmc<TYPE>::GetColumns() const
{
    return m_sCol;
}

/***********************************************
����˵����
    ��ø����С��о���Ԫ��
������
    row�������б�ţ�0��ʼ����
    column�������б�ţ�0��ʼ����
���أ�
    ���и��о���Ԫ��ֵ��
************************************************/
template< class TYPE > TYPE MatrixHxmc<TYPE>::Get(unsigned row, unsigned column) const
{
    assert(row < m_sRow);
    assert(column < m_sCol);

    return m_data[row][column];
}

/*****************************************************
����˵����
    �����LU�ֽ⣺This will create matrices L and U
    such that A=LxU��
������
    row�������б�ţ�0��ʼ����
    column�������б�ţ�0��ʼ����
���أ�
    ���и��о���Ԫ��ֵ��
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
����˵����
    ���þ���ָ���С��е�ֵ��
������
    row�������б�ţ�0��ʼ����
    column�������б�ţ�0��ʼ����
    value�����õ�ֵ��
���أ�
    �ޡ�
******************************************************/
template< class TYPE >void MatrixHxmc<TYPE>::Put(unsigned row, unsigned column, TYPE const& value)
{
    assert(row < m_sRow);
    assert(column < m_sCol);

    m_data[row][column] = value;
}

/********************************************************************
����˵����
    ��ȡ����ָ��λ�ã����У����Ӿ���
    ע�⣺startRow = endRow����startColumn = endColumn����ʾ������ȡ
          һ�еĲ��֣���һ�еĲ��֣���
������
    startRow���Ӿ����ڱ������п�ʼ���У�
    endRow���Ӿ����ڱ������н������У�
    startColumn���Ӿ����ڱ������п�ʼ���У�
    endColumn���Ӿ����ڱ������н������У�
    newOrder����Ϊ�գ������������򣻷�����newOrderָ����˳�������С�
���أ�
    �Ӿ���
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
����˵����
    ��ȡ�����column��(m_sCol*1���󣬼���ʸ��)��
����:
    column����ָ�꣬��0��ʼ��
����ֵ��
    �Ӿ��󣨵�column�е���ʸ������
**********************************************************/
template< class TYPE >MatrixHxmc<TYPE> MatrixHxmc<TYPE>::GetColumn(unsigned column)
{
    return GetSubMatrix(0, m_sRow - 1, column, column);
}

/*********************************************************
����˵����
    ��ȡ�����row��(1*m_sRow���󣬼���ʸ��)��
����:
    row����ָ�꣬��0��ʼ��
����ֵ��
    �Ӿ��󣨵�row�е���ʸ������
**********************************************************/
template< class TYPE >MatrixHxmc<TYPE> MatrixHxmc<TYPE>::GetRow(unsigned row)
{
    return GetSubMatrix(row, row, 0, m_sCol - 1);
}

//-------------------------------------------------------------
    // Place matrix in reduced row echelon form.
    //-------------------------------------------------------------
/*********************************************************
����˵����
    ��ȡ�����row��(1*m_sRow���󣬼���ʸ��)��
����:
    row����ָ�꣬��0��ʼ��
����ֵ��
    �Ӿ��󣨵�row�е���ʸ������
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
����˵����
    ��þ��������ʽֵ���ݹ鷽��ʵ�֣���
����:
    �ޡ�
����ֵ��
    ����ʽֵ��
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
����˵����
    �����������Ӧ����Ԫ֮���ĺ͡�
����:
    otherMatrix���뱾������е������ľ���
����ֵ��
    �����ʸ���������չ����
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
����˵����
    ����һ���������ת�þ���
����:
    �ޡ�
����ֵ��
    ת�þ���
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
˵����
    ת�ñ�����
����:
    �ޡ�
����ֵ��
    �ޡ�
*****************************************/
template< class TYPE > void MatrixHxmc<TYPE>::Transpose()
{
    *this = GetTranspose();
}

/***************************************
˵����
    ����һ������������
����:
    �ޡ�
����ֵ��
    �����
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
˵����
   ��������任���������
����:
    �ޡ�
����ֵ��
    �ޡ�
*****************************************/
template< class TYPE > void  MatrixHxmc<TYPE>::Invert()
{
    *this = GetInverse();
} // Invert


// Operators.
/***************************************
˵����
    ����ļӣ�R = this + otherMatrix��
����:
    otherMatrix�����ӵľ���
����ֵ��
    ��������ĺ͡�
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
˵����
    ����ļӣ�R = *this + otherMatrix��
����:
    otherMatrix�����ӵľ���
����ֵ��
    ��������ĺ͡�
*****************************************/
template< class TYPE > MatrixHxmc<TYPE> const& MatrixHxmc<TYPE>::operator += (MatrixHxmc const& otherMatrix)
{
    *this = *this + otherMatrix;
    return *this;
}

/***************************************
˵����
    ����ļӣ�R = *this - otherMatrix��
����:
    otherMatrix�������ľ���
����ֵ��
    ��������Ĳ
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
˵����
    �Լ��ľ������㣺*this = *this - otherMatrix��
����:
    otherMatrix�������ľ���
����ֵ��
    �Լ����������Ĳ
**************************************************/
template< class TYPE > MatrixHxmc<TYPE> const& MatrixHxmc<TYPE>::operator -= (MatrixHxmc const& otherMatrix)
{
    *this = *this - otherMatrix;
    return *this;
}

/*******************************************
˵����
    ����ĳ˷���R = *this * otherMatrix��
����:
    otherMatrix�����˵ľ���
����ֵ��
    ��������ĳˡ�
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
˵����
    �Գ˷��ľ������㣺*this = *this * otherMatrix��
����:
    otherMatrix�����˵ľ���
����ֵ��
    �Լ����������ĳˡ�
**************************************************/
template< class TYPE >MatrixHxmc<TYPE> const& MatrixHxmc<TYPE>::operator *= (MatrixHxmc const& otherMatrix)
{
    *this = *this * otherMatrix;
    return *this;
}

/*******************************************
˵����
    ���������ĳ˷���R = *this * scalar��
����:
    scalar�����˵�����
����ֵ��
    ÿ������Ԫ���ϸ�����
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
˵����
    ������������ĳ˷���*this = *this * scalar��
����:
    scalar�����˵�������������
����ֵ��
    ÿ������Ԫ���ϸ�����
*********************************************/
template< class TYPE > MatrixHxmc<TYPE> const& MatrixHxmc<TYPE>::operator *= (TYPE const& scalar)
{
    *this = *this * scalar;
    return *this;
}

/*************************************************
˵����
    �����=����otherMatrix�����ݿ�����*this��
����:
    otherMatrix���������ľ���
����ֵ��
    *this��
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
˵����
    �����=�������������п������ݵ�����Ԫ��
    ע�⣺��������ά�ġ�������������һά�ģ��ٶ�������
    ���Դ洢�ģ���������m_sRow*m_sCol����

    Example for 3x2 matrix:
    int const data[ 3 * 2 ] =
    {
       1, 2, 3,
       4, 5, 6
    };
    MatrixHxmc< int > matrix( 3, 2 )��
����:
    data��ָ������һά���顣
����ֵ��
    ÿ������Ԫ���ϸ�����
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
˵����
    �ж����������Ƿ�ȫ�ȡ�
����:
    value������
����ֵ��
    ��������ÿ��Ԫ����ȣ�����true�����򷵻�false��
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
˵����
    �ж����������Ƿ񲻵ȣ�ֻҪһ��Ԫ�ز����������
����:
    value������
����ֵ��
    �������ֻҪһ��Ԫ����ȣ�����true�����򷵻�false��
*****************************************************/
template< class TYPE > bool MatrixHxmc<TYPE>::operator != (MatrixHxmc const& value) const
{
    return !(*this == value);
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
        if (pass)   //���ǶԽ�Ԫ��ȫ����Ϊ0ʱ�����˳�
        {
            break;
        }
    }

    //cout << "����������" << ITERATION - iteration << endl;

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
˵����
    ˫��Jacobi��ת���������n*n�ԳƷ����������������ʸ����
    ����ֵ�ǽ������еģ�����ʸ���ǹ�һ���ġ�
������
    a�������n*n�Գƾ���
    n����������ά�ȣ�
    w�������n������ֵ��
    v�������n*n����ʸ��������ʸ������

���أ�
    ����������������0�����򷵻�1��
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

    MatrixHxmc<TYPE> B = A.GetTranspose() * A;                        //A��ת�ó���A���õ�һ���Գƾ���B
    MatrixHxmc<TYPE> J(IdentityMatrix<TYPE>(columns, columns));
    std::vector <TYPE>S(columns);

    Jacobi(B, columns, S, J);  //��B������ֵ����������
    for (int i = 0; i < S.size(); ++i)
    {
        S[i] = sqrt(S[i]);   //B������ֵ������õ�A������ֵ
    }

    /*����ֵ���ݼ����򣬶�Ӧ��V�е���������ҲҪ������ */
    multimap < TYPE, int > eigen;

    for (int i = 0; i < S.size(); ++i)   //��multimap�ڲ��Զ���key��������
    {
        eigen.insert(make_pair(S[i], i));
    }

    auto iter = --eigen.end();
    //multimap < TYPE, int >::const_iterator iter = --eigen.end();

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
            V.Put(row, i, J.Get(row, index));
        }
    }

    assert(num_eig <= rows);

    for (int i = 0; i < num_eig; ++i)
    {
        MatrixHxmc<TYPE> vi = V.GetColumn(i); //��ȡV�ĵ�i��
        TYPE sigma = E[i];
        MatrixHxmc<TYPE> ui(rows, 1);

        ui = A * vi;
        for (int j = 0; j < rows; ++j)
        {
            U.Put(j, i, ui.Get(j, 0) / sigma);
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
template< class TYPE >
void MatrixHxmc<TYPE>::Orthogonal(MatrixHxmc<TYPE>& matrix, int i, int j, bool& pass, MatrixHxmc<TYPE>& V)
{
    assert(i < j);

    MatrixHxmc<TYPE> Ci = matrix.GetColumn(i);
    MatrixHxmc<TYPE> Cj = matrix.GetColumn(j);

    TYPE ele = ((Ci.GetTranspose()) * Cj).Get(0, 0);

    if (fabs(ele) < THRESHOLD)          //i,j�����Ѿ�����
    {
        return;
    }

    int rows = matrix.GetRows();
    int columns = matrix.GetColumns();

    pass = false;
    TYPE ele1 = ((Ci.GetTranspose()) * Ci).Get(0, 0);
    TYPE ele2 = ((Cj.GetTranspose()) * Cj).Get(0, 0);

    /*ֻҪÿ����תǰ���ѷ�������з���ǰ�棬�Ϳ��Ա�֤���������ֵ�ǵݼ������*/
    if (ele1 < ele2)            //���matrix��i�еķ���С�ڵ�j�У��򽻻�����.ͬʱV����Ҳ����Ӧ�ĵ���
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
                Orthogonal(matrix, i, j, pass, V);      //������εĵ���������V���������
            }
        }
        if (pass)   //���������ж�����ʱ�˳�����
        {
            break;
        }
    }
    //cout << "����������" << ITERATION - iteration << endl;
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
template< class TYPE >
int MatrixHxmc<TYPE>::SVN(MatrixHxmc<TYPE>& matrix, MatrixHxmc<TYPE>& S, MatrixHxmc<TYPE>& U, MatrixHxmc<TYPE>& V)
{
    int rows = matrix.GetRows();
    int columns = matrix.GetColumns();

    assert(rows <= columns);

    Hestens_Jacobi(matrix, V);

    vector<TYPE> E(columns);          //E�д������ֵ
    int none_zero = 0;                //��¼��0����ֵ�ĸ���

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
    * U����ĺ�(rows-none_zero)���Լ�V�ĺ�(columns-none_zero)�оͲ������ˣ�����Ĭ��ֵ0��
    * ��������ֵ�ֽ�A=U*Sigma*V^T������ֻ��ҪU��ǰr�У�V^T��ǰr�У���V��ǰr�У����Ϳ��Իָ�A�ˡ�r��A����
    */
    for (int row = 0; row < rows; ++row)
    {
        S.Put(row, row, E[row]);

        for (int col = 0; col < none_zero; ++col)
        {
            U.Put(row, col, matrix.Get(row, col) / E[col]);
        }
    }

    return none_zero;   //������ֵ�ĸ����༴�������
}

//Protected
/*********************************************************
����˵����
    ��ȡ�����row���е�һ����Ϊ0���б�ţ��е�ǰ��0��������
����:
    row����ָ�꣬��0��ʼ��
����ֵ��
    ��ʼ����һ����Ϊ���ָ�ָ꣬���0��ʼ��
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
����˵����
    ������а���ǰ��0�����Ķ�����������ǰ��0�ٵ���
    ����ǰ�棬���������
    ע�⣺����û�����ţ����Ž�����ڳ�Աm_sRowOrder��
����:
    �ޡ�
����ֵ��
    �ޡ�
**********************************************************/
template< class TYPE > void MatrixHxmc<TYPE>::Reorder()
{
    //����ÿ�е�ǰ��0����
    unsigned* zeros = new unsigned[m_sRow];

    for (unsigned row = 0; row < m_sRow; ++row)
    {
        m_sRowOrder[row] = row;
        zeros[row] = GetLeadingZeros(row);
    }

    //�������򣬽������m_sRowOrder
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
����˵����
    �ø�����ֵ���������������е�ÿ��Ԫ�س����������
����:
    row��    �����б�ţ�0��ʼ��
    divisor����������
����ֵ��
    �ޡ�
**********************************************************/
template< class TYPE > void MatrixHxmc<TYPE>::DivideRow(unsigned row, TYPE const& divisor)
{
    for (unsigned column = 0; column < m_sCol; ++column)
    {
        m_data[row][column] /= divisor;
    }

}

/*****************************************************************
����˵����
    ��addRow�У�0��ʼ������scale���ӵ���row�У�0��ʼ����
    ���µ�row�У�0��ʼ����
����:
    row��    �������б�ţ�0��ʼ��
    addRow�� ���ӵ���һ�б�ţ�0��ʼ��
    scale��  addRow�е��������ӡ�
����ֵ��
    �ޡ�
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
����˵����
    Ԥ����rowNumber*columnNumber����Ĵ洢�ռ䡣
����:
    rowNumber��Ԥ���ռ���������������
    columnNumber�� Ԥ���ռ���������������
����ֵ��
    �ޡ�
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
����˵����
    �ͷž�����ռ�еĴ洢�ռ䣨��Ҫ�������еĴ洢�ռ䣩��
    ����Ԫ����2άstd::vector���Զ��ͷŴ洢�ռ䡣
����:
    rowNumber��Ԥ���ռ���������������
    columnNumber�� Ԥ���ռ���������������
����ֵ��
    �ޡ�
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
˵����
    ���ź��������ڵ���0��������1��С��0�򷵻�-1.
������
    number����������
���أ�
    �������ķ��ŷ���1��-1.
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
����˵����
    ���캯��������һ��rowsParameter�к�columnsParameter�еĵ�λ����
����:
    rowsParameter��    ��������������
    columnsParameter�� ��������������
����ֵ��
    �ޡ�
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
