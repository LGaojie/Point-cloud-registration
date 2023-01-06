#include "pch.h"
#include "MatrixHx.h"

//���캯��
/*********************************************************
����˵����
    ȱʡ���캯��������Ա������0��ָ����ΪNULL��
����:
    �ޡ�
����ֵ��
    �ޡ�
**********************************************************/
template<class Type> MatrixHx<Type>::MatrixHx()	: m_nRow(0), m_nCol(0)
{
    m_ppData = NULL;
    m_pData = NULL;
}

/*********************************************************
����˵����
    ���캯��������һ��nRow*nCol�ľ��������б��nRow
    ��ָ���Ӧ��ָ��ÿ�еĵ�һ�����ݡ�
����:
    nRow����������������
    nCol����������������
����ֵ��
    �ޡ�
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
����˵����
    �������캯��������һ����rhsһ���ľ���
����:
    rhs���������ľ���
����ֵ��
    �ޡ�
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

//����
/*********************************************************
����˵����
    �����������ͷ�����ָ��ָ���ڴ棬����������Ϊ0��
����:
    �ޡ�
����ֵ��
    �ޡ�
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

//�������
/*********************************************************
����˵����
    ����������ع�this������rhs��������ͬ��
����:
    rhs���������ľ���
����ֵ��
    thisָ��ľ���
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
����˵����
    ����������ع�this������rhs��������ͬ��
����:
    rhs���������ľ���
����ֵ��
    thisָ��ľ���
**********************************************************/
template<class Type> MatrixHx<Type> MatrixHx<Type>::operator * (const MatrixHx<Type>& B) const
{
    MatrixHx<Type> C;
    MUL(*this, B, C);
    return C;
}

/*********************************************************
����˵����
    []����������ָ�������ݵ�ָ�롣
����:
    row��ָ�����С�
����ֵ��
    ָ�������ݵ�ָ�롣
**********************************************************/
template<class Type> Type*& MatrixHx<Type>::operator[](unsigned int row)
{
    return m_ppData[row];
}

/*********************************************************
����˵����
    []����������ָ�������ݵ�ָ�루constָ�룩��
����:
    row��ָ�����С�
����ֵ��
    ָ�������ݵ�ָ�롣
**********************************************************/
template<class Type> const Type* MatrixHx<Type>::operator[](unsigned int row) const
{
    return m_ppData[row];
}

//��������
/*********************************************************
����˵����
    ��ȡ����ָ������ʸ����1*m_nCol���󣩡�
����:
    row��ָ�����С�
����ֵ��
    ��ʸ������
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
����˵����
    ��ȡ����ָ������ʸ����m_nRow*1���󣩡�
����:
    col��ָ�����С�
����ֵ��
    ��ʸ������
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
����˵����
    ������ˣ�C = A*B��
����:
    A�����˾���
    B�����˾���
    C���������
����ֵ��
    ���A�����������B������������ȷ���false�����򷵻�true��
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
����˵����
    ���þ���Ϊָ�������������������ú�����ȫ���������ȷ��ֵ����
����:
    row�����ú��������
    col�����ú��������
����ֵ��
    �ޡ�
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
����˵����
    ����ĵ�Ե�ĳ˷���C[i,j] = A[i,j] * B[i,j]��
����:
    A�����˵ľ���
    B����������
    C���������������������A,B��ͬ����
����ֵ��
    ��A,B�������������һ��ʱ������false��һ�㷵��true��
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
����˵����
    ����ĵ�Ե�ĳ˷���C[i,j] = A[i,j] * B[i,j]��
����:
    A�����˵ľ���
    B����������
����ֵ��
    ���ص�Ե�˵Ľ������
    ע�⣺��A,B����������������ʱ���᷵��һ������Ϊ0�Ŀվ���
*************************************************************/
template<class Type> MatrixHx<Type> MatrixHx<Type>::DOTMUL(const MatrixHx<Type>& A, const MatrixHx<Type>& B)
{
    MatrixHx<Type> C;
    DOTMUL(A, B, C);

    return C;
}

/*********************************************************
����˵����
    ����ĵ�Ե�ĳ�����C[i,j] = A[i,j] / B[i,j]��
����:
    A�������ľ���
    B����������
    C���������������������A,B��ͬ����
����ֵ��
    ��A,B�������������һ��ʱ������false��һ�㷵��true��
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
����˵����
    ��̬��Ա�����������ת�ã�B = A~��AΪm*n����BΪn*m����
����:
    A����ת�õľ���
    B��A��ת�þ���
����ֵ��
    ����true��
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
����˵����
    �����ת�ã�B = A~��AΪm*n����BΪn*m����
����:
    �ޡ�
����ֵ��
    ���ر�ת�õľ��󣨷��ؾ�����Ϊn*m���󣩡�
**********************************************************/
template<class Type> MatrixHx<Type> MatrixHx<Type>::Transpose() const
{
    MatrixHx<Type> B;
    Transpose(*this, B);

    return B;
}