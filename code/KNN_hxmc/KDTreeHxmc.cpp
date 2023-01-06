#include "pch.h"
#include "KDTreeHxmc.h"

//�ȽϾ���Ľṹ
struct DistCompare
{
    unsigned int nDataIndex;    // ��������
    float        fDistance;     // ����ֵ

    bool operator < (const DistCompare& B) const
    {
        if (this->fDistance < B.fDistance)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
};

CKDTreeHxmc::CKDTreeHxmc()
{
    m_pRootNode = NULL;
}

CKDTreeHxmc::~CKDTreeHxmc()
{
    ClearTree(m_pRootNode);
}

void CKDTreeHxmc::ClearTree(KDTreeNode*&  pNode)
{
    if (pNode == NULL)
    {
        return;
    }

    ClearTree(pNode->pLeft);
    ClearTree(pNode->pRight);
    delete pNode;

    pNode = NULL;
}

void CKDTreeHxmc::BuildTree(KdTreeDataSet& dataSet)
{
    // ������
    if (m_pRootNode != NULL)
    {
        ClearTree(m_pRootNode);
    }

    // ������ݼ�
    if (dataSet.m_nRow < 1 || dataSet.m_nCol < 1)
    {
        return;
    }

    // �������ݼ�
    m_dataSet = dataSet;

    // �ݹ鹹����
    m_pRootNode = new KDTreeNode();

    IndexList  vecIndexList(dataSet.m_nRow);

    for (unsigned int i = 0; i < vecIndexList.size(); i++)
    {
        vecIndexList[i] = i;
    }

    BuildTree(NULL, m_pRootNode, vecIndexList);
}

void CKDTreeHxmc::BuildTree(KDTreeNode* pParent, KDTreeNode* pNode, const IndexList& vecIndexList)
{
    if (pNode == NULL)
    {
        return;
    }

    pNode->pParent = pParent;

    // ֻʣһ������, ��ΪҶ�ӽڵ�
    if (vecIndexList.size() == 1)
    {
        pNode->nDataIndex = vecIndexList[0];
        pNode->nSplit = KDTreeNode::UNDEFINE_SPLIT;
        pNode->pLeft = NULL;
        pNode->pRight = NULL;

        return;
    }

    // �ҳ�������󷽲��ά��
    float        fMaxDif = 0.0f;        // �������ά���ϵ����ݷ�������ֵ
    unsigned int nMaxDifSplit = 0;      // �����󷽲��ά������
    unsigned int nMaxDifIndex = 0;      // �����ѷָ���������б�����
    for (unsigned int n = 0; n < this->m_dataSet.m_nCol; n++)
    {
        //���nά��ƽ��ֵ
        float fSplitSum = 0.0f;     // ָ���е����ݺ�
        for (unsigned int i = 0; i < vecIndexList.size(); i++)
        {
            unsigned int m = vecIndexList[i];
            fSplitSum += m_dataSet[m][n];
        }

        float fAveValue = fSplitSum / (float)vecIndexList.size(); // ����ָ���е�ƽ��ֵ

        //�󷽲�ͷָ��
        float fStdErr = 0.0f;     // ָ���еķ���ֵ
        float fMinDif = abs(fAveValue - m_dataSet[vecIndexList[0]][n]);
        unsigned int nListIndex = 0;

        for (unsigned int i = 0; i < vecIndexList.size(); i++)
        {
            unsigned int m = vecIndexList[i];
            float fDif = fAveValue - m_dataSet[m][n];
            fStdErr += fDif * fDif;

            if (abs(fDif) < fMinDif)
            {
                fMinDif = abs(fDif);
                nListIndex = i;
            }
        }
        
        if (fStdErr > fMaxDif)
        {
            fMaxDif = fStdErr;
            nMaxDifSplit = n;
            nMaxDifIndex = nListIndex;
        }
    }

    pNode->nSplit = nMaxDifSplit;
    pNode->nDataIndex = vecIndexList[nMaxDifIndex];

    // �����ݷ�Ϊ����������
    IndexList vecLeftIndex;
    IndexList vecRightIndex;

    vecLeftIndex.reserve(vecIndexList.size() * 2 / 3);   // Ԥ�ȷ�����ڴ�, ��ֹ��push_back�����ж���ظ��������Ч��
    vecRightIndex.reserve(vecIndexList.size() * 2 / 3);  // Ԥ�ȷ�����ڴ�, ��ֹ��push_back�����ж���ظ��������Ч��

    //
    unsigned int nMidIdx = vecIndexList[nMaxDifIndex];
    for (unsigned int i = 0; i < vecIndexList.size(); i++)
    {
        if (i == nMaxDifIndex)
        {
            continue;
        }

        
        unsigned int nIndex = vecIndexList[i];

        if (m_dataSet[nIndex][nMaxDifSplit] <= m_dataSet[nMidIdx][nMaxDifSplit])
        {
            vecLeftIndex.push_back(vecIndexList[i]);
        }
        else
        {
            vecRightIndex.push_back(vecIndexList[i]);
        }
    }

    //�ݹ鹹����������
    // ����������
    if (vecLeftIndex.size() == 0)
    {
        pNode->pLeft = NULL;
    }
    else
    {
        pNode->pLeft = new KDTreeNode;
        BuildTree(pNode, pNode->pLeft, vecLeftIndex);
    }

    // ����������
    if (vecRightIndex.size() == 0)
    {
        pNode->pRight = NULL;
    }
    else
    {
        pNode->pRight = new KDTreeNode;
        BuildTree(pNode, pNode->pRight, vecRightIndex);
    }
}