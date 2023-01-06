#include "pch.h"
#include "KDTreeHxmc.h"

//比较距离的结构
struct DistCompare
{
    unsigned int nDataIndex;    // 数据索引
    float        fDistance;     // 距离值

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
    // 清理树
    if (m_pRootNode != NULL)
    {
        ClearTree(m_pRootNode);
    }

    // 检查数据集
    if (dataSet.m_nRow < 1 || dataSet.m_nCol < 1)
    {
        return;
    }

    // 复制数据集
    m_dataSet = dataSet;

    // 递归构建树
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

    // 只剩一个数据, 则为叶子节点
    if (vecIndexList.size() == 1)
    {
        pNode->nDataIndex = vecIndexList[0];
        pNode->nSplit = KDTreeNode::UNDEFINE_SPLIT;
        pNode->pLeft = NULL;
        pNode->pRight = NULL;

        return;
    }

    // 找出具有最大方差的维度
    float        fMaxDif = 0.0f;        // 标记所有维度上的数据方差的最大值
    unsigned int nMaxDifSplit = 0;      // 标记最大方差的维度索引
    unsigned int nMaxDifIndex = 0;      // 标记最佳分割点索引的列表索引
    for (unsigned int n = 0; n < this->m_dataSet.m_nCol; n++)
    {
        //求第n维的平均值
        float fSplitSum = 0.0f;     // 指定列的数据和
        for (unsigned int i = 0; i < vecIndexList.size(); i++)
        {
            unsigned int m = vecIndexList[i];
            fSplitSum += m_dataSet[m][n];
        }

        float fAveValue = fSplitSum / (float)vecIndexList.size(); // 计算指定列的平均值

        //求方差和分割点
        float fStdErr = 0.0f;     // 指定列的方差值
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

    // 将数据分为左右两部分
    IndexList vecLeftIndex;
    IndexList vecRightIndex;

    vecLeftIndex.reserve(vecIndexList.size() * 2 / 3);   // 预先分配好内存, 防止在push_back过程中多次重复分配提高效率
    vecRightIndex.reserve(vecIndexList.size() * 2 / 3);  // 预先分配好内存, 防止在push_back过程中多次重复分配提高效率

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

    //递归构建左右子树
    // 构建左子树
    if (vecLeftIndex.size() == 0)
    {
        pNode->pLeft = NULL;
    }
    else
    {
        pNode->pLeft = new KDTreeNode;
        BuildTree(pNode, pNode->pLeft, vecLeftIndex);
    }

    // 构建右子树
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