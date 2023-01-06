#pragma once

#include <cmath>
#include <vector>
#include "MatrixHx.h"

using std::vector;
typedef MatrixHx<float> KdTreeDataSet;
typedef MatrixHx<float> QueResData;

struct KDTreeNode
{
    enum
    {
        UNDEFINE_SPLIT = -1       // ��ʾδ����ķָ����
    };

    int nSplit;                  // ��ֱ�ڷָ��ķ��������(���ֵΪUNDEFINE_SPLIT, ��ʾ�ýڵ�ΪҶ�ӽڵ�)
    unsigned int nDataIndex;     // �ڵ����ݵ�����
    KDTreeNode* pParent;         // ���ڵ�
    KDTreeNode* pLeft;           // ���ӽڵ�
    KDTreeNode* pRight;          // �Һ��� �ڵ�
};

typedef vector<KDTreeNode*>  NodeList;
typedef vector<unsigned int> IndexList;

class CKDTreeHxmc
{
    CKDTreeHxmc();
    ~CKDTreeHxmc();

    void BuildTree(KdTreeDataSet& dataSet);
    int  SearchNearestNeighbor(QueResData& data);
    int  SearchKNearestNeighbors(const QueResData& data, unsigned int k, IndexList& vecIndexList);

private:
    KDTreeNode*      m_pRootNode;   // ���ڵ�
    KdTreeDataSet    m_dataSet;     // ���ݼ�����n*kά�ľ����ʾ��

    void  BuildTree(KDTreeNode* pParent, KDTreeNode* pNode, const IndexList& vecIndexList);
    void  TraverseTree(KDTreeNode* pNode, NodeList& vecNodeList);
    void  SearchTree(const QueResData& data, NodeList& searchPath);
    float CalculateDistance(const QueResData& data, unsigned int index);
    void  ClearTree(KDTreeNode*& pNode);
};

