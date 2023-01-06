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
        UNDEFINE_SPLIT = -1       // 表示未定义的分割序号
    };

    int nSplit;                  // 垂直于分割超面的方向轴序号(如果值为UNDEFINE_SPLIT, 表示该节点为叶子节点)
    unsigned int nDataIndex;     // 节点数据的索引
    KDTreeNode* pParent;         // 父节点
    KDTreeNode* pLeft;           // 左孩子节点
    KDTreeNode* pRight;          // 右孩子 节点
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
    KDTreeNode*      m_pRootNode;   // 根节点
    KdTreeDataSet    m_dataSet;     // 数据集：用n*k维的矩阵表示。

    void  BuildTree(KDTreeNode* pParent, KDTreeNode* pNode, const IndexList& vecIndexList);
    void  TraverseTree(KDTreeNode* pNode, NodeList& vecNodeList);
    void  SearchTree(const QueResData& data, NodeList& searchPath);
    float CalculateDistance(const QueResData& data, unsigned int index);
    void  ClearTree(KDTreeNode*& pNode);
};

