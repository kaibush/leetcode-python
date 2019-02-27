429. N叉树的层序遍历
给定一个 N 叉树，返回其节点值的层序遍历。 (即从左到右，逐层遍历)。

例如，给定一个 3叉树 :

 

 

返回其层序遍历:

[
     [1],
     [3,2,4],
     [5,6]
]

"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children
"""
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: Node
        :rtype: List[List[int]]
        """
        if not root:
            return []
        deep = self.getDeepth(root.children)
        ans = []
        for i in range(deep + 1):
            j = self.getNodeLst(root, i)
            if j:
                ans.append(j)
        return ans
    
    def getDeepth(self, root):
        if root == []:
            return 0
        cnt = []
        for i in root:
            n = self.getDeepth(i.children)
            cnt.append(n)
        return max(cnt) + 1
    
    def getNodeLst(self, node, n):
        if not node:
            return []
        if n == 0:
            return [node.val]
        if n == 1:
            res = []
            for i in node.children:
                res.append(i.val)
            return res
        else:
            res = []
            for j in node.children:
                res += self.getNodeLst(j, n-1)
            return res
        