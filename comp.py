class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


n1 = TreeNode(1)
n2 = TreeNode(2)
n3 = TreeNode(3)
n4 = TreeNode(4)
n5 = TreeNode(5)
n6 = TreeNode(6)
n7 = TreeNode(7)
n8 = TreeNode(8)
n9 = TreeNode(9)
n10 = TreeNode(10)
n11 = TreeNode(11)
n12 = TreeNode(12)
n13 = TreeNode(13)
n14 = TreeNode(14)
n15 = TreeNode(15)
n1.left = n2
n1.right = n3
n2.left = n4
n2.right = n5
n3.left = n6
n3.right = n7
n4.left = n8
n4.right = n9
n5.left = n10
n5.right = n11
n6.left = n12
n6.right = n13
n7.left = n14
n7.right = n15


# class Solution:
#
#     def maxValue(self, root: TreeNode, k: int) -> int:
#         return self.search(root, k, k)
#
#     def search(self, root, curr_concat_cnt, k):
#         """
#         :param root:
#         :param curr_concat_cnt:
#         :param k:
#         :return:
#         """
#         if not root:
#             return 0
#         v00, v11, v22, v33 = 0, 0, 0, 0
#         if curr_concat_cnt == 0:
#             v00 = self.search(root.left, k, k) + self.search(root.right, k, k)
#         elif curr_concat_cnt == 1:
#             v1 = root.val + self.search(root.left, curr_concat_cnt-1, k) + self.search(root.right, 0, k)
#             v2 = root.val + self.search(root.left, 0, k) + self.search(root.right, curr_concat_cnt-1, k)
#             v3 = root.val + self.search(root.left, 0, k) + self.search(root.right, 0, k)
#             v11 = max(v1, max(v2, v3))
#         elif curr_concat_cnt >= 2:
#             v1 = root.val + self.search(root.left, curr_concat_cnt-1, k) + self.search(root.right, curr_concat_cnt-2, k)
#             v2 = root.val + self.search(root.left, curr_concat_cnt-2, k) + self.search(root.right, curr_concat_cnt-1, k)
#             v22 = max(v1, v2)
#         v33 = self.search(root.left, k, k) + self.search(root.right, k, k)
#         print(root.val, v00, v11, v22, v33)
#         return max(max(v33, v22), max(v11, v00))


class Solution:
    def storeWater(self, bucket, vat):
        pass


if __name__ == '__main__':
    sol = Solution()
    print(sol.storeWater([9,0,1], [0,2,2]))
    # print(sum([i for i in range(1, 16)]))
    # print(8+16+9+10+11+12+13+14+15+2+3)
