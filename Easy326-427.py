# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)

326. 3的幂
给定一个整数，写一个函数来判断它是否是 3 的幂次方。

示例 1:

输入: 27
输出: true

示例 2:

输入: 0
输出: false

示例 3:

输入: 9
输出: true

示例 4:

输入: 45
输出: false
class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        while n > 1:
            if n % 3 != 0:
                return False
            n /= 3
        if n== 1:
            return True
        return False
    
342. 4的幂
给定一个整数 (32 位有符号整数)，请编写一个函数来判断它是否是 4 的幂次方。

示例 1:

输入: 16
输出: true

示例 2:

输入: 5
输出: false

进阶：
你能不使用循环或者递归来完成本题吗？
class Solution(object):
    def isPowerOfFour(self, num):
        """
        :type num: int
        :rtype: bool
        """
        while num > 1:
            if num % 4 != 0:
                return False
            num /= 4
        if num== 1:
            return True
        return False
        
344. 反转字符串
编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。

不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。

你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。

 

示例 1：

输入：["h","e","l","l","o"]
输出：["o","l","l","e","h"]

示例 2：

输入：["H","a","n","n","a","h"]
输出：["h","a","n","n","a","H"]

class Solution(object):
    def reverseString(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        i, j = 0, len(s) - 1
        while i < j:
            s[i], s[j] = s[j], s[i]
            i += 1
            j -= 1
            
345. 反转字符串中的元音字母编写一个函数，以字符串作为输入，反转该字符串中的元音字母。

示例 1:

输入: "hello"
输出: "holle"

示例 2:

输入: "leetcode"
输出: "leotcede"

说明:
元音字母不包含字母"y"。

class Solution(object):
    def reverseVowels(self, s):
        """
        :type s: str
        :rtype: str
        """
        vowels = 'aeiouAEIOU'
        i, j = 0, len(s) - 1
        lst = list(s)
        while i < j:
            if s[i] not in vowels:
                i += 1
            if s[j] not in vowels:
                j -= 1
            if s[i] in vowels and s[j] in vowels:
                lst[i], lst[j] = lst[j], lst[i]
                i += 1
                j -= 1
        return ''.join(lst)
        
349. 两个数组的交集
给定两个数组，编写一个函数来计算它们的交集。

示例 1:

输入: nums1 = [1,2,2,1], nums2 = [2,2]
输出: [2]

示例 2:

输入: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出: [9,4]

说明:

    输出结果中的每个元素一定是唯一的。
    我们可以不考虑输出结果的顺序。

class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        return list(set(nums1) & set(nums2))
        
350. 两个数组的交集 II
给定两个数组，编写一个函数来计算它们的交集。

示例 1:

输入: nums1 = [1,2,2,1], nums2 = [2,2]
输出: [2,2]

示例 2:

输入: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出: [4,9]

说明：

    输出结果中每个元素出现的次数，应与元素在两个数组中出现的次数一致。
    我们可以不考虑输出结果的顺序。
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        def helper(nums):
            cnt = {}
            for i in nums:
                cnt[i] = cnt.get(i, 0) + 1
            return cnt
        
        d = helper(nums1)
        d2 = helper(nums2)
        
        ans = []
        for k in d:
            if k in d2:
                m = min(d2[k], d[k])
                for _ in range(m):
                    ans.append(k)
        return ans
        
367. 有效的完全平方数
给定一个正整数 num，编写一个函数，如果 num 是一个完全平方数，则返回 True，否则返回 False。

说明：不要使用任何内置的库函数，如  sqrt。

示例 1：

输入：16
输出：True

示例 2：

输入：14
输出：False

class Solution(object):
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        for i in xrange(1, num + 1):
            if i * i > num:
                break
            elif i * i < num:
                continue
            else:
                return True
        return False
        
371. 两整数之和
不使用运算符 + 和 - ​​​​​​​，计算两整数 ​​​​​​​a 、b ​​​​​​​之和。

示例 1:

输入: a = 1, b = 2
输出: 3

示例 2:

输入: a = -2, b = 3
输出: 1

class Solution(object):
    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        MAX = 0x7fffffff
        MIN = 0x80000000
        mask = 0xFFFFFFFF
        while b != 0:
            a, b = (a ^ b) & mask, ((a & b) << 1)
        return a if a <= MAX else ~(a ^ mask)
        
374. 猜数字大小
我们正在玩一个猜数字游戏。 游戏规则如下：
我从 1 到 n 选择一个数字。 你需要猜我选择了哪个数字。
每次你猜错了，我会告诉你这个数字是大了还是小了。
你调用一个预先定义好的接口 guess(int num)，它会返回 3 个可能的结果（-1，1 或 0）：

-1 : 我的数字比较小
 1 : 我的数字比较大
 0 : 恭喜！你猜对了！

示例 :

输入: n = 10, pick = 6
输出: 6

# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num):

class Solution(object):
    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        lo, hi = 0, n + 1
        while lo < hi:
            mid = lo + (hi - lo) / 2
            if guess(mid) == 1:
                lo = mid + 1
            elif guess(mid) == -1:
                hi = mid
            else:
                return mid
        
383. 赎金信
给定一个赎金信 (ransom) 字符串和一个杂志(magazine)字符串，判断第一个字符串ransom能不能由第二个字符串magazines里面的字符构成。如果可以构成，返回 true ；否则返回 false。

(题目说明：为了不暴露赎金信字迹，要从杂志上搜索各个需要的字母，组成单词来表达意思。)

注意：

你可以假设两个字符串均只含有小写字母。

canConstruct("a", "b") -> false
canConstruct("aa", "ab") -> false
canConstruct("aa", "aab") -> true

class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        d = {}
        for i in magazine:
            d[i] = d.get(i, 0) + 1
        for s in ransomNote:
            if s not in d:
                return False
            else:
                if d[s] == 0:
                    return False
                d[s] -= 1
        return True
        
387. 字符串中的第一个唯一字符
给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。

案例:

s = "leetcode"
返回 0.

s = "loveleetcode",
返回 2.

 

注意事项：您可以假定该字符串只包含小写字母。
class Solution(object):
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        d = {}
        for i in s:
            d[i] = d.get(i, 0) + 1
        for index, i in enumerate(s):
            if d[i] == 1:
                return index
        return -1
        
389. 找不同
给定两个字符串 s 和 t，它们只包含小写字母。

字符串 t 由字符串 s 随机重排，然后在随机位置添加一个字母。

请找出在 t 中被添加的字母。

 

示例:

输入：
s = "abcd"
t = "abcde"

输出：
e

解释：
'e' 是那个被添加的字母。

from collections import Counter

class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        d = Counter(s)
        d1 = Counter(t)
        return list(d1-d)[0]
        
400. 第N个数字
在无限的整数序列 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...中找到第 n 个数字。

注意:
n 是正数且在32为整形范围内 ( n < 231)。

示例 1:

输入:
3

输出:
3

示例 2:

输入:
11

输出:
0

401. 二进制手表
二进制手表顶部有 4 个 LED 代表小时（0-11），底部的 6 个 LED 代表分钟（0-59）。

每个 LED 代表一个 0 或 1，最低位在右侧。

例如，上面的二进制手表读取 “3:25”。

给定一个非负整数 n 代表当前 LED 亮着的数量，返回所有可能的时间。

案例:

输入: n = 1
返回: ["1:00", "2:00", "4:00", "8:00", "0:01", "0:02", "0:04", "0:08", "0:16", "0:32"]

 

注意事项:

    输出的顺序没有要求。
    小时不会以零开头，比如 “01:00” 是不允许的，应为 “1:00”。
    分钟必须由两位数组成，可能会以零开头，比如 “10:2” 是无效的，应为 “10:02”。

class Solution(object):
    def readBinaryWatch(self, num):
        """
        :type num: int
        :rtype: List[str]
        """
        return ['%d:%02d' % (h, m) for h in range(12) for m in range(60) if (bin(h) + bin(m)).count('1') == num]
        
404. 左叶子之和
计算给定二叉树的所有左叶子之和。

示例：

    3
   / \
  9  20
    /  \
   15   7

在这个二叉树中，有两个左叶子，分别是 9 和 15，所以返回 24

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def helper(node, isLeft):
            if not node:
                return None
            left = helper(node.left, True)
            right = helper(node.right, False)
            ret = 0
            if left is None and right is None and isLeft:
                return node.val
            if left:
                ret += left
            if right:
                ret += right
            return ret
        ret = helper(root, False)
        if ret:
            return ret
        return 0

405. 数字转换为十六进制数
给定一个整数，编写一个算法将这个数转换为十六进制数。对于负整数，我们通常使用 补码运算 方法。

注意:

    十六进制中所有字母(a-f)都必须是小写。
    十六进制字符串中不能包含多余的前导零。如果要转化的数为0，那么以单个字符'0'来表示；对于其他情况，十六进制字符串中的第一个字符将不会是0字符。 
    给定的数确保在32位有符号整数范围内。
    不能使用任何由库提供的将数字直接转换或格式化为十六进制的方法。

示例 1：

输入:
26

输出:
"1a"

示例 2：

输入:
-1

输出:
"ffffffff"

class Solution(object):
    def toHex(self, num):
        """
        :type num: int
        :rtype: str
        """
        if not num:
            return '0'
        result = []
        hexStr = '0123456789abcdef'
        while num and len(result) != 8:
            h = num & 15
            result.append(hexStr[h])
            num >>= 4
        return ''.join(result[::-1])

409. 最长回文串
给定一个包含大写字母和小写字母的字符串，找到通过这些字母构造成的最长的回文串。

在构造过程中，请注意区分大小写。比如 "Aa" 不能当做一个回文字符串。

注意:
假设字符串的长度不会超过 1010。

示例 1:

输入:
"abccccdd"

输出:
7

解释:
我们可以构造的最长的回文串是"dccaccd", 它的长度是 7。


from collections import Counter

class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        d = Counter(s)
        ans = 0
        for i in set(s):
            if d[i] % 2 == 0:
                ans += d[i]
                del d[i]
            else:
                ans += d[i] - 1
        if d:
            return ans + 1
        return ans
        
412. Fizz Buzz
写一个程序，输出从 1 到 n 数字的字符串表示。

1. 如果 n 是3的倍数，输出“Fizz”；

2. 如果 n 是5的倍数，输出“Buzz”；

3.如果 n 同时是3和5的倍数，输出 “FizzBuzz”。

示例：

n = 15,

返回:
[
    "1",
    "2",
    "Fizz",
    "4",
    "Buzz",
    "Fizz",
    "7",
    "8",
    "Fizz",
    "Buzz",
    "11",
    "Fizz",
    "13",
    "14",
    "FizzBuzz"
]

class Solution(object):
    def fizzBuzz(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        ans = []
        for i in xrange(1, n+1):
            k = i % 3
            j = i % 5
            if k == 0 and j != 0:
                ans.append('Fizz')
            elif k != 0 and j == 0:
                ans.append('Buzz')
            elif k == 0 and j == 0:
                ans.append('FizzBuzz')
            else:
                ans.append(str(i))
        return ans
        
414. 第三大的数
给定一个非空数组，返回此数组中第三大的数。如果不存在，则返回数组中最大的数。要求算法时间复杂度必须是O(n)。

示例 1:

输入: [3, 2, 1]

输出: 1

解释: 第三大的数是 1.

示例 2:

输入: [1, 2]

输出: 2

解释: 第三大的数不存在, 所以返回最大的数 2 .

示例 3:

输入: [2, 2, 3, 1]

输出: 1

解释: 注意，要求返回第三大的数，是指第三大且唯一出现的数。
存在两个值为2的数，它们都排第二。

class Solution(object):
    def thirdMax(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        m = list(set(nums))
        m.sort(reverse=True)
        if len(m) >= 3:
            return m[2]
        return max(m)
        
415. 字符串相加
给定两个字符串形式的非负整数 num1 和num2 ，计算它们的和。

注意：

    num1 和num2 的长度都小于 5100.
    num1 和num2 都只包含数字 0-9.
    num1 和num2 都不包含任何前导零。
    你不能使用任何內建 BigInteger 库， 也不能直接将输入的字符串转换为整数形式。

class Solution(object):
    def addStrings(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        carry = 0
        size = max(len(num1), len(num2))
        i, j = len(num1) - 1, len(num2) - 1
        ans = ''
        
        for k in xrange(size):
            a = int(num1[i]) if i >= 0 else 0
            b = int(num2[j]) if j >= 0 else 0
            i, j = i - 1, j - 1
            c = carry
            carry = 0
            sum = a + b + c
            if sum >= 10:
                carry += 1
                ans += str(sum - 10)
            else:
                ans += str(sum)
        if carry:
            ans += '1'
        return ans[::-1]
        

427. 建立四叉树
我们想要使用一棵四叉树来储存一个 N x N 的布尔值网络。网络中每一格的值只会是真或假。树的根结点代表整个网络。对于每个结点, 它将被分等成四个孩子结点直到这个区域内的值都是相同的.

每个结点还有另外两个布尔变量: isLeaf 和 val。isLeaf 当这个节点是一个叶子结点时为真。val 变量储存叶子结点所代表的区域的值。

你的任务是使用一个四叉树表示给定的网络。下面的例子将有助于你理解这个问题：

给定下面这个8 x 8 网络，我们将这样建立一个对应的四叉树：

由上文的定义，它能被这样分割：

 

对应的四叉树应该像下面这样，每个结点由一对 (isLeaf, val) 所代表.

对于非叶子结点，val 可以是任意的，所以使用 * 代替。

提示：

    N 将小于 1000 且确保是 2 的整次幂。
    如果你想了解更多关于四叉树的知识，你可以参考这个 wiki 页面。


"""
# Definition for a QuadTree node.
class Node(object):
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight
"""
class Solution(object):
    def construct(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: Node
        """
        def helper(r1, r2, c1, c2, grid):
            if r1 > r2 or c1 > c2:
                return
            is_leaf = all(grid[i][j] == grid[r1][c1] for i in range(r1, r2 + 1) for j in range(c1, c2 + 1))
            if is_leaf:
                return Node(grid[r1][c1] == 1, True, None, None, None, None)
            topLeft = helper(r1, (r1+r2)/2, c1, (c1+c2)/2, grid)
            topRight = helper(r1, (r1+r2)/2, (c1+c2)/2+1, c2, grid)
            bottomLeft = helper((r1+r2)/2+1, r2, c1, (c1+c2)/2, grid)
            bottomRight = helper((r1+r2)/2+1, r2, (c1+c2)/2+1, c2, grid)
            return Node(False, False, topLeft, topRight, bottomLeft, bottomRight)
        return helper(0, len(grid)-1, 0, len(grid[0])-1, grid)

