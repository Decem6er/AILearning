# AILearning
A revision control when learning AI.

**这玩意儿现在只能识别手写数字/It can only distinguish the number that written by hand**

**结果中最大的值即为对应数字/The maximun value in the output is the corresponding number**
**第一行为标准答案，数组结果从上到下分别为0到9/The first number is the srandard answer, array results are 0 to 9 from top to bottom**


当然，你也可以自己写个数字，拿去识别一下。
做法是：将数字存为28*28像素的图片，再将图像变成浮点数组（传说scipy.misc可行，但我这里好像会报错，嘛，算了），存为.csv，其中csv每行第一个值为你写的数字，投食。



最后感谢 Tariq Rashid ，我在很大程度上参考了他的代码
Thanks you Tariq Rashid. Your code is be of great help.