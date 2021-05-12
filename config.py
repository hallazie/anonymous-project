# coding: utf-8

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False


DATA_PATH = 'I:\\datasets\\kaggle\\human-protein-atlas'

LABEL_DICT = {x.split('.')[0].strip(): x.split('.')[1].strip() for x in """
0.核质
1.核膜
2.核仁
3.核仁纤维中心
4.核斑
5.核机构
6.内质网
7.高尔基体
8.中间丝
9.肌动蛋白丝
10.微管
11.有丝分裂纺锤体
12.中心体
13.质膜
14.线粒体
15.令人讨厌
16.胞质溶胶
17.囊泡和点状胞浆模式
18.负面的
""".split('\n') if x}





