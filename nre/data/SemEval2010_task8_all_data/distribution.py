from tqdm import tqdm
from collections import Counter
import constant
def getRelation(path):
    r=[]
    with open(path,'r',encoding='utf-8') as f:
        data = eval(f.read())
        for d in tqdm(data):
            r.append(d['relation'])
    c=Counter(r)
    print(c)
    return r

def draw_plt(r):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)
    c=Counter(r)

    plt.bar(c.keys(), c.values(), label='graph 1')

    # plt.bar([2, 4, 6, 8, 10], [4, 6, 8, 13, 15], label='graph 2')

    # params

    # x: 条形图x轴
    # y：条形图的高度
    # width：条形图的宽度 默认是0.8
    # bottom：条形底部的y坐标值 默认是0
    # align：center / edge 条形图是否以x轴坐标为中心点或者是以x轴坐标为边缘

    plt.legend()

    plt.xlabel('number')
    plt.ylabel('value')

    plt.title(u'测试例子——条形图', FontProperties=font)

    plt.show()

r=getRelation(constant.train_path)
draw_plt(r)