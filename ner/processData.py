import numpy as np;
def readFile(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        article=f.readlines();
        article_split=[x.split() for x in article]
        row=[]
        result=[]
        for x in article_split:
            if(len(x)==0):
                 result.append(row)
                 row=[]
            else:
                row.append(x)
    return result

# readFile("./CoNLL-2003/eng.train");
