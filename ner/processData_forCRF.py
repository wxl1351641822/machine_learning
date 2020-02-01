def readFile(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        article=f.readlines();
        article_split=[x.split() for x in article]

        result=[]
        dic_word_list=set()
        dic_label_list=set()
        label=[]
        word=[]
        for x in article_split:
            if(len(x)==0):
                if(len(word)>0):
                    # print(row)
                    result.append({'text':word,'label':label})
                    label = []
                    word = []
            else:
                word.append(x[0])
                label.append(x[len(x)-1])

                dic_word_list.add(x[0])
                dic_label_list.add(x[len(x)-1])

        if (len(word) > 0):
            # print(row)
            result.append({'text':word,'label':label})