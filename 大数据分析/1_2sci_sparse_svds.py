# 从scipy创建graph

import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt

# G=nx.read_weighted_edgelist('./yelp.edgelist/yelp.edgelist')
# nx.draw(G,pos = nx.random_layout(G),node_color = 'b',edge_color = 'r',with_labels = True,font_size =18,node_size =20)

def drawGraph(G):
    sparseG=nx.to_scipy_sparse_matrix(G)
    # print(sparseG)
    u,s,vh=sp.sparse.linalg.svds(sparseG,k=10)
    print(sparseG.shape,u.shape,s.shape,vh.shape)
    # 画u
    for i in range(0,9,2):
        x=u[:,i]
        y=u[:,i+1]
        print(x.shape)
        # plt.subplot(10, 2, i/2+1)
        plt.scatter(x, y, marker='.', color='green')
        plt.xlabel('u'+str(i+1))
        plt.ylabel('u'+str(i+2))
        plt.title('u'+str(i+2)+'-'+'u'+str(i+1)+'graph')
        plt.savefig('./data2_'+'u'+str(i+2)+'-'+'u'+str(i+1)+'graph.jpg')
        plt.show()


        x = vh[ i,:]
        print(x.shape)
        y = vh[ i + 1,:]
        # plt.subplot(10, 2, i/2 + 11)
        plt.scatter(x, y, marker='.', color='red')
        plt.xlabel('v' + str(i + 1))
        plt.ylabel('v' + str(i + 2))
        plt.title('v' + str(i + 2) + '-' + 'v' + str(i + 1) + 'graph')
        plt.savefig('./data2_' + 'v' + str(i + 2) + '-' + 'v' + str(i + 1) + 'graph.jpg')
        plt.show()


# G = nx.read_weighted_edgelist('./yelp.edgelist/yelp.edgelist')
G=nx.read_weighted_edgelist('./ratings_Books.csv')
drawGraph(G)
