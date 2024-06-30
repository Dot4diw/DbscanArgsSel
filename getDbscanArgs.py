
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn import metrics
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data =[]
with open("T742-xy.csv") as fInfile:
	for line in fInfile:
		line_list = "[" + line.strip() + "]"
		data.append(line_list)

data = pd.DataFrame(data)
data.columns=['x','y']

#sns.relplot(x="x",y="y",data=data)
rs= []
eps = np.arange(2,20,1) 
min_samples=np.arange(2,25,1)

best_score=0
best_score_eps=0
best_score_min_samples=0


for i in eps:
    for j in min_samples:
        try:
            
            db = DBSCAN(eps=i, min_samples=j).fit(data)
            labels= db.labels_ # get the cluster labels
            k=metrics.silhouette_score(data,labels) 
            raito = len(labels[labels[:] == -1]) / len(labels) 
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            rs.append([i,j,k,raito,n_clusters_])
            
            if k>best_score:
                best_score=k
                best_score_eps=i
                best_score_min_samples=j
        except:
            db=''
        else:
            db=''

rs= pd.DataFrame(rs)
rs.columns=['eps','min_samples','score','raito','n_clusters']
fig_score = sns.relplot(x="eps",y="min_samples", size='score',data=rs)
fig_score.savefig('./DBSCAN_SilhouetteCoefficient.pdf')

fig_raito = sns.relplot(x="eps",y="min_samples", size='raito',data=rs)
fig_raito.savefig('./DBSCAN_SignalToNoiseRatio.pdf')


final_eps = rs[rs['score'] == rs['score'].max()]['eps'].max()
final_minpts = rs[rs['raito'] == rs['raito'].min()]['min_samples'].min()

print("EPS: %d" % final_eps)
print("MinPts: %d" % final_minpts)