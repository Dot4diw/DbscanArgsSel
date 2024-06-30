import time
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from concurrent.futures import ThreadPoolExecutor, as_completed

begin=time.time()
#data = pd.DataFrame(data)
data = pd.read_csv('T742.csv')
data.columns=['x','y']
data = pd.DataFrame(data)
#sns.relplot(x="x",y="y",data=data)

rs= []
eps = np.arange(15,20,5)
min_samples=np.arange(5,10,1)

best_score=0
best_score_eps=0
best_score_min_samples=0

pool_size = 6

def run_dbscan(eps_args, min_samples_args):
    try:
        db = DBSCAN(eps=eps_args, min_samples=min_samples_args).fit(data)
        return db
    except:
        pass
    
with ThreadPoolExecutor(max_workers=pool_size) as t:
    obj_list = []
    for i in eps:
        for j in min_samples:
            args = [i,j]
            print(args)
            obj = t.submit(lambda p: run_dbscan(*p), args)
            obj_list.append(obj)
    for future in as_completed(obj_list):
        db_data = future.result()
        try:
            labels = db_data.labels_
            cur_eps = db_data.eps
            cur_min_samples = db_data.min_samples
            k = metrics.silhouette_score(data,labels)
            raito = len(labels[labels[:] == -1]) / len(labels) 
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            #print(cur_eps, cur_min_samples, k, raito, n_clusters_)
            rs.append([cur_eps, cur_min_samples, k, raito, n_clusters_])
            if (k > best_score):
                best_score = k
                best_score_eps = cur_eps
                best_score_min_samples = cur_min_samples
        except:
            pass
        
rs= pd.DataFrame(rs)
rs.columns=['eps','min_samples','score','raito','n_clusters']
rs.to_csv("T742_dbscan_args_list.csv")
fig_score = sns.relplot(x="eps",y="min_samples", size='score',data=rs)
fig_score.savefig('./DBSCAN_SilhouetteCoefficient.pdf')

fig_raito = sns.relplot(x="eps",y="min_samples", size='raito',data=rs)
fig_raito.savefig('./DBSCAN_SignalToNoiseRatio.pdf')


final_eps = rs[rs['score'] == rs['score'].max()]['eps'].max()
final_minpts = rs[rs['raito'] == rs['raito'].min()]['min_samples'].min()
times = time.time() - begin

print("Totla Time: %f" % times)
print("+"*30)
print("EPS: %d" % final_eps)
print("MinPts: %d" % final_minpts)
