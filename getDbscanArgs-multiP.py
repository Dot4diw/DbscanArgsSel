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
data=[
 [-2.68420713,1.469732895],[-2.71539062,-0.763005825],[-2.88981954,-0.618055245],[-2.7464372,-1.40005944],[-2.72859298,1.50266052],
 [-2.27989736,3.365022195],[-2.82089068,-0.369470295],[-2.62648199,0.766824075],[-2.88795857,-2.568591135],[-2.67384469,-0.48011265],
 [-2.50652679,2.933707545],[-2.61314272,0.096842835],[-2.78743398,-1.024830855],[-3.22520045,-2.264759595],[-2.64354322,5.33787705],
 [-2.38386932,6.05139453],[-2.6225262,3.681403515],[-2.64832273,1.436115015],[-2.19907796,3.956598405],[-2.58734619,2.34213138],
 [1.28479459,3.084476355],[0.93241075,1.436391405],[1.46406132,2.268854235],[0.18096721,-3.71521773],[1.08713449,0.339256755],
 [0.64043675,-1.87795566],[1.09522371,1.277510445],[-0.75146714,-4.504983795],[1.04329778,1.030306095],[-0.01019007,-3.242586915],
 [-0.5110862,-5.681213775],[0.51109806,-0.460278495],[0.26233576,-2.46551985],[0.98404455,-0.55962189],[-0.174864,-1.133170065],
 [0.92757294,2.107062945],[0.65959279,-1.583893305],[0.23454059,-1.493648235],[0.94236171,-2.43820017],[0.0432464,-2.616702525],
 [4.53172698,-0.05329008],[3.41407223,-2.58716277],[4.61648461,1.538708805],[3.97081495,-0.815065605],[4.34975798,-0.188471475],
 [5.39687992,2.462256225],[2.51938325,-5.361082605],[4.9320051,1.585696545],[4.31967279,-1.104966765],[4.91813423,3.511712835],
 [3.66193495,1.0891728],[3.80234045,-0.972695745],[4.16537886,0.96876126],[3.34459422,-3.493869435],[3.5852673,-2.426881725],
 [3.90474358,0.534685455],[3.94924878,0.18328617],[5.48876538,5.27195043],[5.79468686,1.139695065],[3.29832982,-3.42456273]
]
data = pd.DataFrame(data)
data.columns=['x','y']

sns.relplot(x="x",y="y",data=data)
rs= []

eps = np.arange(0.2,4,0.2) 
min_samples=np.arange(2,20,1)

best_score=0
best_score_eps=0
best_score_min_samples=0
pool_size = 4

def run_dbscan(eps_args, min_samples_args):
    try:
        db = DBSCAN(eps=eps_args, min_samples=min_samples_args).fit(data)
        return db
    except:
        pass
    
with ThreadPoolExecutor(max_workers=pool_size) as t:
    obj_list = []
    for i in np.arange(0.2,4,0.2):
        for j in np.arange(2,20,1):
            args = [i,j]
            #print(args)
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
                best_score_eps = i
                best_score_min_samples = j
        except:
            pass
        
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