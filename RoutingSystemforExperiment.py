
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np
import math
import random
import folium
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.grid_objs import Grid, Column

args = sys.argv

#全家のpred,ocpデータを時間別に集約
tlist = ["0","5","10","15","20","25","30"]
hlist = ["01","02","03","04","05"] #houselist
def rdata(h,t):
    return np.genfromtxt('../data/' + hlist[0] + '_prediction_' + t + '.csv',delimiter=",")
for t in tlist:
    temp = np.vstack((rdata(hlist[0],t),rdata(hlist[1],t),rdata(hlist[2],t),rdata(hlist[3],t),rdata(hlist[4],t)))
    np.savetxt('../data/prediction_' + t + '.csv', temp, delimiter=",")
    
#予測時間別に、ロケーションデータx30の読み込みと電力消費,在不在,予測,全データセットの統合
#Read delivery destination data (name,lat,long) in hongo campas
hongo = pd.read_csv('../map/hongo.csv').iloc[0:20,:]
def create_opr(time):
    #Read predcition and real occupancy data of Ecodataset
    pr = pd.read_csv('../data/prediction_' + time + '.csv',header = None)
    pr = pr.T
    #Divide data into 30 mini-dataset
    #prediction
    prdf = pr.iloc[:,0:60].iloc[0].reset_index(drop = True)
    for i in range(29):
        prdf = pd.concat([prdf,pr.iloc[:,60*(i+1):60*(i+2)].iloc[0].reset_index(drop = True)],axis = 1)
    prdf = prdf.T.reset_index(drop = True)

    #occupancy
    ocdf = pr.iloc[:,0:60].iloc[1].reset_index(drop = True)
    for i in range(29):
        ocdf = pd.concat([ocdf,pr.iloc[:,60*(i+1):60*(i+2)].iloc[1].reset_index(drop = True)],axis = 1)
    ocdf = ocdf.T.reset_index(drop = True)

    #ロケーションデータx30の読み込みと電力消費と在不在 / 予測それぞれを統合したデータセットを出力
    #Allocate each mini-dataset to each delivery points
    hongopd = pd.concat([hongo,prdf],axis = 1)
    hongopd.to_csv('createddataset/hongopd'+time+'.csv')
    hongooc = pd.concat([hongo,ocdf],axis = 1)
    hongooc.to_csv('createddataset/hongooc'+time+'.csv')
    
tlist = ["0","5","10","15","20","25","30"]
for t in tlist:
    create_opr(t)

pd0 = pd.read_csv('createddataset/hongopd0.csv')
pd5 = pd.read_csv('createddataset/hongopd5.csv')
pd10 = pd.read_csv('createddataset/hongopd10.csv')
pd15 = pd.read_csv('createddataset/hongopd15.csv')
pd20 = pd.read_csv('createddataset/hongopd20.csv')
pd25 = pd.read_csv('createddataset/hongopd25.csv')
pd30 = pd.read_csv('createddataset/hongopd30.csv')

oc0 = pd.read_csv('createddataset/hongooc0.csv')
oc5 = pd.read_csv('createddataset/hongooc5.csv')
oc10 = pd.read_csv('createddataset/hongooc10.csv')
oc15 = pd.read_csv('createddataset/hongooc15.csv')
oc20 = pd.read_csv('createddataset/hongooc20.csv')
oc25 = pd.read_csv('createddataset/hongooc25.csv')
oc30 = pd.read_csv('createddataset/hongooc30.csv')

#x全地点間の物理距離マトリクスを作成
hongopd0 = pd.read_csv('createddataset/hongopd.csv')
hongopd = hongopd0.iloc[0:20,:]

LATM = 111000
LONM = 91000

#入力距離に対し、将来予測する時間帯を出力する関数を用意　※距離MATRIXの最大値設定：キャンパス内最大距離で１０分
mlist = []
dmax = 0
def create_matrix(dataset):
    dlist = []; i = 0
    while i < len(dataset):
        j = 0; tmp = []
        while j < len(dataset):
            dist = math.sqrt(((dataset[i][2] - dataset[j][2])*LATM) **2 + ((dataset[i][3] - dataset[j][3])*LONM)**2)
            tmp.append(dist)
            j += 1
        i += 1; dlist.append(tmp)
    dmatrix = np.array(dlist)
    return dmatrix

dmatrix = create_matrix(hongopd.values)
dmatrix_df = pd.DataFrame(dmatrix)

hongo['visited']=0
hongo['successed']=0
hongo['visitorder']=0
dMATRIX = pd.concat([hongo,dmatrix_df],axis=1)
distarray = dMATRIX.iloc[:,6:].values

def dist_to_prdtime(dist):
    mlist = []
    gap = dmax / 3
    prdt = 0
    if dist < gap:
        prdt = 0
    elif dist < gap * 2:
        prdt = 1
    elif dist < gap * 3:
        prdt = 2
    return prdt

for i in distarray:
    mlist.append(np.amax(i))
    dmax = np.amax(mlist)

def routegen(onoff,tf,network):
    #初期化
    network['visited']=0
    network['successed'] = 0
    network['visitorder'] = 0
    unvisited = network[network['visited'] == 0].index

    #出発地点登録
    cpos = 0
    visitorder = 0
    network.iat[0,3] = 1
    network.iat[0,4] = 1
    network.iat[0,5] = visitorder
    
    THRESHOLD = 0.8
    
    if onoff == 1:
        WEIGHT = 0.90
    else:
        WEIGHT = 0
    timeframe = tf #random.randrange(10)

    while len(unvisited) > 1:
        mindist = 10000
        unvisited = network[network['visited'] == 0].index     #未訪問のindex抽出
        #print(unvisited)
        #print(len(unvisited))

        candidx = [] #改善アルゴリズム　候補index格納用
        canddist = []#改善アルゴリズム　候補dist格納用

        for i in unvisited:
            #print(i)
            dist = network.iloc[cpos][6:][i] #現在地点からの距離を取得
            if dist_to_prdtime(dist) == 0:
                prd = pd0.iloc[i][4:][timeframe]
                ocp = oc0.iloc[i][4:][timeframe]
                owd = dist*(1-prd * WEIGHT)
            elif dist_to_prdtime(dist) == 1:
                prd = pd5.iloc[i][4:][timeframe]
                ocp = oc5.iloc[i][4:][timeframe]
                owd = dist*(1-prd * WEIGHT)
            elif dist_to_prdtime(dist) == 2:
                prd = pd10.iloc[i][4:][timeframe]
                ocp = oc10.iloc[i][4:][timeframe]
                owd = dist*(1-prd * WEIGHT)
            else:
                print("error")

            #print('dist：{0}\prd:{1}\ocp:{2}\owd:{3}'.format(dist,prd,ocp,owd))    
            if owd < mindist:
                mindist = owd
                minindex = i
                minprd = prd
                minocp = ocp

            if prd > THRESHOLD:
                candidx.append([i])
                canddist.append([dist])
        #print("---")
        #print(minindex)
        #print('minimums: dist：{0}\prd:{1}\ocp:{2}\owd:{3}'.format(mindist,minprd,minocp,mindist))
        #print("---")
        if len(candidx) < 1:
            print("no candidate")
            break
        else:
            nextindex = candidx[np.argmin(canddist)]

        network.loc[minindex,'visited'] = 1
        visitorder += 1

        network.loc[minindex,'visitorder'] = visitorder
        #現在地のインデックス取得
        cpos = minindex
        timeframe += 1

        if minocp == 1:
            network.loc[minindex,'successed'] = 1
        else:
            network.loc[minindex,'successed'] = 0

    #dMATRIX
    srate = 100 * network['successed'].sum() /network['visited'].sum()
    print('success rate: {0}'.format( srate))
    network = network.sort_values('visitorder')
    network.loc[:,'name':'visitorder'].to_csv("output/route_" + str(onoff) + '_' + str(tf) + '.csv',index = False)
    return network
return routegen(args[1],args[2],dMATRIX)


# In[2]:


route1 = routegen(1,4,dMATRIX)
#route1


# In[2]:





dMATRIX = routegen(1,1)
#地図プロット
mapbox_access_token = 'pk.eyJ1Ijoic29oc3VnaSIsImEiOiJjamt6Nm5zZDAwcXhqM3BwOHlsaGt1b2RjIn0.sJbcUk1KZ9zTAlw8JY8fIQ'
tdist = 0
cnt = 0
latlist = []
longlist = []
namelist = []

while cnt < len(dMATRIX.loc[dMATRIX['visited']==1]):    
    data = dMATRIX.loc[dMATRIX['visitorder']==cnt]
    latlist.append(data['latitude'].values[0])
    longlist.append(data['longitude'].values[0])
    namelist.append(str(cnt) + '. ' + data['name'].values[0])
#    tdist += math.sqrt(((curpos[0] - nexpos[0])*LATM)**2 + ((curpos[1] - nexpos[1])*LONM)**2)
    cnt += 1

def showmap_folium(dMATRIX):
    #foliumを使った地図プロット
    hongomap = folium.Map(location=[35.71005,139.76238],zoom_start=17)
    curpos = [35.70811,139.76267]
    nexpos = [35.70811,139.76267]
    tdist = 0
    cnt = 0
    while cnt < len(dMATRIX.loc[dMATRIX['visited']==1]):    
        curpos = nexpos
        data = dMATRIX.loc[dMATRIX['visitorder']==cnt]
        lat = data['latitude'].values[0]
        long = data['longitude'].values[0]
        nexpos = [lat,long]
        name = str(cnt) + '. ' + data['name'].values[0]
        if data['successed'].values[0] == 0:
            folium.Marker(location=[lat, long],popup=name,icon=folium.Icon(color='cloud')).add_to(hongomap)
            folium.PolyLine(locations=[curpos, nexpos], color='red').add_to(hongomap)
        else:
            folium.Marker(location=[lat, long],popup=name).add_to(hongomap)
            folium.PolyLine(locations=[curpos, nexpos], color='blue').add_to(hongomap)
        tdist += math.sqrt(((curpos[0] - nexpos[0])*LATM)**2 + ((curpos[1] - nexpos[1])*LONM)**2)
        cnt += 1
    #未訪問先の表示
    cnt2 = 0
    unvisit = dMATRIX.loc[dMATRIX['visited']==0]
    while cnt2 < len(dMATRIX.loc[dMATRIX['visited']==0]):
        lat = unvisit['latitude'].values[cnt2]
        long = unvisit['longitude'].values[cnt2]
        name = unvisit['name'].values[cnt2]
        folium.Marker(location=[lat, long],popup=name,icon=folium.Icon(color='green')).add_to(hongomap)
        cnt2 += 1
    print('travel dist: {0}'.format(tdist))
    return hongomap
    
def getTrace(num):
    data = go.Scattermapbox(
            lat=latlist[0:num],
            lon=longlist[0:num],
            mode='markers',
            marker=dict(
                size=9
            ),
            text=namelist[0:num],
    )
    return data
snum = 0

data = []
step = []
visibility = [False] * cnt

while snum < cnt:
    visibility = [False] * cnt
    visibility[snum] = True
    data.append(getTrace(snum))
    step.append(
                dict(label = str(snum),
                method = 'update',
                args = [{'title': '5','visible' : visibility}]
            )
    )
    snum +=1
    
updatemenus=list([
    dict(
        buttons = list(step),
        direction = 'left',
        pad = {'r': 10, 't': 10},
        showactive = True,
        type = 'buttons',
        x = 0.1,
        xanchor = 'left',
        y = 1.1,
        yanchor = 'top' 
    ),
])

layout = dict(
    go.Layout(
        autosize=True,
        hovermode='closest',
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=35.71005,
                lon=139.76238
            ),
            pitch=0,
            zoom=15
        ),
    ),updatemenus = updatemenus
)

fig = dict(data = data, layout = layout)
py.iplot(fig,filename='test' )
#plotly.offline.plot(fig, auto_open=True, show_link=False) 

