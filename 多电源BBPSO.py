import math
import random

class Vertex:
    def __init__(self,key):
        self.id = key
        self.connectedTo = {}
        self.Distance=0
        self.Pred=None
        self.Color='white'
    
    def addNeighbor(self,nbr,weight=0):
        self.connectedTo[nbr] = weight
    
    def setDistance(self,distance):
        self.Distance=distance
    
    def setPred(self,vertex):
        self.Pred=vertex
    
    def setColor(self,color):
        self.Color = color

    def __str__(self):
        return str(self.id) + ' connectedTo: '+str([x.id for x in self.connectedTo])
    
    def getConnections(self):
        return self.connectedTo.keys()
    
    def getId(self):
        return self.id
    
    def getWeight(self,nbr):
        return self.connectedTo[nbr]

    def getDistance(self):
        return self.Distance
    
    def getPred(self):
        return self.Pred
        
    def getColor(self):
        return self.Color
    
class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0
        
    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key]=newVertex
        return newVertex
    
    def addNewVertex(self,key,vertex):
        self.numVertices = self.numVertices + 1
        self.vertList[key]=vertex
        return vertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None
        
    def __contains__(self,n):
        return n in self.vertList
    
    def addEdge(self,f,t,cost=0):#f,t为编号key
        if f not in self.vertList:
            nv = self.addVertex(f) 
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t],cost)
        self.vertList[t].addNeighbor(self.vertList[f],cost)
        
    def getVertices(self):
        return self.vertList.keys()
    
    def __iter__(self):
        return iter(self.vertList.values())
class TFU(Vertex): #TFU元件
    def __init__(self,key,name="TFU",Fault_signal=0):
        Vertex.__init__(self,key)
        self.name=name
        self.Fault_signal=Fault_signal #故障电流信号0为无故障电流，1为存在故障电流
        self.Successor_wirelist=[]     #计算期望时候，要设计到的所有馈线，即发送故障会影响到该点的所有馈线
        self.I_S_B=0                  #要求的期望
        self.ulist=[]                  #以节点为分解的上区馈线
        self.dlist=[]                  #以节点为分解的下区馈线
    
    def set_Fault_signal(self,Fault_signal): #设置当前的是否流经故障电流
        self.Fault_signal=Fault_signal
        
    def set_Successor_wirelist(self,Successor_wirelist):
        self.Successor_wirelist=Successor_wirelist

    def set_I_S_B(self,No_S_list,X):
        # print("SET_I_S_B",self.id)
        self.I_S_B=0
        Iju=0
        Ijd=0
        sjd=0               #下区总和
        sju=0               #上区总和
        if sum(X)==0:
            self.I_S_B=0
        elif self.ulist==[] and sum(X)!=0: #上区为空
            self.I_S_B=1
        elif len(self.dlist)==1 and type(self.dlist[0])==int:
            sjd=X[self.dlist[0]-1]
            if self.dlist[0] in No_S_list:  #如果下半区只有一个而且不是电源
                for u in self.ulist:
                    Sju=0           #(1-馈线或逻辑)
                    if u[-1] not in No_S_list:
                        for x in u:
                            Sju=Sju|X[x-1]
                            sju=sju|X[x-1]
                        Iju=(1-Sju)|Iju
                self.I_S_B=Iju*sjd-Ijd*sju
            else:
                Ijd=(1-X[self.dlist[0]-1])
                for u in self.ulist:
                    Sju=0           #(1-馈线或逻辑)
                    if u[-1] not in No_S_list:
                        for x in u:
                            Sju=Sju|X[x-1]
                            sju=sju|X[x-1]
                        Iju=(1-Sju)|Iju
                self.I_S_B=Iju*sjd-Ijd*sju
        else:
            for d in self.dlist:
                Sjd=0           #(1-馈线或逻辑)
                if d[-1] not in No_S_list:
                    for x in d:
                        Sjd=Sjd|X[x-1]
                        sjd=sjd|X[x-1]
                    Ijd=(1-Sjd)|Ijd
            # print("SJD,IJD",Sjd,Ijd)
            for u in self.ulist:
                Sju=0           #(1-馈线或逻辑)
                if u[-1] not in No_S_list:
                    for x in u:
                        Sju=Sju|X[x-1]
                        sju=sju|X[x-1]
                    Iju=(1-Sju)|Iju
            # print("Sju,Iju",Sju,Iju)
            self.I_S_B=Iju*sjd-Ijd*sju
            
    
    def set_ulist(self,ulist):
        self.ulist=ulist
    
    def set_dlist(self,dlist):
        self.dlist=dlist

    def get_Fault_signal(self): #获得当前数据
        return self.Fault_signal

    def get_Successor_wirelist(self):
        return self.Successor_wirelist
    
    def get_I_S_B(self):
        return self.I_S_B
    def get_ulist(self):
        return self.ulist
    
    def get_dlist(self):
        return self.dlist
class Queue:
    def __init__(self):
        self.items = []
        
    def isEmpty(self):
        return self.items == []
    
    def enqueue(self, item): #复杂度为O(n)
        self.items.insert(0, item)
        
    def dequeue(self):      #复杂度为O(1)
        return self.items.pop()
    
    def size(self):
        return len(self.items)
def bfs(g,start):
    start.setDistance(0)
    start.setPred(None)
    vertQueue = Queue()
    vertQueue.enqueue(start)
    while (vertQueue.size() > 0):
        currentVert = vertQueue.dequeue()
        for nbr in currentVert.getConnections():
            if (nbr.getColor() == "white"):
                nbr.setColor('gray')
                nbr.setDistance(currentVert.getDistance()+1)
                nbr.setPred(currentVert)
                vertQueue.enqueue(nbr)
            currentVert.setColor('black')
def traverse(y):
    x=y
    while (x.getPred()):
        print(x.getId())
        x=x.getPred()
    print(x.getId())
def traverselist(y):
    x=y
    lst=[]
    while (x.getPred()):
        lst.append(x.getId())
        x=x.getPred()
    lst.append(x.getId())
    return lst
def sign(r):
    if r>=0.1:
        return 1
    else:
        return -1
def sigmoid(v):
    if v>4:
        return 0.98
    elif v<-4:
        return -0.98
    else:
        return 1/(1+math.e**(-v))

class Particle:
    def __init__(self,X,V): #初始速度和初始位置，均通过随机产生
        self.X=X            #粒子的空间位置
        self.V=V            #粒子当前的速度
        self.pX=0           #粒子的最优位置
        self.k=0            #当前迭代次数
        self.t=0            #进入探索阶段后的迭代次数
        self.state=0        #粒子当前阶段 0为捕食阶段，1位探索阶段
        self.F=0            #粒子当前的适应度,评价函数评价值
        self.min_F=1000     #粒子当前的最优平均值
    
    def set_X(self,X):
        self.X=X
    
    def set_V(self,V):
        self.V=V

    def replace_X(self,NewV):
        NewX=[]
        for i in range(len(NewV)):
            v=NewV[i]
            sigmoid_v=sigmoid(v)
            if random.random()<sigmoid_v:
                x=1
            else:
                x=0
            NewX.append(x)
        self.X=NewX

    def replace_V(self,V,state,pgX,hpX=[]):
        NewV=[]
        if state==0: #为捕食状态
            for i in range(len(V)):
                v=V[i]
                px=self.pX[i]
                x=self.X[i]
                pgx=pgX[i]
                w=w_max-(w_max-w_min)*(self.k/T)
                newv=w*v+c1*random.random()*(px-x)+c2*random.random()*(pgx-x)
                # if newv>4:
                #     newv=4
                # elif newv<-4:
                #     newv=-4
                NewV.append(newv)
        elif state==1:  #为探索状态
            for i in range(len(V)):
                v=V[i]
                px=self.pX[i]
                x=self.X[i]
                pgx=pgX[i]
                hpx=hpX[i]
                c3=random.random()
                w=w_max-(w_max-w_min)*(self.t/(T-self.k+self.t))
                r=random.random()
                newv=w*sign(r)*v+c1*random.random()*(px-x)+c2*random.random()*(pgx-x)+c3*random.normalvariate(0,1)*(x-hpx)
                # if newv>4:
                #     newv=4
                # elif newv<-4:
                #     newv=-4
                NewV.append(newv)
        self.V=NewV

    def set_pX(self,X):#将当前设置为最优位置
        self.pX=X

    def replace_k(self):
        self.k=self.k+1
    
    def replace_t(self):
        self.t=self.t+1
    
    def replace_state(self,state):
        self.state=state

    def replace_F(self,g,wire_list):
        self.F=0
        w_=0.5      #权重
        for i in range(len(wire_list)):
            vertex=g.getVertex(i+1)
            vertex.set_I_S_B(No_S_list,self.X)
            ISB=vertex.get_I_S_B()
            self.F=self.F+abs(wire_list[i]-ISB)
            # print("wire_list[i]",wire_list[i],"ISB",ISB,"I",abs(wire_list[i]-ISB))
        self.F=self.F+w_*sum(self.X)
    def replace_min_F(self,F):
        self.min_F=F
        
    def get_X(self):
        return self.X
    
    def get_V(self):
        return self.V
    
    def get_pX(self):
        return self.pX
    
    def get_k(self):
        return self.k
    def get_t(self):
        return self.t
    
    def get_state(self):
        return self.state
    
    def get_F(self):
        return self.F
        
    def get_min_F(self):
        return self.min_F

vertexslist=[i for i in range(1,34)]
edgelist=[(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(2,19),(19,20),(20,21),(21,22),(3,23),(23,24),(24,25),(6,26),(26,27),(27,28),(28,29),(29,30),(30,31),(31,32),(32,33)]
wire_list=[1 for i in range(5)]+[-1 for i in range(22-5)]+[0 for i in range(3)]+[-1 for i in range(33-25)]
S_list=[1] #系统电源和
DG_list=[22,18,33]
No_S_list=[25]
noandslist=S_list+DG_list+No_S_list

g=Graph()
for i in vertexslist:
    g.addNewVertex(i,TFU(key=i,name="TFU"))
for edge in edgelist:#边
    g.addEdge(f=edge[0],t=edge[1],cost=None)

S_load_traverse_list=[]
for S in noandslist:
    g=Graph()
    for i in vertexslist:
        g.addNewVertex(i,TFU(key=i,name="TFU"))
    for edge in edgelist:#边
        g.addEdge(f=edge[0],t=edge[1],cost=None)
    bfs(g,g.getVertex(S)) #对图g中的节点进行广义搜索标记
    load_traverse_list={}
    for i in vertexslist:
        load_traverse_list[i]=traverselist(g.getVertex(i))
    S_load_traverse_list.append(load_traverse_list)

#用于设置期望做准备,找出节点的故障导致的馈线
for i in vertexslist:
    tfu=g.getVertex(i)
    Successor_wirelist=[]
    for load_traverse_list in S_load_traverse_list:
        Successor_wirelist.append(load_traverse_list[i])
    tfu.set_Successor_wirelist(Successor_wirelist)
#设置相关的上下游
for i in vertexslist:
    ulist=[]
    dlist=[]
    tfu=g.getVertex(i)
    Successor_wirelist=tfu.get_Successor_wirelist()
    if len(Successor_wirelist[0])==1:
        dlist=vertexslist
        tfu.set_ulist(ulist)
        tfu.set_dlist(dlist)
    else:
        for predlist in Successor_wirelist:
            if len(predlist)==1:
                dlist=predlist
                Successor_wirelist.remove(predlist)
                ulist=[lst[1:] for lst in Successor_wirelist]
                break
            else:
                if(predlist[1])==Successor_wirelist[0][1]:
                    ulist.append(predlist[1:])
                else:
                    dlist.append(predlist)
        tfu.set_dlist(dlist)
        tfu.set_ulist(ulist)

T=200
N=500           #总种群数
D=33            #粒子解空间维度
Particle_swarm=[]
for i in range(N):
    X=[]
    V=[]
    for j in range(D):
        X.append(random.randint(0,1))
        V.append(random.uniform(-4,4))
    particle=Particle(X,V)
    particle.replace_F(g,wire_list)
    particle.replace_min_F(particle.get_min_F())
    Particle_swarm.append(particle)
w_max=0.9
w_min=0.4
c1=1.494
c2=1.794
hpX=[]      #探索阶段的粒子的所有解中的最全局优解
pgX=[]      #所有解空间内粒子的全局最优解
pgV=[]
g_min_F=10000
h_min_F=10000
F=[]        #记录每次迭代的最优F
M=1         #每次前往探索的粒子数
m_1=0         #当前探索的粒子数
for a in range(T):      #循环迭代T次
    for particle in Particle_swarm:         #遍历所有的粒子
        nowF=0                             #粒子更新前的适应值
        if particle.get_state()==0:         #如果粒子是捕获状态
            nowF=particle.get_F()          #记录更新前的F值
            if g_min_F>nowF:               #如果最小的适应度小于当前,设置群体最优解
                g_min_F=nowF
                pgX=particle.get_X()
                pgV=particle.get_V()
            if particle.get_min_F()>nowF:   #设置粒子最优解
                particle.replace_min_F(nowF)#替换个体最优时间
                particle.set_pX(particle.X) #替换个体最优位置
            particle.replace_V(particle.get_V(),0,pgX)#更新速度
            particle.replace_X(particle.get_V())#更新当前位置
            particle.replace_F(g,wire_list)                #更新适应度F
            particle.replace_k()
        elif particle.get_state()==1:
            # print(a)
            nowF=particle.get_F()          #记录更新前的F值
            if g_min_F>nowF:               #如果最小的适应度小于当前,设置群体最优解
                g_min_F=nowF
                pgX=particle.get_X()
                pgV=particle.get_V()
            if particle.get_min_F()>nowF:   #设置粒子最优解
                particle.replace_min_F(nowF)#替换个体最优时间
                particle.set_pX(particle.X) #替换个体最优位置
            if h_min_F>nowF:
                h_min_F=nowF
                hpX=particle.get_X()
            particle.replace_V(particle.get_V(),1,pgX,hpX)#更新速度
            particle.replace_X(particle.get_V())#更新当前位置
            particle.replace_F(g,wire_list)                #更新适应度F
            particle.replace_k()
            particle.replace_t()
    F.append(g_min_F)
    if (a>=3):
        f=(F[a]-F[a-1])/(F[a-1]+F[a-2]+0.000000001)
        if 0<=f and f<=1 and m_1<N-1:
            for m in range(M):
                if Particle_swarm[m_1].get_state()==0:
                    X=[]
                    V=[]
                    for j in range(D):
                        X.append(random.randint(0,1))
                        V.append(random.uniform(-4,4))
                    Particle_swarm[m_1].replace_state(1)        #更新状态
                    Particle_swarm[m_1].set_X(X)                #初始化X值
                    Particle_swarm[m_1].set_V(V)                #初始化V值
                    Particle_swarm[m_1].replace_F(g,wire_list)  #求出当前F值
                    Particle_swarm[m_1].replace_min_F(particle.get_min_F())#重设当前粒子的最优解
                    m_1=m_1+1
print(g_min_F,T)
print(pgX,pgV)