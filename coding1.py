import random
import time
import numpy as np
import math

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
class Node(Vertex): #元件
    def __init__(self,key,name):
        Vertex.__init__(self,key)
        self.name=name
        self.failure_number=0 #元件故障次数
        self.failure_time=0   #元件故障总时间
        self.mean_failure_time=0 #元件平均每次故障的时间
    
    def set_name(self,name):
        self.name=name

    def set_failure_number(self,n=1):
        self.failure_number=self.failure_number+n
    
    def set_failure_time(self,time):
        self.failure_time=self.failure_time+time

    def get_name(self):
        return self.name
    
    def get_failure_number(self):
        return self.failure_number
    
    def get_failure_time(self):
        return self.failure_time

    def get_mean_failure_time(self):
        if self.failure_number!=0:
            mean_failure_tim=self.failure_time/self.failure_number
            return mean_failure_tim
        else:
            return 0

class Transformer(Node):#变压器
    def __init__(self, key, name='transformer',failure_rate=0, repair_time=0, replace_number=0,replace_time=1):
        super().__init__(key, name)
        self.failure_rate=failure_rate #故障率
        self.repair_time=repair_time   #故障修复时间
        self.replace_number=replace_number #可更换次数
        self.TWT=0                        #元件无故障工作时间
        self.TRT=0                     #元件修复故障时间
        self.replace_time=replace_time#更换所需时间

    def set_Trouble_free_working_time(self,r):
        if self.failure_rate!=0:
            self.TWT=-math.log(r)/self.failure_rate
        else:
            raise Exception("未设置故障率")

    def set_Trouble_repair_time(self,r):
        if self.failure_rate!=0:
            self.TRT=-math.log(r)/self.repair_time
        else:
            raise Exception("未设置修复时间")
        
    def get_Trouble_free_working_time(self):
        return self.TWT

    def get_Trouble_repair_time(self):
        return self.TRT

    def get_failure_rate(self):
        return self.failure_rate
class Wire(Node):#馈线
    def __init__(self, key, name='wire',failure_rate=0,lenth=0, repair_time=0):
        super().__init__(key, name)
        self.failure_rate=failure_rate*lenth #故障率
        self.repair_time=repair_time   #故障修复时间
        self.TWT=0                        #元件无故障工作时间
        self.TRT=0                     #元件修复故障时间

    def set_Trouble_free_working_time(self,r):
        if self.failure_rate!=0:
            self.TWT=-math.log(r)/self.failure_rate
        else:
            raise Exception("未设置故障率")

    def set_Trouble_repair_time(self,r):
        if self.failure_rate!=0:
            self.TRT=-math.log(r)/self.repair_time
        else:
            raise Exception("未设置修复时间")
        
    def get_Trouble_free_working_time(self):
        return self.TWT

    def get_Trouble_repair_time(self):
        return self.TRT

    def get_failure_rate(self):
        return self.failure_rate

class Fuse(Node):#熔断器
    def __init__(self, key, name='fuse',failure_rate=0, repair_time=0,):
        super().__init__(key, name)
        self.failure_rate=failure_rate #故障率
        self.repair_time=repair_time   #故障修复时间
        self.TWT=0                     #元件无故障工作时间
        self.TRT=0                     #元件修复故障时间
        self.search_state=0           #未搜索为0，已搜索为1

    def set_Trouble_free_working_time(self,r):
        if self.failure_rate!=0:
            self.TWT=-math.log(r)/self.failure_rate
        else:
            raise Exception("未设置故障率")

    def set_Trouble_repair_time(self,r):
        if self.failure_rate!=0:
            self.TRT=-math.log(r)/self.repair_time
        else:
            raise Exception("未设置修复时间")

    def set_search_state(self,search_state):
        self.search_state=search_state

    def get_Trouble_free_working_time(self):
        return self.TWT

    def get_Trouble_repair_time(self):
        return self.TRT
        
    def get_failure_rate(self):
        return self.failure_rate

    def get_repair_time(self):
        return self.repair_time

    def get_search_state(self):
        return self.search_state
class Breaker(Node):#断路器
    def __init__(self, key, name='breaker',failure_rate=0, repair_time=0):
        super().__init__(key, name)
        self.failure_rate=failure_rate #故障率
        self.repair_time=repair_time   #故障修复时间
        self.TWT=0                     #元件无故障工作时间
        self.TRT=0                     #元件修复故障时间
        self.search_state=0           #未搜索为0，已搜索为1
        self.openstate=1              #当前开关状态，1为闭合，0为断开
        
    def set_Trouble_free_working_time(self,r):
        if self.failure_rate!=0:
            self.TWT=-math.log(r)/self.failure_rate
        else:
            raise Exception("未设置故障率")
    
    def set_Trouble_repair_time(self,r):
        if self.failure_rate!=0:
            self.TRT=-math.log(r)/self.repair_time
        else:
            raise Exception("未设置修复时间")

    def set_search_state(self,search_state):
        self.search_state=search_state
    
    def set_openstate(self,openstate):
        self.openstate=openstate
    
    def get_Trouble_free_working_time(self):
        return self.TWT
    
    def get_Trouble_repair_time(self):
        return self.TRT

    def get_repair_time(self):
        return self.repair_time
        
    def get_search_state(self):
        return self.search_state

    def get_openstate(self):
        return self.openstate
class Sectional_switch(Node):#分段开关
    def __init__(self, key, name='sectional_switch',failure_rate=0,repair_time=0,open_time=0):
        super().__init__(key, name)
        self.failure_rate=failure_rate #故障率
        self.repair_time=repair_time   #故障修复时间
        self.open_time=open_time      #打断开关需要的时间
        self.search_state=0           #未搜索为0，已搜索为1
        self.openstate=1              #当前开关状态，1为闭合，0为断开

    def set_search_state(self,search_state):
        self.search_state=search_state
    
    def set_openstate(self,openstate):
        self.openstate=openstate
    
    def get_failure_rate(self):
        return self.failure_rate

    def get_repair_time(self):
        return self.repair_time
        
    def get_search_state(self):
        return self.search_state

    def get_openstate(self):
        return self.openstate

    def get_open_time(self):
        return self.open_time
class Contact_switch(Node):#联络开关
    def __init__(self, key, name='contact_switch',failure_rate=0,repair_time=0,open_time=0):
        super().__init__(key, name)
        self.failure_rate=failure_rate #故障率
        self.repair_time=repair_time   #故障修复时间
        self.open_time=open_time      #打断开关需要的时间
        self.search_state=0           #未搜索为0，已搜索为1
        self.openstate=1              #当前开关状态，1为闭合，0为断开

    def set_search_state(self,search_state):
        self.search_state=search_state
    
    def set_openstate(self,openstate):
        self.openstate=openstate
    
    def get_failure_rate(self):
        return self.failure_rate
        
    def get_repair_time(self):
        return self.repair_time
        
    def get_search_state(self):
        return self.search_state

    def get_openstate(self):
        return self.openstate

    def get_open_time(self):
        return self.open_time
class Load(Node):#负荷
    def __init__(self, key, name='load',user_number=0):
        Node.__init__(self,key, name)
        self.failure_rate=0     #负荷故障停运率          failure_number/normal_operation
        self.U=0                #年平均停运时间   failure_time/failure_time+normal_operation*8760
        self.r=0                #每次平均停运时间 failure_time/faliure_number
        self.normal_operation=0 #负荷正常运行总时间  总时间-故障停运时间
        self.failure_time=0     #故障修复总时间（父类有，但是这里也重新定义一下）
        self.failure_number=0   #故障总次数
        self.user_number=user_number #用户数量
    
    def add_failure_number(self):
        self.failure_number=self.failure_number+1
    
    def add_failure_time(self,t):
        self.failure_time=self.failure_time+t

    def add_normal_operation(self,t):
        self.normal_operation=self.normal_operation+t

    def get_failure_rate(self):#单位为年每次
        if self.normal_operation!=0:
            self.failure_rate=(self.failure_number/self.normal_operation)
            return self.failure_rate
        else:
            raise Exception("存在一直停运的负荷，有问题")
    
    def get_U(self):#单位为年每小时
        if (self.failure_time+self.normal_operation)!=0:
            self.U=self.failure_time/(self.failure_time+self.normal_operation)
            return self.U
        else:
            raise Exception("模拟总时间为0，出现错误")
    
    def get_r(self):#次每小时
        if (self.failure_number)!=0:
            self.r=self.failure_time/self.failure_number
            return self.r
        else:
            return 0

    def get_normal_operation(self):
        return self.normal_operation
    
    def get_failure_number(self):
        return super().get_failure_number()
    
    def get_failure_time(self):
        return super().get_failure_time()

    def get_user_number(self):
        return self.user_number
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

#初始化参数
allnumber=104#元件总数
Transformers=[4,8,12,16,20,25,30,36,41,45,49,53,58,63,67,73,77,81,86,91,95,99,103]#变压器编号
Fuses=[i-1 for i in Transformers]#熔断器编号
Wires=[2,6,10,14,18,22,23,27,28,32,34,38,39,43,47,51,56,60,61,65,69,71,75,79,84,88,89,93,97,101]#馈线编号
Wires_lenth=[2.8,2.5,1.6,0.9,1.6,2.5,0.6,1.6,0.8,0.9,3.2,1.6,0.8,2.8,2.5,3.2,3.2,2.8,0.6,3.5,1.6,2.8,3.2,2.5,2.8,2.5,0.75,1.6,3.2,2.8]#馈线长度
Breakers=[1,33,70,83]#断路器编号
Loads=[i+1 for i in Transformers]#负荷编号
Loads_user=[147,126,1,1,132,147,1,79,1,76,1,1,79,1,76,79,76,1,79,1,1,76,1]#负荷用户数
Sectional_switchs=[55]#分段开关
Failures=Transformers+Wires+Breakers  #存在可以主动发生故障，需要修理的元件数
edgelist=[(1,2),(2,3),(3,4),(4,5),(2,6),(6,7),(7,8),(8,9),(2,10),(10,11),(11,12),(12,13),(10,14),(14,15),(15,16),(16,17),(14,18),(18,19),(19,20),(20,21),(18,22),(22,23),(23,24),(24,25),(25,26),(22,27),(27,28),(28,29),(29,30),(30,31),(27,32),(32,33),(33,34),(34,35),(35,36),(36,37),(34,38),(38,39),(39,40),(40,41),(41,42),(38,43),(43,44),(44,45),(45,46),(43,47),(47,48),(48,49),(49,50),(47,51),(51,52),(52,53),(53,54),(32,55),(55,56),(56,57),(57,58),(58,59),(55,60),(60,61),(61,62),(62,63),(63,64),(60,65),(65,66),(66,67),(67,68),(65,69),(69,70),(70,71),(71,72),(72,73),(73,74),(71,75),(75,76),(76,77),(77,78),(75,79),(79,80),(80,81),(81,82),(69,83),(83,84),(84,85),(85,86),(86,87),(84,88),(88,89),(89,90),(90,91),(91,92),(88,93),(93,94),(94,95),(95,96),(93,97),(97,98),(98,99),(99,100),(97,101),(101,102),(102,103),(103,104)]

#构造图
g=Graph()
for i in Transformers:#变压器
    transformer=Transformer(i,'transformer',failure_rate=0.015,repair_time=200,replace_number=0,replace_time=0)#不考虑更替
    g.addNewVertex(i,transformer)
for i in Fuses:#熔断器
    fuse=Fuse(key=i,name='fuse',failure_rate=0,repair_time=0)#数值上不考虑熔断器熔断，熔断器熔断为0
    g.addNewVertex(i,fuse)
for i in range(len(Wires)):#馈线
    wire=Wire(key=Wires[i],name='wire',failure_rate=0.05,lenth=Wires_lenth[i],repair_time=4)
    g.addNewVertex(Wires[i],wire)
for i in Breakers:#断路器
    breaker=Breaker(key=i,name='breaker',failure_rate=0.02,repair_time=4)
    g.addNewVertex(i,breaker)
for i in range(len(Loads)):#负荷
    load=Load(key=Loads[i],name='load',user_number=Loads_user[i])
    g.addNewVertex(Loads[i],load)
for i in Sectional_switchs:#分段开关
    sectional_switch=Sectional_switch(key=i,name="sectional_switch",failure_rate=0,repair_time=1.5/8760,open_time=1/3)
    g.addNewVertex(i,sectional_switch)
for edge in edgelist:#边
    g.addEdge(f=edge[0],t=edge[1],cost=None)

bfs(g,g.getVertex(1)) #对图g中的节点进行广义搜索标记
load_traverse_list={}#构造元件搜索数据图
for i in Loads:
    load_traverse_list[i]=traverselist(g.getVertex(i))

#初始化数据
x_0=time.time() #初始化种子为系统时间
x=x_0           #记录初始的x_0
j=0             #初始化累加次数
TTF=0
min_id=0
TTR=0
#设限仿真时间单位小时
M=2*8760
#当前仿真时间
MCTime=0

#系统最小正常工作时间累加大于设限时间则停止
#蒙特卡洛法
#设限仿真时间单位小时
M=2*8760
x_0=time.time() #初始化种子为系统时间
x=x_0           #记录初始的x_0
j=0             #初始化累加次数
TTF=0
min_id=0
TTR=0
#当前仿真时间
MCTime=0
#统计次数
f=0
#系统最小正常工作时间累加大于设限时间则停止
while MCTime<M:
    f=f+1
    TTF=0
    for i in range(len(Failures)):
        j=j+1
        if j>50:
            j=1
            x=math.sqrt(x*x_0)
            random.seed(x)
            r=random.random()
        else:
            random.seed(x)
            r=random.random()
            x=(j+1.02)/(j+1.01)*x*r
            random.seed(x)
            r=random.random()
        g.getVertex(Failures[i]).set_Trouble_free_working_time(r)
        TWT = g.getVertex(Failures[i]).get_Trouble_free_working_time()
        if TTF==0:
            TTF=TWT
        if TTF>TWT and TTF!=0:
            TTF=TWT
            min_id=Failures[i]
    j=j+1
    if j>50:
        j=1
        x=math.sqrt(x*x_0)
        random.seed(x)
        r=random.random()
    else:
        random.seed(x)
        r=random.random()
        x=(j+1.02)/(j+1.01)*x*r
        random.seed(x)
        r=random.random()
    g.getVertex(Failures[i]).set_Trouble_repair_time(r)
    TTR=g.getVertex(Failures[i]).get_Trouble_repair_time()
    for predlist in load_traverse_list.values():
        if min_id in predlist:
            failure_load=g.getVertex(predlist[0]) #取出故障的节点
            failure_load.add_failure_number()
            failure_load.add_failure_time(TTR)
            failure_load.add_normal_operation(TTF)
        else:
            failure_load=g.getVertex(predlist[0]) #取出故障的节点
            failure_load.add_normal_operation(TTF+TTR)
    MCTime=TTF+MCTime+TTR
    #print(MCTime)   #查看进度
#求四个参数
SAIFI=0
SAIDI=0
CAIDI=0
ASAI=0
alluser_number=0
failure_user_number=0
for i in Loads:
    alluser_number=alluser_number+g.getVertex(i).get_user_number()
    SAIFI=SAIFI+g.getVertex(i).get_failure_rate()*g.getVertex(i).get_user_number()
    SAIDI=SAIDI+g.getVertex(i).get_U()*g.getVertex(i).get_user_number()
    if g.getVertex(i).get_failure_number()!=0:
        failure_user_number+=g.getVertex(i).get_user_number()
        CAIDI+=g.getVertex(i).get_U()*g.getVertex(i).get_user_number()
CAIDI/=SAIFI
SAIFI/=alluser_number
SAIDI/=alluser_number
CAIDI/=failure_user_number
ASAI=(1-SAIDI/8760)
#输出
print(SAIFI,SAIDI,CAIDI,ASAI)