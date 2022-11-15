from math import floor
import random
from typing import List

def unique(arr):
    unique_set = set(arr)
    return (list(unique_set))

def deep_copy2d(copy):
    return [[i for i in row] for row in copy]

def deep_copy1d(copy):
    return [i for i in copy]

def most_frequent(arr):
    counter = 0
    num = arr[0]
     
    for i in arr:
        curr_frequency = arr.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
    return num

def distance_btwn(point1,point2):
    return ((point1[0]-point2[0])**2+(point1[1]-point2[1])**2+(point1[2]-point2[2])**2+(point1[3]-point2[3])**2)**0.5 #euclidean distance
class KMeansClusterClassifier:
    def __init__(self, n_cluster=3, m_iter=100):
        self.n_cluster = n_cluster
        self.m_iter = m_iter
        self.clusters=[[] for i in range(n_cluster)] # boş clusterlar tanımlanıyor
        self.centroids=[]

    def fit(self, X: List[List[float]], y: List[int]):
        if(len(X)<1): return;
        r=len(X)
        c=len(X[0])
        data = deep_copy2d(X)
        for i in range(len(y)):#y değerleriyle X değerleri eşleştiriliyor
            data[i].append(y[i])
        self.centroids, self.clusters = self.find_clusters_and_centroids(data)
        
    def find_clusters_and_centroids(self,data):
        centroids = []
        for _ in range(self.n_cluster):
            centroids.append(random.choice(data))#rastgele centroidler seçiliyor
        for _ in range(self.m_iter):
            clusters = [[] for i in range(self.n_cluster)]
            for point in data:
    
                closesest=0
                closesest_dist = 9223372036854775806#maximum value of a 64 bit integer
                for close,centroid in enumerate(centroids):
                    distance = distance_btwn(point,centroid)#mevcut centroid ile nokta arasındaki mesafeyi hesaplıyor
                    if distance < closesest_dist: #en kısa mesafeye sahip olan centroid seçiliyor
                        closesest_dist = distance 
                        closesest = close
                clusters[closesest].append(point) #en kısa mesafeye sahip olan centroidin clusterına nokta ekleniyor
            for num,_ in enumerate(centroids):#centroidler yeni değerlerine atanıyor
                if(len(clusters[num])>0):
                    centroids[num][0] = sum(row[0] for row in clusters[num])/len(clusters[num])#o clusterdaki ortalama column_0 değeri
                    centroids[num][1] = sum(row[1] for row in clusters[num])/len(clusters[num])#o clusterdaki ortalama column_1 değeri
                    centroids[num][2] = sum(row[2] for row in clusters[num])/len(clusters[num])#o clusterdaki ortalama column_2 değeri
                    centroids[num][3] = sum(row[3] for row in clusters[num])/len(clusters[num])#o clusterdaki ortalama column_3 değeri
            
            
        return centroids,clusters
        
    
    def predict(self, X: List[List[float]]):
        predictions = []
        for point in X:
            closesest=0
            closesest_dist = 9223372036854775806
            for close,centroid in enumerate(self.centroids):#noktaya en yakın centroid seçiliyor
                    distance = distance_btwn(point,centroid)
                    if distance < closesest_dist:
                        closesest_dist = distance
                        closesest = close

            label = most_frequent([row[4] for row in self.clusters[closesest]])#clusters[closesest] içindeki en çok tekrar eden y değeri tahmin olarak seçiliyor
            predictions.append(label)
        return predictions
    
    