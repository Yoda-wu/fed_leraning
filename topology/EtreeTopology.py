import numpy as np
from fedml.core.distributed.topology.base_topology_manager import BaseTopologyManager
import sys

class EtreeTopologyManager(BaseTopologyManager):
    """
    The topology definition is determined by this initialization method.
    It a tree-base topology
    Arguments:
        args : delayMatrix and each node's accuracy
        n (int): number of nodes in the topology.
        client_id_list_in_total(list): client id list 
        groups(list): len(groups) == layers, represent the cluster number of each layer 
    """
    def __init__(self, args, client_id_list_in_total, groups, n, layers=3, K=5):
        self.args = args
        self.client_id_list_in_total = client_id_list_in_total
        self.n = n
        self.layers = layers
        self.groups = groups
        self.topology = []  # n * n if topology[i][j] = 1 means client i and client j is connect
        self.clusterOfAllLayer = [[] for i in range(layers -1)] # clusters
        self.centroidsOfAllLayers = [[] for i in range(layers -1)] 
    
    def generate_topology(self):
        layers = self.layers
        resultsOfAllLayers = [[] for x in range(layers - 1)]
        centroidsOfAllLayers = [[] for x in range(layers - 1)]
        
        lastNodeIndexes = [x for x in range(len(self.client_id_list_in_total))]
        for layer in range(layers - 1):
            if layer > 0 :
                res, centroids = self.KMA(lastNodeIndexes,self.groups[layer])
            else:
                res, centroids, iteration, is_retain, total_difference, is_contain_single_node, num_of_group, mean_avg_acc_group, std_avg_acc_group, mean_acc_group, std_acc_group = self.KMA(lastNodeIndexes,self.groups)
            # update the nodes to be clustered
            lastNodeIndexes = centroids
            centroidsOfAllLayers[layer] = centroids
            resultsOfAllLayers[layer] = res
        self.centroidsOfAllLayers = resultsOfAllLayers
        self.clusterOfAllLayer = centroidsOfAllLayers

        # topology -- networkx 
        # we know if topology[i][j] = 1 means client i and client j is connect
        # in etree , node only can communicate with cluster head.
        topology = []
        
        cluster = resultsOfAllLayers[len(resultsOfAllLayers) - 1][0]
        centerId = lastNodeIndexes[cluster[len(cluster) - 1]]
        numOfChildren = len(cluster) - 1
        host_info = {
            "name": "n" + str(centerId),
            "layer": layers,
            "sync": 1,
            "child_num": numOfChildren,
        }
        topology.append(host_info)


        lastNodeIndexes = [cluster[ind] for ind in range(len(cluster) - 1)]
        currentLayer = layers - 1
        while currentLayer > 1:
            newLastNodeIndexes = []
            curRes = resultsOfAllLayers[currentLayer - 2]
            for l1 in lastNodeIndexes:
                m = 0
                while curRes[m][len(curRes[m]) - 1] != l1:
                    m += 1
                cluster = curRes[m]
                centerId = lastNodeIndexes[l1]
                numOfChildren = len(cluster) - 1
                layer = currentLayer
                host_info = {
                    "name": "n" + str(centerId),
                    "layer": layer,
                    "sync": 1,
                    "child_num": numOfChildren,
                }
                topology.append(host_info)
                for l2 in range(numOfChildren):
                    newLastNodeIndexes.append(cluster[l2])
            lastNodeIndexes = newLastNodeIndexes
            currentLayer -= 1
        self.topology = topology
        
    def KMA(self,nodeIdList, k, delta=0.5, iterations=50 ):
        """
        KMA algorithm
        """
        # Randomly select K nodes as initial center nodes, and initialize a cluster for each center node;
        centroids = np.random.choice(nodeIdList,self.k,replace=False)
        finalCentroids = centroids
         # init the cluster
        clusterList = []
        for i in range(self.k):
            clusterList.append([])
        # calc the global accuracy 
        globalAcc = np.mean(self.args.accuracies)
        # all client's accuracy
        accuracies = self.args.accuracies
        # centroids's average accuracies
        centroidsAvgAccuracies = [0.0 for x in range(self.k)]
        # centroids's accuracies
        centroidsAccuracies= [0.0 for x in range(self.k)]
        # init the two centroids accuracies list 
        for i in range(k):
            centroidsAvgAccuracies[i] = accuracies[centroids[i]] 
            centroidsAccuracies[i] = accuracies[centroids[i]]
        mdm = self.args.mdm
        delta = self.args.delta
        end = False
        iteration = 0
        size = len(nodeIdList)
        while not end and iteration < iterations :
            end = True
            for i in range(size):
                currentNodeId = nodeIdList[i]
                # Sort the current K center nodes according to their distance to ni in an ascending order;
                centroids_copy = sorted(centroids, lambda x : mdm[currentNodeId][x])
                acc_of_node = accuracies[currentNodeId]
                add = False
                for j in range(self.k):
                    avgAcc = centroidsAvgAccuracies[j] # cluster_j 's avg accuracy
                    current_size = len(clusterList[j])
                    newAvg = (avgAcc * current_size + acc_of_node) / (current_size + 1)
                    diff = newAvg - globalAcc
                    diff = np.abs(diff)
                    if diff < delta:
                        add = True
                        clusterList[j].append(currentNodeId)
                        centroidsAvgAccuracies = newAvg
                        centroidsAccuracies = newAvg * (current_size + 1)
                # Assign ni to the cluster whose center node has the shortest transmission delay to it ;
                if not add :
                    nearestCluster = centroids_copy[0]
                    avgAcc = centroidsAvgAccuracies[nearestCluster]
                    clusterSize = len(clusterList[nearestCluster])
                    newAvg = (avgAcc * clusterSize + acc_of_node) / (clusterSize + 1)
                    clusterList[nearestCluster].append(currentNodeId    )
                    centroidsAccuracies[nearestCluster] = newAvg * (clusterSize + 1)
                    centroidsAvgAccuracies[nearestCluster] = newAvg

                # compute the newest centroid  
                # Update the center node of the clusters using Equation (1); (in the essay)
                for i in range(self.k):
                    minTotalDelay = sys.maxsize
                    newCentroid = clusterList[i][0]
                    for j in range(len(clusterList[i])):
                        totalDelay = 0
                        for id in clusterList[i] :
                            if id == clusterList[i][j] :
                                continue
                            totalDelay += mdm[id][clusterList[i][j]]
                        if totalDelay < minTotalDelay:
                            minTotalDelay = totalDelay
                            newCentroid = clusterList[i][j]
                    
                    if newCentroid != centroids[i]:
                        end = False
                        centroids[i] = newCentroid
                    
            finalCentroids = centroids    
            iteration += 1
        # end while

        # put centroids in the last 
        # so we can easily get the centroids easily
        for i in range(len(finalCentroids)):
            clusterList.append(finalCentroids[i])
        is_contain_single_node = 0
        for i in clusterList:
            num_in_group = len(i)
            if num_in_group == 1:
                is_contain_single_node = 1

        total_difference = 0.0
        for i in centroidsAvgAccuracies:
            difference_value = i[0] - globalAcc
            difference_value = np.abs(difference_value)
            total_difference = total_difference + difference_value

        mean_avg_acc_group = np.mean(centroidsAvgAccuracies)
        std_avg_acc_group = np.std(centroidsAvgAccuracies)
        num_of_group = len(clusterList)

        mean_acc_group = np.mean(centroidsAccuracies)
        std_acc_group = np.std(centroidsAccuracies)

        if iteration < iterations:
            is_retain = 1
        else:
            is_retain = 0

        return clusterList, finalCentroids,iteration, is_retain, total_difference, is_contain_single_node, num_of_group, mean_avg_acc_group, std_avg_acc_group, mean_acc_group, std_acc_group        

        

    def get_in_neighbor_idx_list(self, node_index):
        """
        input: node_index current client index,
        output: node list which current client can get info from.
        """
        neighbor_in_idx_list = []
        
        return neighbor_in_idx_list

    def get_out_neighbor_idx_list(self, node_index):
        """
        input: node_index current client index,
        utput: node list which current client can send info to.
        """
        neighbor_out_idx_list = []

        return neighbor_out_idx_list
        