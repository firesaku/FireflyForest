from scipy.spatial import KDTree
from dadapy.data import Data
import warnings
import numpy as np
from treelib import Tree
import dataPreprocessing
import skdim
from diptest import diptest
from sklearn.datasets import make_blobs
import math
from sklearn.decomposition import PCA
import copy


def find_max_k(data,Dthr = 23.928,omega=1,max_k=100,IQR_threhold=1.5):
    lpca = skdim.id.lPCA().fit_pw(data,
                                  n_neighbors=max_k,
                                  n_jobs=1)


    id=np.mean(lpca.dimension_pw_)
    pca = PCA(n_components=math.ceil(id))
    transformed_data = pca.fit_transform(data)
    tree = KDTree(transformed_data)
    distances, indices = tree.query(transformed_data, k=min(max_k, data.shape[0]))
    dissimilarity=np.power(distances,id)
    V_matrix = np.diff(dissimilarity, axis=1)*omega

    
    list_k = [-1] * len(data)
    list_rou = [-1] * len(data)
    list_error = [-1] * len(data)
    list_light = [-1] * len(data)
    for i in range(len(data)):
        Dk_flag = False 
        now_k = 0 
        while True:
            now_k += 1
            j = indices[i][now_k]
            Dk = -2 * now_k * (np.log(np.sum(V_matrix[i][:now_k])) + np.log(np.sum(V_matrix[j][:now_k])) - 2 * np.log(
                np.sum(V_matrix[i][:now_k]) + np.sum(V_matrix[j][:now_k])) + np.log(4))
            Dk1 = -2 * now_k * (np.log(np.sum(V_matrix[i][:now_k+1])) + np.log(np.sum(V_matrix[j][:now_k+1])) - 2 * np.log(
                np.sum(V_matrix[i][:now_k+1]) + np.sum(V_matrix[j][:now_k+1])) + np.log(4))
            if Dk<Dthr:
                Dk_flag=True
            if ((Dk1 >= Dthr) and (Dk_flag==True)) or (now_k==min(max_k-1,data.shape[0])) == True: #如果【达到阈值】 或者 【遍历到最大近邻数】 则停止遍历
                V_list = copy.copy(V_matrix[i][:now_k])
                Q1 = np.percentile(V_list, 25)
                Q3 = np.percentile(V_list, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - IQR_threhold * IQR
                upper_bound = Q3 + IQR_threhold * IQR
                outlier_indices = [index for index, value in enumerate(V_list) if value<lower_bound or value > upper_bound]
                V_list = [value for index, value in enumerate(V_list) if index not in outlier_indices]
                list_k[i] = len(V_list)
                list_rou[i] = list_k[i] / np.sum(V_list)
                list_error[i] = np.sqrt((4 * list_k[i] + 2) / ((list_k[i] - 1) * list_k[i]))
                list_light[i] = np.log(list_rou[i]) / list_error[i]
                break



    return list_k, list_rou, list_error, list_light, distances, indices


def Model(data, Dthr = 23.928, omega=1, Mthr=10, max_k=100,Ithr=2):
    list_k, list_rou, list_error, list_light, distances, indices = find_max_k(data,Dthr,omega,max_k,Ithr) 
    root_firefly = []

    firefly_tree_list = [] 
    label = [-1] * data.shape[0] 
    label_index = 0 
    sorted_list_light_id = sorted(range(len(list_light)), key=lambda k: list_light[k], reverse=True) 


    for i in range(0, len(sorted_list_light_id)): 
        root_firefly_flag = True 
        for j in range(1, list_k[sorted_list_light_id[i]]): 
            higher_light_point = indices[sorted_list_light_id[i]][j] 
            if list_light[higher_light_point] > list_light[sorted_list_light_id[i]]: 
                root_firefly_flag = False
                break
        if root_firefly_flag == True: 
            root_firefly.append(sorted_list_light_id[i]) 
            label[sorted_list_light_id[i]] = label_index 
            tree = Tree() 
            tree.create_node(identifier=sorted_list_light_id[i], data=sorted_list_light_id[i])
            firefly_tree_list.append(tree) 
            label_index += 1 


        else: 
            firefly_tree_list[label[higher_light_point]].create_node(parent=higher_light_point,
                                                             identifier=sorted_list_light_id[i],
                                                             data=sorted_list_light_id[i]) 
            label[sorted_list_light_id[i]] = label[higher_light_point] 

    while True:
        for i in range(0,len(firefly_tree_list)):
            nodes=firefly_tree_list[i].all_nodes()
            for j in range(0,len(nodes)):
                label[nodes[j].data]=i
        merge_array = np.full((len(root_firefly), len(root_firefly)),np.inf)  
        for i in range(0, len(root_firefly)):  
            leaf_firefly_i = firefly_tree_list[i].leaves() 
            leaf_firefly_list_i = []  
            for leaf_firefly in leaf_firefly_i:
                leaf_firefly_list_i.append(leaf_firefly.data)
            leaf_firefly_set_i = set(leaf_firefly_list_i)
            for j in leaf_firefly_set_i:  
                for k in indices[j][:list_k[j]]:  
                    if label[root_firefly[i]] != label[k]:  
                        if list_light[root_firefly[label[k]]]-list_light[j]<merge_array[label[root_firefly[i]]][label[k]]:
                            merge_array[label[root_firefly[i]]][label[k]] = list_light[root_firefly[label[k]]]-list_light[j]
        min_value = np.min(merge_array) 
        if min_value>=Mthr:
            break
        max_loc=np.where(merge_array==min_value)
        row_index = max_loc[0][0]
        col_index = max_loc[1][0]
        if list_light[root_firefly[row_index]]>=list_light[root_firefly[col_index]]:
            firefly_tree_list[row_index].paste(firefly_tree_list[row_index].root,firefly_tree_list[col_index])# 树合并
            root_firefly.remove(root_firefly[col_index])
            firefly_tree_list.remove(firefly_tree_list[col_index])
        else:
            firefly_tree_list[col_index].paste(firefly_tree_list[col_index].root, firefly_tree_list[row_index])  # 树合并
            root_firefly.remove(root_firefly[row_index])
            firefly_tree_list.remove(firefly_tree_list[row_index])

    for i in range(0,len(firefly_tree_list)):
        nodes=firefly_tree_list[i].all_nodes()
        for j in range(0,len(nodes)):
            label[nodes[j].data]=i
    return label,merge_array


