import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def degree_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]
    return Dn

def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


def get_uniform_graph(num_node, self_link, neighbor):
    A = normalize_digraph(edge2mat(neighbor + self_link, num_node))
    return A


def get_uniform_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    A = I - N
    return A


def get_distance_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    N = normalize_digraph(edge2mat(neighbor, num_node))
    A = np.stack((I, N))
    return A


def get_spatial_graph(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def get_spatial_relation_graph(num_node, self_link, inward, outward, neighbor):

    A_1 = degree_undigraph(edge2mat(neighbor, num_node))

    In = edge2mat(inward, num_node)
    Out = edge2mat(outward, num_node)
    A_2 = In + Out

    A = np.stack((A_1, A_2))
    return A

def get_spatial_relation_pair_graph(num_node, self_link, inward, outward, neighbor):

    list_link = [[], [], [], [], [],
                 [], [], [], [], [],
                 [], [], [], [], [],
                 [], [], [], [], [],
                 [], [], [], [], []]
    lenl = np.zeros((num_node, 1))
    for link in inward:
        list_link[link[0]].append(link)
        lenl[link[0]] += 1
    for link in outward:
        list_link[link[0]].append(link)
        lenl[link[0]] += 1

    numl = int(max(lenl)[0])
    AA = []
    AA.append(edge2mat(self_link, num_node))

    for num in range(numl):
        A = np.zeros((num_node, num_node))
        for node in range(num_node):
            ll = len(list_link[node])
            if num+1 <= ll:
                i =  list_link[node][num][0]
                j =  list_link[node][num][1]
                A[j, i] = 1
        AA.append(A)

    AA = np.stack((AA[0], AA[1], AA[2], AA[3], AA[4]))
    return AA

def get_complete_relation_pair_graph(num_node, self_link, inward, outward, neighbor):
    list_link = [[], [], [], [], [],
                 [], [], [], [], [],
                 [], [], [], [], [],
                 [], [], [], [], [],
                 [], [], [], [], []]

    for node in range(num_node):
        for j in range(num_node):
            if j != node:
                list_link[node].append(j)

    numl = num_node - 1
    AA = []
    AA.append(edge2mat(self_link, num_node))

    for num in range(numl):
        A = np.zeros((num_node, num_node))
        for node in range(num_node):
            ll = len(list_link[node])
            if num+1 <= ll:
                i =  node
                j =  list_link[node][num]
                A[j, i] = 1
        AA.append(A)

    AA = np.stack((AA[0], AA[1], AA[2], AA[3], AA[4],
                   AA[5], AA[6], AA[7], AA[8], AA[9],
                   AA[10], AA[11], AA[12], AA[13], AA[14],
                   AA[15], AA[16], AA[17], AA[18], AA[19],
                   AA[20], AA[21], AA[22], AA[23], AA[24]
                   ))
    return AA

def get_DAD_graph(num_node, self_link, neighbor):
    A = normalize_undigraph(edge2mat(neighbor + self_link, num_node))
    return A


def get_DLD_graph(num_node, self_link, neighbor):
    I = edge2mat(self_link, num_node)
    A = I - normalize_undigraph(edge2mat(neighbor, num_node))
    return A