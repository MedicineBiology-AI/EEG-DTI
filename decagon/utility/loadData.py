import numpy as np
from matplotlib import pyplot
import scipy.sparse as sp
import time
import sklearn.preprocessing as prep


def minmax_scale(x_train):
    '''
    do min-max normalization
    '''
    preprocessor = prep.MinMaxScaler()
    x_train = preprocessor.fit_transform(x_train)
    return x_train


def standrad_scale(x_train):
    '''
    do standard normalization
    '''
    preprocessor = prep.StandardScaler()
    x_train = preprocessor.fit_transform(x_train)
    return x_train


def drawHist(mat):
    pyplot.hist(mat, 100)
    pyplot.xlabel('Numbers Distribution between 0 and 1')
    pyplot.ylabel('Frequency')
    pyplot.title('Cell-Gene Edges Total')
    pyplot.savefig('./sc_ppi/zeisel/cell_gene_distribution.png',dpi=300)


def drawHist_gene_gene(mat):
    pyplot.hist(mat, 100)
    pyplot.xlabel('Numbers Distribution between 0 and 1')
    pyplot.ylabel('Frequency')
    pyplot.title('Gene-Gene Edges Total')


def print_count(mat):
    print('[0.0~0.1)=', sum(sum(mat >= 0.0)) - sum(sum(mat >= 0.1)))
    print('[0.1~0.2)=', sum(sum(mat >= 0.1)) - sum(sum(mat >= 0.2)))
    print('[0.2~0.3)=', sum(sum(mat >= 0.2)) - sum(sum(mat >= 0.3)))
    print('[0.3~0.4)=', sum(sum(mat >= 0.3)) - sum(sum(mat >= 0.4)))
    print('[0.4~0.5)=', sum(sum(mat >= 0.4)) - sum(sum(mat >= 0.5)))
    print('[0.5~0.6)=', sum(sum(mat >= 0.5)) - sum(sum(mat >= 0.6)))
    print('[0.6~0.7)=', sum(sum(mat >= 0.6)) - sum(sum(mat >= 0.7)))
    print('[0.7~0.8)=', sum(sum(mat >= 0.7)) - sum(sum(mat >= 0.8)))
    print('[0.8~0.9)=', sum(sum(mat >= 0.8)) - sum(sum(mat >= 0.9)))
    print('[0.9~1.0]=', sum(sum(mat >= 0.9)) - sum(sum(mat > 1.0)))

# load_protein_drug_interactions and drug proten
def load_protein_drug_interactions(threshold=0,toone=0,draw=0,path=''):
    time1 = time.time()
    print('==========Loading.....Part one: drug-protein=============')
    protein_drug_interactions = np.loadtxt(path)
    print(protein_drug_interactions.shape)

    # If draw the histPic?
    if draw==0:pass
    else:drawHist(protein_drug_interactions)

    print('Before:')
    print_count(protein_drug_interactions)

    # If change by the threshold?
    if threshold==0:pass
    else:protein_drug_interactions[protein_drug_interactions < threshold]=0

    # If change all data to be 1?
    if toone==0:pass
    else:protein_drug_interactions[(protein_drug_interactions !=0)] = 1

    # protein_drug_interactions[(protein_drug_interactions != 0)] = 1
    drug_proten_interactions = protein_drug_interactions.transpose()
    print('COUNT:   After threshold chose num_count = ',sum(sum(protein_drug_interactions>0)))
    print('COUNT:   sparse rate = ',str(round(100*sum(sum(protein_drug_interactions>threshold))/(protein_drug_interactions.shape[0]*protein_drug_interactions.shape[1]),3))+'%')
    print('COUNT:   Drug    numbers = ',drug_proten_interactions.shape[0])
    print('COUNT:   Protein numbers = ', drug_proten_interactions.shape[1])

    # Sparse process.....
    protein_drug_interactions = sp.csr_matrix(protein_drug_interactions)
    drug_proten_interactions = sp.csr_matrix(drug_proten_interactions)
    time2 = time.time()
    # print(time2)
    print('load time is = ',round(time2-time1,3))
    return  drug_proten_interactions,protein_drug_interactions

def Load_Drug_Adj_Togerther(drug_drug_path = '',drug_drug_sim_chemical_path = '',
                            drug_drug_sim_interaction_path = '',drug_drug_sim_se_path='',\
                            drug_drug_sim_disease_path=''):
    print('Load_Drug_Adj_Togerther')
    th = 0.5
    a = np.loadtxt(drug_drug_path)
    b = np.loadtxt(drug_drug_sim_chemical_path)
    c = np.loadtxt(drug_drug_sim_interaction_path)
    d = np.loadtxt(drug_drug_sim_se_path)
    e = np.loadtxt(drug_drug_sim_disease_path)

    print('-------------Before--------------')
    print_count(a)
    b[b >= th] = 1
    b[b < th] = 0

    c[c >= th] = 1
    c[c < th] = 0

    d[d >= th] = 1
    d[d < th] = 0

    e[e >= th] = 1
    e[e < th] = 0

    Final = a+b+c+d+e
    # Final = a
    Final[Final >= 1] = 1

    for i in range(Final.shape[0]):
        for j in range(Final.shape[1]):
            if i == j:
                Final[i][j] = 0

    for i in range(Final.shape[0]):
        for j in range(Final.shape[1]):
            if Final[i][j] == 1:
                Final[j][i] = 1
    print('-------------After--------------')
    print_count(Final)
    Final = sp.csr_matrix(Final)

    return Final

def Load_Protein_Adj_Togerther(protein_protein_path='', protein_protein_sim_sequence_path='',
                            protein_protein_sim_disease_path='', protein_protein_sim_interaction_path=''):
    print('Load_Protein_Adj_Togerther')
    th = 0.5
    a = np.loadtxt(protein_protein_path)
    b = np.loadtxt(protein_protein_sim_sequence_path)
    b = b/100
    c = np.loadtxt(protein_protein_sim_disease_path)
    d = np.loadtxt(protein_protein_sim_interaction_path)
    print('Before')
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i == j:
                a[i][j] = 0
    print_count(a)
    b[b >= th] = 1
    b[b < th] = 0
    c[c >= 0.8] = 1
    c[c < 0.8] = 0
    d[d >= th] = 1
    d[d < th] = 0
    Final = a + b + c +d

    Final[Final >= 1] = 1

    for i in range(Final.shape[0]):
        for j in range(Final.shape[1]):
            if i == j:
                Final[i][j] = 0

    for i in range(Final.shape[0]):
        for j in range(Final.shape[1]):
            if Final[i][j] == 1:
                Final[j][i] = 1
    print('After')
    print_count(Final)
    Final = sp.csr_matrix(Final)

    return Final


def load_Adj_adj(threshold=0,toone=0,draw=0,sim_path=''):
    print('=========Loading.....: and feat=========='+sim_path)
    print('threshold = ', threshold)
    time1 = time.time()
    Adj = np.loadtxt(sim_path)

    # If draw the histPic?
    if draw==0:pass
    else:drawHist_gene_gene(Adj)

    print('Before:')
    print_count(Adj)

    # If change by the threshold?
    if threshold==0:pass
    else:Adj[Adj < threshold]=0

    # If change all data to be 1?
    if toone==0:pass
    else:Adj[(Adj !=0)] = 1
    print('COUNT:    numbers=',Adj.shape[0])

    # get adj of cell_gene and feature
    # Protein_feature = np.loadtxt(feat_path)
    # print('COUNT:   protein feature=', Protein_feature.shape)
    print('COUNT:   After threshold chose num_count is ',sum(sum(Adj>0)))
    print('COUNT:   sparse rate is ',str(round(100*sum(sum(Adj>0))/(Adj.shape[0]*Adj.shape[1]),3))+'%')
    # print(Protein_Protein_sim)

    Adj =  sp.csr_matrix(Adj)
    # Protein_feature = sp.csr_matrix(Protein_feature)
    time2 = time.time()
    print('load time is = ',round(time2-time1,3))
    return Adj

def load_Adj_adj_transpose(threshold=0,toone=0,draw=0,sim_path=''):
    print('=========Loading.....: and feat=========='+sim_path)
    print('threshold = ', threshold)
    time1 = time.time()
    Adj = np.loadtxt(sim_path)
    Adj = Adj.transpose()
    # If draw the histPic?
    if draw==0:pass
    else:drawHist_gene_gene(Adj)

    print('Before:')
    print_count(Adj)

    # If change by the threshold?
    if threshold==0:pass
    else:Adj[Adj < threshold]=0

    # If change all data to be 1?
    if toone==0:pass
    else:Adj[(Adj !=0)] = 1
    print('COUNT:    numbers=',Adj.shape[0])

    # get adj of cell_gene and feature
    # Protein_feature = np.loadtxt(feat_path)
    # print('COUNT:   protein feature=', Protein_feature.shape)
    print('COUNT:   After threshold chose num_count is ',sum(sum(Adj>0)))
    print('COUNT:   sparse rate is ',str(round(100*sum(sum(Adj>0))/(Adj.shape[0]*Adj.shape[1]),3))+'%')
    # print(Protein_Protein_sim)

    Adj =  sp.csr_matrix(Adj)
    # Protein_feature = sp.csr_matrix(Protein_feature)
    time2 = time.time()
    print('load time is = ',round(time2-time1,3))
    return Adj
