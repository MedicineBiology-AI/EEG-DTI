import numpy as np
from matplotlib import pyplot
import scipy.sparse as sp
import time
import scipy.stats

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

def drawHist_cell_gene(mat):
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

# load_protein_drug_interactions and drug proten
def load_protein_drug_interactions(threshold=0,toone=0,draw=0,margin = 0.,path=''):
    time1 = time.time()
    print('==========Loading.....Part one: drug-protein=============')
    protein_drug_interactions = np.loadtxt(path, dtype=str, delimiter='\t')
    print('Before:', protein_drug_interactions.shape)
    protein_drug_interactions = protein_drug_interactions[1:, ]
    protein_drug_interactions = protein_drug_interactions[:, 1:]
    protein_drug_interactions = np.array(protein_drug_interactions,dtype=int)
    # protein_drug_interactions = np.loadtxt(path)
    print(protein_drug_interactions.shape)

    # If draw the histPic?
    if draw==0:pass
    else:drawHist_cell_gene(protein_drug_interactions)

    print('Before:')
    print('(0.0~0.1]=', sum(sum(protein_drug_interactions > 0.0)) - sum(sum(protein_drug_interactions >= 0.1)))
    print('(0.1~0.2]=', sum(sum(protein_drug_interactions > 0.1)) - sum(sum(protein_drug_interactions >= 0.2)))
    print('(0.2~0.3]=', sum(sum(protein_drug_interactions > 0.2)) - sum(sum(protein_drug_interactions >= 0.3)))
    print('(0.3~0.4]=', sum(sum(protein_drug_interactions > 0.3)) - sum(sum(protein_drug_interactions >= 0.4)))
    print('(0.4~0.5]=', sum(sum(protein_drug_interactions > 0.4)) - sum(sum(protein_drug_interactions >= 0.5)))
    print('(0.5~0.6]=', sum(sum(protein_drug_interactions > 0.5)) - sum(sum(protein_drug_interactions >= 0.6)))
    print('(0.6~0.7]=', sum(sum(protein_drug_interactions > 0.6)) - sum(sum(protein_drug_interactions >= 0.7)))
    print('(0.7~0.8]=', sum(sum(protein_drug_interactions > 0.7)) - sum(sum(protein_drug_interactions >= 0.8)))
    print('(0.8~0.9]=', sum(sum(protein_drug_interactions > 0.8)) - sum(sum(protein_drug_interactions >= 0.9)))
    print('(0.9~1.0]=', sum(sum(protein_drug_interactions > 0.9)) - sum(sum(protein_drug_interactions > 1.0)))
    # print('1,0=', sum(sum(protein_drug_interactions >= 1.0)))
    # If change by the threshold?
    if threshold==0:pass
    else:protein_drug_interactions[protein_drug_interactions < threshold]=0

    # If change all data to be 1?
    if toone==0:pass
    else:protein_drug_interactions[(protein_drug_interactions !=0)] = 1

    protein_drug_interactions[(protein_drug_interactions != 0)] = 1
    # get adj of cell_gene and feature
    drug_proten_interactions = protein_drug_interactions.transpose()
    interactions_margin = drug_proten_interactions
    print('COUNT:   After threshold chose num_count = ',sum(sum(protein_drug_interactions>0)))
    print('COUNT:   sparse rate = ',str(round(100*sum(sum(protein_drug_interactions>threshold))/(protein_drug_interactions.shape[0]*protein_drug_interactions.shape[1]),3))+'%')
    # print(gene_cell_adj)
    print('COUNT:   Drug    numbers = ',drug_proten_interactions.shape[0])
    print('COUNT:   Protein numbers = ', drug_proten_interactions.shape[1])

    protein_drug_interactions =  sp.csr_matrix(protein_drug_interactions)
    drug_proten_interactions = sp.csr_matrix(drug_proten_interactions)
    time2 = time.time()
    # print(time2)
    print('load time is = ',round(time2-time1,3))
    return  drug_proten_interactions,protein_drug_interactions,interactions_margin


def load_adj_by_p_valuethreshold(threshold=0,toone=0,draw=0,sim_path=''):
    print('===============Loading.....Part two: Sim and feat================')
    # print('threshold = ', threshold)
    Drug_Drug_sim = np.loadtxt(sim_path, dtype=str, delimiter='\t')
    print('Before:', Drug_Drug_sim.shape)
    Drug_Drug_sim = Drug_Drug_sim[1:, ]
    Drug_Drug_sim = Drug_Drug_sim[:, 1:]
    Drug_Drug_sim = np.array(Drug_Drug_sim, dtype=float)
    print('Before select by p_value')
    print_count(Drug_Drug_sim)
    mean = np.mean(Drug_Drug_sim)
    std = np.std(Drug_Drug_sim)

    for i in range(Drug_Drug_sim.shape[0]):
        for j in range(Drug_Drug_sim.shape[1]):
            z_score = (float(Drug_Drug_sim[i][j]) - mean) / std
            #scipy.stats.norm.ppf(0.95)
            if z_score >= 1.64:
                Drug_Drug_sim[i][j]=1
            else:
                Drug_Drug_sim[i][j]=0

    for i in range(Drug_Drug_sim.shape[0]):
        for j in range(Drug_Drug_sim.shape[1]):
            if i == j:
                Drug_Drug_sim[i][j] = 0


    for i in range(Drug_Drug_sim.shape[0]):
        for j in range(Drug_Drug_sim.shape[1]):
            if Drug_Drug_sim[i][j] != Drug_Drug_sim[j][i]:
                Drug_Drug_sim[j][i] = Drug_Drug_sim[i][j]


    print('After select by p_value')
    print_count(Drug_Drug_sim)
    Drug_Drug_sim =  sp.csr_matrix(Drug_Drug_sim)
    return Drug_Drug_sim

