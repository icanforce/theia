import config
import pickle
prefix = config.Config.results
#-----------------------------------------------------------------------------------------------------        
def _find(U,V, M, N, K, th=0.5):
    module_mi = []
    module_m = []
    for k in range(K):
        module_mi.append([])
        module_m.append([])
        for i in range(M):
            if U[i,k] > th:
                module_mi[k].append(i)
        for j in range(N):
            if V[j,k] > th:        
                module_m[k].append(j)                
    return module_mi, module_m
#-----------------------------------------------------------------------------------------------------        
def find(name_mi, name_m, U,V, M, N, K, th=0.5):
    mi, m = _find(U,V, M, N, K, th=0.5)
    modules = []
    module_mi = []
    module_m = []
    
    fd = open(prefix + 'module.txt', 'w')
    for k in range(K):
        module_mi.append([])
        module_m.append([])    
        for i in range(len(mi[k])):  
            module_mi[k].append(name_mi[mi[k][i]])
        for j in range(len(m[k])):  
            module_m[k].append(name_m[m[k][j]])
        modules.append([module_mi[k], module_m[k]])
    fd.write("[\n")        
    for k in range(K):            
        # fd.write("Group %d: miRNAs %s, genes %s" % (k, str(module_mi[k]), str(module_m[k]))  + '\n')    
        fd.write("[%s, %s]" % (str(module_mi[k]), str(module_m[k]))  + '\n')    
    fd.write("]")        
    fd.write('\n\n\n')
    for k in range(K):            
        # fd.write("Group %d: miRNAs %s, genes %s" % (k, str(mi[k]), str(m[k]))  + '\n')    
        fd.write("[%s, %s]" % (str(mi[k]), str(m[k]))  + '\n')    
    fd.close()
    
    with open(prefix + 'module.dat', 'wb') as f:
        pickle.dump(modules, f)
        
    return module_mi, module_m
#-----------------------------------------------------------------------------------------------------        
def find_set(U,V, M, N, K, th=0.5):
    mi, m = _find(U, V, M, N, K, th=0.5)
    gene = lambda x: 'G'+str(x)
    
    clusters = []
    for k in xrange(K):
        candidate = []
        candidate += map(str, mi[k])
        candidate += map(gene, m[k])
        clusters.append(set(candidate))
    return clusters
        
    
    
    
    
