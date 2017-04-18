from scipy.stats import itemfreq

# Calculate the gini index for a  group
def gini_index(ys):
    '''
    #input
    groups: numpy array of size N about group label
    #output
    gini: scalar of gini index for the group ys
    '''
    if len(ys)==0: return 0
    freqs = itemfreq(ys)[:,1]
    freqs = freqs/sum(freqs)
    gini = 1-sum(freqs**2)
    return gini