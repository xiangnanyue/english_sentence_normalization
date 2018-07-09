# this algorithm calculate the LevenshteinDistcance

import numpy as np

def Levenshtein_Distcance(str1, str2):
    m = len(str1)
    n = len(str2)
    table = np.zeros((m+1, n+1))

    for i in range(n+1):
        table[0, i] = i
    for j in range(m+1):
        table[j, 0] = j
    
    cost = 0
    for i in range(1,m+1):
        for j in range(1,n+1):
            if str1[i-1] == str2[j-1]:
                cost = 0
            else:
                cost = 1
            
            table[i, j] = min(table[i-1, j]+1, table[i, j-1]+1, table[i-1, j-1]+cost)
    
    return table[m, n]


if __name__ == '__main__':
    str1 = "kitten"
    str2 = "sitting"
    print Levenshtein_Distcance(str1, str2)