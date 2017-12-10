#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 09:46:45 2017

@author: jacques
"""

import numpy as np


# The puzzle can only be started where there are 0 and 9 (6/4 at borders)

# first find the indecies for the 0 and 9 in the puzzel matrix


# A is a 1D array
B = np.where(A==value)[0]  # Gives the indexes in A for which value = 2

x = np.arange(9.).reshape(3, 3)

# Initialise empty picture matrix, filled with NaNs
picture = np.full(fullPic.shape, np.nan)

def newSolver(partialNumb, partialPic):
    
    
    
    # Get the size of the matrix
    nrows = partialPic.shape[0]
    ncols = partialPic.shape[1]
    
    # Here we define a few methods, to optimise the loops
    isany = np.any
    isitnull = pd.isnull
    isitnotnull = pd.notnull
    donansum = np.nansum
    isitnan = np.isnan
    
    
    # gives pair of indecies ([i],[j]) for 2D matrix
    indPairs = np.where( (partialNumb == 9) | (partialNumb == 0) ) 
    indi = indPairs[0]
    indj = indPairs[1]
    
    #indinew = np.array([[]]) 
    kIndex = np.array([[]])
    
    # this loops over the pairs of indecies
    for i,j in zip(indi, indj):
        

        
        # Define the local group of squares
        localSqrs = partialPic[max(0, i-1):min(nrows, i+2), 
                    max(0, j-1):min(ncols, j+2)]
        
       
        missSqrs = isitnull(localSqrs).sum()
        
        if (missSqrs > 0):
            # Count black squares
            blackSqrs = donansum(localSqrs)
            diff = (partialNumb[i, j] - blackSqrs)
            
            #print(missSqrs)
            
            if (diff == 0) or ((diff - missSqrs) == 0):
                # Update the picture by solving what can be solved
                localSqrs[isitnan(localSqrs)] = (partialNumb[i, j] - blackSqrs)/missSqrs
                # record indecies
                
                kIndex = np.append(kIndex, (i*ncols +j) )
                
                
    print(kIndex)
    # new we use these to get new indecies
    #first get all new k, get rid of copies
    
    newIndexK = np.unique(np.concatenate(( (kIndex-2), (kIndex-1), 
                                          (kIndex+1), (kIndex+2) ), axis=0)) 
    print(newIndexK)
    
    # limit the k-values
    #newIndexK[newIndexK >= 0...]
    
    # Concert new k indecies to i and j
    #k//ncols = i
    #k%ncols = j
            
        
    print(partialPic)

    return 


array+1





##