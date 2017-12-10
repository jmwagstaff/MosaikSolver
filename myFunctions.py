"""

"""

# First let us import the required packages
import numpy as np
import pandas as pd
from progressbar import ProgressBar



#########################################


def expandMatrix(x, n):
    
    nrows = x.shape[0]
    ncols = x.shape[1]
    totsize = nrows*ncols
    
    x1 = x.reshape((1, totsize))*np.ones((n, totsize))
    
    x1A = x1[:n, :ncols]
    for i in range(nrows-1):
        x1A = np.concatenate([x1A, x1[:n, (i+1)*ncols:(i+2)*ncols]])
        
        
    x2 = x1A.reshape((totsize*n, 1))*np.ones((totsize*n, n))
    newx = x2.reshape((nrows*n, ncols*n))
    
    return newx


#########################################
    

# collapseMatrix
def collapseMatrix(x, n):
    ''' This functions takes a full image, collapses it to a grid of
    zeros and ones which can then be used for the solver'''
    # this one line collapses the matrix and converts 0 to 0, and 255 to 1
    x = np.array(x[::n, ::n], dtype = 'float64')
    x[x == 0] = 1
    x[x == 255] = 0
    
    return x


#########################################

# This function takes as argument a partial picture matrix i.e. a matrix
# filled with black squares (1), white squares (0) and empty squares (NaN)
# and returns count matrices for black and empty squares around local groups 

def countBlackNaN(partialPic):
    '''This function builds the black-squares-count matrix and the 
    missing-values-count matrix. The sums are done over local squares'''
    
    # This initialises empty matrices of correct shape
    missSqrs = np.empty(partialPic.shape)
    blackSqrs = np.empty(partialPic.shape)
    
    # Get the size of the matrix
    nrows = partialPic.shape[0]
    ncols = partialPic.shape[1]
    
    # This loops over every element in the matrices:
    for i in range(nrows):
        for j in range(ncols):
            
            # Define local squares:
            localSqrs = partialPic[max(0, i-1):min(nrows, i+2), 
                                max(0, j-1):min(ncols, j+2)]     
            # The max sorts out the i=j=0 problems, 
            # so that i and j don't go negative.
            # The min sorts out problems at the other end!
            
            # Write the count values:
            # Count missing values
            missSqrs[i, j] = pd.isnull(localSqrs).sum() 
            # Count black squares
            blackSqrs[i, j] = np.nansum(localSqrs)
            
    # return count matrices
    return missSqrs, blackSqrs



##############################
    
 
def newPicMatrix(partialPuzNum, partialPic):
    '''This functions takes in a partially filled Numbers matrix and a 
    partially completed picture matrix and solves what it can. It returns 
    an updated version of the picture matrix'''
    
    # First we get the local number of black sqrs and missing values
    missSqrs = countBlackNaN(partialPic)[0]
    blackSqrs = countBlackNaN(partialPic)[1]
    
    # We can define a difference matrix. If an element of diffMat is zero
    # then the local group has all blk sqrs filled in
    diffMat = partialPuzNum - blackSqrs
    
    # Get the size of the matrix
    nrows = partialPic.shape[0]
    ncols = partialPic.shape[1]
    
    # This loops over every element in the matrices:
    for i in range(nrows):
        for j in range(ncols):
            
            # Define the local group of squares
            localSqrs = partialPic[max(0, i-1):min(nrows, i+2), 
                                max(0, j-1):min(ncols, j+2)]     
            # The max sorts out the i=j=0 problems, 
            # so that i and j don't go negative.
            # The min sorts out problems at the other end!
            
            # Look for where there is a puzzel number, are where there 
            # are missing values in local group
            if pd.notnull(partialPuzNum[i, j]) and missSqrs[i, j] > 0:
                
                # If difference is equal to missing, fill with blks
                if diffMat[i, j] == missSqrs[i, j]:
                    localSqrs[np.isnan(localSqrs)] = 1
                    
                # If all blk are filled, fill with white squares
                elif diffMat[i, j] == 0:
                    localSqrs[np.isnan(localSqrs)] = 0
                    
    return partialPic




####################################
      
# Count the total missing values in the picture matrix
def totMissing(partialPic): 
    countNaN = pd.isnull(partialPic).sum()
    return countNaN


####################################
    
# Here we count the number of 0 and 9 in the puzzel matrix
# This works only if the puzzleMat is an np array, not a DF
def count09(puzzleMat): 
    
    count09 = [puzzleMat[puzzleMat == 0].shape[0],
               puzzleMat[puzzleMat == 9].shape[0]]
    
    return count09

####################################
    

    
# Here we find the indecies for the 0 and 9 in the puzzel matrix
def index09(puzzleMat): 
    
    # Get the size of the matrix
    nrows = puzzleMat.shape[0]
    ncols = puzzleMat.shape[1]
    
    puzzleMat = puzzleMat.reshape((1,nrows*ncols))

    ind09 = np.array([[]]) 
    for i in range(nrows*ncols):
        if puzzleMat[0, i] == 0 or puzzleMat[0, i] == 9: 
            ind09 = np.append(ind09, i)
    
    
    return ind09

# create randomized indecies with 0 and 9 at front:
def index09_random(puzzleMat):
    
    # Get the size of the matrix
    nrows = puzzleMat.shape[0]
    ncols = puzzleMat.shape[1]
    
    indIs09 = index09(puzzleMat)
    indNot09 = np.setdiff1d(np.arange(nrows*ncols), indIs09)
    # randomise the rest
    np.random.shuffle(indNot09)
    # put indecies of 0 and 9 first, then randomise rest
    return np.append(indIs09, indNot09)
    

####################################
    
# This function solves the puzzle, and returns partialPic
def solvePuzzel(partialPuzNum, partialPic, noPrint = False):
    
    if not noPrint:
        print('\n'+'Processing...')
        
    solved = True
    
    # While there are still some missing values, solve
    while totMissing(partialPic) > 0: 
        solved = False
        
        # Count missing values in our picture  
        check = totMissing(partialPic)
        
        # Update the picture by solving what can be solved
        partialPic = newPicMatrix(partialPuzNum, partialPic)
        
        # This is to break the loop if the puzzle cannot be solved
        if totMissing(partialPic) == check:
            if not noPrint: print('CANNOT BE SOLVED!')
            break
        elif totMissing(partialPic) == 0:
            solved = True
            if not noPrint: print('\n'+'SOLVED!')
        
    return  partialPic, solved


####################################
    
# This function solves the puzzle, updates partialPic and returns True if
# solved and False if the puzzle cannot be solved
def solvePuzzelFast(partialPuzNum, partialPic):
    
    # Get the size of the matrix
    nrows = partialPic.shape[0]
    ncols = partialPic.shape[1]
    
    
    # Here we define a few methods, to optimise the loops
    isany = np.any
    isitnull = pd.isnull
    isitnotnull = pd.notnull
    donansum = np.nansum
    isitnan = np.isnan
    
    
    # While there are still some missing values, solve
    while isany(isitnull(partialPic)): 
        
        # How many missing values
        countNaN = isitnull(partialPic).sum()
        
        # This loops over every element in the matrices:
        for i in range(nrows):
            for j in range(ncols):
                
                if isitnotnull(partialPuzNum[i, j]): # is not NA
                    # Define the local group of squares
                    localSqrs = partialPic[max(0, i-1):min(nrows, i+2), 
                                max(0, j-1):min(ncols, j+2)]
                    # Count missing values
                    missSqrs = isitnull(localSqrs).sum()
                    
                    if (missSqrs > 0):
                        # Count black squares
                        blackSqrs = donansum(localSqrs)
                        x = (partialPuzNum[i, j] - blackSqrs)
                        if (x == 0) or ((x - missSqrs) == 0):
                            # Update the picture by solving what can be solved
                            localSqrs[isitnan(localSqrs)] = (partialPuzNum[i, j] - blackSqrs)/missSqrs
                            
        if (countNaN == isitnull(partialPic).sum()):
            #print("Cannot be Solved!")
            return False
    
    return True
    


######################################
    

def creatPartialPuzzel(fullPicMatrix):
    
    # Get full puzzel matrix
    fullPuzzelNum = countBlackNaN(fullPicMatrix)[1]
    
    # Make a copy for building the partial number matrix 
    partialPuzzelNum = fullPuzzelNum.copy()
    
    # Get the size of the matrix
    nrows = fullPicMatrix.shape[0]
    ncols = fullPicMatrix.shape[1]
        
    # This loops over every element in the matrices:
    for i in range(nrows):
        for j in range(ncols):
            
            # Initialise empty picture matrix, filled with NaNs
            picture = np.full(fullPicMatrix.shape, np.nan)
            
            # remove first element
            partialPuzzelNum[i,j] = np.nan
            
            # Test if the puzzle cannot be solved, 
            # if so re-sub in element from fullPuzz num
            #if not solvePuzzel(partialPuzzelNum, picture, noPrint = True)[1]:
            if not solvePuzzelFast(partialPuzzelNum, picture):
                partialPuzzelNum[i, j] = fullPuzzelNum[i, j]
                
    return partialPuzzelNum
                    

######################################
    

def creatPartialPuzzel_random(fullPicMatrix):
    
    # Get full puzzel matrix
    fullPuzzelNum = countBlackNaN(fullPicMatrix)[1]
    
    # Make a copy for building the partial number matrix 
    partialPuzzelNum = fullPuzzelNum.copy()
    
    # Get the size of the matrix
    nrows = fullPicMatrix.shape[0]
    ncols = fullPicMatrix.shape[1]
        
    # This loops over every element in the matrices at random:
    for i in np.random.choice(np.arange(nrows*ncols), replace=False, 
                              size=nrows*ncols):
            
            # Initialise empty picture matrix, filled with NaNs
            picture = np.full(fullPicMatrix.shape, np.nan)
            
            # This removes element 
            partialPuzzelNum.reshape((1,nrows*ncols))[0, i] = np.nan
            
            # Test if the puzzle cannot be solved, 
            # if so re-sub in element from fullPuzz num
            if not solvePuzzel(partialPuzzelNum, picture, noPrint = True)[1]:
                partialPuzzelNum.reshape((1,nrows*ncols))[0, i] = fullPuzzelNum.reshape((1,nrows*ncols))[0, i]
                
                
    return partialPuzzelNum
                    

######################################
    

def creatPartialPuzzel_09_random(fullPicMatrix):
    
    # Get full puzzel matrix
    fullPuzzelNum = countBlackNaN(fullPicMatrix)[1]
    
    # Make a copy for building the partial number matrix 
    partialPuzzelNum = fullPuzzelNum.copy()
    
    # Get the size of the matrix
    nrows = fullPicMatrix.shape[0]
    ncols = fullPicMatrix.shape[1]
    
    pbar = ProgressBar()
        
    # This loops over every element in the matrices at random:
    for i in pbar(np.array(index09_random(fullPuzzelNum), dtype = int)):
            
            # Initialise empty picture matrix, filled with NaNs
            picture = np.full(fullPicMatrix.shape, np.nan)
            
            # This removes element 
            partialPuzzelNum.reshape((1,nrows*ncols))[0, i] = np.nan
            
            # Test if the puzzle cannot be solved, 
            # if so re-sub in element from fullPuzz num
            #if not solvePuzzel(partialPuzzelNum, picture, noPrint = True)[1]:
            if not solvePuzzelFast(partialPuzzelNum, picture):
                partialPuzzelNum.reshape((1,nrows*ncols))[0, i] = fullPuzzelNum.reshape((1,nrows*ncols))[0, i]
                
                
    return partialPuzzelNum

####################################
    
# This function solves the puzzle, updates partialPic and returns True if
# solved and False if the puzzle cannot be solved
def creatPartialPuzzel_Fast(fullPicMatrix):
    
    # partialPuzNum, partialPic
    
    # Get full puzzel matrix
    fullPuzzelNum = countBlackNaN(fullPicMatrix)[1]
    
    # Make a copy for building the partial number matrix 
    partialPuzNum = fullPuzzelNum.copy()
    
    # Get the size of the matrix
    nrows = fullPicMatrix.shape[0]
    ncols = fullPicMatrix.shape[1]
    
    # Initialise empty picture matrix, filled with NaNs
    partialPic = np.full(fullPicMatrix.shape, np.nan)
    
    # reset picture matrix: partialPic[:, :] = np.nan
    
    # This loops over every element in the Numb matrix:
    for k in range(nrows):
        for l in range(ncols):
            
            # remove first element
            partialPuzNum[k, l] = np.nan
    
            # While there are still some missing values, solve
            while np.any(pd.isnull(partialPic)): 
                
                # How many missing values
                countNaN = pd.isnull(partialPic).sum()
                
                # This loops over every element in the matrices:
                for i in range(nrows):
                    for j in range(ncols):
                        
                        if pd.notnull(partialPuzNum[i, j]): # is not NA
                            # Define the local group of squares
                            localSqrs = partialPic[max(0, i-1):min(nrows, i+2), 
                                        max(0, j-1):min(ncols, j+2)]
                            # Count missing values
                            missSqrs = pd.isnull(localSqrs).sum()
                            
                            if (missSqrs > 0):
                                # Count black squares
                                blackSqrs = np.nansum(localSqrs)
                                x = (partialPuzNum[i, j] - blackSqrs)
                                if (x == 0) or ((x - missSqrs) == 0):
                                    # Update the picture by solving what can be solved
                                    localSqrs[np.isnan(localSqrs)] = (partialPuzNum[i, j] - blackSqrs)/missSqrs
                                    
                if (countNaN == pd.isnull(partialPic).sum()):
                    partialPuzNum[k, l] = fullPuzzelNum[k, l]
                    break
            
            partialPic[:, :] = np.nan
            
    return partialPuzNum
    


######################################