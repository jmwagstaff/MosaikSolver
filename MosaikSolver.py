"""
MosaikSolver.py
~~~~~~~~~~~~~~~

Programm to solve Mosaik puzzels

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.



Here we want to write the Image matrix as an image
Using GIMP Image Editor:
    1. Image → Mode → Greyscale
    2. Image → Scale, eg image is (8,9 sqrs) hence (160,180 px)
    3. Filters → Blur → Pixelise (20x20 px)
    4. Colours → Levels, or Contrast
    
# to run the whole script:
%run MosaikSolver.py 

"""

#%%

#### Libraries
# First let us import the required packages
import sys
import numpy as np
import pandas as pd
from skimage import io
from matplotlib import pyplot as plt
import myFunctions as myfn # Import my functions
# Use this to reload when making changes to modules
import importlib; importlib.reload(myfn) 

#from progressbar import ProgressBar



#%%

'''

# Here we load the picture csv file:
picFileName = input("Enter picture file name: ") # ask for picture file name
picFileName = 'data/' + picFileName + '.csv' # create path name
picData = pd.read_csv(picFileName, header = None) # load with pandas to creat dataFrame

# We explicitly copy the picture data values into an array, 
# and convert array to be float type so that we can introduce NaNs later
fullPic = picData.values.copy() 
fullPic = np.array(fullPic, dtype='float64')

'''

#%%

# Puzzle size: 60x100 sqrs =  1200x2000 px (for 20x20 pixel sqrs)

# Here we read-in an png image file
picFileName = input("Enter image file name: ") # ask for picture file name
picFileName = 'imageFiles/' + picFileName + '.png' # create path name

n = int(input("Enter image square size n: ")) # ask for picture file name

# This should be a greyscale image, i.e. a 2D array
# Here the image size and square size (n) from pixalation should be known
image = io.imread(picFileName)
#print(type(image))
#plt.imshow(image);
plt.imshow(image, cmap='gray', interpolation='nearest');
plt.show()

# Collapse image file to create a picture numbers matrix
fullPic = myfn.collapseMatrix(image, n)

# Get the size
nrows = fullPic.shape[0]
ncols = fullPic.shape[1]
totsize = nrows*ncols


# not sure if we need this
#bigpic = expandMatrix(fullPic, 10)
#plt.imshow(bigpic, cmap='gray', interpolation='nearest');



#%%
# Here we run solvePuzzel on the full puzzel number to see if the picture can
# be solved with the standard method:

# Get full puzzel matrix and plug into solvePuzzel
fullPuzzelNum = myfn.countBlackNaN(fullPic)[1]
# Initialise empty picture matrix, filled with NaNs
picture = np.full(fullPic.shape, np.nan)
  
# Run solver 2: Fast
if myfn.solvePuzzelFast(fullPuzzelNum, picture): 
    print('\n'+'This Puzzle can be SOLVED!\n')
else: 
    print('CANNOT BE SOLVED! Try a new Picture!') 
    sys.exit() # and break script
    
# Here we count the number of 0 and 9 in the full puzzel
print('The number of 0 and 9 in the full puzzle are:\n')
print(myfn.count09(fullPuzzelNum)) 

#%%
# This function call creats a partial puzzel numbers matrix

'''
# %time on testImage (8,9): 
# CPU times: user 10.8 s, sys: 544 ms, total: 11.3 s Wall time: 10.3 s
newPuzzel = myfn.creatPartialPuzzel(fullPic)

# %time on testImage (8,9): with solvePuzzleFast
# CPU times: user 1.86 s, sys: 4 ms, total: 1.87 s Wall time: 1.85 s
newPuzzel = myfn.creatPartialPuzzel(fullPic)

# %time on testImage (8,9): with creatPartialPuzzel_Fast
# CPU times: user 2.03 s, sys: 84 ms, total: 2.11 s Wall time: 2.02 s
newPuzzel = myfn.creatPartialPuzzel_Fast(fullPic)



# randomised indecies
# %time on testImage (8,9): 
# CPU times: user 13.8 s, sys: 524 ms, total: 14.3 s Wall time: 13.3 s
newPuzzel_random = myfn.creatPartialPuzzel_random(fullPic)



# 0 and 9 in first, then randomised indecies

# %time on testImage (8,9):
# CPU times: user 13.6 s, sys: 1.06 s, total: 14.7 s Wall time: 12.7 s
newPuzzel_09_random = myfn.creatPartialPuzzel_09_random(fullPic)

# %time on testImage (8,9): with solvePuzzleFast
# CPU times: user 2.54 s, sys: 172 ms, total: 2.71 s Wall time: 2.45 s
newPuzzel_09_random = myfn.creatPartialPuzzel_09_random(fullPic)


## Let's test our faster code on a bigger puzzle
# %time on testImage (20,20): with solvePuzzleFast
# CPU times: user 2min 51s, sys: 5.4 s, total: 2min 56s Wall time: 2min 46s
newPuzzel = myfn.creatPartialPuzzel_09_random(fullPic)


## Let's test our optimised code on a bigger puzzle
# %time on testImage (20,20): with solvePuzzleFast, '.'-optimised
# CPU times: user 3min 2s, sys: 5.53 s, total: 3min 7s
newPuzzel = myfn.creatPartialPuzzel_09_random(fullPic)
# this didn't seem to make things faster! Maybe for bigger loops though.

'''
# Profiling my code on testImage (20,20)
%prun myfn.creatPartialPuzzel_09_random(fullPic)



want_to_creat = input("Do you want to creat a puzzle? (yes/no): ")
if want_to_creat == 'yes':
    newPuzzel = myfn.creatPartialPuzzel_09_random(fullPic)
    print('\n Puzzle completed! \n')

else: sys.exit()

# Here we count the number of 0 and 9 in the partial puzzel
print('The number of 0 and 9 in the new puzzle are:\n')
print(myfn.count09(newPuzzel)) 


# Print the results
print('\n'+'The Number Puzzel Matrix is:\n')
print(newPuzzel)


# Finally we export the puzzel to cvs file, we ask first
expToFile = input("Do you want to export to file? (yes/no): ")
if expToFile == 'yes':
    # Ask for a puzzel file name
    puzzFileName = input("Enter puzzel file name to save: ")
    puzzFileName = 'output/' + puzzFileName + '.csv' # create path name
    newPuzzel = pd.DataFrame(newPuzzel)
    newPuzzel.to_csv(puzzFileName, index = False, header = False)


#%%
###########################

'''
# To test the above solve
testPuzzel = pd.read_csv('data/testPuzzel.csv', header = None) 
testPuzzel = np.array(testPuzzel, dtype='float64')

testpicture = np.full(testPuzzel.shape, np.nan)

myfn.solvePuzzel(testPuzzel, testpicture)

myfn.totMissing(testpicture)
'''

##############################


