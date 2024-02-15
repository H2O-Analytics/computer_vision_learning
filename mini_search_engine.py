# Coding the Matrix 0.6.7 Mini-Search Engine
DATA_PATH = '/Users/tawate/Documents/Self Learning/Moding the Matrix'
f = open(DATA_PATH + '/stories_small.txt')
for line in f:
    strlist = line.split()

# Create a function that returns the string as the key, and a set of numbers relating to the document number from documents where that string is present. 
# (i.e if Eddie is in lines 2 and 5 of documents then {'Eddie', (2, 5)} would be that key:value pair
def makeInverseIndex(strlist):
    return dict(enumerate(strlist))

dict_list = makeInverseIndex(strlist = strlist)