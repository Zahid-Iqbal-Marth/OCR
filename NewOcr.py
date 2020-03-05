import numpy as np
import cv2
import csv
from sklearn.ensemble import RandomForestClassifier
import glob
from PIL import Image
from operator import itemgetter
from spellchecker import SpellChecker

#funtions that will return hog discripter. it takes 2 arguments ; size is the Size of the image
# and xTimes is the Num of Blocks in the Image
def HogDiscripter(size,xTimes):
    block_size = (size[0] // (xTimes//2), size[1] // (xTimes//2))
    block_stride = (size[0] // xTimes, size[1] // xTimes)
    cell_size = block_stride
    num_bins = 9
    return cv2.HOGDescriptor(small_size, block_size, block_stride,
                            cell_size, num_bins)
#function that will take image of each folder an them calculates their hog discripter
def Train(ListO,LableValue,trainIndex):
    NewSize=80
    for x in ListO:
        img=cv2.imread(x,0)
        img=cv2.resize(img,(NewSize,NewSize))
        HogGrid[trainIndex] = hog.compute(img)[:,0]
        trainLables[trainIndex]=LableValue
        trainIndex+=1
        
    return trainIndex

#creating hog discripter
NewSize=80
xTimes=10
small_size = (NewSize, NewSize)
NumOfBlocks=9
BlockSize=4
NumOFbins=9
hog = HogDiscripter(small_size,xTimes)


totalTypeOfCharacters=79
d=NumOFbins*BlockSize*NumOfBlocks*NumOfBlocks
totalNumberOfTranImages=2370
HogGrid=np.zeros((totalNumberOfTranImages,d),np.float)
trainLables=np.zeros((totalNumberOfTranImages),np.uint8)
EvaluationArray=[]

incrementer=0
with open('./TestLables.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        EvaluationArray.append(row[0])
        
# print(EvaluationArray)
PathLetters="./trainingData/"
trainIndex=0
#Preaparing hog grid for traning
for y in range(totalTypeOfCharacters):
    listO=glob.glob(PathLetters+str(y+1)+"/*.png")
    trainIndex=Train(listO,y+1,trainIndex)
# print(trainIndex)
classifier = RandomForestClassifier(n_estimators = 50, max_depth=None,
min_samples_split=2, random_state=0)
classifier.fit(HogGrid,trainLables)


#print (classifier.score(HogGrid,trainLables))
#reading the test image
img = cv2.imread("Test.png", 0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
connectivity = 8 



rows,cols=thresh1.shape
IsFirstRow=True
PreviousLine=0
ListOfLines=[]
RowStarted=False
#Saperating each line in a paragraph
for y in range(rows):
    if(np.sum(thresh1[y,:])==0 and (not IsFirstRow) and RowStarted):
        ListOfLines.append(thresh1[PreviousLine:y,:])
        RowStarted=False
        PreviousLine=y
        
    elif(np.sum(thresh1[y,:])>0):
        RowStarted=True
        IsFirstRow=False
Hog=np.zeros((1,d),np.float32)
IsSpace=False
spell = SpellChecker()
FullLine=""
#Picking Letters from each line a classifying them
for Lines in range(len(ListOfLines)):
    output = cv2.connectedComponentsWithStats(ListOfLines[Lines] , connectivity, cv2.CV_32S)
    # cv2.imshow("Q1.png",ListOfLines[Lines])

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    rows,cols=ListOfLines[Lines].shape
    #print(output[2])

    Word=""
    output=(sorted(output[2],key=lambda l:(l[0])))
    NewImage=np.zeros((NewSize,NewSize),np.float32)
    i=1
    while (i<len(output)):
        initialY=output[i][1]
        maxX=output[i][2]+output[i][0]
        maxY=output[i][3]+initialY
        initialX=output[i][0]   
        if(i<len(output)-1): 
            maxXNext=output[i+1][2]+output[i+1][0]  
            initialYNext=output[i+1][1]
            maxYNext=output[i+1][3]+initialYNext
            initialXNext=output[i+1][0] 
            if(maxXNext==maxX): # concatinates dot of i and j
                initialY=min(initialY,initialYNext)
                initialX=min(initialX,initialXNext)
                maxX=max(maxX,maxXNext)
                maxY=max(maxY,maxYNext)
                i+=1
            elif initialXNext > maxX+4: # checker for space
                IsSpace=True
            # print(output[i][2]+output[i][0],"\n",output[i+1][0],"\n")
        #Adding padding to the image
        NewImage[10:70,10:70]=cv2.resize(ListOfLines[Lines][initialY:maxY,initialX:maxX],(60,60))
        # print(NewImage)
        ret,thresh1 = cv2.threshold(NewImage,127,255,cv2.THRESH_BINARY_INV)
        
        # cv2.imshow("Q1.png",thresh1)
        Hog[0]=hog.compute(thresh1.astype(np.uint8))[:,0]
        # print(EvaluationArray[int(classifier.predict(Hog))-1])
        Word+=EvaluationArray[int(classifier.predict(Hog))-1]
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
        if IsSpace or i==len(output)-1:
            IsSpace=False
            # find those words that may be misspelled
            Word=spell.correction(Word)
            # print(Word+' ')
            FullLine+=Word+' '
            Word=""
        # cv2.imwrite(str(i*(Lines+1))+'.png',thresh1[initialY:maxY,initialX:maxX])
        i+=1
    print(FullLine)
    print('\n')
    FullLine=""
