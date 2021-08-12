import pickle
import re
import pandas as pd
import numpy as np

import bz2
import pickle
import _pickle as cPickle

class treeDb():
    def __init__(self):
        self.aDict = {'endOfTheWorld':[]}
        self.keys = []
    
    def addEdge(self,root,child):
        if self.addNode(root) ==  False :
            return False
        if child == None :
            self.aDict['endOfTheWorld'].append(root)
            return True
        if self.addNode(child) ==False :
            root, child = child, root
        self.aDict[child] = root
        return True
    
    def addNode(self,root):
        if root not in self.keys:
            self.keys.append(root)
            return True
        return False
    
    def getTree(self):
        return self.aDict
    
    def getRoot(self,child):
        if child in self.aDict :
            key = child
            while True:
                if key not in self.aDict : break
                key = self.aDict[key]
            return key
        if child in self.keys : return child
        return False
    
    def getSimilar(self,key):
        result = [key]
        child = key 
        if key in self.aDict: # get root
            while True: 
                if child not in self.aDict : break 
                result.append(self.aDict[child])
                child = self.aDict[child]
        if key in self.keys:    
            while True: # get child
                if child == 'endOfTheWorld' : break
                valueDict = list(self.aDict.values())
                if child not in valueDict : break
                newChild = list(self.aDict.keys())[valueDict.index(child)]
                result.append(newChild)
                child = newChild
        return set(result)
    def getOOV(self):
        return self.aDict['endOfTheWorld']
    def getFeatureNames(self):
        return self.aDict.keys()

def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

def similarToken_client(sentences):
    trees = decompress_pickle('trees.pbz2')
    newString = ""
    if type(sentences) == float : False
    for item in re.findall('[A-Za-z]{3,}',sentences):
        item = item.lower()
        root = trees.getRoot(item)
        if root == False :  
            continue
            newString += " " + item 
        else:
            newString += " " + root
    return newString

def similarToken2_client(sentences):
    trees,ekspor,impor,jual,beli = pickle.load(open("all_final.pickle","rb"))
    newString =""
    for item in re.findall('[A-Za-z]{3,}',sentences):
        item = item.lower()
        root = trees.getRoot(item)
        if root == False :  
            continue
            newString += " " + item 
        else:
            newString += " " + root
    return newString

def similarityInput(aList):
    return [similarToken_client(item.lower()) for item in aList]

# def corrInput(keluaran,masukan):
#     trees,ekspor,impor,jual,beli = pickle.load(open("all_final.pickle","rb"))
#     db,C, setImpor,setEkspor = pickle.load(open("corr_all.p","rb"))
#     # C = decompress_pickle('C.pbz2')
#     # setImpor = decompress_pickle('setImpor.pbz2')
#     # setEkspor = decompress_pickle('setEkspor.pbz2')   
#     cDict = {}
#     # tIdx = [setImpor.index(impor[each]) for each in masukan]
#     tIdx = []
#     for each in masukan:
#         if each in impor:
#             tIdx.append(setImpor.index(impor[each]))
#     for each in keluaran:
#         if each not in ekspor : continue
#         cDict[each]  = np.sum(C[tIdx,setEkspor.index(ekspor[each])])
#     return {k: v for k, v in sorted(cDict.items(), key=lambda item: item[1],reverse=False)}
#     # tIdx = [setImpor.index(impor[each]) for each in masukan]
#     # for each in keluaran:
#     #     cDict[each]  = np.sum(C[tIdx,setEkspor.index(ekspor[each])])
#     # return {k: v for k, v in sorted(cDict.items(), key=lambda item: item[1],reverse=False)}

# def corrInput(idxnum):
#     trees,ekspor,impor,jual,beli = pickle.load(open("all_final.pickle","rb"))
# #     db,C, setImpor,setEkspor = pickle.load(open("corr_all.p","rb"))
#     db,C, setImpor,setEkspor = decompress_pickle("cordata.pbz2")
#     num = int(idxnum)
#     masukan = set(db.impor[num])
#     keluaran = set(db.ekspor[num])
#     cDict = {}
#     tIdx = [setImpor.index(impor[each]) for each in masukan]
#     for each in keluaran:
#         cDict[each]  = np.sum(C[tIdx,setEkspor.index(ekspor[each])])
#     return {k: v for k, v in sorted(cDict.items(), key=lambda item: item[1],reverse=False)}
