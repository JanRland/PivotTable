#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:03:37 2024

@author: jan
"""

import numpy as np 
import itertools
import math


class CreateTable:
    def __init__(self, data, labels, split, verbose=False):
        self.data=data
        self.split=split
        self.labels=labels
        self.N_features=data.shape[1]
        self.verbose=verbose
        
        self.epsilons, self.partitions=self.getEpsilons()
        self.clippedData=self.getClippedData()

        self.maxCache=[0]
        self.getMaxCache()

        if self.verbose:
            self.resultBreak()
            print("Max Cache:")
            print(self.maxCache)
            self.resultBreak()
            
        self.projectedData=self.lambdaMergeTransform(self.clippedData)

        if self.verbose:
            self.resultBreak()
            print("Clipped Data:")
            print(self.clippedData[0:20,:])
            print("Projected Data:")
            print(self.projectedData[0:20])
            self.resultBreak()
        if self.verbose:
            self.resultBreak()
            print("Number of features: " + str(self.N_features))
            print("Splits: " + str(self.split))
            print("Epsilons: " + str(self.epsilons))
            self.resultBreak()
            
    def indicatorFunction(self, x, A):
        """

        Parameters
        ----------
        x : int, float, double
            Real number that should be checked if it is part of 
            the set A. 
        A : List, tuple
            Indicator set. 

        Returns
        -------
        res : Boolean
            Returns 1 if x is element of A and 0 otherwise. 

        """
        
        res=0
        if x>=A[0] and x<A[1]:
            res=1
        return res
    
    
    def getEpsilons(self):
        """
        

        Returns
        -------
        epsilons : List
            List of binning intervals according to the number of splits
            specified in the class.
        partitions : List of Lists
            List of intervals [a,b] in which the value range is split. 
            Note: Assuming right-continous intervals

        """
        epsilons=[]
        partitions=[]
        for f in range(self.N_features):
            partition=[]
            f_max=np.amax(self.data[:,f])
            f_min=np.amin(self.data[:,f])

            epsilon=(f_max-f_min)/(self.split[f])
            epsilons.append(epsilon)

            for s in range(self.split[f]):
                if s==self.split[f]-1:
                    partition.append([s*epsilon+f_min,(s+1)*epsilon+f_min+0.00001])
                else:
                    partition.append([s*epsilon+f_min,(s+1)*epsilon+f_min]) ## falsch
            partitions.append(partition)
            
        return epsilons, partitions
            
    def getClippedData(self):
        clippedData=self.data.copy()

        for i in range(self.N_features):
            for i_2, d in enumerate(self.data[:,i]):
                d_hat=0
                for i_3, v in enumerate(self.partitions[i]):
                    d_hat+=i_3*self.indicatorFunction(d, v)
                clippedData[i_2,i]=d_hat
        
        return clippedData

    def getXYaxis(self, factor=0):
        """
        

        Parameters
        ----------
        factor : Int, optional
            The decimal for which the output should be rounded

        Returns
        -------
        axis_x : List of lists
            Values of the Blocks on the X-axis with repeating entries.
        axis_y : List of lists
            Values of the Blocks on the Y-axis with repeating entries. 

        """
        axis_x=[]
        axis_y=[]

        for i in range(self.N_features-1, -1,-1):
            if i<math.ceil(self.N_features/2.):
                if axis_x==[]:
                    values=[]

                    for i,interv in enumerate(self.partitions[i]):
                        values.append(round((interv[1]-interv[0])/2.+interv[0], factor))
                    axis_x.append(values)

                    
                else:
                    values=[]
                    for i,interv in enumerate(self.partitions[i]):
                        values.append(round((interv[1]-interv[0])/2.+interv[0], factor))

                    values=values*len(axis_x[-1])
                    axis_x.append(values)


            else:
                if axis_y==[]:
                    values=[]

                    for i,interv in enumerate(self.partitions[i]):
                        values.append(round((interv[1]-interv[0])/2.+interv[0], factor))

                    axis_y.append(values)

                else:
                    values=[]
                    for i,interv in enumerate(self.partitions[i]):
                        values.append(round((interv[1]-interv[0])/2.+interv[0], factor))
                    values=values*len(axis_y[-1])
                    axis_y.append(values)

        return axis_x, axis_y
    
    def getRedXYaxis(self, factor=0):
        """
        

        Parameters
        ----------
        factor : Int, optional
            The decimal for which the output should be rounded

        Returns
        -------
        axis_x : List of lists
            Values of the Blocks on the X-axis. Not repeating entries!
        axis_y : List of lists
            Values of the Blocks on the Y-axis. Not repeating entries!

        """
        
        axis_x=[]
        axis_y=[]

        for i in range(self.N_features-1, -1,-1):
            if i<math.ceil(self.N_features/2.):
                if axis_x==[]:
                    values=[]

                    for i,interv in enumerate(self.partitions[i]):
                        values.append(round((interv[1]-interv[0])/2.+interv[0], factor))
                    axis_x.append(values)

                    
                else:
                    values=[]
                    for i,interv in enumerate(self.partitions[i]):
                        values.append(round((interv[1]-interv[0])/2.+interv[0], factor))

                    values=values
                    axis_x.append(values)


            else:
                if axis_y==[]:
                    values=[]

                    for i,interv in enumerate(self.partitions[i]):
                        values.append(round((interv[1]-interv[0])/2.+interv[0], factor))

                    axis_y.append(values)

                else:
                    values=[]
                    for i,interv in enumerate(self.partitions[i]):
                        values.append(round((interv[1]-interv[0])/2.+interv[0], factor))
                    values=values
                    axis_y.append(values)

        return axis_x, axis_y

    def lambdaMergeTransform(self, matrix):
        """
        

        Parameters
        ----------
        matrix : numpy array
            NxM Matrix with N samples and M features

        Returns
        -------
        p_projected : List
            Bijective mapping 

        """
        p_projected=[0]*matrix.shape[0]
        for i in range(self.N_features):
            p=matrix[:,i]
            for i_2, v in enumerate(p):
                p_projected[i_2]+=(self.maxCache[i]+1)*(v+1)
        return p_projected

    def getTotalProjection(self):
        """

        Returns
        -------
        projectedValues : List
            List containing all possible projected values.
        """
        values=[]
        for f in range(self.N_features):
            values.append([x for x in range(len(self.partitions[f]))])

        if self.verbose:
            self.resultBreak()
            print("Values")
            print(values)
            print("Iter Product:")
            print(list(set(list(itertools.product(*values)))))
            self.resultBreak()
        possibleValues=np.array(list(set(list(itertools.product(*values)))))
        if self.verbose:
            self.resultBreak()
            print("Possible Values")
            print(possibleValues)
            self.resultBreak()
        projectedValues=self.lambdaMergeTransform(possibleValues)
        if self.verbose:
            self.resultBreak()
            print("Projected Values")
            print(projectedValues)
            self.resultBreak()
        
        projectedValues.sort()

        return projectedValues
        
    def getMaxCache(self):
        """
        Calculates the maximal values for the bijective function

        Returns
        -------
        None.

        """
        values=[]
        for f in range(self.N_features):
            values.append([x for x in range(len(self.partitions[f]))])

        possibleValues=np.array(list(itertools.product(*values)))
        maxCache=0
        p_projected=[0]*possibleValues.shape[0]
        for i in range(self.N_features):
            p= possibleValues[:,i]

            for i_2, v in enumerate(p):
                p_projected[i_2]+=(maxCache+1)*(v+1)
            maxCache=np.amax(p_projected)+1
            self.maxCache.append(maxCache)
        
    def getLambdaMergeTransform(self, values):
        """
        

        Parameters
        ----------
        values : List
            List of lambda transformed values. 

        Returns
        -------
        res : Int
            Bijective mapping of the list values

        """
        res=0
        for i, v in enumerate(values):
            res+=(self.maxCache[i]+1)*(v+1)
        return res
        
    def createMatrix(self):
        """
        

        Returns
        -------
        matrix : Numpy Array
            Content of the pivot table.

        """
        x_axis, y_axis=self.getXYaxis()
        entries_x = len(x_axis[-1])
        entries_y = len(y_axis[-1])
        if self.verbose:
            self.resultBreak()
            print("X-axis")
            print(x_axis)
            print("Y-axis")
            print(y_axis)
            self.resultBreak()
        matrix=np.zeros((entries_y, entries_x))
        possibleValues=self.getTotalProjection()

        if self.verbose:
            self.resultBreak()
            print("All Projections")
            print(possibleValues)
            print("Projections: ")
            print(self.projectedData[0:20])
            self.resultBreak()
        for i,p in enumerate(possibleValues):
            n = np.where(self.projectedData==p)[0]
            if n.shape[0]==0:
                matrix[int(i/float(entries_x)),i%entries_x]=-1
                #print(p)
            else:
                matrix[int(i/float(entries_x)),i%entries_x]=np.mean(self.labels[n])
        return matrix

    def resultBreak(self):
        print("###################################")
        
        
def writeText(ax, rx,ry, w, h, msg, rotation=0., mcolor="black", fs=12):
    """
    

    Parameters
    ----------
    ax : matplotlib.axis
        axis for which the annotation should be written.
    rx : float
        x-coordinate of text
    ry : float
        y-coordinate of text
    w : float
        width of box surrounding the text
    h : float
        heigt of box surrounding the text
    msg : string
        message to write at the placement
    rotation : Int, optional
        rotation of the text. The default is 0.
    mcolor : string
        Color of . The default is "black".
    fs : Int
        Font size

    Returns
    -------
    None.
    
    """
    cx=rx + w/2.
    cy=ry + h/2.
    ax.annotate(msg, (cx, cy), color=mcolor, fontsize=fs, ha='center', va='center', rotation=rotation)
        
class Pivotframe:
    def __init__(self, axis_x, axis_y, matrix, sep, featureSep, featureBox, x_names, y_names):
        """
        

        Parameters
        ----------
        axis_x : List of lists
            List of feature blocks derived from CreateTable class.
        axis_y : List of lists 
            List of feature blocks derived from CreateTable class.
        matrix : numpy array
            numpy array derived from CreateTable class.
        sep : float
            separation of the boxes
        featureSep : float
            separation of the axis boxes
        featureBox : float
            Width/Length of the boxes
        x_names : List of strings
            Names of the boxes in x-axis
        y_names : List of strings
            Names of the boxes in y-axis

        Returns
        -------
        None.

        """
        self.verbose=True
        
        N_x=len(axis_x)
        N_y=len(axis_y)
        
        if self.verbose:
            print("X-axis:")
            print(axis_x)
            print("Y-axis:")
            print(axis_y)
            self.resultBreak()

        M_x=matrix.shape[1]
        M_y=matrix.shape[0]
        
        featureFrame_x=1.-(N_y*featureSep+N_y*featureBox)-sep*2
        featureFrame_y=1.-(N_x*featureSep+N_x*featureBox)-sep*2

        if self.verbose:
            print("Feature Frames:")
            print(featureFrame_x)
            print(featureFrame_y)
            self.resultBreak()

        self.y_boxes=[]
        self.y_annot=[]
        self.x_boxes=[]
        self.x_annot=[]

        n_repeat=1
        cx=featureSep
        cy=1
        for y_i, values in enumerate(axis_y):
            y_name=y_names[y_i]
            n_boxes=len(values)
            width=featureBox
            height=(featureFrame_y-(n_repeat*n_boxes-1)*sep)/(n_repeat*n_boxes)
            self.y_annot.append([cx-sep, 1-featureFrame_y/2., y_name])
            for i in range(n_boxes*n_repeat):
                if i==0:
                    cy=cy-height
                    self.y_boxes.append([cx, cy, width, height, values[i%len(values)]])
                else:
                    cy=cy-height-sep
                    self.y_boxes.append([cx, cy, width, height, values[i%len(values)]])
                
            cx+=featureBox+featureSep
            cy=1
            n_repeat*=n_boxes

        m_height=height
        n_repeat=1
        x_start=1-featureFrame_x
        cx=x_start
        cy=featureSep
        for x_i, values in enumerate(axis_x):
            x_name=x_names[x_i]
            n_boxes=len(values)
            print(n_boxes)
            height=featureBox
            width=(featureFrame_x-(n_repeat*n_boxes-1)*sep)/(n_repeat*n_boxes)
            self.x_annot.append([1-featureFrame_x/2., cy-sep, x_name])
            for i in range(n_boxes*n_repeat):
                if i==0:
                    self.x_boxes.append([cx, cy, width, height, values[i%len(values)]])
                else:
                    cx=cx+width+sep
                    self.x_boxes.append([cx, cy, width, height, values[i%len(values)]])
            cx=x_start
            cy+=featureBox+featureSep
            n_repeat*=n_boxes
                    
        m_width=width

        self.matrixBoxes=[]
        cy=1-m_height
        for i in range(M_y):
            cx=x_start
            for j in range(M_x):
                self.matrixBoxes.append([cx, cy, m_width, m_height, matrix[i,j]])
                cx=cx+m_width+sep
            cy=cy-m_height-sep
            
    def resultBreak(self):
        print("###################################")
