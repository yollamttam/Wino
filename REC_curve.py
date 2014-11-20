#!/usr/bin/python

#++++++++++++++++++++++++++++++++++++++
#
# NAME: REC_curve.py
#
# PURPOSE: This script draws the REC
#          (Regression Error Characteristic)
#          curves for a given set of data
#
# DEPENDENCIES: numpy, matplotlib
#
# INPUTS: command line inputs for:
#         true label file,
#         predicted label file,
#         plot output file (if not specified,
#         then plots display on screen)
#          
# OUTPUTS: Plot shown to screen or saved to file
#          showing REC curve and a few relevant measures
#
# BY: Alan Meert
#
# DATE: 14 NOV 2014
#
#-----------------------------------------
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import splrep, splev

def get_args(): 
    """collect arguments from the user"""
    parser = argparse.ArgumentParser(description='Draw the REC Curve given the user input.')

    parser.add_argument("truelab",  action="store", type=str,
                        help="File of true labels")
    parser.add_argument("predlab",  action="store", type=str,
                       help="File of predicted labels")
    parser.add_argument("-o", "--outfile", action="store", 
                      type=str, dest="outfile",default=None,
                      help="optional outfile where plots can be saved as pdf")
    args = parser.parse_args()
    return args
    
class rec_curve(object):
    """class for calculating all the necessary statistics of the REC curve"""
    def __init__(self, truelabels, predlabels):
        self.truelabels = truelabels
        self.predlabels = predlabels
        self.accuracy = []
        self.tol = []
        self.AOC = -1.0
        self.power = 1.0 # for error
        self.binnum = 1000
        return


    def error_function(self):
        """error"""
        return np.abs(self.truelabels-self.predlabels)**self.power

    def calc_rec(self,minval, maxval):
        error = self.error_function()
        self.REC,self.tol = np.histogram(error, bins = self.binnum, range=(0.0, np.max(error)))
        self.REC = np.cumsum(self.REC)/float(self.truelabels.size)
        
        #ensure that it starts at 0,0
        self.REC = np.insert(self.REC, 0, 0.0) 
        self.AOC = self.calcAOC()
        return

    def calcAOC(self):
        """Calculates the AOC from the REC given"""
        return simps(1.0-self.REC, x=self.tol)

    def get_accuracy(self, tolerance):
        """returns the accuracy at a given tolerance. Uses a simple spline"""
        tck = splrep(self.tol, self.REC)
        return splev(tolerance, tck)

    def display(self, outfile):
        """generates the output to either the screen or the pdf file"""
        fig = plt.figure(figsize=(6.0,5.0))
        tols = self.get_accuracy(np.array([0.25,0.5, 1.0]))
        plt.plot(self.tol, self.REC)
        plt.title("REC Curve")
        plt.xlabel("Tolerance")
        plt.ylabel("Accuracy")
        plt.text(0.8, 0.17,'AOC=%0.3f' %self.AOC,transform=plt.gca().transAxes)
        plt.text(0.8, 0.13,'tol$_{0.25}$=%0.3f' %tols[0],transform=plt.gca().transAxes)
        plt.text(0.8, 0.09,'tol$_{0.5}$=%0.3f' %tols[1],transform=plt.gca().transAxes)
        plt.text(0.8, 0.05,'tol$_{1.0}$=%0.3f' %tols[2],transform=plt.gca().transAxes)
        plt.plot([0.25,0.25], plt.ylim(), 'k--', linewidth=2)
        plt.plot([0.5,0.5], plt.ylim(), 'k--', linewidth=2)
        plt.plot([1.0,1.0], plt.ylim(), 'k--', linewidth=2)

        if outfile is not None:
            plt.savefig(outfile)
        else:
            plt.show()
        return


def main():
    args = get_args()

    truelabels = np.loadtxt(args.truelab)
    predlabels = np.loadtxt(args.predlab)

    rec_output = rec_curve(truelabels, predlabels)
    rec_output.calc_rec(0.0, 10.0)

    confusion = np.histogram2d(truelabels,predlabels, bins=11,
                               range=[[-0.5,10.5], [-0.5,10.5]])#, weights=None)

    plt.imshow(np.log10(confusion[0]), origin='lower', interpolation='none', vmin=0)
    plt.colorbar()
    plt.title('Confusion Matrix')
    plt.xlabel('true label')
    plt.ylabel('fitted label')
    
    rec_output.display(args.outfile)
    plt.show()

    return 0

if __name__ == "__main__":
   main()
