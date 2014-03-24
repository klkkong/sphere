#!/usr/bin/env python

from sphere import *
import subprocess
import sys

def passed():
    return "\tPassed"

def failed():
    return "\tFailed"

def test(statement, string):
    if (statement == True):
        print(string + passed())
    else:
        print(string + failed())
        raise Exception("Failed")

def compare(first, second, string):
    returnvalue = (first == second)
    if (returnvalue == True or returnvalue > 0):
        print(string + passed())
    else:
        print(string + failed() + ' (' + str(returnvalue) + ')')
        raise Exception("Failed")
        return(returnvalue)

def compareFloats(first, second, string, tolerance=1e-3):
    #if abs(first-second) < tolerance:
    if abs((first-second)/first) < tolerance:
        print(string + passed())
    else :
        print(string + failed())
        print("First: " + str(first))
        print("Second: " + str(second))
        print("Abs. difference: " + str(second-first))
        print("Rel. difference: " + str(abs((first-second)/first)))
        raise Exception("Failed")
        return(1)

def compareNumpyArrays(first, second, string):
    if ((first == second).all()):
        print(string + passed())
    else :
        print(string + failed())
        raise Exception("Failed")
        return(1)

def compareNumpyArraysClose(first, second, string, tolerance=1e-5):
    if (numpy.allclose(first, second, atol=tolerance)):
        print(string + passed())
    else :
        print(string + failed())
        raise Exception("Failed")
        return(1)
