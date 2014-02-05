#!/usr/bin/env python

from sphere import *
import subprocess
import sys

def passed():
    return "\tPassed"

def failed():
    raise Exception("Failed")
    return "\tFailed"

def compare(first, second, string):
  if (first == second):
    print(string + passed())
  else:
    print(string + failed())
    return(1)

def compareFloats(first, second, string, tolerance=1e-3):
    #if abs(first-second) < tolerance:
    if abs((first-second)/first) < tolerance:
        print(string + passed())
    else :
        print(string + failed())
        print("First: " + str(first))
        print("Second: " + str(second))
        print("Difference: " + str(second-first))
        return(1)

def compareNumpyArrays(first, second, string):
    if ((first == second).all()):
        print(string + passed())
    else :
        print(string + failed())
        return(1)

def compareNumpyArraysClose(first, second, string, tolerance=1e-5):
    if (numpy.allclose(first, second, atol=tolerance)):
        print(string + passed())
    else :
        print(string + failed())
        return(1)


def cleanup(spherebin):
    'Remove temporary files'
    subprocess.call("rm -f ../input/" + spherebin.sid + ".bin", shell=True)
    subprocess.call("rm -f ../output/" + spherebin.sid + ".status.dat", shell=True)
    subprocess.call("rm -f ../output/" + spherebin.sid + ".*.bin", shell=True)
    subprocess.call("rm -f ../output/" + spherebin.sid + ".*.vtu", shell=True)
    subprocess.call("rm -f ../output/fluid-" + spherebin.sid + ".*.vti", shell=True)
    subprocess.call("rm -f ../output/" + spherebin.sid + "-conv.png", shell=True)
    subprocess.call("rm -f ../output/" + spherebin.sid + "-conv.log", shell=True)
    print("")
