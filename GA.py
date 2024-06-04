import numpy as np
import math
from numpy.random import rand
from featureExtraction import *

templateFile='101_1.txt'
queryFile='102_1.txt'

def readData(filePath):
    with open(filePath,'r') as file:
        return file.readlines()

template_set =[line.split() for line in readData(templateFile)]
query_set = [line.split() for line in readData(queryFile)]


def convert_list_to_integer(lst):
    sum = 0
    x = 1
    for i in range(len(lst)):
        sum = sum + x*lst[i]
        x = x*2
    return sum

def get_value_from_chromosome(single_chromo):

    s_list = single_chromo[:5]
    theta_list = single_chromo[5:11]
    tx_list = single_chromo[11:19]
    ty_list = single_chromo[19:27]
    
    s = convert_list_to_integer(s_list)*0.01 + 0.9
    theta = convert_list_to_integer(theta_list) - 30
    tx = convert_list_to_integer(tx_list)-128
    ty = convert_list_to_integer(ty_list)-128
    
    return s, theta, tx, ty

def fitness_function(s, theta, tx, ty, mx, my, qx, qy, thold):
    theta = math.radians(theta) 
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    num = 0
    for i in range(len(mx)):
        for j in range(i,len(qx)):
            px = s*(mx[i]*cos_theta - my[i]*sin_theta) - qx[j] + tx
            py = s*(mx[i]*sin_theta + my[i]*cos_theta) - qy[j] + ty

            if(math.sqrt(px*px + py*py)< thold):
                num = num + 1
    return num

def selection(pop, scores):
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop),2):
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

def crossover(p1, p2, r_cross):
    c1, c2 = p1.copy(), p2.copy()
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        if rand() < r_mut:
            bitstring[i] = 1 - bitstring[i]
        return bitstring

def genetic_algorithm(n_iter, n_pop, r_cross, r_mut, thold, mx, my, qx, qy):

    arr1 = readPopulation(templateFile)
    arr2 = readPopulation(queryFile)
    pop = np.int_(np.concatenate((arr1, arr2))).tolist()
    s, theta, tx, ty = get_value_from_chromosome(pop[0])
#     print(s, theta, tx, ty)
    best, best_eval = 0, fitness_function(s, theta, tx, ty, mx, my, qx, qy, thold)
    for gen in range(n_iter):
        scores = []
        for i in range(100):
            s, theta, tx, ty = get_value_from_chromosome(pop[i])
            scores.append(fitness_function(s, theta, tx, ty, mx, my, qx, qy, thold))

        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print("new best (%s) = %.3f" % (pop[i], scores[i]))

        selected = [selection(pop, scores) for i in range(n_pop)]

        children = list()
        for i in range(0, n_pop, 2):
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                mutation(c, r_mut)
                children.append(c)
        pop = children
        
    return [best, best_eval]

def Genetic(minutiae_set1, minutiae_set2):
    mx = []
    my = []
    for i in range(len(minutiae_set1)):
        mx.append(float(minutiae_set1[i][0]))
        my.append(float(minutiae_set1[i][1]))
    qx = []
    qy = []
    for i in range(len(minutiae_set2)):
        qx.append(float(minutiae_set2[i][0]))
        qy.append(float(minutiae_set2[i][1]))
    n_iter = 100
    n_pop = 100
    r_cross = 0.9
    r_mut=0.1
    thold = 100
    best, score = genetic_algorithm(n_iter, n_pop, r_cross, r_mut, thold, mx, my, qx, qy)



Genetic(template_set,query_set)

