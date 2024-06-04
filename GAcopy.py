

from numpy.random import randint
import math
from numpy.random import rand

def readData(filePath):
    with open(filePath,'r') as file:
        return file.readlines()


# In[96]:


templateFile='101_1.txt'
queryFile='101_2.txt'
template_set =[line.split() for line in readData(templateFile)]
query_set = [line.split() for line in readData(queryFile)]


def convert_list_to_integer(lst):
    sum = 0
    base = 1
    for i in range(len(lst)):
        sum = sum + base*lst[i]
        base = base*2
        
    return sum

def get_value_from_chromosome(single_chromo):
    #     first 5 bits represents scale
    s_list = single_chromo[:5]
#     Next 6 bits represents angle
    theta_list = single_chromo[5:11]
#     Next 8 bits represents movement in x axis
    tx_list = single_chromo[11:19]
#     Next 8 bits represents movement in y axis
    ty_list = single_chromo[19:27]
    
    s = convert_list_to_integer(s_list)*0.01 + 0.9
    theta = convert_list_to_integer(theta_list) - 30
    tx = convert_list_to_integer(tx_list)-128
    ty = convert_list_to_integer(ty_list)-128
    
    return s, theta, tx, ty

def fitness_function(s, theta, tx, ty, mx, my, qx, qy, thold):
    #     Converting degrees to radian
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
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop),2):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]

def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]

def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]
        return bitstring


# In[98]:


def genetic_algorithm(n_iter, n_pop, r_cross, r_mut, thold, mx, my, qx, qy):    
    # initial population of random bitstring
    pop = [randint(0, 2, 27).tolist() for i in range(n_pop)]
    
    # keep track of best solution
    s, theta, tx, ty = get_value_from_chromosome(pop[0])
#     print(s, theta, tx, ty)
    best, best_eval = 0, fitness_function(s, theta, tx, ty, mx, my, qx, qy, thold)
    # enumerate generations
    for gen in range(n_iter):
        
        # evaluate all candidates in the population
        scores = []
        for i in range(100):
            s, theta, tx, ty = get_value_from_chromosome(pop[i])
            scores.append(fitness_function(s, theta, tx, ty, mx, my, qx, qy, thold))
        
        # check for new best solution
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print("new best (%s) = %.3f" % (pop[i], scores[i]))
                
        # select parents
        selected = [selection(pop, scores) for i in range(n_pop)]
        
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
        
    return [best, best_eval]


# In[99]:


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
    #print(mx)

    # define the total iterations
    n_iter = 100

    # define the population size. This varries as the number of miniuate of images varries
    n_pop = 100

    # crossover rate
    r_cross = 0.9

    # mutation rate
    #r_mut = 1.0 / (float(27) * 2)
    r_mut=0.1

    thold = 100

    # perform the genetic algorithm search
    best, score = genetic_algorithm(n_iter, n_pop, r_cross, r_mut, thold, mx, my, qx, qy)
    if(score>final_thres):
        return True
    else:
        return False

Genetic(template_set,query_set)

