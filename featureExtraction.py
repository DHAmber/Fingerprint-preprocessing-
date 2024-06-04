from numpy.random import randint

def genearte_population():
    with open('param.txt', 'a+') as f:
        for i in range(150):
            f.writelines(str(randint(0, 2, 27))+'\n')

def readPopulation(user):
    with open('param.txt','r') as f:
        ll=f.readlines()
        lines=[]
        for line in ll:
            lines.append(line.replace('[', '').replace(']', '').split())
        if user[:3]=='101':
            return lines[:50]
        elif user[:3]=='102':
            return lines[50:100]
        elif user[:3]=='103':
            return lines[100:]
        else:
            return None