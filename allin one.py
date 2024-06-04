import math as m
import numpy as np
#Hough transform

def readData(filePath):
   with open(filePath, 'r') as file:
      return file.readlines()

   template_set = [line.split() for line in readData(templateFile)]
   query_set = [line.split() for line in readData(queryFile)]
   txytheta=[]
   theta=[]
   qxytheta=[]
   qtheta=[]
   for i in range(len(template_set)):
       tx.append(float(template_set[i][0]))
       ty.append(float(template_set[i][1]))
       ttheta.append(float(template_set[i][2]))
       txytheta.append((float(template_set[i][0]),float(template_set[i][1]),float(template_set[i][2])))

   for i in range(len(query_set)):
       qx.append(float(query_set[i][0]))
       qy.append(float(query_set[i][1]))
       qtheta.append(float(query_set[i][2]))
       qxytheta.append((float(query_set[i][0]),float(query_set[i][1]),float(query_set[i][2])))
   #print(tx)
   #print(ty)
   #print(qtheta)
   return(qxytheta,txytheta)
def hough_transform(Q_set, T_set):
    A = {}
    max_val, max_val_key = -float('inf'), None
    for i in range(len(Q_set)):
        for j in range(len(T_set)):
            x_q, y_q, theta_q = Q_set[i]
            x_t, y_t, theta_t = T_set[j]

            del_theta = theta_t - theta_q
            del_x = x_t - x_q * math.cos(del_theta) - y_q * math.sin(del_theta)
            del_y = y_t + x_q * math.sin(del_theta) - y_q * math.cos(del_theta)
            k = (int(del_theta), int(del_x), int(del_y))
            if k not in A:
                A[k] = 1
            else:
                A[k] += 1
            if A[k] > max_val:
                max_val = A[k]
                max_val_key = k
    #     print("Dictionary:",A)
    return max_val_key,del_x,del_y,del_theta


# In[7]:


'''
    Alignment operator

'''


def align(query_set, del_theta, del_x, del_y):
    aligned_query = []
    for current_query in query_set:
        x, y, theta = current_query
        new_theta = theta + del_theta
        new_x = x + del_x * math.cos(new_theta) - del_y * math.sin(new_theta)
        new_y = y + del_y * math.cos(new_theta) + del_x * math.sin(new_theta)
        aligned_query.append((new_x, new_y, new_theta))
    return aligned_query


DISTANCE_THRESHOLD = 10
ROTATION_THRESHOLD = 20


def minutiae_pairing(Q_set, T_set,del_x, del_y, del_theta):
    f_T = [False] * len(T_set)
    f_Q = [False] * len(Q_set)
    count = 0
    ret = []

    #     aligned_Q_set = align(Q_set,del_theta,del_x,del_y)
    aligned_Q_set = Q_set
    #     aligned_T_set = T_set

    for i in range(len(Q_set)):
        Qx, Qy, Qtheta = aligned_Q_set[i]
        for j in range(len(T_set)):
            if (not f_T[j] and not f_Q[i]):
                new_Tx, new_Ty, new_Ttheta = T_set[j]
                new_del_theta = new_Ttheta - Qtheta
                new_del_x = new_Tx - (Qx * math.cos(new_del_theta)) - (Qy * math.sin(new_del_theta))
                new_del_y = new_Ty + (Qx * math.sin(new_del_theta)) - (Qy * math.cos(new_del_theta))
                d_i_j = math.sqrt(new_del_x ** 2 + new_del_y ** 2)
                #                 print("Distance:",d_i_j,"Rotation:",abs(new_del_theta))
                if d_i_j < DISTANCE_THRESHOLD and abs(new_del_theta) < ROTATION_THRESHOLD:
                    f_T[j] = True
                    f_Q[i] = True
                    count += 1
                    ret.append((j, i))
    return ret


def match_score(matched_minutiae, T_set, Q_set, image1_area, image2_area):
    #     MATCH_THRESHOLD = min(20,len(T_set)//2)
    MATCH_THRESHOLD = 10
    T_points = []
    Q_points = []
    for i, j in matched_minutiae:  # i in T_set, j in Q_set
        T_points.append((T_set[i][0], T_set[i][1]))
        Q_points.append((Q_set[j][0], Q_set[j][1]))
    #     if min(len(T_points),len(Q_points)) >= 3:
    #         hull_T = ConvexHull(T_points)
    #         hull_Q = ConvexHull(Q_points)
    #         area_T = hull_T.volume
    #         area_Q = hull_Q.volume

    #         feature_2 = min((area_T/image1_area) , (area_Q/image2_area))
    #     else:
    #         feature_2 = 0

    #     feature_1 = len(matched_minutiae)/min(len(T_set),len(Q_set))
    feature_1 = len(matched_minutiae)
    feature_2 = 0

    print("Match Score:",0.9*feature_1+0.1*feature_2)
    return 0.9 * feature_1 + 0.1 * feature_2 >= MATCH_THRESHOLD


#Q_set,T_set=readData(r'E:\Phd\6th sem\FingerPrint_matching_using_GA\')
templateFile = '101_1.txt'
queryFile = '101_2.txt'
Q_set,T_set=readData(templateFile)
max_key,del_x,del_y,del_theta=hough_transform(Q_set, T_set)
newx,newy,new_theta=align(Q_set, del_theta, del_x, del_y)
ret=minutiae_pairing(Q_set, T_set,del_x, del_y, del_theta)
score=match_score(ret, T_set, Q_set)
print("matching score=",match)

