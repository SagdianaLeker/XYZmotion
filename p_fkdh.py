import numpy as np
import p_trigd as tr

#initial parameter
#[j0 j1 j2 j3;d0 d1 d2 d3;a0 a1 a2 a3;t0 t1 t2 t3]
#theta3=360-theta1-theta2
def dh_par(j1,j2,j3): # D-H parameter table 
    j4 = 360-(j2+j3)
    j = np.array([[j1, j2, j3, j4],[10, 0, 0, 0],[5, 25, 25, 10],[90, 0, 0, 0]], dtype=object)
    return j

def dh(theta,d,a,alpha): # homogeneous transformation matrix
    dhm = np.array([[tr.cosd(theta),-tr.sind(theta)*tr.cosd(alpha),tr.sind(theta)*tr.sind(alpha),a*tr.cosd(theta)],
    [tr.sind(theta),tr.cosd(theta)*tr.cosd(alpha),-tr.cosd(theta)*tr.sind(alpha),a*tr.sind(theta)],
    [0,tr.sind(alpha),tr.cosd(alpha),d],
    [0,0,0,1]])
    return dhm

def dh_kine(j): # cumulative transformation matrix
    #collum 1=joint angle, 2=joint offset, 3=link length, 4=twist angle
    T01 = dh(j[0,0],j[1,0],j[2,0],j[3,0])
    T12 = dh(j[0,1],j[1,1],j[2,1],j[3,1])
    T23 = dh(j[0,2],j[1,2],j[2,2],j[3,2])
    T34 = dh(j[0,3],j[1,3],j[2,3],j[3,3])
    
    T02 = T01@T12
    T03 = T02@T23
    T04 = T03@T34
    return np.hstack((T01, T02, T03, T04))

def el_xyzpos(m): #el_xyzpos(dh_kine(j)) extract xyz
    Q1 = np.array([0,0,m[0,3],m[0,7],m[0,11],m[0,15]])
    Q2 = np.array([0,0,m[1,3],m[1,7],m[1,11],m[1,15]])
    Q3 = np.array([0,m[2,3],m[2,3],m[2,7],m[2,11],m[2,15]])
    Q = np.vstack((Q1,Q2,Q3))
    return Q

def el_pos2base(Q): # converts relative to base
    X,Y,Z = Q[0,:],Q[1,:],Q[2,:]
    p0 = np.array([X[0],Y[0],Z[0]]) #base
    p1 = np.array([X[1],Y[1],Z[1]])
    p2 = np.array([X[2],Y[2],Z[2]])
    p3 = np.array([X[3],Y[3],Z[3]])
    p4 = np.array([X[4],Y[4],Z[4]])
    p5 = np.array([X[5],Y[5],Z[5]]) #end effector
    return p0,p1,p2,p3,p4,p5


def jacobian(dhkine):
    Z0 = np.array([[0],[0],[1]])
    O = np.array([[0],[0],[0]])
    O4 = dhkine[0:3,15:]
    Jac1 = np.cross(Z0,(O4-O), axis=0)

    Z1 = dhkine[0:3,2:3]
    O1 = dhkine[0:3,3:4]
    Jac2 = np.cross(Z1,(O4-O1), axis=0)

    Z2 = dhkine[0:3,6:7]
    O2 = dhkine[0:3,7:8]
    Jac3 = np.cross(Z2,(O4-O2), axis=0)

    Z3 = dhkine[0:3,10:11]
    O3 = dhkine[0:3,11:12]
    Jac4 = np.cross(Z3,(O4-O3), axis=0)
    return np.hstack((Jac1,Jac2,Jac3,Jac4))

def PinvJac(jacobian): # pseudoinverse Jacobian
    return np.linalg.pinv(jacobian)



dh_kine.__doc__="homogeneous transformation matrix for dh parameters"
