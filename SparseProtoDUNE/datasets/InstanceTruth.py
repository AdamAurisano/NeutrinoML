import numpy as np
import torch, math

def gauss_pdf(c, sigmainv, mu, dim=3):
    prob = []
    sigma = np.linalg.inv(sigmainv)
    den = (2*np.pi)**dim*np.linalg.det(sigma)
    N = 1/np.sqrt(den)
    mu = np.array(mu)
    for k in c:
        k = np.array(k)
        p = (k-mu).dot(sigmainv).dot((k-mu).transpose())
        f = N*np.exp(p*-0.5)
        prob.append(f)
    return prob

def get_InstanceTruth(c, vox_id, y, width):                                   

   # background                                                               
    stuff_list = [1,2,4] # delta, diffuse, michel 
    for k in stuff_list:
        vox_id[y ==k] = 0
  
    sigma = np.identity(3)*width                                                                      
    sigmainv = np.linalg.inv(sigma)                                                                   
    medoids = []
    offset = np.ones((c.shape[0],c.shape[1]))*(-1)
    chtm = np.zeros(c.shape[0])                                                                       
    u, ct = np.unique(vox_id, return_counts=True) 
    for p, _nvox in zip(u[:1], ct):
        #if p == 0: continue
        if _nvox < 5:
            vox_id[vox_id ==p ]= 0
            continue
        mask = vox_id ==p
        ci = c[mask]
        a = np.zeros(ci.shape[0])                                                                     
        l = tuple(np.linalg.norm((k -ci),axis=1).sum() for k in ci)                                   
        ind = l.index((min(l)))                                                                       
        medoids.append(np.array(ci[ind]))
        Off = tuple(np.array(k-ci[ind]) for k in ci)                                                  
        chtm[mask]= gauss_pdf(ci,sigmainv,ci[ind])    
        chtm[mask] /= chtm[mask].max().item() 
        offset[mask] = Off
        
    chtm = torch.tensor(chtm.reshape([chtm.shape[0],1])) 
    return torch.tensor(np.array(medoids)), chtm, torch.tensor(np.array(offset)), vox_id  
