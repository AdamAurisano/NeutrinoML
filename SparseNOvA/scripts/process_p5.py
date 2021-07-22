import h5py, numpy as np, matplotlib.pyplot as plt

f = h5py.File("/data/p5/test/fardet_genie_N1810j0211a_nonswap_genierw_fhc_v08_1000_r00015387_s24_c000_R19-11-18-prod5reco.x_v1_20191108_032900_sim.h5caf.h5")

#print(f.keys())
#print(f["rec.training.cvnmaps"].keys())

#print(f["rec.training.cvnmaps"]["cvnmap"][()].shape)
im = f["rec.training.cvnmaps"]["cvnmap"][0].reshape(2,100,80) 
#plt.imshow(im[1])
#plt.savefig("test.png")
mask = im[0].nonzero()
print(mask)

print(im[0][mask])

lab = f["rec.training.cvnmaps"]["cvnlabmap"][0].reshape(2, 100, 80)
print(lab[0][mask])

obj = f["rec.training.cvnmaps"]["cvnobjmap"][0].reshape(2, 100, 80)
print(obj[0][mask])
#print(obj)



