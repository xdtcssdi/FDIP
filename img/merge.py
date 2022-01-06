# import numpy as np
# import cv2
# from glob import glob
# dip_path = "gt/*.png"
# our_path = "online/our/*.png"
# other_path = "online/transPose/*.png"
# outs = list(glob(our_path))
# gts = list(glob(dip_path))
# other_paths = list(glob(other_path))
# cnt = 0
# for g, o, t in zip(gts, outs, other_paths):
#     g = cv2.imread(g)
#     o = cv2.imread(o)
#     t = cv2.imread(t)
#     m = np.concatenate((g, o, t), axis=1)
#     cv2.imwrite(f"merge/{cnt}.png",m)
#     cnt+=1


import numpy as np
import cv2
from glob import glob
dip_path = "gt/*.png"
our_online = "online/our/*.png"
other_online = "online/transPose/*.png"
our_offline = "offline/our/*.png"
other_offline = "offline/transPose/*.png"

gts = list(glob(dip_path))

outs_online = list(glob(our_online))
outs_offline = list(glob(our_offline))

other_online = list(glob(other_online))
other_offline = list(glob(other_offline))




cnt = 0
for g, ooff, toff, oon, ton in zip(gts, outs_offline, other_offline, outs_online, other_online):
    g = cv2.imread(g)
    ooff = cv2.imread(ooff)
    toff = cv2.imread(toff)
    oon = cv2.imread(oon)
    ton = cv2.imread(ton)
    
    m = np.concatenate((g, toff, ooff,ton, oon,), axis=1)
    cv2.imwrite(f"merge/{cnt}.png",m)
    cnt+=1