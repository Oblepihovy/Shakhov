from src.config.libloader import xp

#TD_KN=0.1
TD_KN = 0.09
TD_PR=0.72
TD_W=0.81
#TD_W = 0.2

X_LEFT, X_RIGHT = 0, 1
XI_LEFT, XI_RIGHT = -5, 5

def F_BEG_N(x):
    return xp.where(x <= 0.5, 1., 0.125)
    #return 1

def F_BEG_U(x):
    return xp.zeros_like(x)

def F_BEG_T(x):
    return xp.where(x <= 0.5, 1., 0.8)
    #return 1

dtype = xp.float64