import numpy as np

def rician_estimate(img):
    X = img.reshape(-1) 
    t=np.arange(0,len(X),1) 
    size=len(X)-2
    sigma=0
    for i in range(1,len(X)-2):
        a = (t[i+1] - t[i]) / (t[i+1] - t[i-1])
        b = (t[i] - t[i-1]) / (t[i+1] - t[i-1])
        c = 1 / (a * a + b * b + 1)        
        
        sigma += c * c * (a * X[i-1] + b * X[i+1] - X[i]) * (a * X[i-1] + b * X[i+1] - X[i])

    return np.sqrt(sigma / size)
