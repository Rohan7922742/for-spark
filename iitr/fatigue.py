import numpy as np
import matplotlib.pyplot as plt
a = 1e-5
m = 2
sigma = 100

def true_solution(N):
    return 1-np.exp(-a*sigma**m * N)
N = np.linspace(0, 5000, 200).reshape(1,-1)
D = true_solution(N)

np.random.seed(42)
def init_prams():
    w1=np.random.randn(20,1)*0.1
    b1=np.zeros((20,1))
    w2=np.random.randn(20,20)*0.1
    b2=np.zeros((20,1))
    w3=np.random.randn(1,20)*0.1
    b3=np.zeros((1,1))
    return w1,b1,w2,b2,w3,b3
def forward(N,p):
    w1,b1,w2,b2,w3,b3=p
    z1=w1@N+b1
    A1=np.tanh(z1)
    z2=w2@A1+b2 
    A2=np.tanh(z2)
    z3=w3@A2+b3 
    predict_D=z3 
    cache=(N,z1,A1,z2,A2,z3)
    return predict_D, cache
def physics(N, predict_D):
    dN = N[0,1] - N[0,0]
    dDbydN = np.gradient(predict_D, dN, axis=1)
    residual = dDbydN - a * sigma**m * (1 - predict_D)
    return residual
def backward( predict_D, D, params, cache, pinn=False, physics_lambda=1.0):
    w1,b1,w2,b2,w3,b3=params
    N,z1,A1,z2,A2,z3=cache
    m=N.shape[1]
    dz3=( predict_D - D)/m
    if pinn:
        residulal=physics(N, predict_D)
        dz3+=physics_lambda*residulal/m
    dw3=dz3@A2.T
    db3=np.sum(dz3, axis=1, keepdims=True)
    
    dA2=w3.T@dz3
    dz2=dA2*(1-np.tanh(z2)**2)
    
    dw2=dz2@A1.T 
    db2=np.sum(dz2, axis=1, keepdims=True)
    
    dA1=w2.T@dz2
    dz1=dA1*(1-np.tanh(z1)**2)
    
    dw1=dz1@N.T 
    db1=np.sum(dz1, axis=1, keepdims=True)
    grads=( dw1, db1, dw2, db2, dw3, db3 )
    return grads
def update_params(params, grads, learning_rate):
    return tuple(p-learning_rate*g for p, g in zip(params, grads))
def tarin(pinn=False):
    params=init_prams()
    learning_rate=0.001
    epochs = 3000
    
    for i in range(epochs):
        predict_D, cache = forward(N, params)
        grads = backward(predict_D, D, params, cache, pinn=pinn)
        params = update_params(params, grads, learning_rate)
        
    return params

print("Training in standard NN...")
params_nn=tarin(pinn=False)
print("Training in pinn...")
params_pinn = tarin(pinn=True)

D_nn, _ = forward(N, params_nn)
D_pinn, _ = forward(N, params_pinn)


plt.figure(figsize=(10,6))
plt.plot(N.flatten(), D.flatten(), label="True")
plt.plot(N.flatten(), D_nn.flatten(), "--", label="NN")
plt.plot(N.flatten(), D_pinn.flatten(),"_", label="PINN")
plt.xlabel("cycles(N)")
plt.ylabel("Damage(D)")
plt.legend()
plt.title("FAtigue Damage Prediction")
plt.show()

plt.figure()
plt.plot(N.flatten(), np.abs(D_nn - D).flatten(), label="NN Error")
plt.plot(N.flatten(), np.abs(D_pinn - D).flatten(), label="Pinn_Error")
plt.xlabel("cycles(N)")
plt.ylabel("Absolute Error")
plt.legend()
plt.title("Prediction Error")
plt.show()

sigma_values=np.linspace(50, 150, 50)
N_contour=np.linspace(0, 50, 200)
Damage_map=np.zeros((len(sigma_values), len(N_contour)))

for i, sigma in enumerate(sigma_values):
    temp_D = 1-np.exp(-a*sigma**m * N_contour)
    Damage_map[i,:] = temp_D
    
plt.figure()
X, Y=np.meshgrid(N_contour, sigma_values)
cp=plt.contourf(X, Y, Damage_map, levels=40, cmap="viridis")
plt.colorbar(cp)
plt.xlabel("cycles(N)")
plt.ylabel("Stress (σ)")
plt.title("Damage Contour")
plt.show()