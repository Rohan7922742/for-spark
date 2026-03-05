import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torch.optim as optim


N=torch.linspace(0,50,200).unsqueeze(1)
sigma=torch.linspace(50,150,50).unsqueeze(1)

N_grid, sigma_grid = torch.meshgrid(N.squeeze(), sigma.squeeze(), indexing='ij')

N_flat=N_grid.reshape(-1,1)
sigma_flaat=sigma_grid.reshape(-1,1)

N_norm=N_flat/50
sigma_norm=sigma_flaat/150
X_train=torch.cat([N_norm, sigma_norm], dim=1)

def true_solution(N_grid, sigma_grid, a, m):
    return 1-torch.exp(-a*sigma_grid**m * N_grid)

D_true = true_solution(N_grid, sigma_grid, 1e-5, 2.0)

D_train=true_solution(N_flat, sigma_flaat, 1e-5, 2.0)

class PINN(nn.Module):
    def __init__(self):
        super(PINN,self).__init__()
        
        self.net=nn.Sequential(
            nn.Linear(2,32),
            nn.Tanh(),
            nn.Linear(32,32),
            nn.Tanh(),
            nn.Linear(32,1),
            nn.Sigmoid()
            )
        self.a = nn.Parameter(torch.tensor(1e-5))
        self.m = nn.Parameter(torch.tensor(2.0))

        
    def forward(self,x):
        return self.net(x)
    
model=PINN()

def physics(model, N_norm, sigma_norm, sigma_real):
    X=torch.cat([N_norm, sigma_norm], dim=1)
    X.requires_grad_(True)
    
    D_pred=model(X)
    grad=torch.autograd.grad(
        D_pred,
        X,
        grad_outputs=torch.ones_like(D_pred),
        create_graph=True
    )[0]
    dD_dN=grad[:,0:1]/5000.00
    residual=dD_dN -model.a * sigma_real**model.m * (1-D_pred)
    return residual
optimizer=optim.Adam(model.parameters(), lr=1e-3)
loss_fn=nn.MSELoss()
physics_Lambda=0.001

epochs=4000
for epoch in range(epochs):
    optimizer.zero_grad()
    D_pred=model(X_train)
    data_loss=loss_fn(D_pred, D_train)
    
    residual=physics(
        model, 
        N_norm, 
        sigma_norm, 
        sigma_flaat
    )
    
    physics_loss=torch.mean(residual**2)
    
    loss=data_loss + physics_Lambda*physics_loss
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(epoch, loss.item(), model.a.item(), model.m.item())
        
optimizer=optim.LBFGS(model.parameters(), lr=0.1)

def closure():
    optimizer.zero_grad()
    D_pred = model(X_train)
    data_loss = loss_fn(D_pred, D_train)
    residual = physics(model, N_norm, sigma_norm, sigma_flaat)
    physics_loss = torch.mean(residual**2)
    loss = data_loss + physics_Lambda * physics_loss
    loss.backward()
    return loss

optimizer.step(closure)

model.eval()

with torch.no_grad():
    D_pred_full = model(X_train).reshape(200, 50).numpy()
    
N_plot = N.numpy()
sigma_plot = sigma.numpy()

X_mesh, Y_mesh = np.meshgrid(N_plot, sigma_plot, indexing='ij')

plt.figure(figsize=(10,6))
cp=plt.contourf(X_mesh, Y_mesh, D_pred_full, levels=40, cmap="viridis")
plt.colorbar(cp)
plt.xlabel("Cycles (N)")
plt.ylabel("Stress (sigma)")
plt.title("Damage contour from PINN")
plt.show()