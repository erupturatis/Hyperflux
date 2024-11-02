
Only weights, lr = 0.0008, no scheduler
* 70.5% convergence, peaks at 72% somewhere

Only weights, lr = 0.0008, scheduler 0.1 ** (epoch//2)
* 73% convergence smoothly, with bigger scheduler values still same convergence but more unstable


Weights and Flipping, lr_w = 0.0008, lr_flip = lr_w * 10, no scheduler
* Same convergence as only weights

Normal scheduler helps the network without pruning, but it does not help the network after pruning.

Weights + Pruning + Flipping, lr_w = 0.0008, lr_flip = lr_w * 10
scheduler slowly decreasing until desired pruning and then aggresively decreasing while adding weights