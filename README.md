# MPC_pathfollowing_sim
未実装
## matplotlibcpp
### install
Python3.8のとき
```
sudo apt-get install python3.8-dev
```
### コンパイル
solver
```
g++ -Wall mpc_solver.cpp `pkg-config --libs casadi` -DWITHOUT_NUMPY -I /usr/include/python3.8 -lpython3.8
```
