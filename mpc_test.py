from casadi import *
import numpy as np
import math
import matplotlib.pyplot as plt

# 問題設定
T = 5.0     # ホライゾン長さ
N = 100     # ホライゾン離散化グリッド数
dt = T / N  # 離散化ステップ
nx = 3      # 状態空間の次元
nu = 3      # 制御入力の次元
# 目標状態(経路)
path = []
for i in range(1000):
    path.append([i,-1,0.0])

inipos=[0.0,-2.0,0.0]

# 以下で非線形計画問題(NLP)を定式化
w   = []  # 最適化変数を格納する list
w0  = []  # 最適化変数(w)の初期推定解を格納する list
lbw = []  # 最適化変数(w)の lower bound を格納する list
ubw = []  # 最適化変数(w)の upper bound を格納する list
J = 0     # コスト関数
g = []    # 制約（等式制約，不等式制約どちらも）を格納する list
lbg = []  # 制約関数(g)の lower bound を格納する list
ubg = []  # 制約関数(g)の upper bound を格納する list

Xk = MX.sym('X0', nx) # 初期時刻の状態ベクトル x0

w += [Xk]             # x0 を 最適化変数 list (w) に追加
# 初期状態は given という条件を等式制約として考慮
lbw += inipos # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
ubw += inipos # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
w0  += [0.0,0.0,0.0] # x0 の初期推定解


#重み
Q     = [1.0, 1.0, 0.01] # 状態への重み
R     = [1.0, 1.0, 1.0]  # 制御入力への重み
# 離散化ステージ 0~N-1 までのコストと制約を設定
k_max=0
for k in range(N):
    if len(path)<=k:
        break
    Uk = MX.sym('U_' + str(k), nu) # 時間ステージ k の制御入力 uk を表す変数
    w   += [Uk]                    # uk を最適化変数 list に追加
    lbw += [-1.0,-1.0,-1.0]        # uk の lower-bound
    ubw += [1.0,1.0,1.0]           # uk の upper-bound
    w0  += [0.0,0.0,0.0]                 # uk の初期推定解

    #運動方程式
    x   = Xk[0]    # X座標[m]
    y   = Xk[1]    # Y座標[m]
    yaw = Xk[2]    # ロボット角度[rad]
    #制御入力
    vx      = Uk[0]   # vx[m/s]
    vy      = Uk[1]   # vy[m/s]
    angular = Uk[2]   # w[rad/s]
    # ステージコストのパラメータ
    k_max+=1
    x_ref = path[k]           # 目標状態
    L = 0 # ステージコスト
    for i in range(nx):
        L += 0.5 * Q[i] * (Xk[i]-x_ref[i])**2
    for i in range(nu):
        L += 0.5 * R[i] * Uk[i]**2
    J = J + L * dt # コスト関数にステージコストを追加

    # Forward Euler による離散化状態方程式
    Xk_next = vertcat(x + vx * dt,
                      y + vy * dt,
                      yaw + angular * dt)
    Xk1 = MX.sym('X_' + str(k+1), nx)  # 時間ステージ k+1 の状態 xk+1 を表す変数
    w   += [Xk1]                       # xk+1 を最適化変数 list に追加
    lbw += [-inf, -inf, -inf]    # xk+1 の lower-bound （指定しない要素は -inf）
    ubw += [inf, inf, inf]       # xk+1 の upper-bound （指定しない要素は inf）
    w0  += [0.0,0.0,0.0]       # xk+1 の初期推定解

    # 状態方程式(xk+1=xk+fk*dt) を等式制約として導入
    g   += [Xk_next-Xk1]
    lbg += [0.0,0.0,0.0] # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
    ubg += [0.0,0.0,0.0] # 等式制約は lower-bo und と upper-bound を同じ値にすることで設定
    Xk = Xk1

# 終端コストのパラメータ
Vf = 0                            # 終端コスト
for i in range(nx):
    Vf += 0.5 * Q[i] * (Xk[i]-x_ref[i])**2
for i in range(nu):
    Vf += 0.5 * R[i] * Uk[i]**2
J = J + Vf

# 非線形計画問題(NLP)
nlp = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
# Ipopt ソルバー，最小バリアパラメータを0.001に設定
solver = nlpsol('solver', 'ipopt', nlp, {'ipopt':{'mu_min':0.001}})

# NLPを解く
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x'].full().flatten()

# 解をプロット
x1_opt = np.array(w_opt[0::6])
x2_opt = np.array(w_opt[1::6])
x3_opt = np.array(w_opt[2::6])
u1_opt  = np.array(w_opt[3::6])
u2_opt  = np.array(w_opt[4::6])
u3_opt  = np.array(w_opt[5::6])

tgrid = np.array([dt*k for k in range(k_max+1)])
plt.figure(1)
plt.clf()
plt.subplot(1, 2, 1)
plt.plot(tgrid, x1_opt, '-')
plt.plot(tgrid, x2_opt, '-')
plt.plot(tgrid, x3_opt, '-')
plt.step(tgrid, np.array(vertcat(DM.nan(1), u1_opt)), '-.')
plt.step(tgrid, np.array(vertcat(DM.nan(1), u2_opt)), '-.')
plt.step(tgrid, np.array(vertcat(DM.nan(1), u3_opt)), '-.')
plt.xlabel('t')
plt.legend(['x','y', 'yaw', 'vx','vy','angular'])
plt.grid()
plt.subplot(1, 2, 2)
path=np.array(path)
plt.xlim([0,10])
#plt.ylim([0,2.0])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(path[:,0], path[:,1], '-')
plt.plot(x1_opt, x2_opt, '.')
plt.grid()
plt.show()
