from casadi import *
import numpy as np
import math
import matplotlib.pyplot as plt

class MPC(object):
    # 問題設定
    T = 30.0     # ホライゾン長さ
    N = 10     # ホライゾン離散化グリッド数
    dt = T / N  # 離散化ステップ
    nx = 3      # 状態空間の次元
    nu = 3      # 制御入力の次元
    Vmax = 1.0
    Angularmax = 1.0
    #ステージコスト
    Q = [100.0, 100.0, 50.0] # 状態への重み
    R = [1.0, 0.0, 1.0]  # 制御入力への重み
    #終端コスト
    Qf = [1.0, 1.0, 0.5] # 状態への重み
    Rf = [0.1, 0.0, 0.1]  # 制御入力への重み
    inipos=[]
    path=[]

    def __init__(self):
        pass
    def __init__(self,T,dt,Vmax,Angularmax):
        self.T=T
        self.dt=dt
        self.N=int(T/dt)
        self.Vmax=Vmax
        self.Angularmax=Angularmax

    def calc_vel(self,path):#inipos=[x0,y0,yaw0] path=[[x,y,yaw],...]
        cpath=[]
        size=len(path)
        acc = size/4
        v=0.0
        theta_pre=0.0
        for i in range(size):
            x = path[i][0]
            y = path[i][1]
            if i<(size-1):
                x_ = path[i+1][0]
                y_ = path[i+1][1]
            else:
                x_ = path[-1][0]
                y_ = path[-1][1]
            theta = math.atan2(y_-y,x_-x)
            diff_angle=theta-theta_pre
            theta_pre=theta
            #angular=diff_angle*self.dt
            angular=diff_angle
            if i<acc:
                v+=self.Vmax/acc
            elif i>(size-1)-acc:
                v-=self.Vmax/acc
            if v>self.Vmax:
                v=self.Vmax
            if v<0.0:
                v=0.0
            cpath.append([x,y,theta,v,0.0,angular])
            #print(cpath)
        return np.array(cpath)

    def bezier_curve(self,points, nTimes=1000):
        """
        Given a set of control points, return the
        bezier curve defined by the control points.
        points should be a list of lists, or list of tuples
        such as [ [1,1], 
                    [2,3], 
                    [4,5], ..[Xn, Yn] ]
            nTimes is the number of time steps, defaults to 1000
            See http://processingjs.nihongoresources.com/bezierinfo/
        """
        from scipy.special import comb
        def bernstein_poly(i, n, t):
            """
            The Bernstein polynomial of n, i as a function of t
            """
            return comb(n, i) * ( t**(n-i) ) * (1 - t)**i
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        return xvals, yvals

    def bezier_path(self,path_,n=1000):
        #path=np.vstack((init,path))#初期位置をpathに追加
        xvals, yvals = self.bezier_curve(path_, nTimes=n)#ベジェ曲線で経路を滑らかにする
        cpath=np.flipud(np.array(list(map(list, zip(xvals,yvals)))))#xvalsとyvalsの結合と反転
        #結果表示
        plt.plot(path_[:,0],path_[:,1])
        plt.plot(cpath[:,0],cpath[:,1],color = "red")
        plt.show()
        return cpath

    def conversion_path(self,path):
        l=0.0
        for i in range(len(path)-1):
            p=path[i]
            p_next=path[i+1]
            print(p,p_next)
            l+=math.sqrt((p[0]-p_next[0])**2+(p[1]-p_next[1])**2)
        n=int(l/self.dt)
        print("l:",l,"ntimes:",n)
        bpath=self.bezier_path(path,n)
        cpath=self.calc_vel(bpath)
        return cpath
        

    def cost(self,inipos,iniv,path):#inipos=[x0,y0,yaw0] path=[[x,y,yaw,vx,vy,angular],...]
        J = 0 # コスト関数
        for k in range(self.N):
            Uk = MX.sym('U_' + str(k), self.nu) # 時間ステージ k の制御入力 uk を表す変数
            self.w   += [Uk]                    # uk を最適化変数 list に追加
            #制御入力の制約条件
            self.lbw += [-self.Vmax,0.0,-self.Angularmax]        # uk の lower-bound
            self.ubw += [self.Vmax,0.0,self.Angularmax]           # uk の upper-bound

            self.w0  += iniv                 # uk の初期推定解

            #運動方程式
            x   = self.Xk[0]    # X座標[m]
            y   = self.Xk[1]    # Y座標[m]
            yaw = self.Xk[2]    # ロボット角度[rad]
            #制御入力
            vx      = Uk[0]   # vx[m/s]
            #vy      = Uk[1]   # vy[m/s]
            angular = Uk[2]   # w[rad/s]
            # ステージコストのパラメータ
            self.k_max+=1
            x_ref = path[k][0:3]           # 目標状態
            u_ref = path[k][3:6]           # 目標状態
            L = 0 # ステージコスト
            for i in range(self.nx):
                L += 0.5 * self.Q[i] * (self.Xk[i]-x_ref[i])**2
            for i in range(self.nu):
                L += 0.5 * self.R[i] * (Uk[i]-u_ref[i])**2
            J = J + L * self.dt # コスト関数にステージコストを追加

            # Forward Euler による離散化状態方程式
            Xk_next = vertcat(x + vx*cos(yaw)*self.dt,
                            y + vx*sin(yaw)*self.dt,
                            yaw + angular*self.dt)
            Xk1 = MX.sym('X_' + str(k+1), self.nx)  # 時間ステージ k+1 の状態 xk+1 を表す変数
            self.w   += [Xk1]                       # xk+1 を最適化変数 list に追加
            self.lbw += [-inf, -inf, -inf]    # xk+1 の lower-bound （指定しない要素は -inf）
            self.ubw += [inf, inf, inf]       # xk+1 の upper-bound （指定しない要素は inf）
            self.w0  += inipos       # xk+1 の初期推定解

            self.g   += [Xk_next-Xk1] # 状態方程式(xk+1=xk+fk*dt) を等式制約として導入
            self.lbg += [0.0,0.0,0.0] # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
            self.ubg += [0.0,0.0,0.0] # 等式制約は lower-bo und と upper-bound を同じ値にすることで設定
            self.Xk = Xk1

        # 終端コストのパラメータ
        Vf = 0                            # 終端コスト
        for i in range(self.nx):
            Vf += 0.5 * self.Qf[i] * (self.Xk[i]-x_ref[i])**2
        for i in range(self.nu):
            Vf += 0.5 * self.Rf[i] * (Uk[i]-u_ref[i])**2
        J = J + Vf
        return J
    def solve(self,inipos,iniv,path):#inipos=[x0,y0,yaw0] path=[[x,y,yaw,vx,vy,angular],...]
        self.inipos=inipos
        self.path=path
        # 以下で非線形計画問題(NLP)を定式化
        self.w   = []  # 最適化変数を格納する list
        self.w0  = []  # 最適化変数(w)の初期推定解を格納する list
        self.lbw = []  # 最適化変数(w)の lower bound を格納する list
        self.ubw = []  # 最適化変数(w)の upper bound を格納する list
        self.g = []    # 制約（等式制約，不等式制約どちらも）を格納する list
        self.lbg = []  # 制約関数(g)の lower bound を格納する list
        self.ubg = []  # 制約関数(g)の upper bound を格納する list

        self.Xk = MX.sym('X0', self.nx) # 初期時刻の状態ベクトル x0

        self.w += [self.Xk]             # x0 を 最適化変数 list (w) に追加
        # 初期状態は given という条件を等式制約として考慮
        self.lbw += inipos # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
        self.ubw += inipos # 等式制約は lower-bound と upper-bound を同じ値にすることで設定
        self.w0  += inipos # x0 の初期推定解
        self.k_max=0
        #現在位置から一番近い経路の点取得
        d=(path[:,0]-inipos[0])**2+(path[:,1]-inipos[1])**2
        min_d=np.argmin(d)
        path_=list(path[min_d:,:])#経路の現在位置から一番近い経路の点からゴールまでを抜き出し
        #サイズNのリストのあまりをゴールの座標で埋める
        for i in range((self.N-len(path[min_d:,:]))):
            path_.append(path[-1,:])
        path_=np.array(path_)
        #評価関数式作成
        J=self.cost(inipos,iniv,path_)
        # 非線形計画問題(NLP)
        nlp = {'f': J, 'x': vertcat(*self.w), 'g': vertcat(*self.g)}
        # Ipopt ソルバー，最小バリアパラメータを0.001に設定
        solver = nlpsol('solver', 'ipopt', nlp)

        # NLPを解く
        sol = solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg)
        w_opt = sol['x'].full().flatten()
        print(self.k_max)
        tgrid = np.array([self.dt*k for k in range(self.k_max+1)])
        return w_opt, tgrid

    def goal_check(self,lim,alim):
        d=math.sqrt((self.path[-1,0]-self.inipos[0])**2+(self.path[-1,1]-self.inipos[1])**2)
        print(d,math.fabs(self.path[-1,2]-self.inipos[2]))
        if d<lim:
            if math.fabs(self.path[-1,2]-self.inipos[2])<alim:
                return True
        return False