from casadi import *
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from MPC.mpc import *
import time

# 目標状態(経路)
path = []
def f(x):
    return 2*sin(x*0.1)
for i in np.arange(0,50):
    path.append([i*0.05,f(i),0.0])
path=np.array(path)
#初期位置
inipos=[0.0,0.0,0.0]

class model(object):
    def __init__(self):
        self.x=0
        self.y=0
        self.yaw=0
        self.vx=0
        self.vy=0
        self.w=0
    def update(self,vx,vy,w,dt):
        self.x+=vx*cos(self.yaw)*dt
        self.y+=vx*sin(self.yaw)*dt
        self.yaw+=w*dt
        self.vx=vx
        self.vy=vy
        self.w=w

def drawRobo(ax,x,y,yaw):
    size = 1.0
    #描画更新
    def relative_rectangle(w: float, h: float, center_tf, **kwargs):
        rect_origin_to_center = Affine2D().translate(w / 2, h / 2)
        return Rectangle(
            (0, 0), w, h, transform=rect_origin_to_center.inverted() + center_tf, **kwargs)
    #body
    to_body_center_tf = Affine2D().rotate(yaw).translate(x,y) + ax.transData
    body = relative_rectangle(size,size, to_body_center_tf, edgecolor='black',fill=False)
    ax.add_patch(body)
    return body

def main():
    mpc = MPC(2.0,0.01,1.0,2.0)
    robo=model()
    robo.x=inipos[0]
    robo.y=inipos[1]
    robo.yaw=inipos[2]
    x=[robo.x]
    y=[robo.y]
    t_path=mpc.conversion_path(path)
    #t=np.arange(0,len(t_path))
    #plt.plot(t,t_path[:,0],c="r")
    #plt.plot(t,t_path[:,1],c="g")
    #plt.plot(t,t_path[:,2],c="b")
    #plt.plot(t,t_path[:,3],c="m")
    #plt.plot(t,t_path[:,5],c="y")
    #plt.grid()
    #plt.show()
    w_opt, tgrid = mpc.solve([robo.x,robo.y,robo.yaw],[robo.vx,robo.vy,robo.w],t_path)
    while True:
        if mpc.goal_check(0.1,0.1):#収束判定
            break
        #最適化処理
        w_opt, tgrid = mpc.solve([robo.x,robo.y,robo.yaw],[robo.vx,robo.vy,robo.w],t_path)
        # 解をプロット
        x1_opt = np.array(w_opt[0::6])#x
        x2_opt = np.array(w_opt[1::6])#y
        x3_opt = np.array(w_opt[2::6])#yaw
        u1_opt = np.array(w_opt[3::6])#vx
        u2_opt = np.array(w_opt[4::6])#vy 対向２輪なので常に0
        u3_opt = np.array(w_opt[5::6])#angular
        #グラフ描画
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.grid()
        plt.step(tgrid, np.array(vertcat(DM.nan(1), u1_opt)), '-.')
        plt.step(tgrid, np.array(vertcat(DM.nan(1), u3_opt)), '-.')
        plt.xlabel('t')
        plt.legend(['vx','angular'])
        ax=plt.subplot(1, 2, 2)
        ax.set_aspect('equal')
        plt.xlim(-1,9)
        plt.ylim(-5,5)
        plt.grid()
        plt.plot(path[:,0],path[:,1],c="b")
        plt.plot(x1_opt, x2_opt, '.',c="g")
        #モデル更新
        robo.update(u1_opt[1],u2_opt[1],u3_opt[1],mpc.dt)
        x.append(robo.x)
        y.append(robo.y)
        #ロボット軌道表示
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(x,y,c="r")
        #ロボット表示
        body=drawRobo(ax,robo.x,robo.y,robo.yaw)
        plt.pause(0.01)
        body.remove()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
