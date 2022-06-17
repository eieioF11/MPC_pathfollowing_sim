from casadi import *
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import time
from mpc import *
import time

# 目標状態(経路)
path = []
for i in range(50):
    path.append([i,50*cos(i*0.1),math.pi/4])
for i in range(50,100):
    path.append([i,50*cos(i*0.1),math.pi/2])
for i in range(100,110):
    path.append([i,50*cos(i*0.1),math.pi])
path=np.array(path)
#初期位置
inipos=[0.0,100.0,0.0]

class model(object):
    def __init__(self):
        self.x=0
        self.y=0
        self.yaw=0
        self.vx=0
        self.vy=0
        self.w=0
    def update(self,vx,vy,w,dt):
        self.x+=vx*dt
        self.y+=vy*dt
        self.yaw+=w*dt
        self.vx=vx
        self.vy=vy
        self.w=w

def drawRobo(ax,x,y,yaw):
    size = 10.0
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
def runMPC(robo,mpc,x,y,t):
    w_opt, tgrid = mpc.solve([robo.x,robo.y,robo.yaw],[robo.vx,robo.vy,robo.w],path)
    #w_opt, tgrid = mpc.Solve([robo.x,robo.y,robo.yaw],[robo.vx,robo.vy,robo.w],path,dt)
    # 解をプロット
    x1_opt = np.array(w_opt[0::6])
    x2_opt = np.array(w_opt[1::6])
    x3_opt = np.array(w_opt[2::6])
    u1_opt = np.array(w_opt[3::6])
    u2_opt = np.array(w_opt[4::6])
    u3_opt = np.array(w_opt[5::6])
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.grid()
    plt.step(tgrid, np.array(vertcat(DM.nan(1), u1_opt)), '-.')
    plt.step(tgrid, np.array(vertcat(DM.nan(1), u2_opt)), '-.')
    plt.step(tgrid, np.array(vertcat(DM.nan(1), u3_opt)), '-.')
    plt.xlabel('t')
    plt.legend(['vx','vy','angular'])
    ax=plt.subplot(1, 2, 2)
    ax.set_aspect('equal')
    plt.grid()
    plt.plot(path[:,0],path[:,1],c="b")
    plt.plot(x1_opt, x2_opt, '.',c="g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x,y,c="r")
    plt.legend(['Path','Horizon','Robot Path'])
    body=drawRobo(ax,robo.x,robo.y,robo.yaw)
    robo.update(u1_opt[0],u2_opt[0],u3_opt[0],mpc.dt)
    x.append(robo.x)
    y.append(robo.y)
    if t==-1:
        plt.show()
    else:
        plt.pause(t)
    body.remove()

def main():
    mpc = MPC()
    robo=model()
    robo.x=inipos[0]
    robo.y=inipos[1]
    robo.yaw=inipos[2]
    x=[robo.x]
    y=[robo.y]
    runMPC(robo,mpc,x,y,5)
    while True:
        d=math.sqrt((path[-1,0]-robo.x)**2+(path[-1,1]-robo.y)**2)
        runMPC(robo,mpc,x,y,0.01)
        if d<=0.01:
            runMPC(robo,mpc,x,y,-1)
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
