#include <stdio.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <unistd.h>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

class model
{
    private:
        std::chrono::system_clock::time_point now, old;
        //std::vector<double> acc{0, 0, 0}; // Ax,Ay,angular_Vz
        inline double trapezoid(double a,double b,double h)
        {
            return ((a+b)*h)/2.0;
        }
    public:
        std::vector<double> pos{0, 0, 0}; // X,Y,Z
        std::vector<double> vel{0, 0, 0}; // Vx,Vy,angular_z
        model();
        double deltaT();
        void update(std::vector<double> acc,std::vector<double> vel0,std::vector<double> pos0,double dt);
};

model::model()
{
    this->now = std::chrono::system_clock::now();
    this->old = std::chrono::system_clock::now();
}
double model::deltaT()
{
    this->now = std::chrono::system_clock::now();
    double dt = std::chrono::duration_cast<std::chrono::seconds>(this->now - this->old).count();
    this->old = this->now;
    return dt;
}
void model::update(std::vector<double> acc, std::vector<double> vel0, std::vector<double> pos0, double dt)
{
    std::cout << dt<< "sec.\n";
    for(int i=0;i<3;i++)
    {
        this->vel[i] = this->trapezoid(vel0[i], acc[i]*dt, dt);
        this->pos[i] = this->trapezoid(pos0[i], vel[i]*dt, dt);
    }
    for (int i = 0; i < 3; i++)
    {
        std::cout <<"("<< acc[i] << "[m/s^2],";
        std::cout << vel[i] << "[m/s],";
        std::cout << pos[i] << "[m])\n";
    }
}

class MPC
{
    private:
        int horizon=10;
        model m;
    public:
        MPC();

};

int main()
{
    model m;
    std::vector<double> x{0};
    std::vector<double> y{0};
    int c = 0;
    while(1)
    {
        m.update({0.1,0,0},m.vel,m.pos,1);
        //sleep(1);
        x.push_back(m.pos[0]);
        y.push_back(m.pos[1]);
        c++;
        plt::plot(x,y,"b");
        plt::pause(0.1);
    }
    return 0;
}
