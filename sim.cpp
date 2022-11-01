#include <stdio.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <unistd.h>
#include <vector>
#include "matplotlibcpp.h"

using namespace casadi;
//目標状態（経路）

double Y = 0.0;
std::vector<std::vector<double>> path =
    {
        {1.0, Y, 0.0},
        {2.0, Y, 0.0},
        {3.0, Y, 0.0},
        {4.0, Y, 0.0},
        {5.0, Y, 0.0},
        {6.0, Y, 0.0},
        {7.0, Y, 0.0},
        {8.0, Y, 0.0},
        {9.0, Y, 0.0},
        {10.0, Y, 0.0},
        {11.0, Y, 0.0},
        {12.0, Y, 0.0},
        {13.0, Y, 0.0},
        {14.0, Y, 0.0},
        {15.0, Y, 0.0},
        {16.0, Y, 0.0},
        {17.0, Y, 0.0},
        {18.0, Y, 0.0},
        {19.0, Y, 0.0},
        {20.0, Y, 0.0},
        {21.0, Y, 0.0},
        {22.0, Y, 0.0},
        {23.0, Y, 0.0},
        {24.0, Y, 0.0},
        {25.0, Y, 0.0},
        {26.0, Y, 0.0},
        {27.0, Y, 0.0},
        {28.0, Y, 0.0},
        {29.0, Y, 0.0},
        {30.0, Y, 0.0},
        {31.0, Y, 0.0},
        {32.0, Y, 0.0},
        {33.0, Y, 0.0},
        {34.0, Y, 0.0},
        {35.0, Y, 0.0},
        {36.0, Y, 0.0},
        {37.0, Y, 0.0},
        {38.0, Y, 0.0},
        {39.0, Y, 0.0},
        {40.0, Y, 0.0},
        {41.0, Y, 0.0},
        {42.0, Y, 0.0},
        {43.0, Y, 0.0},
        {44.0, Y, 0.0},
        {45.0, Y, 0.0},
        {46.0, Y, 0.0},
        {47.0, Y, 0.0},
        {48.0, Y, 0.0},
        {49.0, Y, 0.0},
        {50.0, Y, 0.0},
        {49.0, Y, 0.0},
        {50.0, Y, 0.0},
        {51.0, Y, 0.0},
        {52.0, Y, 0.0},
        {53.0, Y, 0.0},
        {54.0, Y, 0.0},
        {55.0, Y, 0.0},
        {56.0, Y, 0.0},
        {57.0, Y, 0.0},
        {58.0, Y, 0.0},
        {59.0, Y, 0.0},
        {60.0, Y, 0.0},
        {71.0, Y, 0.0},
        {72.0, Y, 0.0},
        {73.0, Y, 0.0},
        {74.0, Y, 0.0},
        {75.0, Y, 0.0},
        {76.0, Y, 0.0},
        {77.0, Y, 0.0},
        {78.0, Y, 0.0},
        {79.0, Y, 0.0},
        {80.0, Y, 0.0},
        {81.0, Y, 0.0},
        {82.0, Y, 0.0},
        {83.0, Y, 0.0},
        {84.0, Y, 0.0},
        {85.0, Y, 0.0},
        {86.0, Y, 0.0},
        {87.0, Y, 0.0},
        {88.0, Y, 0.0},
        {89.0, Y, 0.0},
        {90.0, Y, 0.0},
        {91.0, Y, 0.0},
        {92.0, Y, 0.0},
        {93.0, Y, 0.0},
        {94.0, Y, 0.0},
        {95.0, Y, 0.0},
        {96.0, Y, 0.0},
        {97.0, Y, 0.0},
        {98.0, Y, 0.0},
        {99.0, Y, 0.0},
        {100.0, Y, 0.0},
        {101.0, Y, 0.0},
        {102.0, Y, 0.0}};
std::vector<double> inipos = {0.0, 1.0, 0.0};


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
        void update(std::vector<double> acc, std::vector<double> vel0, std::vector<double> pos0, double dt);
        void update(std::vector<double> vel, std::vector<double> pos0, double dt);
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
        this->pos[i] = this->trapezoid(pos0[i], this->vel[i] * dt, dt);
    }
    for (int i = 0; i < 3; i++)
    {
        std::cout <<"("<< acc[i] << "[m/s^2],";
        std::cout << vel[i] << "[m/s],";
        std::cout << pos[i] << "[m])\n";
    }
}
void model::update(std::vector<double> vel, std::vector<double> pos0, double dt)
{
    std::cout << dt << "sec.\n";
    for (int i = 0; i < 3; i++)
    {
        this->vel[i] = vel[i];
        this->pos[i] = this->trapezoid(pos0[i], this->vel[i] * dt, dt);
    }
    for (int i = 0; i < 3; i++)
    {
        std::cout << "(" << acc[i] << "[m/s^2],";
        std::cout << this->vel[i] << "[m/s],";
        std::cout << this->pos[i] << "[m])\n";
    }
}

int main()
{
    model m;
    std::vector<double> x{0};
    std::vector<double> y{0};
    int c = 0;
    while(1)
    {
        m.update({0.1, 0, 0}, m.vel, m.pos, 1);
        //sleep(1);
        x.push_back(m.pos[0]);
        y.push_back(m.pos[1]);
        c++;
        plt::plot(x,y,"b");
        plt::pause(0.1);
    }
    return 0;
}
