#include <iostream>
#include <chrono>
#include <vector>
#include <map>
#include <unistd.h>
#include "casadi/casadi.hpp"
using namespace casadi;
using namespace std;
#define Vmax 2.0

class MPC
{
    private:
        double T = 5.0;
        int N = 10;
        double dt = T / N;
        const int nx = 3;
        const int nu = 3;
        const vector<double> Q = {1.0, 1.0, 0.01}; //状態への重み
        const vector<double> R = {1.0, 1.0, 1.0};  //制御入力への重み

        vector<MX> w;        //最適化変数list
        vector<MX> w0;       //最適化変数wの初期推定解を格納
        vector<MX> lbw;      //最適化変数w lower bound
        vector<MX> ubw;      //最適化変数w upper bound
        MX J = MX::zeros(1); //コスト関数
        vector<MX> g;        //制約(等式制約,不等式制約)
        vector<MX> lbg;      //制約関数g lower bound
        vector<MX> ubg;      //制約関数g upper bound

        MX Xk = MX::sym("X0", nx); //初期時刻状態ベクトルx0
        vector<double> x_ref;
        MX Uk;
        int k_max = 0;
        
        MX solver(vector<double> inipos, vector<double> iniv, vector<vector<double>> path);
    public:
        MPC();
        std::map<std::string, DM> update(vector<double> inipos, vector<double> iniv, vector<vector<double>> path);
        vector<double> Tgrid();
};
MPC::MPC()
{
}
MX MPC::solver(vector<double> inipos, vector<double> iniv, vector<vector<double>> path)
{
    MX J = MX::zeros(1); //コスト関数
    for (int k = 0; k < this->N; k++)
    {
        this->Uk = MX::sym("U_" + to_string(k), this->nu);
        this->w.push_back(this->Uk);
        this->lbw.push_back(-Vmax);
        this->lbw.push_back(-Vmax);
        this->lbw.push_back(-Vmax);
        this->ubw.push_back(Vmax);
        this->ubw.push_back(Vmax);
        this->ubw.push_back(Vmax);
        this->w0.push_back(iniv[0]);
        this->w0.push_back(iniv[1]);
        this->w0.push_back(iniv[2]);
        //運動方程式
        MX x = this->Xk(0); // X座標[m]
        MX y = this->Xk(1); // Y座標[m]
        MX yaw = this->Xk(2); //ロボット角度[rad]
        //制御入力
        MX vx = this->Uk(0); // vx[m/s]
        MX vy = this->Uk(1); // vy[m/s]
        MX angular = this->Uk(2); // w[rad/s]
        this->k_max += 1;
        MX L = MX::zeros(1); //ステージコスト
        this->x_ref = path[k];
        cout << path[k];
        for (int i = 0; i < this->nx; i++)
            L += 0.5 * this->Q[i] * pow((this->Xk(i) - this->x_ref[i]), 2);
        for (int i = 0; i < this->nu; i++)
            L += 0.5 * this->R[i] * pow(this->Uk(i), 2);
        J = J + L * dt; //コスト関数にステージコストを追加
        // Forward Euler による離散化状態方程式
        MX Xk_next = vertcat(x + vx * dt, y + vy * dt, yaw + angular * dt);
        MX Xk1 = MX::sym("X_" + to_string(k + 1), this->nu); //時間ステージ k+1 の状態 xk+1 を表す変数
        this->w.push_back(Xk1);
        this->lbw.push_back(-inf);
        this->lbw.push_back(-inf);
        this->lbw.push_back(-inf);
        this->ubw.push_back(inf);
        this->ubw.push_back(inf);
        this->ubw.push_back(inf);
        this->w0.push_back(inipos[0]);
        this->w0.push_back(inipos[1]);
        this->w0.push_back(inipos[2]);
        //状態方程式(xk+1=xk+fk*dt) を等式制約として導入
        this->g.push_back(Xk_next - Xk1);
        this->lbg.push_back(0.0);
        this->lbg.push_back(0.0);
        this->lbg.push_back(0.0);
        this->ubg.push_back(0.0);
        this->ubg.push_back(0.0);
        this->ubg.push_back(0.0);
        this->Xk = Xk1;
    }
    MX Vf = MX::zeros(1); //終端コスト
    for (int i = 0; i < this->nx; i++)
        Vf += 0.5 * this->Q[i] * pow((this->Xk(i) - this->x_ref[i]), 2);
    for (int i = 0; i < this->nu; i++)
        Vf += 0.5 * this->R[i] * pow(this->Uk(i), 2);
    J = J + Vf;
    return J;
}
std::map<std::string, DM> MPC::update(vector<double> inipos, vector<double> iniv, vector<vector<double>> path)
{
    this->w.push_back(Xk);

    this->lbw.push_back(inipos[0]);
    this->lbw.push_back(inipos[1]);
    this->lbw.push_back(inipos[2]);

    this->ubw.push_back(inipos[0]);
    this->ubw.push_back(inipos[1]);
    this->ubw.push_back(inipos[2]);

    this->w0.push_back(0.0);
    this->w0.push_back(0.0);
    this->w0.push_back(0.0);
    MX J = solver(inipos, iniv,path);
    // NLP
    MXDict nlp = {{"f", J}, {"x", vertcat(this->w)}, {"g", vertcat(this->g)}};
    // Create NLP solver and buffers
    Function solver = nlpsol("solver", "ipopt", nlp);
    std::map<std::string, DM> arg, res;
    // Solve the NLP
    arg["x0"] = w0;
    arg["lbx"] = lbw;
    arg["ubx"] = ubw;
    arg["lbg"] = lbg;
    arg["ubg"] = ubg;
    res = solver(arg);
    return res;
}
vector<double> MPC::Tgrid()
{
    vector<double> tgrid;
    for (int k = 0; k < this->k_max + 1; k++)
        tgrid.push_back(this->dt * k);
    return tgrid;
}

vector<double> line_excerpt(vector<vector<double>> val, int line, int col = 0)
{
    vector<double> cut;
    if (col == 0)
        col = val.size();
    for (int i = 0; i < col; i++)
        cut.push_back(val[i][line]);
    return cut;
}

vector<double> slice(vector<double> val, int ini, int step)
{
    vector<double> cut;
    for (int i = ini; i < (int)val.size(); i += step)
        cut.push_back(val[i]);
    return cut;
}
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;
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

int main()
{
    MPC mpc;
    std::vector<double> inipos = {0.0, 1.0, 0.0};
    std::vector<double> iniv = {0.0, 0.0, 0.0};

    std::map<std::string, DM> res = mpc.update(inipos, iniv, path);
    auto x1_opt = slice((vector<double>)res.at("x"), 0, 6);
    auto x2_opt = slice((vector<double>)res.at("x"), 1, 6);
    auto x3_opt = slice((vector<double>)res.at("x"), 2, 6);
    auto u1_opt = slice((vector<double>)res.at("x"), 3, 6);
    auto u2_opt = slice((vector<double>)res.at("x"), 4, 6);
    auto u3_opt = slice((vector<double>)res.at("x"), 5, 6);
    // cout << x1_opt << endl;
    // cout << x2_opt << endl;
    // cout << x3_opt << endl;
    // cout << u1_opt << endl;
    // cout << u2_opt << endl;
    // cout << u3_opt << endl;
    vector<double> tgrid = mpc.Tgrid();
    plt::subplot(1, 2, 1);
    plt::grid(true);
    plt::plot(tgrid, x1_opt, "-b");
    plt::plot(tgrid, x2_opt, "-g");
    plt::plot(tgrid, x3_opt, "-r");

    plt::plot(tgrid, (vector<double>)vertcat(DM::nan(1), u1_opt), "-.");
    plt::plot(tgrid, (vector<double>)vertcat(DM::nan(1), u2_opt), "-.");
    plt::plot(tgrid, (vector<double>)vertcat(DM::nan(1), u3_opt), "-.");
    plt::subplot(1, 2, 2);
    plt::grid(true);
    plt::plot(line_excerpt(path, 0), line_excerpt(path, 1), "-r");
    plt::plot(x1_opt, x2_opt, ".g");
    plt::show();
    return 0;
}
