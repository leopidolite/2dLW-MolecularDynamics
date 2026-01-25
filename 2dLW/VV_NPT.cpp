#include <iostream>
#include <fstream>
#include <vector> 
#include <string>
#include <list>
#include <cmath>
#include <tuple> 
#include <chrono>
#include <omp.h>
#include <iomanip>

using namespace std; 

//// 2d Lewis-Wanhstrom: 3 atoms/molecule, indexed i, i + N, i + 2*N
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double rho_star = 0.25;
int N = 300;
double TARGET_TEMP = 3.0; 
double TARGET_PRESSURE = 3.5; 
int M = 5; 

string prefix = "300_NPT_cool_3.5"; 
string read_prefix = "/users/lli190/scratch/MD_full_trajectories/300_NPT_3T";               
string folderpath = "/users/lli190/scratch/MD_full_trajectories/"; 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//define global variables 
double m = 1.0;  
double epsilon_kB = 198.8;  // epsilon / kB
double sigma = 1.0 ; //reduced; real value = 0.341 
double epsilon = 1.0; // reduced energy
double L = sqrt(N/rho_star); // simulation box side length in \rho^*
double L2 = L/2; 
double r_c = 2.5; // cutoff distance
double r_l = 3; // skin radius 
double dt = 0.001; // timestep
//linked list parameters
int s_c =  static_cast<int>(floor(L/r_c)) -1; //  s_c such that l = L/s_c is slightly larger than r_c
double l = L/s_c; // side length of each cell
vector<vector<list<int>>> cells(s_c, vector<list<int>>(s_c)); // s_c x s_c grid
//molecule parameters 
double apical = 5*M_PI/12; //75 degree apical angle
double d_23 = 2*sin(0.5*apical); // d_23 imaginary bond length

// Thermostat/Barostat Parameters
double v_eps = 0.0; // barostat velocity
double W_eps; // Barostat mass
double tau_baro = dt * 1000.0; // barostat characteristic time
double physical_virial = 0.0; 
double constraint_virial = 0.0;

double tau_NHC = dt*500; //thermostat characteristic time


// Create Output Files 
std::ofstream kineticE(folderpath + prefix + "_KE.txt");
std::ofstream potentialE(folderpath + prefix + "_PE.txt");
std::ofstream vol_traj(folderpath + prefix + "_volume.txt"); 
std::ofstream press_traj(folderpath + prefix + "_pressure.txt");
std::ofstream trajectory(folderpath + prefix + "_trajectory.bin", std::ios::out | std::ios::binary);

// output files for adjusted temperature: 
std::ofstream x_init(folderpath + prefix + "_x_initial.txt");
std::ofstream y_init(folderpath + prefix + "_y_initial.txt"); 
std::ofstream vx_init(folderpath + prefix + "_vx_initial.txt"); 
std::ofstream vy_init(folderpath + prefix + "_vy_initial.txt");
std::ofstream vol_init(folderpath + prefix + "_vol_initial.txt");

//global vectors of coordinates
std::vector<double> x;
std::vector<double> y;
std::vector<double> vx;
std::vector<double> vy;
std::vector<double> xprev(3*N);
std::vector<double> yprev(3*N);
std::vector<double> vx_12(3*N);
std::vector<double> vy_12(3*N);
 
// global vectors of Nosé-Hoover Chain thermostat
std::vector<double> eta(M);
std::vector<double> p_eta(M);
std::vector<double> G(M); 
std::vector<double> Q(M); 

// global vectors of Nosé-Hoover Chain barostat
std::vector<double> eta_baro(M);
std::vector<double> p_eta_baro(M);
std::vector<double> G_baro(M); 
std::vector<double> Q_baro(M);

std::vector<double> weights =  {  //Suzuki-Yoshida weights, 7th order
    0.784513610477560,
    0.235573213359357,
    -1.17767998417887,
    1-(2*0.784513610477560 + 2*0.235573213359357 + 2*(-1.17767998417887)), 
    -1.17767998417887,
    0.235573213359357,
    0.784513610477560
};

//FUNCTION - store values 
inline void store_initial(){
    for (int i=0; i<3*N; i++){
        x_init << x[i] << "\n";
        y_init<< y[i] << "\n";
        vx_init << vx[i] << "\n";
        vy_init << vy[i] << "\n";
         
    }
    vol_init << L*L<< "\n";  // store final volume
    cout << "Initial state stored" << "\n";
}

inline void store_binary() {
    trajectory.write(reinterpret_cast<char*>(x.data()), sizeof(double) * x.size());
    trajectory.write(reinterpret_cast<char*>(y.data()), sizeof(double) * y.size());
    trajectory.write(reinterpret_cast<char*>(&L), sizeof(double));
}

//FUNCTION - read from file into vector<double> 
std::vector<double> read(string path) {
    std::ifstream file(path);
    std::vector<double> read_vector;
    double line;
    while (file >> line) {
                read_vector.push_back(line);
            }
    return read_vector;
}

void init_L(){
    std::string filename = read_prefix + "_vol_init.txt";
    std::ifstream file(filename);
    double vol_in = 0.0;
    if (file >> vol_in) {
        L = std::sqrt(vol_in);
    } 
}

//FUNCTION - periodic boundary conditions
inline double periodic(double dr) {
    if (dr > L2) dr -= L;
    if (dr < -L2) dr += L;
    return dr;
}

//FUNCTION - Make Cell Lists
void make_cell_lists(){
    // for(int i=0; i<s_c; i++) for(int j=0; j<s_c; j++) cells[i][j].clear();
    for (int i = 0; i < 3*N; i++){
        int w = floor((x[i] + L2) / l);
        int h = floor((y[i] + L2) / l); 
        w = (w % s_c + s_c) % s_c;
        h = (h % s_c + s_c) % s_c;
        cells[h][w].push_back(i);
    }
}


void update_lists() {

    for (int h = 0; h < s_c; h++) {
        for (int w = 0; w < s_c; w++) {
            auto& currentlist = cells[h][w];
            for (auto it = currentlist.begin(); it != currentlist.end(); ) {
                int i = *it;
                int newh = floor((y[i] + L2) / l);
                int neww = floor((x[i] + L2) / l);
                // wrap indices
                newh = (newh % s_c + s_c) % s_c;
                neww = (neww % s_c + s_c) % s_c;

                if (neww != w || newh != h) {
                    cells[newh][neww].push_back(i);
                    it = currentlist.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }
}

//update lists method accounting for rescaling of volume
void update_lists_NPT() {
    int new_s_c = static_cast<int>(floor(L / r_c)) - 1;
    if (new_s_c != s_c) {
        s_c = new_s_c;
        l = L / s_c;
        cells.assign(s_c, std::vector<std::list<int>>(s_c));
        make_cell_lists(); 
    } else {
        l = L / s_c;
        update_lists();
    }
}

// force computation in half adjacent cells
const vector<pair<int, int>> neighbor_offsets = { // add these to h,w 
    {0, 0},   // current cell
    {0, 1},   // right cell
    {1, 1},  // top-right cell
    {1, 0},   // top cell 
    {-1, 1}    // bottom-right cell 
};

//FUNCTION - linked list force computation
std::tuple<vector<double>, vector<double>> compute_forces_LL() {
    std::vector<double> f_x(3*N, 0.0);
    std::vector<double> f_y(3*N, 0.0);
    double local_virial = 0.0; 

    int h_current;
    int w_current;
    #pragma omp parallel for collapse(2) private(h_current, w_current) reduction(+:local_virial)
    for(int h=0; h<s_c; h++){
        for(int w= 0; w<s_c; w++){
            for(int i : cells[h][w]){
                for (const pair<int,int>& offset : neighbor_offsets){
                    h_current = (h+ offset.first+s_c)%s_c;
                    w_current = (w+ offset.second+s_c)%s_c;
                    for (int j: cells[h_current][w_current]){
                        if (j == i || j%N == i%N) continue; 
                        
                        double r_ijx = periodic(x[i]-x[j]);
                        double r_ijy = periodic(y[i]-y[j]);
                        double r2 = r_ijx*r_ijx + r_ijy*r_ijy;
                        
                        if (r2 < r_c *r_c){
                             double r2inv = 1/r2;
                             double r6 = r2inv*r2inv*r2inv;
                             double r12 = r6*r6;
                             double f_i = 24*epsilon*r2inv*(2*(r12) - (r6));
                             double pair_virial = f_i * r2; 

                            if(h_current == h && w_current == w){
                                #pragma omp atomic
                                f_x[i] += r_ijx * f_i;
                                #pragma omp atomic
                                f_y[i] += r_ijy * f_i;
                                local_virial += pair_virial;
                            }
                            else{
                                #pragma omp atomic
                                f_x[i] += r_ijx * f_i;
                                #pragma omp atomic
                                f_y[i] += r_ijy * f_i;
                                #pragma omp atomic
                                f_x[j] -= r_ijx * f_i;
                                #pragma omp atomic
                                f_y[j] -= r_ijy * f_i;
                                local_virial += pair_virial;
                            }
                        }
                    }
                }
            }
        }
    }
    physical_virial = local_virial;
    return make_tuple(f_x, f_y);
}

// Compute energies with linked list method
void compute_energies_LL(){
    double KE = 0;
    double PE = 0;

    for(int h=0; h<s_c; h++){
        for(int w= 0; w<s_c; w++){
            for(int i : cells[h][w]){
                KE += 0.5*(vx[i]*vx[i] +  vy[i]*vy[i]); 
                for (const pair<int,int>& offset : neighbor_offsets){
                    int h_current = (h+ offset.first+s_c)%s_c;
                    int w_current = (w+ offset.second+s_c)%s_c;
                    for (int j: cells[h_current][w_current]){
                        if (i==j|| i%N == j%N) continue;
                        double r_ijx = periodic(x[i]-x[j]);
                        double r_ijy = periodic(y[i]-y[j]);
                        double r2 = r_ijx*r_ijx + r_ijy*r_ijy;
                        if (r2 <= r_c *r_c){
                             double r2inv = 1/r2;
                             double r6 = r2inv*r2inv*r2inv;
                             double r12 = r6*r6;
                            if(h_current == h && w_current == w){
                                if(i<j) PE += 4*((r12) - (r6)); // Only count once
                            }
                            else{
                                PE += 4*((r12) - (r6));
                            }
                        }
                    }
                }
            }
        }
    }
    kineticE << KE << "\n";
    potentialE << PE << "\n";
}

void r_constraints(){ 
    double local_w_constr = 0.0;

    #pragma omp parallel for reduction(+:local_w_constr)
    for (int i =0; i<N; i++){
        double r_12, r_13, r_23;
        double g12, g13, g23;
        double dx_12, dy_12, dx_13, dy_13, dx_23, dy_23, dx, dy;
        double virial_term = 0.0;

        dx_12 = periodic(x[i] - x[N+i]);
        dy_12 = periodic(y[i] - y[N+i]);
        r_12= dx_12*dx_12 + dy_12*dy_12;
        dx_13 = periodic(x[i]-x[i+2*N]);  
        dy_13 = periodic(y[i]-y[i+2*N]);
        r_13 = dx_13*dx_13 + dy_13*dy_13;
        dx_23 = periodic(x[N+i]-x[i+2*N]);
        dy_23 = periodic(y[N+i]-y[i+2*N]);
        r_23 = dx_23*dx_23 + dy_23*dy_23;
        
        int iter = 0; 
        while((fabs(r_12-1.0)>10e-7 || fabs(r_13-1.0)>10e-7 || fabs(r_23-d_23*d_23)>10e-7) && iter < 500){
            iter++;
            g12 = (r_12 - 1) / ((4*(dx_12)*periodic(xprev[i]-xprev[N+i]) + 4*(dy_12)*periodic(yprev[i]-yprev[N+i]))*dt);
            g13 = (r_13 - 1) / ((4*(dx_13)*periodic(xprev[i]-xprev[2*N+i]) + 4*(dy_13)*periodic(yprev[i]-yprev[2*N+i]))*dt);
            g23 = (r_23 - d_23*d_23) / ((4*(dx_23)*periodic(xprev[N+i]-xprev[2*N+i]) + 4*(dy_23)*periodic(yprev[N+i]-yprev[2*N+i]))*dt);
            
            virial_term += (g12 * 1.0 + g13 * 1.0 + g23 * (d_23*d_23));

            x[i] = periodic(x[i] - (g12*periodic(xprev[i]-xprev[i+N])+g13*periodic(xprev[i]-xprev[i+2*N]))*dt);
            y[i] = periodic(y[i]- (g12*periodic(yprev[i]-yprev[i+N])+g13*periodic(yprev[i]-yprev[i+2*N]))*dt);
            x[i+N] =periodic(x[i+N] +  (g12*periodic(xprev[i]-xprev[i+N])-g23*periodic(xprev[N+i]-xprev[i+2*N]))*dt);
            y[i+N] = periodic(y[i+N] +(g12*periodic(yprev[i]-yprev[i+N])-g23*periodic(yprev[N+i]-yprev[i+2*N]))*dt);
            x[i+2*N] =periodic(x[i+2*N] + (g13*periodic(xprev[i]-xprev[i+2*N])+g23*periodic(xprev[N+i]-xprev[i+2*N]))*dt);
            y[i+2*N] = periodic(y[i+2*N] +(g13*periodic(yprev[i]-yprev[i+2*N])+g23*periodic(yprev[N+i]-yprev[i+2*N]))*dt);
            
            dx = periodic(x[i] - x[N+i]);
            dy = periodic(y[i] - y[N+i]);
            r_12 = dx*dx + dy*dy;
            dx = periodic(x[i]-x[i+2*N]);  
            dy = periodic(y[i]-y[i+2*N]);
            r_13 = dx*dx + dy*dy;
            dx = periodic(x[N+i]-x[i+2*N]);
            dy = periodic(y[N+i]-y[i+2*N]);
            r_23 = dx*dx + dy*dy;
        }
        local_w_constr += virial_term / dt;
    }
    constraint_virial = local_w_constr;
}

//ROLL-safe velocity constraints 
double k_12, k_13, k_23;
double a, b, c, d, e, f, det, vr_12, vr_13, vr_23;
double x12,x13,x23,y12,y13,y23;
void vel_constraints_analytical(double v_eps_current){
    for (int i = 0; i<N; i++){
        x12 = periodic(x[i]-x[i+N]);  y12 = periodic(y[i]-y[i+N]);
        x13 = periodic(x[i]-x[i+2*N]);  y13 = periodic(y[i]-y[i+2*N]);
        x23 = periodic(x[i+N]-x[i+2*N]);  y23 = periodic(y[i+N]-y[i+2*N]);

        a = x12*x12 + y12*y12; 
        b = x13*x13 + y13*y13; 
        c = x23*x23 + y23*y23; 

        d = x12*x13 + y12*y13; 
        e = x12*x23 + y12*y23; 
        f = x13*x23 + y13*y23; 

        double vrel_12 = (vx[i]-vx[i+N])*x12 + (vy[i]-vy[i+N])*y12;
        double vrel_13 = (vx[i]-vx[i+2*N])*x13 + (vy[i]-vy[i+2*N])*y13;
        double vrel_23 = (vx[i+N]-vx[i+2*N])*x23 + (vy[i+N]-vy[i+2*N])*y23;

        vr_12 = -1.0 * (vrel_12 + v_eps_current * a);
        vr_13 = -1.0 * (vrel_13 + v_eps_current * b);
        vr_23 = -1.0 * (vrel_23 + v_eps_current * c);

        // The determinant and matrix inversion logic below is CORRECT
        det = 8*a*b*c - 2*a*f*f - 2*c*d*d - 2*d*e*f - 2*b*e*e;        
      
        k_12 = (vr_12 * (4*b*c - f*f) + vr_13 * (-2*c*d - e*f) + vr_23 * (d*f + 2*e*b))*(1/det)*0.5;
        k_13 = (vr_12 * (-2*d*c - f*e) + vr_13 * (4*a*c - e*e) + vr_23 * (-2*a*f - e*d))*(1/det)*0.5;
        k_23 = (vr_12 * (d*f + 2*b*e) + vr_13 * (-2*a*f - d*e) + vr_23 * (4*a*b - d*d))*(1/det)*0.5;

        // Update velocities
        vx[i] += k_12*x12 + k_13*x13;
        vy[i] += k_12*y12 + k_13*y13;

        vx[i+N] += -k_12*x12 + k_23*x23;
        vy[i+N] += -k_12*y12 + k_23*y23;
    
        vx[i+2*N] += -k_23*x23 - k_13*x13;
        vy[i+2*N] += -k_23*y23 - k_13*y13;
    }
}

//// Nosé-Hoover Thermostat

double Nf = 3*N-2; 

void init_NHC(){
    //Thermostat
    Q[0] = Nf * TARGET_TEMP*tau_NHC*tau_NHC;
    for (int i = 1; i<M; i++){
        Q[i] = TARGET_TEMP*tau_NHC*tau_NHC;
    }
    // Barostat Thermostat (1 DOF)
    W_eps = (Nf + 1) * TARGET_TEMP * tau_baro * tau_baro; //mass of barostat
    Q_baro[0] = 1.0 * TARGET_TEMP * tau_NHC * tau_NHC;
    for (int i = 1; i<M; i++){
        Q_baro[i] = TARGET_TEMP * tau_NHC * tau_NHC;
    }
}
double kinetic_(){
    double KE = 0;
    for (int i =0; i<3*N; i++){
        KE += (vx[i]*vx[i] + vy[i]*vy[i])/2;
    }
    return KE;
}
void update_G(){ 
    G[0] = 2*kinetic_() - Nf*TARGET_TEMP;
    for(int j=1;j<M;j++){
        G[j] = p_eta[j-1]*p_eta[j-1]/Q[j-1]-TARGET_TEMP;
    }
}
void update_G_baro(){ 
    G_baro[0] = W_eps * v_eps * v_eps - TARGET_TEMP; 
    for(int j=1; j<M; j++){
        G_baro[j] = p_eta_baro[j-1]*p_eta_baro[j-1]/Q_baro[j-1] - TARGET_TEMP;
    }
}

void NHC_propagator(double step){
    update_G();
    p_eta[M-1] = p_eta[M-1] + step/4*G[M-1];
    for (int j=M-2;j>=0; j--){
        p_eta[j] = p_eta[j] * exp( (-step/8.0) * p_eta[j+1]/Q[j+1] );
        p_eta[j] = p_eta[j] + step/4 * G[j];
        p_eta[j] = p_eta[j] * exp( (-step/8.0) * p_eta[j+1]/Q[j+1] );
    }
    #pragma omp parallel for
    for (int i = 0; i<3*N; i++){
        vx[i] = vx[i]*exp(-step/2*p_eta[0] / Q[0]);
        vy[i] = vy[i]*exp(-step/2*p_eta[0] / Q[0]);
    }
    for (int j = 0; j<M; j++){
        eta[j] = eta[j] + (step/2) * p_eta[j]/Q[j];
    }
    update_G();
    for (int j=0;j<M-1; j++){
        p_eta[j] = p_eta[j] * exp( (-step/8.0) * p_eta[j+1]/Q[j+1] );
        p_eta[j] = p_eta[j] + step/4 * G[j];
        p_eta[j] = p_eta[j] * exp( (-step/8.0) * p_eta[j+1]/Q[j+1] );
    }
    p_eta[M-1] = p_eta[M-1] + step/4*G[M-1];  
}

// Barostat Propagator 
void NHC_baro_propagator(double step){
    update_G_baro();
    p_eta_baro[M-1] += step/4 * G_baro[M-1];
    for (int j=M-2; j>=0; j--){
        p_eta_baro[j] *= exp( -step/8.0 * p_eta_baro[j+1]/Q_baro[j+1] );
        p_eta_baro[j] += step/4 * G_baro[j];
        p_eta_baro[j] *= exp( -step/8.0 * p_eta_baro[j+1]/Q_baro[j+1] );
    }
    v_eps *= exp(-step/2 * p_eta_baro[0] / Q_baro[0]);
    for (int j=0; j<M; j++) eta_baro[j] += step/2 * p_eta_baro[j]/Q_baro[j];
    update_G_baro();
    for (int j=0; j<M-1; j++){
        p_eta_baro[j] *= exp( -step/8.0 * p_eta_baro[j+1]/Q_baro[j+1] );
        p_eta_baro[j] += step/4 * G_baro[j];
        p_eta_baro[j] *= exp( -step/8.0 * p_eta_baro[j+1]/Q_baro[j+1] );
    }
    p_eta_baro[M-1] += step/4 * G_baro[M-1];
}

void nose_hoover(double tstep){
    for(int i = 0; i<7; i++){
        NHC_propagator(weights[i] * tstep);
    }
}

void nose_hoover_baro(double tstep){
    for(int i = 0; i<7; i++){
        NHC_baro_propagator(weights[i] * tstep);
    }
}



//FUNCTION - NPT ROLL Integrator, Yu et al. 2010
void npt_roll(int tsteps, bool store){

    double Vol = L*L;
    
    for (int t=0; t<tsteps; t++){
        
        update_lists_NPT();
        //half step thermo,baro-stats
        nose_hoover(dt/2.0);
        nose_hoover_baro(dt/2.0);

        //barstat half step
        double current_press = (2.0*kinetic_() + physical_virial + constraint_virial) / (2.0 * Vol);
        double F_eps = 2.0 * Vol * (current_press - TARGET_PRESSURE) + (2.0/3.0)*TARGET_TEMP;
        v_eps += (dt/2.0) * F_eps / W_eps;

        //scaling variables
        double alpha = v_eps; 
        double Delta = exp(-alpha * dt / 2.0);
        double sinh_term = (fabs(alpha*dt/4.0) < 1e-6) ? 1.0 : sinh(alpha*dt/4.0)/(alpha*dt/4.0);
        double Tilde_Delta = exp(-alpha*dt/4.0) * sinh_term;
        double D_pos = exp(v_eps * dt);
        double sinh_pos = (fabs(v_eps*dt/2.0) < 1e-6) ? 1.0 : sinh(v_eps*dt/2.0)/(v_eps*dt/2.0);
        double Tilde_D_pos = exp(v_eps*dt/2.0) * sinh_pos;

        // VV velocity half step
        auto forces1 = compute_forces_LL();
        std::vector<double>& a_x12 = get<0>(forces1);
        std::vector<double>& a_y12 = get<1>(forces1);

        #pragma omp parallel for
        for (int i=0; i<3*N; i++){
            vx_12[i] = vx[i]*Delta + a_x12[i]*dt*0.5*Tilde_Delta;
            vy_12[i] = vy[i]*Delta + a_y12[i]*dt*0.5*Tilde_Delta;
            
            xprev[i] = x[i]; 
            yprev[i] = y[i];
            x[i] = periodic(x[i]*D_pos + vx_12[i]*dt*Tilde_D_pos);
            y[i] = periodic(y[i]*D_pos + vy_12[i]*dt*Tilde_D_pos);
        }

        Vol *= D_pos * D_pos; 
        L = sqrt(Vol);
        L2 = L/2;

        r_constraints(); 
        #pragma omp parallel for
        for (int i = 0; i<3*N; i++){
            vx[i] = (periodic(x[i] - xprev[i]*D_pos)) / (dt * Tilde_D_pos); 
            vy[i] = (periodic(y[i] - yprev[i]*D_pos)) / (dt * Tilde_D_pos);
        }
        auto forces2 = compute_forces_LL();
        std::vector<double>& a_x = get<0>(forces2);
        std::vector<double>& a_y = get<1>(forces2);

        #pragma omp parallel for
        for (int i = 0; i<3*N; i++){
            vx[i] = vx[i]*Delta + 0.5*a_x[i]*dt*Tilde_Delta;
            vy[i] = vy[i]*Delta + 0.5*a_y[i]*dt*Tilde_Delta;
        }
        vel_constraints_analytical(v_eps);
        double new_press = (2.0*kinetic_() + physical_virial + constraint_virial) / (2.0 * Vol);
        F_eps = 2.0 * Vol * (new_press - TARGET_PRESSURE) + (2.0/3.0)*TARGET_TEMP;
        v_eps += (dt/2.0) * F_eps / W_eps;

        // thermo,baro-stats half step
        nose_hoover_baro(dt/2.0);
        nose_hoover(dt/2.0);

        if(t%10 == 0){
             compute_energies_LL();
             if (store == true){
                store_binary();
             }
             press_traj << current_press << "\n" << endl;
             vol_traj << Vol << endl;
        }
    }
}

void adjust_temp(double t_temp, double rate_tau){
    while (TARGET_TEMP > t_temp){
        TARGET_TEMP = TARGET_TEMP-rate_tau;
        npt_roll((tau_NHC*2)/dt, false);
    }
}


//MAIN FUNCTION 
int main(){

    //// Read from initial 
    x = read(read_prefix+"_x_initial.txt");
    y = read(read_prefix+"_y_initial.txt");
    vx = read(read_prefix+"_vx_initial.txt");
    vy = read(read_prefix+"_vy_initial.txt");  
    cout<< "Initial Values read \n" << flush;

    init_L(); //TURN OFF when initializing from non-NPT state 
    make_cell_lists();

    //// SIMULATION
    cout << prefix << " started..." << endl; 
    init_NHC();
    adjust_temp(1.0, 2.0/10000);
    npt_roll(10/dt, true); 

    store_initial();
    cout << prefix << " done." << endl; 

    kineticE.close();
    potentialE.close();
    vol_traj.close(); 

    trajectory.close();

    x_init.close();
    y_init.close();
    vx_init.close();
    vy_init.close();
    vol_init.close();

    return  0;
}
