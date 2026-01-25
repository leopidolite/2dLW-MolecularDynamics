#include <iostream>
#include <fstream>
#include <vector> 
#include <string>
#include <list>
#include <cmath>
#include <tuple> 
#include <chrono>
#include <omp.h>

using namespace std; 

//// 2d Lewis-Wanhstrom: 3 atoms/molecule, indexed i, i + N, i + 2*N
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double rho_star = 0.25;
int N = 300; 
double T_star = 5.0;
string prefix = "300_0.7T"; // Name outputs
string read_prefix = "/Users/leo/C++/MD_Inputs/300_1000tau_melt"; // Read input from
string folderpath = "/Users/leo/C++/Outputs/"; // Write output to this folder 
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//define global variables 
double m = 1.0;  //set m=1
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


// Create Output Files 
// std::ofstream xtraj(folderpath + prefix + "_xtraj.txt");
// std::ofstream ytraj(folderpath + prefix + "_ytraj.txt"); 
// std::ofstream vxtraj(folderpath + prefix + "_vxtraj.txt"); 
// std::ofstream vytraj(folderpath + prefix + "_vytraj.txt");
std::ofstream kineticE(folderpath + prefix + "_KE.txt");
std::ofstream potentialE(folderpath + prefix + "_PE.txt");

std::ofstream trajectory(folderpath + prefix + "_trajectory.bin", std::ios::out | std::ios::binary);

// Output files for adjusted temperature: 
std::ofstream x_init(folderpath + prefix + "_x_initial.txt");
std::ofstream y_init(folderpath + prefix + "_y_initial.txt"); 
std::ofstream vx_init(folderpath + prefix + "_vx_initial.txt"); 
std::ofstream vy_init(folderpath + prefix + "_vy_initial.txt");

//define global vectors of trajectory 
std::vector<double> x;
std::vector<double> y;
std::vector<double> vx;
std::vector<double> vy;
std::vector<double> xprev(3*N);
std::vector<double> yprev(3*N);
std::vector<double> vx_12(3*N);
std::vector<double> vy_12(3*N);


//FUNCTION - store values 

// inline void store(){
//     for (int i=0; i<3*N; i++){
//         xtraj << x[i] << "\n";
//         ytraj << y[i] << "\n";
//         vxtraj << vx[i] << "\n";
//         vytraj << vy[i] << "\n";
//     }
// }

inline void store_initial(){
    for (int i=0; i<3*N; i++){
        x_init << x[i] << "\n";
        y_init<< y[i] << "\n";
        vx_init << vx[i] << "\n";
        vy_init << vy[i] << "\n";
    }
    cout << "Initial state stored" << "\n";
}

inline void store_binary() {
    trajectory.write(reinterpret_cast<char*>(x.data()), sizeof(double) * x.size());
    trajectory.write(reinterpret_cast<char*>(y.data()), sizeof(double) * y.size());
    trajectory.write(reinterpret_cast<char*>(vx.data()), sizeof(double) * vx.size());
    trajectory.write(reinterpret_cast<char*>(vy.data()), sizeof(double) * vy.size());
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



//FUNCTION - periodic boundary conditions
inline double periodic(double dr) {
    if (dr > L2) dr -= L;
    if (dr < -L2) dr += L;
    return dr;
}

// inline double periodic(double dr){
//     return dr - L * static_cast<int>(dr/L2);
// }

 
//FUNCTION - Neighbor list
std::tuple<std::vector<int>, std::vector<int>> neighbors() {
    std::vector<int> point(3*N+1);
    std::vector<int> list;

    int num = 0;
    for (int i = 0; i < 3*N; i++) {
        point[i] = num;
        for (int j = 0; j<3*N; j++){
            if (i==j || i%N == j%N){
                continue;
            }
            double dx = periodic(x[i]-x[j]);
            double dy = periodic(y[i]-y[j]);
            double dr = dx*dx + dy*dy;

            if(dr < r_l*r_l){
                list.push_back(j);
                num++;
            }
        }
    }
    point[3*N] = list.size();

    return std::make_tuple(point, list);
}

//FUNCTION - Make Cell Lists
void make_cell_lists(){
    for (int i =0; i<3*N; i++){
        int w = floor((x[i]+L2)/l);
        int h = floor((y[i]+L2)/l); 
        list<int>& currentlist = cells[h][w];
        currentlist.push_back(i);
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
                if (neww != w || newh != h) {
                    cells[newh%s_c][neww%s_c].push_back(i);
                    it = currentlist.erase(it);
                } else {
                    ++it;
                }
            }
        }
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
    int h_current;
    int w_current;
    #pragma omp parallel for private(h_current, w_current)
    for(int h=0; h<s_c; h++){
        for(int w= 0; w<s_c; w++){
            // list<int>& currentlist = cells[h][w];
            for(int i : cells[h][w]){
                for (const pair<int,int>& offset : neighbor_offsets){
                    h_current = (h+ offset.first+s_c)%s_c;
                    w_current = (w+ offset.second+s_c)%s_c;
                    for (int j: cells[h_current][w_current]){
                        if (j == i || j%N == i%N) continue; //Avoid atoms of same molecule
                        double r_ijx = periodic(x[i]-x[j]);
                        double r_ijy = periodic(y[i]-y[j]);
                        double r2 = r_ijx*r_ijx + r_ijy*r_ijy;
                        if (r2 < r_c *r_c){
                             double r2inv = 1/r2;
                             double r6 = r2inv*r2inv*r2inv;
                             double r12 = r6*r6;
                             double f_i = 24*epsilon*r2inv*(2*(r12) - (r6));
                            if(h_current == h && w_current == w){
                                #pragma omp atomic
                                f_x[i] += r_ijx * f_i;
                                #pragma omp atomic
                                f_y[i] += r_ijy * f_i;
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
                            }
                        }
                    }
                }
            }
        }
    }
    return make_tuple(f_x, f_y);
}

//FUNCTION - neighbor list force computation
std::tuple<vector<double>, vector<double>> compute_forces(vector<int> point, vector<int> list) {
    std::vector<double> f_x(N*3);
    std::vector<double> f_y(N*3);
    
    for (int i = 0; i<N; i++){
        for (int j = point[i]; j<point[i+1]; j++){
            double r_ijx = periodic(x[i]-x[list[j]]);
            double r_ijy = periodic(y[i]-y[list[j]]);
            double r2 = (r_ijx*r_ijx + r_ijy*r_ijy);
            if(r2 <= r_c*r_c){
                double r2inv = 1/r2;
                double r6 = r2inv*r2inv*r2inv;
                double r12 = r6*r6;
                double f_i = 24*epsilon*r2inv*(2*(r12) - (r6));
                f_x[i] += r_ijx * f_i;
                f_y[i] += r_ijy * f_i;
            }   
        }
    }
    return make_tuple(f_x, f_y);
}

//FUNCTION - neighbor list energy computation
void compute_energies(vector<int>point, vector<int>list){
    // Kinetic Energy
    double KE = 0;
    double PE = 0;
    for (int i = 0; i<N; i++){
        KE += 0.5*(vx[i]*vx[i] + vy[i]*vy[i]);
        for (int j = point[i]; j<point[i+1]; j++){
            double r_ijx = periodic(x[i]-x[list[j]]);
            double r_ijy = periodic(y[i]-y[list[j]]);
            double r2 = (r_ijx*r_ijx + r_ijy*r_ijy);
            if(r2 <= r_c*r_c){
                double r2inv = 1/r2;
                double r6 = r2inv*r2inv*r2inv;
                double r12 = r6*r6;
                PE += 2*(r12-r6);
            }
            
        }
    }
    kineticE << KE << "\n";
    potentialE << PE << "\n";
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
                                PE += 2*((r12) - (r6));
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

//// define variables for position constraints
double r_12, r_13, r_23;
double g12, g13, g23;
double dx_12, dy_12, dx_13, dy_13, dx_23, dy_23, dx, dy;

void r_constraints(){ 
    // #pragma omp parallel for private(dx_12, dy_12, dx_13, dy_13, dx_23, dy_23, dx, dy,g12, g13, g23, r_12, r_13, r_23)
    for (int i =0; i<N; i++){
        dx_12 = periodic(x[i] - x[N+i]);
        dy_12 = periodic(y[i] - y[N+i]);
        r_12= dx_12*dx_12 + dy_12*dy_12;
        dx_13 = periodic(x[i]-x[i+2*N]);  
        dy_13 = periodic(y[i]-y[i+2*N]);
        r_13 = dx_13*dx_13 + dy_13*dy_13;
        dx_23 = periodic(x[N+i]-x[i+2*N]);
        dy_23 = periodic(y[N+i]-y[i+2*N]);
        r_23 = dx_23*dx_23 + dy_23*dy_23;
        if(fabs(r_12-1.0)<=10e-7 && fabs(r_13-1.0)<=10e-7 && fabs(r_23-d_23*d_23)<=10e-7) continue;

        while(fabs(r_12-1.0)>10e-7 || fabs(r_13-1.0)>10e-7 || fabs(r_23-d_23*d_23)>10e-7){
            g12 = (r_12 - 1) / ((4*(dx_12)*periodic(xprev[i]-xprev[N+i]) + 4*(dy_12)*periodic(yprev[i]-yprev[N+i]))*dt);
            g13 = (r_13 - 1) / ((4*(dx_13)*periodic(xprev[i]-xprev[2*N+i]) + 4*(dy_13)*periodic(yprev[i]-yprev[2*N+i]))*dt);
            g23 = (r_23 - d_23*d_23) / ((4*(dx_23)*periodic(xprev[N+i]-xprev[2*N+i]) + 4*(dy_23)*periodic(yprev[N+i]-yprev[2*N+i]))*dt);
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
    }
}

//// define variables for velocity constraints
double k_12, k_13, k_23;
double a, b, c, d, e, f, det, vr_12, vr_13, vr_23;
double theta = 7*M_PI / 24;
double d2 = d_23*d_23;
double x12,x13,x23,y12,y13,y23;

void vel_constraints(){ 
    // cout << "INITIAL VELOCITY MAGNITUDES\n";
    // #pragma omp parallel for private(k_12, k_13, k_23, a, b, c, d, e, f, det, vr_12, vr_13, vr_23, x12,x13,x23,y12,y13,y23)
    for (int i =0; i<N; i++){
        dx_12 = periodic(x[i]-x[i+N]);
        dx_13 = periodic(x[i]-x[i+2*N]);
        dx_23 = periodic(x[i+N]-x[i+2*N]);
        dy_12 = periodic(y[i]-y[N+i]);
        dy_13 = periodic(y[i]-y[2*N+i]);
        dy_23 = periodic(y[i+N]-y[2*N+i]);
        while(fabs(((vx[i]-vx[i+N])*dx_12) + ((vy[i]-vy[i+N])*dy_12))>10e-7||
               fabs(((vx[i]-vx[i+2*N])*dx_13) + ((vy[i]-vy[i+2*N])*dy_13)) >10e-7||
               fabs(((vx[N+i]-vx[i+2*N])*dx_23) + ((vy[N+i]-vy[i+2*N])*dy_23))>10e-7 ){
            k_12 = (dx_12*(vx[i]-vx[i+N]) + dy_12*(vy[i]-vy[i+N]))/(2*1);
            k_13 = (dx_13*(vx[i]-vx[i+2*N]) + dy_13*(vy[i]-vy[i+2*N]))/(2*(1));
            k_23 = (dx_23*(vx[i+N]-vx[i+2*N]) + dy_23*(vy[i+N]-vy[i+2*N]))/(2*(d_23*d_23));

            vx[i]-=(k_12*dx_12+k_13*dx_13);
            vx[i+N]+=k_12*dx_12-k_23*dx_23;

            vy[i]-=k_12*dy_12+k_13*dy_13;
            vy[i+N]+=k_12*dy_12-k_23*dy_23;

            vx[i+2*N]+=k_13*dx_13+k_23*dx_23;
            vy[i+2*N]+=k_13*dy_13+k_23*dy_23;

        }
    }
}


void vel_constraints_analytical(){
    for (int i = 0; i<N; i++){
        
        x12 = periodic(x[i]-x[i+N]);  y12 = periodic(y[i]-y[i+N]);
        x13 = periodic(x[i]-x[i+2*N]);  y13 = periodic(y[i]-y[i+2*N]);
        x23 = periodic(x[i+N]-x[i+2*N]);  y23 = periodic(y[i+N]-y[i+2*N]);

        a = x12*x12 + y12*y12; //r12•r12
        b = x13*x13 + y13*y13; //r13•r13
        c = x23*x23 + y23*y23; //r23•r23

        d = x12*x13 + y12*y13; //r12•r13
        e = x12*x23 + y12*y23; //r12•r23
        f = x13*x23 + y13*y23; //r13•r23

        vr_12 = -1*((vx[i]-vx[i+N])*x12 + (vy[i]-vy[i+N])*y12); //-v'12•r12
        vr_13 = -1*((vx[i]-vx[i+2*N])*x13 + (vy[i]-vy[i+2*N])*y13); //-v'13•r13
        vr_23 = -1*((vx[i+N]-vx[i+2*N])*x23 + (vy[i+N]-vy[i+2*N])*y23); //-v'23•r23

        det = 8*a*b*c - 2*a*f*f - 2*c*d*d - 2*d*e*f - 2*b*e*e;        
        // det = 8*d2 - 4*d2*cos(theta) + 2*cos(5*M_PI/12)*d_23*cos(theta)*cos(theta) - 2*d2*cos(5*M_PI/12)*cos(5*M_PI/12);

        k_12 = (vr_12 * (4*b*c - f*f) 
                + vr_13 * (-2*c*d - e*f)
                    + vr_23 * (d*f + 2*e*b))*(1/det)*0.5;
        k_13 = (vr_12 * (-2*d*c - f*e) 
                + vr_13 * (4*a*c - e*e)
                    + vr_23 * (-2*a*f - e*d))*(1/det)*0.5;
        k_23 = (vr_12 * (d*f + 2*b*e) 
                + vr_13 * (-2*a*f - d*e)
                    + vr_23 * (4*a*b - d*d))*(1/det)*0.5;


        vx[i] = vx[i]+ k_12*x12 + k_13*x13;
        vy[i] = vy[i]+ k_12*y12 + k_13*y13;

        vx[i+N] = vx[i+N] - k_12*x12 + k_23*x23;
        vy[i+N] = vy[i+N] - k_12*y12 + k_23*y23;
    
        vx[i+2*N] = vx[i+2*N] - k_23*x23 - k_13*x13;
        vy[i+2*N] = vy[i+2*N] - k_23*y23 - k_13*y13;
    }
}

void scale_vel(double scale){
    for(int i = 0; i<3*N; i++){
        vx[i] *= scale;
        vy[i] *= scale;
    }
}

double measured_temp = 0;
//FUNCTION - verlocity verlet for given number of timesteps
void vel_verlet(int tsteps, bool measure_temp, bool store_traj){
    // Uncomment when using neighbor lists 
    // std::tuple<vector<int>,vector<int>> nbgh;
    // std::vector<int> npoint;
    // std::vector<int> nlist;

    for (int t=0; t<tsteps; t++){

        //// Neighbor list update lists
        // if (t%10 ==0){
        //     nbgh = neighbors(x, y);
        //     npoint = get<0>(nbgh);
        //     nlist = get<1>(nbgh);
        // }
        
        //// Linked list update linked lists
        update_lists(); // update every timestep/

        // auto forces1 = compute_forces(npoint, nlist);  // compute forces half step
        auto forces1 = compute_forces_LL(); // compute forces - Linked list method

        std::vector<double>& a_x12 = get<0>(forces1);
        std::vector<double>& a_y12 = get<1>(forces1);
        

        for (int i=0; i<3*N; i++){
            vx_12[i] = vx[i] + a_x12[i]*dt*0.5;
            vy_12[i] = vy[i] + a_y12[i]*dt*0.5;
            xprev[i] = x[i]; // Temporarily store previous positions 
            yprev[i] = y[i]; // Temporarily store previous positions
            x[i] = periodic(x[i] + vx_12[i]*dt);
            y[i] = periodic(y[i] + vy_12[i]*dt);
        }
        // RATTLE position constraints 
        r_constraints();
        
        for (int i = 0; i<3*N; i++){
            vx_12[i] = periodic(x[i] - xprev[i]) / dt;
            vy_12[i] = periodic(y[i] - yprev[i]) / dt;
        }

        // auto forces2 = compute_forces(npoint, nlist);  // compute forces full step, neighbor lists
        auto forces2 = compute_forces_LL();

        std::vector<double>& a_x = get<0>(forces2);
        std::vector<double>& a_y = get<1>(forces2);

        for (int i = 0; i<3*N; i++){
            vx[i] = vx_12[i] + 0.5*a_x[i]*dt;
            vy[i] = vy_12[i] + 0.5*a_y[i]*dt;
        }

        // RATTLE velocity constraints 
        vel_constraints_analytical();

        // Energy and trajectory storage, temperature measurement
        if(t%10 == 0){
            if(store_traj== true){
                compute_energies_LL();             
                store_binary(); // Store every 10 timesteps, and compute energies when storing
            }
            else {
                if(measure_temp == true){
                    double KE = 0;
                    for(int i = 0; i<3*N; i++){
                        KE += 0.5*(vx[i]*vx[i] + vy[i]*vy[i]);
                    }
                    measured_temp += (2.0 / (3.0*N)) * KE /(tsteps/10);
                }
            }
        }

    }
}

void adjust_temp(double target_temp){
    vel_verlet(10/dt, true, false); // Measure temperature
    double mtemp = measured_temp;
    cout<< "Beginning temperature: "<< measured_temp << "\n";
    while (mtemp > (target_temp +0.01) || mtemp < (target_temp - 0.01)){
        double tscale = sqrt((target_temp)/mtemp);
        scale_vel(tscale);
        vel_verlet(20/dt, false, false); // relax, no measurement
        vel_verlet(10/dt, true, false); // re-measure temperature
        mtemp = measured_temp;
        measured_temp = 0;
        cout << "Adjusted temperature: " << mtemp << "\n";
    }
}

//MAIN FUNCTION 
int main(){

//// Read from initial 

    x = read(read_prefix+"_x_initial.txt");
    y = read(read_prefix+"_y_initial.txt");
    vx = read(read_prefix+"_vx_initial.txt");
    vy = read(read_prefix+"_vy_initial.txt");  

    // Store initial values from x0,y0,vx0,vy0 and make cell lists
    make_cell_lists();

    //// SIMULATION  - modify at will 
    adjust_temp(0.7);
    cout << "Relaxing 1000tau... "<<"\n";
    vel_verlet(1000/dt, false, false);
    cout << "RUN 0.7T, 10,000" << "\n";
    vel_verlet(10000/dt, false, true);
    store_initial();
    // xtraj.close();
    // ytraj.close();
    // vxtraj.close();
    // vytraj.close();
    kineticE.close();
    potentialE.close();

    trajectory.close();

    x_init.close();
    y_init.close();
    vx_init.close();
    vy_init.close();

    return  0;
}                                                                                                                                                                                                                                                                                                                                                                                         
