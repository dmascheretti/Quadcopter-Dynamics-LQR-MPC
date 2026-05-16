#include "mpc_wrapper.h"
#include "mpc_controller_code.h"

#define MAX_IW 100000
#define MAX_W  1000000

// Utile per visulizzare corretta funzione nel blocco C caller

static long long int iw[MAX_IW]; 
static double w[MAX_W];        

void my_mpc_wrapper(const double x_curr[12], const double target[3], const double u_prev[4], const double Ad[144], const double Bd[48], double U_opt[4]) {
    
    const double* arg[5] = {x_curr, target, u_prev, Ad, Bd};
    
    double* res[1] = {U_opt};
    
    mpc_solver(arg, res, iw, w, 0);
}