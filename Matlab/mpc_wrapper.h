#ifndef MPC_WRAPPER_H
#define MPC_WRAPPER_H

void my_mpc_wrapper(const double x_curr[12], const double target[3], const double u_prev[4], const double Ad[144], const double Bd[48], double U_opt[4]);

#endif