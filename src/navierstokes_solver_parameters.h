#ifndef NAVIERSTOKES_SOLVER_PARAMETERS_H_
#define NAVIERSTOKES_SOLVER_PARAMETERS_H_

//// Parameters for the iterative Jacobi solver

// Define `CFDDEMCOUPLING` in order to enable the two-way coupling between the
// fluid and particle phase.
#define CFDDEMCOUPLING

// Solver parameter, used in velocity prediction and pressure iteration
// 1.0: Use old pressures for fluid velocity prediction (see Langtangen et al.
// 2002)
// 0.0: Do not use old pressures for fluid velocity prediction (Chorin's
// original projection method, see Chorin (1968) and "Projection method (fluid
// dynamics)" page on Wikipedia.
// The best results precision and performance-wise are obtained by using BETA=0
// and a very low tolerance criteria value (e.g. 1.0e-9)
#define BETA 0.0

// Under-relaxation parameter, used in solution of Poisson equation. The value
// should be within the range ]0.0;1.0]. At a value of 1.0, the new estimate of
// epsilon values is used exclusively. At lower values, a linear interpolation
// between new and old values is used. The solution typically converges faster
// with a value of 1.0, but instabilities may be avoided with lower values.
#define THETA 1.0

// Smoothing parameter. The epsilon (pressure) values are smoothed by including
// the average epsilon value of the six closest (face) neighbor cells. This
// parameter should be in the range [0.0;1.0[. The higher the value, the more
// averaging is introduced. A value of 0.0 disables all averaging.
#define GAMMA 0.5
//#define GAMMA 0.0

// Tolerance criteria for the normalized residual
//const double tolerance = 1.0e-3;
const double tolerance = 1.0e-4;
//const double tolerance = 1.0e-5;
//const double tolerance = 1.0e-7;
//const double tolerance = 1.0e-8;
//const double tolerance = 1.0e-9;

// The maximum number of iterations to perform
const unsigned int maxiter = 1e4;

// The number of iterations to perform between checking the norm. residual value
const unsigned int nijacnorm = 10;

// Write max. residual during the latest solution loop to logfile
// 'max_res_norm.dat'
// 0: False, 1: True
const int write_res_log = 0;

// Report epsilon values during Jacobi iterations to stdout
// 0: False, 1: True
const int report_epsilon = 1;
const int report_even_more_epsilon = 1;

// Report the number of iterations it took before convergence to logfile
// 'output/<sid>-conv.dat'
// 0: False, 1: True
const int write_conv_log = 1;

// The interval between iteration number reporting in 'output/<sid>-conv.log'
const int conv_log_interval = 10;
//const int conv_log_interval = 1;

#endif
