// darcy.cuh
// CUDA implementation of Darcy porous flow

#include <iostream>
#include <cuda.h>
//#include <cutil_math.h>
#include <helper_math.h>

#include "vector_arithmetic.h"  // for arbitrary precision vectors
#include "sphere.h"
#include "datatypes.h"
#include "utility.h"
#include "constants.cuh"
#include "debug.h"

// Initialize memory
void DEM::initDarcyMemDev(void)
{
    // size of scalar field
    unsigned int memSizeF = sizeof(Float)*darcyCells();

    // size of cell-face arrays in staggered grid discretization
    //unsigned int memSizeFface = sizeof(Float)*darcyCellsVelocity();

    //cudaMalloc((void**)&dev_darcy_dpdt, memSizeF);  // Backwards Euler gradient
    cudaMalloc((void**)&dev_darcy_dp_expl, memSizeF); // Expl. pressure change
    cudaMalloc((void**)&dev_darcy_p_old, memSizeF); // old pressure
    cudaMalloc((void**)&dev_darcy_p, memSizeF);     // hydraulic pressure
    cudaMalloc((void**)&dev_darcy_p_new, memSizeF); // updated pressure
    cudaMalloc((void**)&dev_darcy_v, memSizeF*3);   // cell hydraulic velocity
    //cudaMalloc((void**)&dev_darcy_vp_avg, memSizeF*3); // avg. particle velocity
    //cudaMalloc((void**)&dev_darcy_d_avg, memSizeF); // avg. particle diameter
    cudaMalloc((void**)&dev_darcy_phi, memSizeF);   // cell porosity
    cudaMalloc((void**)&dev_darcy_dphi, memSizeF);  // cell porosity change
    cudaMalloc((void**)&dev_darcy_norm, memSizeF);  // normalized residual
    cudaMalloc((void**)&dev_darcy_f_p, sizeof(Float4)*np); // pressure force
    cudaMalloc((void**)&dev_darcy_k, memSizeF);        // hydraulic permeability
    cudaMalloc((void**)&dev_darcy_grad_k, memSizeF*3);  // grad(permeability)
    //cudaMalloc((void**)&dev_darcy_div_v_p, memSizeF3); // divergence(v_p)

    checkForCudaErrors("End of initDarcyMemDev");
}

// Free memory
void DEM::freeDarcyMemDev()
{
    //cudaFree(dev_darcy_dpdt);
    cudaFree(dev_darcy_dp_expl);
    cudaFree(dev_darcy_p_old);
    cudaFree(dev_darcy_p);
    cudaFree(dev_darcy_p_new);
    cudaFree(dev_darcy_v);
    //cudaFree(dev_darcy_vp_avg);
    //cudaFree(dev_darcy_d_avg);
    cudaFree(dev_darcy_phi);
    cudaFree(dev_darcy_dphi);
    cudaFree(dev_darcy_norm);
    cudaFree(dev_darcy_f_p);
    cudaFree(dev_darcy_k);
    cudaFree(dev_darcy_grad_k);
    //cudaFree(dev_darcy_div_v_p);
}

// Transfer to device
void DEM::transferDarcyToGlobalDeviceMemory(int statusmsg)
{
    checkForCudaErrors("Before attempting cudaMemcpy in "
            "transferDarcyToGlobalDeviceMemory");

    //if (verbose == 1 && statusmsg == 1)
    //std::cout << "  Transfering fluid data to the device:           ";

    // memory size for a scalar field
    unsigned int memSizeF  = sizeof(Float)*darcyCells();

    //writeNSarray(ns.p, "ns.p.txt");

    cudaMemcpy(dev_darcy_p, darcy.p, memSizeF, cudaMemcpyHostToDevice);
    checkForCudaErrors("transferDarcytoGlobalDeviceMemory after first "
            "cudaMemcpy");
    cudaMemcpy(dev_darcy_v, darcy.v, memSizeF*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_darcy_phi, darcy.phi, memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_darcy_dphi, darcy.dphi, memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_darcy_f_p, darcy.f_p, sizeof(Float4)*np,
            cudaMemcpyHostToDevice);

    checkForCudaErrors("End of transferDarcyToGlobalDeviceMemory");
    //if (verbose == 1 && statusmsg == 1)
    //std::cout << "Done" << std::endl;
}

// Transfer from device
void DEM::transferDarcyFromGlobalDeviceMemory(int statusmsg)
{
    if (verbose == 1 && statusmsg == 1)
        std::cout << "  Transfering fluid data from the device:         ";

    // memory size for a scalar field
    unsigned int memSizeF  = sizeof(Float)*darcyCells();

    cudaMemcpy(darcy.p, dev_darcy_p, memSizeF, cudaMemcpyDeviceToHost);
    checkForCudaErrors("In transferDarcyFromGlobalDeviceMemory, dev_darcy_p", 0);
    cudaMemcpy(darcy.v, dev_darcy_v, memSizeF*3, cudaMemcpyDeviceToHost);
    cudaMemcpy(darcy.phi, dev_darcy_phi, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(darcy.dphi, dev_darcy_dphi, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(darcy.f_p, dev_darcy_f_p, sizeof(Float4)*np,
            cudaMemcpyDeviceToHost);
    cudaMemcpy(darcy.k, dev_darcy_k, memSizeF, cudaMemcpyDeviceToHost);

    checkForCudaErrors("End of transferDarcyFromGlobalDeviceMemory", 0);
    if (verbose == 1 && statusmsg == 1)
        std::cout << "Done" << std::endl;
}

// Transfer the normalized residuals from device to host
void DEM::transferDarcyNormFromGlobalDeviceMemory()
{
    cudaMemcpy(darcy.norm, dev_darcy_norm, sizeof(Float)*darcyCells(),
            cudaMemcpyDeviceToHost);
    checkForCudaErrors("End of transferDarcyNormFromGlobalDeviceMemory");
}

// Transfer the pressures from device to host
void DEM::transferDarcyPressuresFromGlobalDeviceMemory()
{
    cudaMemcpy(darcy.p, dev_darcy_p, sizeof(Float)*darcyCells(),
            cudaMemcpyDeviceToHost);
    checkForCudaErrors("End of transferDarcyNormFromGlobalDeviceMemory");
}

// Get linear index from 3D grid position
__inline__ __device__ unsigned int d_idx(
        const int x, const int y, const int z)
{
    // without ghost nodes
    //return x + dev_grid.num[0]*y + dev_grid.num[0]*dev_grid.num[1]*z;

    // with ghost nodes
    // the ghost nodes are placed at x,y,z = -1 and WIDTH
    return (x+1) + (devC_grid.num[0]+2)*(y+1) +
        (devC_grid.num[0]+2)*(devC_grid.num[1]+2)*(z+1);
}

// Get linear index of velocity node from 3D grid position in staggered grid
__inline__ __device__ unsigned int d_vidx(
        const int x, const int y, const int z)
{
    // without ghost nodes
    //return x + (devC_grid.num[0]+1)*y
    //+ (devC_grid.num[0]+1)*(devC_grid.num[1]+1)*z;

    // with ghost nodes
    // the ghost nodes are placed at x,y,z = -1 and WIDTH+1
    return (x+1) + (devC_grid.num[0]+3)*(y+1)
        + (devC_grid.num[0]+3)*(devC_grid.num[1]+3)*(z+1);
}

// The normalized residuals are given an initial value of 0, since the values at
// the Dirichlet boundaries aren't written during the iterations.
__global__ void setDarcyNormZero(
        Float* __restrict__ dev_darcy_norm)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // check that we are not outside the fluid grid
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {
        __syncthreads();
        dev_darcy_norm[d_idx(x,y,z)] = 0.0;
    }
}

// Set an array of scalars to 0.0 inside devC_grid
    template<typename T>
__global__ void setDarcyZeros(T* __restrict__ dev_scalarfield)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // check that we are not outside the fluid grid
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {
        __syncthreads();
        dev_scalarfield[d_idx(x,y,z)] = 0.0;
    }
}


// Update a field in the ghost nodes from their parent cell values. The edge
// (diagonal) cells are not written since they are not read. Launch this kernel
// for all cells in the grid using
// setDarcyGhostNodes<datatype><<<.. , ..>>>( .. );
    template<typename T>
__global__ void setDarcyGhostNodes(
        T* __restrict__ dev_scalarfield,
        const int bc_bot,
        const int bc_top)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        const T val = dev_scalarfield[d_idx(x,y,z)];

        // x
        if (x == 0)
            dev_scalarfield[idx(nx,y,z)] = val;
        if (x == nx-1)
            dev_scalarfield[idx(-1,y,z)] = val;

        // y
        if (y == 0)
            dev_scalarfield[idx(x,ny,z)] = val;
        if (y == ny-1)
            dev_scalarfield[idx(x,-1,z)] = val;

        // z
        if (z == 0 && bc_bot == 0)
            dev_scalarfield[idx(x,y,-1)] = val;     // Dirichlet
        if (z == 0 && bc_bot == 1)
            dev_scalarfield[idx(x,y,-1)] = val;     // Neumann
        if (z == 0 && bc_bot == 2)
            dev_scalarfield[idx(x,y,nz)] = val;     // Periodic -z

        if (z == nz-1 && bc_top == 0)
            dev_scalarfield[idx(x,y,nz)] = val;     // Dirichlet
        if (z == nz-2 && bc_top == 1)
            dev_scalarfield[idx(x,y,nz)] = val;     // Neumann
        if (z == nz-1 && bc_top == 2)
            dev_scalarfield[idx(x,y,-1)] = val;     // Periodic +z
    }
}

// Find the porosity in each cell on the base of a sphere, centered at the cell
// center. 
__global__ void findDarcyPorosities(
        const unsigned int* __restrict__ dev_cellStart,   // in
        const unsigned int* __restrict__ dev_cellEnd,     // in
        const Float4* __restrict__ dev_x_sorted,          // in
        const unsigned int iteration,                     // in
        const unsigned int np,                            // in
        const Float c_phi,                                // in
        Float*  __restrict__ dev_darcy_phi,               // in + out
        Float*  __restrict__ dev_darcy_dphi)              // in + out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell dimensions
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // Cell sphere radius
    const Float R = fmin(dx, fmin(dy,dz)) * 0.5; // diameter = cell width
    //const Float R = fmin(dx, fmin(dy,dz));       // diameter = 2*cell width
    const Float cell_volume = 4.0/3.0*M_PI*R*R*R;

    Float void_volume = cell_volume;
    Float4 xr;  // particle pos. and radius

    // check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        if (np > 0) {

            // Cell sphere center position
            const Float3 X = MAKE_FLOAT3(
                    x*dx + 0.5*dx,
                    y*dy + 0.5*dy,
                    z*dz + 0.5*dz);

            Float d, r;
            Float phi = 1.00;
            //Float4 v;
            unsigned int n = 0;

            //Float3 v_avg = MAKE_FLOAT3(0.0, 0.0, 0.0);
            //Float  d_avg = 0.0;

            // Read old porosity
            __syncthreads();
            Float phi_0 = dev_darcy_phi[d_idx(x,y,z)];

            // The cell 3d index
            const int3 gridPos = make_int3((int)x,(int)y,(int)z);

            // The neighbor cell 3d index
            int3 targetCell;

            // The distance modifier for particles across periodic boundaries
            Float3 dist, distmod;

            unsigned int cellID, startIdx, endIdx, i;

            // Iterate over 27 neighbor cells, R = cell width
            for (int z_dim=-1; z_dim<2; ++z_dim) { // z-axis
                for (int y_dim=-1; y_dim<2; ++y_dim) { // y-axis
                    for (int x_dim=-1; x_dim<2; ++x_dim) { // x-axis

            // Iterate over 125 neighbor cells, R = 2*cell width
            /*for (int z_dim=-2; z_dim<3; ++z_dim) { // z-axis
                for (int y_dim=-2; y_dim<3; ++y_dim) { // y-axis
                    for (int x_dim=-2; x_dim<3; ++x_dim) { // x-axis*/

                        // Index of neighbor cell this iteration is looking at
                        targetCell = gridPos + make_int3(x_dim, y_dim, z_dim);

                        // Get distance modifier for interparticle
                        // vector, if it crosses a periodic boundary
                        distmod = MAKE_FLOAT3(0.0, 0.0, 0.0);
                        if (findDistMod(&targetCell, &distmod) != -1) {

                            // Calculate linear cell ID
                            cellID = targetCell.x
                                + targetCell.y * devC_grid.num[0]
                                + (devC_grid.num[0] * devC_grid.num[1])
                                * targetCell.z;

                            // Lowest particle index in cell
                            __syncthreads();
                            startIdx = dev_cellStart[cellID];

                            // Make sure cell is not empty
                            if (startIdx != 0xffffffff) {

                                // Highest particle index in cell
                                __syncthreads();
                                endIdx = dev_cellEnd[cellID];

                                // Iterate over cell particles
                                for (i=startIdx; i<endIdx; ++i) {

                                    // Read particle position and radius
                                    __syncthreads();
                                    xr = dev_x_sorted[i];
                                    //v  = dev_vel_sorted[i];
                                    r = xr.w;

                                    // Find center distance
                                    dist = MAKE_FLOAT3(
                                            X.x - xr.x, 
                                            X.y - xr.y,
                                            X.z - xr.z);
                                    dist += distmod;
                                    d = length(dist);

                                    // Lens shaped intersection
                                    if ((R - r) < d && d < (R + r)) {
                                        void_volume -=
                                            1.0/(12.0*d) * (
                                                    M_PI*(R + r - d)*(R + r - d)
                                                    *(d*d + 2.0*d*r - 3.0*r*r
                                                        + 2.0*d*R + 6.0*r*R
                                                        - 3.0*R*R) );
                                        //v_avg += MAKE_FLOAT3(v.x, v.y, v.z);
                                        //d_avg += r+r;
                                        n++;
                                    }

                                    // Particle fully contained in cell sphere
                                    if (d <= R - r) {
                                        void_volume -= 4.0/3.0*M_PI*r*r*r;
                                        //v_avg += MAKE_FLOAT3(v.x, v.y, v.z);
                                        //d_avg += r+r;
                                        n++;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            //if (phi < 0.999) {
            //v_avg /= n;
            //d_avg /= n;
            //}

            // Make sure that the porosity is in the interval [0.0;1.0]
            phi = fmin(0.9, fmax(0.1, void_volume/cell_volume));
            //phi = fmin(0.99, fmax(0.01, void_volume/cell_volume));
            //phi = void_volume/cell_volume;

            Float dphi = phi - phi_0;
            if (iteration == 0)
                dphi = 0.0;

            // report values to stdout for debugging
            //printf("%d,%d,%d\tphi = %f dphi = %f\n", x,y,z, phi, dphi);
            //printf("%d,%d,%d\tphi = %f dphi = %f v_avg = %f,%f,%f d_avg = %f\n",
            //       x,y,z, phi, dphi, v_avg.x, v_avg.y, v_avg.z, d_avg);

            // Save porosity and porosity change
            __syncthreads();
            //phi = 0.5; dphi = 0.0; // disable porosity effects
            const unsigned int cellidx = d_idx(x,y,z);
            dev_darcy_phi[cellidx]   = phi*c_phi;
            dev_darcy_dphi[cellidx] += dphi*c_phi;
            //dev_darcy_vp_avg[cellidx] = v_avg;
            //dev_darcy_d_avg[cellidx]  = d_avg;

#ifdef CHECK_FLUID_FINITE
            (void)checkFiniteFloat("phi", x, y, z, phi);
            (void)checkFiniteFloat("dphi", x, y, z, dphi);
            //(void)checkFiniteFloat3("v_avg", x, y, z, v_avg);
            //(void)checkFiniteFloat("d_avg", x, y, z, d_avg);
#endif
        } else {

            __syncthreads();
            const unsigned int cellidx = d_idx(x,y,z);

            //Float phi = 0.5;
            //Float dphi = 0.0;
            //if (iteration == 20 && x == nx/2 && y == ny/2 && z == nz/2) {
            //phi = 0.4;
            //dphi = 0.1;
            //}
            //dev_darcy_phi[cellidx]  = phi;
            //dev_darcy_dphi[cellidx] = dphi;
            dev_darcy_phi[cellidx]  = 0.999;
            dev_darcy_dphi[cellidx] = 0.0;

            //dev_darcy_vp_avg[cellidx] = MAKE_FLOAT3(0.0, 0.0, 0.0);
            //dev_darcy_d_avg[cellidx]  = 0.0;
        }
    }
}

// Find the particle velocity divergence at the cell center from the average
// particle velocities on the cell faces
__global__ void findDarcyParticleVelocityDivergence(
        const Float* __restrict__ dev_darcy_v_p_x,  // in
        const Float* __restrict__ dev_darcy_v_p_y,  // in
        const Float* __restrict__ dev_darcy_v_p_z,  // in
        Float* __restrict__ dev_darcy_div_v_p)      // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    if (x < nx && y < ny && z < nz) {

        // read values
        __syncthreads();
        Float v_p_xn = dev_darcy_v_p_x[d_vidx(x,y,z)];
        Float v_p_xp = dev_darcy_v_p_x[d_vidx(x+1,y,z)];
        Float v_p_yn = dev_darcy_v_p_y[d_vidx(x,y,z)];
        Float v_p_yp = dev_darcy_v_p_y[d_vidx(x,y+1,z)];
        Float v_p_zn = dev_darcy_v_p_z[d_vidx(x,y,z)];
        Float v_p_zp = dev_darcy_v_p_z[d_vidx(x,y,z+1)];

        // cell dimensions
        const Float dx = devC_grid.L[0]/nx;
        const Float dy = devC_grid.L[1]/ny;
        const Float dz = devC_grid.L[2]/nz;

        // calculate the divergence using first order central finite differences
        const Float div_v_p =
            (v_p_xp - v_p_xn)/dx +
            (v_p_yp - v_p_yn)/dy +
            (v_p_zp - v_p_zn)/dz;

        __syncthreads();
        dev_darcy_div_v_p[d_idx(x,y,z)] = div_v_p;

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat("div_v_p", x, y, z, div_v_p);
#endif
    }
}

// Find particle-fluid interaction force as outlined by Zhou et al. 2010, and
// originally by Gidaspow 1992. All terms other than the pressure force are
// neglected. The buoyancy force is included.
__global__ void findDarcyPressureForce(
    const Float4* __restrict__ dev_x,           // in
    const Float*  __restrict__ dev_darcy_p,     // in
    const Float*  __restrict__ dev_darcy_phi,   // in
    Float4* __restrict__ dev_force,             // out
    Float4* __restrict__ dev_darcy_f_p)         // out
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x; // Particle index

    if (i < devC_np) {

        // read particle information
        __syncthreads();
        const Float4 x = dev_x[i];

        // determine fluid cell containing the particle
        const unsigned int i_x =
            floor((x.x - devC_grid.origo[0])/(devC_grid.L[0]/devC_grid.num[0]));
        const unsigned int i_y =
            floor((x.y - devC_grid.origo[1])/(devC_grid.L[1]/devC_grid.num[1]));
        const unsigned int i_z =
            floor((x.z - devC_grid.origo[2])/(devC_grid.L[2]/devC_grid.num[2]));
        const unsigned int cellidx = d_idx(i_x, i_y, i_z);

        // determine cell dimensions
        const Float dx = devC_grid.L[0]/devC_grid.num[0];
        const Float dy = devC_grid.L[1]/devC_grid.num[1];
        const Float dz = devC_grid.L[2]/devC_grid.num[2];

        // read fluid information
        __syncthreads();
        const Float phi = dev_darcy_phi[cellidx];
        const Float p_xn = dev_darcy_p[d_idx(i_x-1,i_y,i_z)];
        //const Float p    = dev_darcy_p[cellidx];
        const Float p_xp = dev_darcy_p[d_idx(i_x+1,i_y,i_z)];
        const Float p_yn = dev_darcy_p[d_idx(i_x,i_y-1,i_z)];
        const Float p_yp = dev_darcy_p[d_idx(i_x,i_y+1,i_z)];
        const Float p_zn = dev_darcy_p[d_idx(i_x,i_y,i_z-1)];
        const Float p_zp = dev_darcy_p[d_idx(i_x,i_y,i_z+1)];

        // find particle volume (radius in x.w)
        const Float V = 4.0/3.0*M_PI*x.w*x.w*x.w;

        // determine pressure gradient from first order central difference
        const Float3 grad_p = MAKE_FLOAT3(
                (p_xp - p_xn)/(dx + dx),
                (p_yp - p_yn)/(dy + dy),
                (p_zp - p_zn)/(dz + dz));

        // find pressure gradient force plus buoyancy force.
        // buoyancy force = weight of displaced fluid
        // f_b = -rho_f*V*g
        const Float3 f_p = -1.0*grad_p*V/(1.0 - phi);
            //- devC_params.rho_f*V*MAKE_FLOAT3(
                    //devC_params.g[0],
                    //devC_params.g[1],
                    //devC_params.g[2]);

        /*printf("%d,%d,%d findPF:\n"
                "\tphi    = %f\n"
                "\tp      = %f\n"
                "\tgrad_p = % f, % f, % f\n"
                "\tf_p    = % f, % f, % f\n",
                i_x, i_y, i_z,
                phi, p,
                grad_p.x, grad_p.y, grad_p.z,
                f_p.x, f_p.y, f_p.z);*/

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat3("f_p", i_x, i_y, i_z, f_p);
#endif
        // save force
        __syncthreads();
        dev_force[i]    += MAKE_FLOAT4(f_p.x, f_p.y, f_p.z, 0.0);
        dev_darcy_f_p[i] = MAKE_FLOAT4(f_p.x, f_p.y, f_p.z, 0.0);
    }
}

// Set the pressure at the top boundary to new_pressure
__global__ void setDarcyTopPressure(
    const Float new_pressure, Float* __restrict__ dev_darcy_p)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    
    // check that the thread is located at the top boundary
    if (x < devC_grid.num[0] &&
        y < devC_grid.num[1] &&
        z == devC_grid.num[2]-1) {

        const unsigned int cellidx = idx(x,y,z);

        // Write the new pressure the top boundary cells
        __syncthreads();
        dev_darcy_p[cellidx] = new_pressure;
    }
}

// Set the pressure at the top wall to new_pressure
__global__ void setDarcyTopWallPressure(
    const Float new_pressure,
    const unsigned int wall0_iz,
    Float* __restrict__ dev_darcy_p)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    
    // check that the thread is located at the top boundary
    if (x < devC_grid.num[0] &&
        y < devC_grid.num[1] &&
        z == wall0_iz) {

        const unsigned int cellidx = idx(x,y,z);

        // Write the new pressure the top boundary cells
        __syncthreads();
        dev_darcy_p[cellidx] = new_pressure;
    }
}


// Find the cell permeabilities from the Kozeny-Carman equation
__global__ void findDarcyPermeabilities(
        const Float k_c,                            // in
        const Float* __restrict__ dev_darcy_phi,    // in
        Float* __restrict__ dev_darcy_k)            // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    if (x < nx && y < ny && z < nz) {

        // 1D thread index
        const unsigned int cellidx = d_idx(x,y,z);

        __syncthreads();
        Float phi = dev_darcy_phi[cellidx];

        // avoid division by zero
        if (phi > 0.9999)
            phi = 0.9999;

        Float k = k_c*pow(phi,3)/pow(1.0 - phi, 2);

        /*printf("%d,%d,%d findK:\n"
                "\tphi    = %f\n"
                "\tk      = %e\n",
                x, y, z,
                phi, k);*/

        // limit permeability [m*m]
        // K_gravel = 3.0e-2 m/s => k_gravel = 2.7e-9 m*m
        //k = fmin(2.7e-9, k);
        k = fmin(2.7e-10, k);

        __syncthreads();
        dev_darcy_k[cellidx] = k;

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat("k", x, y, z, k);
#endif
    }
}

// Find the spatial gradients of the permeability.
__global__ void findDarcyPermeabilityGradients(
        const Float*  __restrict__ dev_darcy_k,   // in
        Float3* __restrict__ dev_darcy_grad_k)    // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell size
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    if (x < nx && y < ny && z < nz) {

        // 1D thread index
        const unsigned int cellidx = d_idx(x,y,z);

        // read values
        __syncthreads();
        const Float k_xn = dev_darcy_k[d_idx(x-1,y,z)];
        const Float k_xp = dev_darcy_k[d_idx(x+1,y,z)];
        const Float k_yn = dev_darcy_k[d_idx(x,y-1,z)];
        const Float k_yp = dev_darcy_k[d_idx(x,y+1,z)];
        const Float k_zn = dev_darcy_k[d_idx(x,y,z-1)];
        const Float k_zp = dev_darcy_k[d_idx(x,y,z+1)];

        // gradient approximated by first-order central difference
        const Float3 grad_k = MAKE_FLOAT3(
                (k_xp - k_xn)/(dx+dx),
                (k_yp - k_yn)/(dy+dy),
                (k_zp - k_zn)/(dz+dz));

        // write result
        __syncthreads();
        dev_darcy_grad_k[cellidx] = grad_k;

        /*printf("%d,%d,%d findK:\n"
                "\tk_x     = %e, %e\n"
                "\tk_y     = %e, %e\n"
                "\tk_z     = %e, %e\n"
                "\tgrad(k) = %e, %e, %e\n",
                x, y, z,
                k_xn, k_xp,
                k_yn, k_yp,
                k_zn, k_zp,
                grad_k.x, grad_k.y, grad_k.z);*/

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat3("grad_k", x, y, z, grad_k);
#endif
    }
}

// Find the temporal gradient in pressure using the backwards Euler method
__global__ void findDarcyPressureChange(
        const Float* __restrict__ dev_darcy_p_old,    // in
        const Float* __restrict__ dev_darcy_p,        // in
        const unsigned int iter,                      // in
        const unsigned int ndem,                      // in
        Float* __restrict__ dev_darcy_dpdt)           // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {

        // 1D thread index
        const unsigned int cellidx = d_idx(x,y,z);

        // read values
        __syncthreads();
        const Float p_old = dev_darcy_p_old[cellidx];
        const Float p     = dev_darcy_p[cellidx];

        Float dpdt = (p - p_old)/(ndem*devC_dt);

        // Ignore the large initial pressure gradients caused by solver "warm
        // up" towards hydrostatic pressure distribution
        if (iter < 2)
            dpdt = 0.0;

        // write result
        __syncthreads();
        dev_darcy_dpdt[cellidx] = dpdt;

        /*printf("%d,%d,%d\n"
                "\tp_old = %e\n"
                "\tp     = %e\n"
                "\tdt    = %e\n"
                "\tdpdt  = %e\n",
                x,y,z,
                p_old, p,
                devC_dt, dpdt);*/

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat("dpdt", x, y, z, dpdt);
#endif
    }
}

__global__ void firstDarcySolution(
        const Float*  __restrict__ dev_darcy_p,       // in
        const Float*  __restrict__ dev_darcy_k,       // in
        const Float*  __restrict__ dev_darcy_phi,     // in
        const Float*  __restrict__ dev_darcy_dphi,    // in
        const Float3* __restrict__ dev_darcy_grad_k,  // in
        const Float beta_f,                           // in
        const Float mu,                               // in
        const int bc_bot,                             // in
        const int bc_top,                             // in
        const unsigned int ndem,                      // in
        const unsigned int wall0_iz,                  // in
        Float* __restrict__ dev_darcy_dp_expl)        // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell size
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // Perform the epsilon updates for all non-ghost nodes except the Dirichlet
    // boundaries at z=0 and z=nz-1.
    // Adjust z range if a boundary has the Dirichlet boundary condition.
    /*int z_min = 0;
    int z_max = nz-1;
    if (bc_bot == 0)
        z_min = 1;
    if (bc_top == 0)
        z_max = nz-2;*/

    //if (x < nx && y < ny && z >= z_min && z <= z_max) {
    if (x < nx && y < ny && z < nz) {

        // 1D thread index
        const unsigned int cellidx = d_idx(x,y,z);

        // read values
        __syncthreads();
        const Float  k      = dev_darcy_k[cellidx];
        const Float3 grad_k = dev_darcy_grad_k[cellidx];
        const Float  phi    = dev_darcy_phi[cellidx];
        const Float  dphi   = dev_darcy_dphi[cellidx];

        const Float p_xn  = dev_darcy_p[d_idx(x-1,y,z)];
        const Float p     = dev_darcy_p[cellidx];
        const Float p_xp  = dev_darcy_p[d_idx(x+1,y,z)];
        const Float p_yn  = dev_darcy_p[d_idx(x,y-1,z)];
        const Float p_yp  = dev_darcy_p[d_idx(x,y+1,z)];
        Float p_zn = dev_darcy_p[d_idx(x,y,z-1)];
        Float p_zp = dev_darcy_p[d_idx(x,y,z+1)];

        // Neumann BCs
        if (z == 0 && bc_bot == 1)
            p_zn = p;
        if (z == nz-1 && bc_top == 1)
            p_zp = p;

        // upwind coefficients for grad(p) determined from values of k
        // k =  1.0: backwards difference
        // k = -1.0: forwards difference
        /*const Float3 e_k = MAKE_FLOAT3(
                copysign(1.0, grad_k.x),
                copysign(1.0, grad_k.y),
                copysign(1.0, grad_k.z));

        // gradient approximated by first-order forward differences
        const Float3 grad_p = MAKE_FLOAT3(
                ((1.0 + e_k.x)*(p - p_xn) + (1.0 - e_k.x)*(p_xp - p))/(dx + dx),
                ((1.0 + e_k.y)*(p - p_yn) + (1.0 - e_k.y)*(p_yp - p))/(dy + dy),
                ((1.0 + e_k.z)*(p - p_zn) + (1.0 - e_k.z)*(p_zp - p))/(dz + dz)
                );*/

        // gradient approximated by first-order central differences
        const Float3 grad_p = MAKE_FLOAT3(
                (p_xp - p_xn)/(dx + dx),
                (p_yp - p_yn)/(dy + dy),
                (p_zp - p_zn)/(dz + dz));

        // laplacian approximated by second-order central differences
        const Float laplace_p =
                (p_xp - (p + p) + p_xn)/(dx*dx) +
                (p_yp - (p + p) + p_yn)/(dy*dy) +
                (p_zp - (p + p) + p_zn)/(dz*dz);

        Float dp_expl =
            + (ndem*devC_dt)/(beta_f*phi*mu)*(k*laplace_p + dot(grad_k, grad_p))
            - dphi/(beta_f*phi*(1.0 - phi));

        // Dirichlet BC at dynamic top wall. wall0_iz will be larger than the
        // grid if the wall isn't dynamic
        if ((bc_bot == 0 && z == 0) || (bc_top == 0 && z == nz-1)
                || (z >= wall0_iz))
            dp_expl = 0.0;

#ifdef REPORT_FORCING_TERMS
            const Float dp_diff = (ndem*devC_dt)/(beta_f*phi*mu)
                *(k*laplace_p + dot(grad_k, grad_p));
            const Float dp_forc = -dphi/(beta_f*phi*(1.0 - phi));
        printf("\n%d,%d,%d updateDarcySolution\n"
                "p           = %e\n"
                "p_x         = %e, %e\n"
                "p_y         = %e, %e\n"
                "p_z         = %e, %e\n"
                "dp_expl     = %e\n"
                "laplace_p   = %e\n"
                "grad_p      = %e, %e, %e\n"
                "grad_k      = %e, %e, %e\n"
                "dp_diff     = %e\n"
                "dp_forc     = %e\n"
                "dphi        = %e\n"
                "dphi/dt     = %e\n"
                ,
                x,y,z,
                p,
                p_xn, p_xp,
                p_yn, p_yp,
                p_zn, p_zp,
                dp_expl,
                laplace_p,
                grad_p.x, grad_p.y, grad_p.z,
                grad_k.x, grad_k.y, grad_k.z,
                dp_diff, dp_forc,
                dphi, dphi/(ndem*devC_dt));
#endif

        // save explicit integrated pressure change
        __syncthreads();
        dev_darcy_dp_expl[cellidx] = dp_expl;

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat("dp_expl", x, y, z, dp_expl);
#endif
    }
}
// A single jacobi iteration where the pressure values are updated to
// dev_darcy_p_new.
// bc = 0: Dirichlet, 1: Neumann
__global__ void updateDarcySolution(
        const Float*  __restrict__ dev_darcy_p_old,   // in
        //const Float*  __restrict__ dev_darcy_dpdt,    // in
        const Float*  __restrict__ dev_darcy_dp_expl, // in
        const Float*  __restrict__ dev_darcy_p,       // in
        const Float*  __restrict__ dev_darcy_k,       // in
        const Float*  __restrict__ dev_darcy_phi,     // in
        const Float*  __restrict__ dev_darcy_dphi,    // in
        const Float3* __restrict__ dev_darcy_grad_k,  // in
        const Float beta_f,                           // in
        const Float mu,                               // in
        const int bc_bot,                             // in
        const int bc_top,                             // in
        const unsigned int ndem,                      // in
        const unsigned int wall0_iz,                  // in
        Float* __restrict__ dev_darcy_p_new,          // out
        Float* __restrict__ dev_darcy_norm)           // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell size
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // Perform the epsilon updates for all non-ghost nodes except the Dirichlet
    // boundaries at z=0 and z=nz-1.
    // Adjust z range if a boundary has the Dirichlet boundary condition.
    /*int z_min = 0;
    int z_max = nz-1;
    if (bc_bot == 0)
        z_min = 1;
    if (bc_top == 0)
        z_max = nz-2;*/

    //if (x < nx && y < ny && z >= z_min && z <= z_max) {
    if (x < nx && y < ny && z < nz) {

        // 1D thread index
        const unsigned int cellidx = d_idx(x,y,z);

        // read values
        __syncthreads();
        const Float k     = dev_darcy_k[cellidx];
        const Float3 grad_k = dev_darcy_grad_k[cellidx];
        const Float  phi    = dev_darcy_phi[cellidx];
        const Float  dphi   = dev_darcy_dphi[cellidx];

        const Float p_old   = dev_darcy_p_old[cellidx];
        const Float dp_expl = dev_darcy_dp_expl[cellidx];

        const Float p_xn  = dev_darcy_p[d_idx(x-1,y,z)];
        const Float p     = dev_darcy_p[cellidx];
        const Float p_xp  = dev_darcy_p[d_idx(x+1,y,z)];
        const Float p_yn  = dev_darcy_p[d_idx(x,y-1,z)];
        const Float p_yp  = dev_darcy_p[d_idx(x,y+1,z)];
        Float p_zn = dev_darcy_p[d_idx(x,y,z-1)];
        Float p_zp = dev_darcy_p[d_idx(x,y,z+1)];

        // Neumann BCs
        if (z == 0 && bc_bot == 1)
            p_zn = p;
        if (z == nz-1 && bc_top == 1)
            p_zp = p;

        // upwind coefficients for grad(p) determined from values of k
        // k =  1.0: backwards difference
        // k = -1.0: forwards difference
        /*const Float3 e_k = MAKE_FLOAT3(
                copysign(1.0, grad_k.x),
                copysign(1.0, grad_k.y),
                copysign(1.0, grad_k.z));

        // gradient approximated by first-order forward differences
        const Float3 grad_p = MAKE_FLOAT3(
                ((1.0 + e_k.x)*(p - p_xn) + (1.0 - e_k.x)*(p_xp - p))/(dx + dx),
                ((1.0 + e_k.y)*(p - p_yn) + (1.0 - e_k.y)*(p_yp - p))/(dy + dy),
                ((1.0 + e_k.z)*(p - p_zn) + (1.0 - e_k.z)*(p_zp - p))/(dz + dz)
                );*/

        // gradient approximated by first-order central differences
        const Float3 grad_p = MAKE_FLOAT3(
                (p_xp - p_xn)/(dx + dx),
                (p_yp - p_yn)/(dy + dy),
                (p_zp - p_zn)/(dz + dz));

        // laplacian approximated by second-order central differences
        const Float laplace_p =
                (p_xp - (p + p) + p_xn)/(dx*dx) +
                (p_yp - (p + p) + p_yn)/(dy*dy) +
                (p_zp - (p + p) + p_zn)/(dz*dz);

        //Float p_new = p_old
        Float dp_impl =
            + (ndem*devC_dt)/(beta_f*phi*mu)*(k*laplace_p + dot(grad_k, grad_p))
            - dphi/(beta_f*phi*(1.0 - phi));

        // Dirichlet BC at dynamic top wall. wall0_iz will be larger than the
        // grid if the wall isn't dynamic
        if ((bc_bot == 0 && z == 0) || (bc_top == 0 && z == nz-1)
                || (z >= wall0_iz))
            dp_impl = 0.0;
            //p_new = p;

        // choose integration method, parameter in [0.0; 1.0]
        //    epsilon = 0:   explicit
        //    epsilon = 0.5: Crank-Nicolson
        //    epsilon = 1:   implicit
        const Float epsilon = 1.0;
        Float p_new = p_old + (1.0 - epsilon)*dp_expl + epsilon*dp_impl;

        // add underrelaxation
        const Float theta = 0.1;
        p_new = p*(1.0 - theta) + p_new*theta;

        // normalized residual, avoid division by zero
        //const Float res_norm = (p_new - p)*(p_new - p)/(p_new*p_new + 1.0e-16);
        const Float res_norm = (p_new - p)/(p + 1.0e-16);

#ifdef REPORT_FORCING_TERMS
            const Float dp_diff = (ndem*devC_dt)/(beta_f*phi*mu)
                *(k*laplace_p + dot(grad_k, grad_p));
            const Float dp_forc = -dphi/(beta_f*phi*(1.0 - phi));
        printf("\n%d,%d,%d updateDarcySolution\n"
                "p_new       = %e\n"
                "p           = %e\n"
                "p_x         = %e, %e\n"
                "p_y         = %e, %e\n"
                "p_z         = %e, %e\n"
                "dp_expl     = %e\n"
                "p_old       = %e\n"
                "laplace_p   = %e\n"
                "grad_p      = %e, %e, %e\n"
                "grad_k      = %e, %e, %e\n"
                "dp_diff     = %e\n"
                "dp_forc     = %e\n"
                "dphi        = %e\n"
                "dphi/dt     = %e\n"
                "res_norm    = %e\n"
                ,
                x,y,z,
                p_new, p,
                p_xn, p_xp,
                p_yn, p_yp,
                p_zn, p_zp,
                dp_expl,
                p_old,
                laplace_p,
                grad_p.x, grad_p.y, grad_p.z,
                grad_k.x, grad_k.y, grad_k.z,
                dp_diff, dp_forc,
                dphi, dphi/(ndem*devC_dt),
                res_norm);
#endif

        // save new pressure and the residual
        __syncthreads();
        dev_darcy_p_new[cellidx] = p_new;
        dev_darcy_norm[cellidx]  = res_norm;

        /*printf("%d,%d,%d\tp = % f\tp_new = % f\tres_norm = % f\n",
                x,y,z,
                p,
                p_new,
                res_norm);*/

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat("p_new", x, y, z, p_new);
        checkFiniteFloat("res_norm", x, y, z, res_norm);
#endif
    }
}

__global__ void findNewPressure(
        const Float* __restrict__ dev_darcy_dp,     // in
        Float* __restrict__ dev_darcy_p)            // in+out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        const unsigned int cellidx = d_idx(x,y,z);

        const Float dp = dev_darcy_dp[cellidx];

        // save new pressure
        __syncthreads();
        dev_darcy_p[cellidx] += dp;

        /*printf("%d,%d,%d\tp = % f\tp_new = % f\tres_norm = % f\n",
                x,y,z,
                p,
                p_new,
                res_norm);*/

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat("dp", x, y, z, dp);
#endif
    }
}

// Find cell velocities
__global__ void findDarcyVelocities(
        const Float* __restrict__ dev_darcy_p,      // in
        const Float* __restrict__ dev_darcy_phi,    // in
        const Float* __restrict__ dev_darcy_k,      // in
        const Float mu,                             // in
        Float3* __restrict__ dev_darcy_v)           // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell size
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        const unsigned int cellidx = d_idx(x,y,z);

        __syncthreads();
        const Float p_xn = dev_darcy_p[d_idx(x-1,y,z)];
        const Float p_xp = dev_darcy_p[d_idx(x+1,y,z)];
        const Float p_yn = dev_darcy_p[d_idx(x,y-1,z)];
        const Float p_yp = dev_darcy_p[d_idx(x,y+1,z)];
        const Float p_zn = dev_darcy_p[d_idx(x,y,z-1)];
        const Float p_zp = dev_darcy_p[d_idx(x,y,z+1)];

        const Float k   = dev_darcy_k[cellidx];
        const Float phi = dev_darcy_phi[cellidx];

        // approximate pressure gradient with first order central differences
        const Float3 grad_p = MAKE_FLOAT3(
                (p_xp - p_xn)/(dx + dx),
                (p_yp - p_yn)/(dy + dy),
                (p_zp - p_zn)/(dz + dz));

        // Flux [m/s]: q = -k/nu * dH
        // Pore velocity [m/s]: v = q/n

        // calculate flux
        //const Float3 q = -k/mu*grad_p;

        // calculate velocity
        //const Float3 v = q/phi;
        const Float3 v = (-k/mu * grad_p)/phi;

        // Save velocity
        __syncthreads();
        dev_darcy_v[cellidx] = v;
    }
}

// Print final heads and free memory
void DEM::endDarcyDev()
{
    freeDarcyMemDev();
}

// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
