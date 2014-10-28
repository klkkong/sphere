// navierstokes.cuh
// CUDA implementation of porous flow

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
    unsigned int memSizeFface = sizeof(Float)*darcyCellsVelocity();

    cudaMalloc((void**)&dev_darcy_p, memSizeF);     // hydraulic pressure
    cudaMalloc((void**)&dev_darcy_p_old, memSizeF); // prev. hydraulic pressure
    cudaMalloc((void**)&dev_darcy_v, memSizeF*3);   // cell hydraulic velocity
    cudaMalloc((void**)&dev_darcy_v_p, memSizeF*3); // predicted cell velocity
    cudaMalloc((void**)&dev_darcy_vp_avg, memSizeF*3); // avg. particle velocity
    cudaMalloc((void**)&dev_darcy_d_avg, memSizeF); // avg. particle diameter
    cudaMalloc((void**)&dev_darcy_phi, memSizeF);   // cell porosity
    cudaMalloc((void**)&dev_darcy_dphi, memSizeF);  // cell porosity change
    cudaMalloc((void**)&dev_darcy_norm, memSizeF);  // normalized residual
    cudaMalloc((void**)&dev_darcy_f_p, sizeof(Float4)*np); // pressure force
    cudaMalloc((void**)&dev_darcy_k, memSizeF);        // hydraulic permeability
    cudaMalloc((void**)&dev_darcy_grad_k, memSizeF3);  // grad(permeability)
    cudaMalloc((void**)&dev_darcy_div_v_p, memSizeF3); // divergence(v_p)

    checkForCudaErrors("End of initDarcyMemDev");
}

// Free memory
void DEM::freeDarcyMemDev()
{
    cudaFree(dev_darcy_p);
    cudaFree(dev_darcy_p_old);
    cudaFree(dev_darcy_v);
    cudaFree(dev_darcy_vp_avg);
    cudaFree(dev_darcy_d_avg);
    cudaFree(dev_darcy_phi);
    cudaFree(dev_darcy_dphi);
    cudaFree(dev_darcy_norm);
    cudaFree(dev_darcy_f_p);
    cudaFree(dev_darcy_k);
    cudaFree(dev_darcy_grad_k);
    cudaFree(dev_darcy_div_v_p);
}

// Transfer to device
void DEM::transferDarcyToGlobalDeviceMemory(int statusmsg)
{
    checkForCudaErrors("Before attempting cudaMemcpy in "
                       "transferDarcyToGlobalDeviceMemory");

    //if (verbose == 1 && statusmsg == 1)
    //std::cout << "  Transfering fluid data to the device:           ";

    // memory size for a scalar field
    unsigned int memSizeF  = sizeof(Float)*NScells();

    //writeNSarray(ns.p, "ns.p.txt");

    cudaMemcpy(dev_darcy_p, darcy.p, memSizeF, cudaMemcpyHostToDevice);
    checkForCudaErrors("transferNStoGlobalDeviceMemory after first cudaMemcpy");
    cudaMemcpy(dev_darcy_v, darcy.v, memSizeF*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_darcy_phi, darcy.phi, memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_darcy_dphi, darcy.dphi, memSizeF, cudaMemcpyHostToDevice);

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
    cudaMemcpy(darcy.v_x, dev_darcy_v_x, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(darcy.v_y, dev_darcy_v_y, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(darcy.v_z, dev_darcy_v_z, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(darcy.phi, dev_darcy_phi, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(darcy.dphi, dev_darcy_dphi, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(darcy.norm, dev_darcy_norm, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(darcy.f_p, dev_darcy_f_p, sizeof(Float4)*np,
            cudaMemcpyDeviceToHost);

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

// Find averaged cell velocities from cell-face velocities. This function works
// for both normal and predicted velocities. Launch for every cell in the
// dev_darcy_v array. This function does not set the averaged
// velocity values in the ghost node cells.
__global__ void findDarcyAvgVel(
    Float3* __restrict__ dev_darcy_v,    // out
    const Float* __restrict__ dev_darcy_v_x,  // in
    const Float* __restrict__ dev_darcy_v_y,  // in
    const Float* __restrict__ dev_darcy_v_z)  // in
{

    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // check that we are not outside the fluid grid
    if (x<devC_grid.num[0] && y<devC_grid.num[1] && z<devC_grid.num[2]-1) {
        const unsigned int cellidx = d_idx(x,y,z);

        // Read cell-face velocities
        __syncthreads();
        const Float v_xn = dev_darcy_v_x[d_vidx(x,y,z)];
        const Float v_xp = dev_darcy_v_x[d_vidx(x+1,y,z)];
        const Float v_yn = dev_darcy_v_y[d_vidx(x,y,z)];
        const Float v_yp = dev_darcy_v_y[d_vidx(x,y+1,z)];
        const Float v_zn = dev_darcy_v_z[d_vidx(x,y,z)];
        const Float v_zp = dev_darcy_v_z[d_vidx(x,y,z+1)];

        // Find average velocity using arithmetic means
        const Float3 v_bar = MAKE_FLOAT3(
            amean(v_xn, v_xp),
            amean(v_yn, v_yp),
            amean(v_zn, v_zp));

        // Save value
        __syncthreads();
        dev_darcy_v[d_idx(x,y,z)] = v_bar;
    }
}

// Find cell-face velocities from averaged velocities. This function works for
// both normal and predicted velocities. Launch for every cell in the
// dev_darcy_v array. Make sure that the averaged velocity ghost
// nodes are set
// beforehand.
__global__ void findDarcyCellFaceVel(
    const Float3* __restrict__ dev_darcy_v,    // in
    Float* __restrict__ dev_darcy_v_x,  // out
    Float* __restrict__ dev_darcy_v_y,  // out
    Float* __restrict__ dev_darcy_v_z)  // out
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
    if (x < nx && y < ny && x < nz) {
        const unsigned int cellidx = d_idx(x,y,z);

        // Read the averaged velocity from this cell as well as the required
        // components from the neighbor cells
        __syncthreads();
        const Float3 v = dev_darcy_v[d_idx(x,y,z)];
        const Float v_xn = dev_darcy_v[d_idx(x-1,y,z)].x;
        const Float v_xp = dev_darcy_v[d_idx(x+1,y,z)].x;
        const Float v_yn = dev_darcy_v[d_idx(x,y-1,z)].y;
        const Float v_yp = dev_darcy_v[d_idx(x,y+1,z)].y;
        const Float v_zn = dev_darcy_v[d_idx(x,y,z-1)].z;
        const Float v_zp = dev_darcy_v[d_idx(x,y,z+1)].z;

        // Find cell-face velocities and save them right away
        __syncthreads();

        // Values at the faces closest to the coordinate system origo
        dev_darcy_v_x[d_vidx(x,y,z)] = amean(v_xn, v.x);
        dev_darcy_v_y[d_vidx(x,y,z)] = amean(v_yn, v.y);
        dev_darcy_v_z[d_vidx(x,y,z)] = amean(v_zn, v.z);

        // Values at the cell faces furthest from the coordinate system origo.
        // These values should only be written at the corresponding boundaries
        // in order to avoid write conflicts.
        if (x == nx-1)
            dev_darcy_v_x[d_vidx(x+1,y,z)] = amean(v.x, v_xp);
        if (y == ny-1)
            dev_darcy_v_x[d_vidx(x+1,y,z)] = amean(v.y, v_yp);
        if (z == nz-1)
            dev_darcy_v_x[d_vidx(x+1,y,z)] = amean(v.z, v_zp);
    }
}


// The normalized residuals are given an initial value of 0, since the values at
// the Dirichlet boundaries aren't written during the iterations.
__global__ void setDarcyNormZero(Float* __restrict__ dev_darcy_norm)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // check that we are not outside the fluid grid
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {
        __syncthreads();
        const unsigned int cellidx = d_idx(x,y,z);
        dev_darcy_norm[d_idx(x,y,z)]    = 0.0;
    }
}


// Copy the values from one cell to another
__device__ void copyDarcyValsDev(
    const unsigned int read,
    const unsigned int write,
    Float*  __restrict__ dev_darcy_p,
    Float3* __restrict__ dev_darcy_v,
    Float*  __restrict__ dev_darcy_phi,
    Float*  __restrict__ dev_darcy_dphi)
{
    // Coalesced read
    const Float  p       = dev_darcy_p[read];
    const Float3 v       = dev_darcy_v[read];
    const Float  phi     = dev_darcy_phi[read];
    const Float  dphi    = dev_darcy_dphi[read];

    // Coalesced write
    __syncthreads();
    dev_darcy_p[write]       = p;
    dev_darcy_v[write]       = v;
    dev_darcy_phi[write]     = phi;
    dev_darcy_dphi[write]    = dphi;
}


// Update ghost nodes from their parent cell values. The edge (diagonal) cells
// are not written since they are not read. Launch this kernel for all cells in
// the grid
__global__ void setDarcyGhostNodesDev(
    Float*  __restrict__ dev_darcy_p,
    Float3* __restrict__ dev_darcy_v,
    Float*  __restrict__ dev_darcy_phi,
    Float*  __restrict__ dev_darcy_dphi,
    Float*  __restrict__ dev_darcy_epsilon)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // 1D thread index
    const unsigned int cellidx = d_idx(x,y,z);

    // 1D position of ghost node
    unsigned int writeidx;

    // check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        if (x == 0) {
            writeidx = d_idx(nx,y,z);
            copyDarcyValsDev(cellidx, writeidx,
                          dev_darcy_p, dev_darcy_v,
                          dev_darcy_phi, dev_darcy_dphi);
        }
        if (x == nx-1) {
            writeidx = d_idx(-1,y,z);
            copyDarcyValsDev(cellidx, writeidx,
                          dev_darcy_p, dev_darcy_v,
                          dev_darcy_phi, dev_darcy_dphi);
        }

        if (y == 0) {
            writeidx = d_idx(x,ny,z);
            copyDarcyValsDev(cellidx, writeidx,
                          dev_darcy_p, dev_darcy_v,
                          dev_darcy_phi, dev_darcy_dphi);
        }
        if (y == ny-1) {
            writeidx = d_idx(x,-1,z);
            copyDarcyValsDev(cellidx, writeidx,
                          dev_darcy_p, dev_darcy_v,
                          dev_darcy_phi, dev_darcy_dphi);
        }

        // Z boundaries fixed
        if (z == 0) {
            writeidx = d_idx(x,y,-1);
            copyDarcyValsDev(cellidx, writeidx,
                          dev_darcy_p, dev_darcy_v,
                          dev_darcy_phi, dev_darcy_dphi);
        }
        if (z == nz-1) {
            writeidx = d_idx(x,y,nz);
            copyDarcyValsDev(cellidx, writeidx,
                          dev_darcy_p, dev_darcy_v,
                          dev_darcy_phi, dev_darcy_dphi);
        }

        // Z boundaries periodic
        /*if (z == 0) {
          writeidx = d_idx(x,y,nz);
          copyNSvalsDev(cellidx, writeidx,
          dev_ns_p,
          dev_ns_v, dev_ns_v_p,
          dev_ns_phi, dev_ns_dphi,
          dev_ns_epsilon);
          }
          if (z == nz-1) {
          writeidx = d_idx(x,y,-1);
          copyNSvalsDev(cellidx, writeidx,
          dev_ns_p,
          dev_ns_v, dev_ns_v_p,
          dev_ns_phi, dev_ns_dphi,
          dev_ns_epsilon);
          }*/
    }
}

// Update a field in the ghost nodes from their parent cell values. The edge
// (diagonal) cells are not written since they are not read. Launch this kernel
// for all cells in the grid using
// setDarcyGhostNodes<datatype><<<.. , ..>>>( .. );
template<typename T>
__global__ void setDarcyGhostNodes(T* __restrict__ dev_scalarfield)
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

        if (x == 0)
            dev_scalarfield[d_idx(nx,y,z)] = val;
        if (x == nx-1)
            dev_scalarfield[d_idx(-1,y,z)] = val;

        if (y == 0)
            dev_scalarfield[d_idx(x,ny,z)] = val;
        if (y == ny-1)
            dev_scalarfield[d_idx(x,-1,z)] = val;

        if (z == 0)
            dev_scalarfield[d_idx(x,y,-1)] = val;     // Dirichlet
        //dev_scalarfield[d_idx(x,y,nz)] = val;    // Periodic -z
        if (z == nz-1)
            dev_scalarfield[d_idx(x,y,nz)] = val;     // Dirichlet
        //dev_scalarfield[d_idx(x,y,-1)] = val;    // Periodic +z
    }
}

// Update a field in the ghost nodes from their parent cell values. The edge
// (diagonal) cells are not written since they are not read.
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
            dev_scalarfield[d_idx(nx,y,z)] = val;
        if (x == nx-1)
            dev_scalarfield[d_idx(-1,y,z)] = val;

        // y
        if (y == 0)
            dev_scalarfield[d_idx(x,ny,z)] = val;
        if (y == ny-1)
            dev_scalarfield[d_idx(x,-1,z)] = val;

        // z
        if (z == 0 && bc_bot == 0)
            dev_scalarfield[d_idx(x,y,-1)] = val;     // Dirichlet
        //if (z == 1 && bc_bot == 1)
        if (z == 0 && bc_bot == 1)
            dev_scalarfield[d_idx(x,y,-1)] = val;     // Neumann
        if (z == 0 && bc_bot == 2)
            dev_scalarfield[d_idx(x,y,nz)] = val;     // Periodic -z

        if (z == nz-1 && bc_top == 0)
            dev_scalarfield[d_idx(x,y,nz)] = val;     // Dirichlet
        if (z == nz-2 && bc_top == 1)
            dev_scalarfield[d_idx(x,y,nz)] = val;     // Neumann
        if (z == nz-1 && bc_top == 2)
            dev_scalarfield[d_idx(x,y,-1)] = val;     // Periodic +z
    }
}

// Update a field in the ghost nodes from their parent cell values. The edge
// (diagonal) cells are not written since they are not read.
// Launch per face.
// According to Griebel et al. 1998 "Numerical Simulation in Fluid Dynamics"
template<typename T>
__global__ void setDarcyGhostNodesFace(
    T* __restrict__ dev_scalarfield_x,
    T* __restrict__ dev_scalarfield_y,
    T* __restrict__ dev_scalarfield_z,
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
    //if (x <= nx && y <= ny && z <= nz) {
    if (x < nx && y < ny && z < nz) {

        const T val_x = dev_scalarfield_x[d_vidx(x,y,z)];
        const T val_y = dev_scalarfield_y[d_vidx(x,y,z)];
        const T val_z = dev_scalarfield_z[d_vidx(x,y,z)];

        // x (periodic)
        if (x == 0) {
            dev_scalarfield_x[d_vidx(nx,y,z)] = val_x;
            dev_scalarfield_y[d_vidx(nx,y,z)] = val_y;
            dev_scalarfield_z[d_vidx(nx,y,z)] = val_z;
        }
        if (x == 1) {
            dev_scalarfield_x[d_vidx(nx+1,y,z)] = val_x;
        }
        if (x == nx-1) {
            dev_scalarfield_x[d_vidx(-1,y,z)] = val_x;
            dev_scalarfield_y[d_vidx(-1,y,z)] = val_y;
            dev_scalarfield_z[d_vidx(-1,y,z)] = val_z;
        }

        // z ghost nodes at x = -1 and z = nz,
        // equal to the ghost node at x = nx-1 and z = nz
        if (z == nz-1 && x == nx-1 && bc_top == 0) // Dirichlet +z
            dev_scalarfield_z[d_vidx(-1,y,nz)] = val_z;

        if (z == nz-1 && x == nx-1 && (bc_top == 1 || bc_top == 2)) //Neumann +z
            dev_scalarfield_z[d_vidx(-1,y,nz)] = 0.0;

        if (z == 0 && x == nx-1 && bc_top == 3) // Periodic +z
            dev_scalarfield_z[d_vidx(-1,y,nz)] = val_z;

        // z ghost nodes at y = -1 and z = nz,
        // equal to the ghost node at y = ny-1 and z = nz
        if (z == nz-1 && y == ny-1 && bc_top == 0) // Dirichlet +z
            dev_scalarfield_z[d_vidx(x,-1,nz)] = val_z;

        if (z == nz-1 && y == ny-1 && (bc_top == 1 || bc_top == 2)) //Neumann +z
            dev_scalarfield_z[d_vidx(x,-1,nz)] = 0.0;

        if (z == 0 && y == ny-1 && bc_top == 3) // Periodic +z
            dev_scalarfield_z[d_vidx(x,-1,nz)] = val_z;


        // x ghost nodes at x = nx and z = -1,
        // equal to the ghost nodes at x = 0 and z = -1
        // Dirichlet, Neumann free slip or periodic -z
        if (z == 0 && x == 0 && (bc_bot == 0 || bc_bot == 1 || bc_bot == 3))
            dev_scalarfield_x[d_vidx(nx,y,-1)] = val_x;

        if (z == 0 && x == 0 && bc_bot == 2) // Neumann no slip -z
            dev_scalarfield_x[d_vidx(nx,y,-1)] = -val_x;

        // y ghost nodes at y = ny and z = -1,
        // equal to the ghost node at x = 0 and z = -1
        // Dirichlet, Neumann free slip or periodic -z
        if (z == 0 && y == 0 && (bc_bot == 0 || bc_bot == 1 || bc_bot == 3))
            dev_scalarfield_y[d_vidx(x,ny,-1)] = val_y;

        if (z == 0 && y == 0 && bc_bot == 2) // Neumann no slip -z
            dev_scalarfield_y[d_vidx(x,ny,-1)] = -val_y;


        // z ghost nodes at x = nx and z = nz
        // equal to the ghost node at x = 0 and z = nz
        if (z == nz-1 && x == 0 && (bc_top == 0 || bc_top == 3)) // D. or p. +z
            dev_scalarfield_z[d_vidx(nx,y,nz)] = val_z;

        if (z == nz-1 && x == 0 && (bc_top == 1 || bc_top == 2)) // N. +z
            dev_scalarfield_z[d_vidx(nx,y,nz)] = 0.0;

        // z ghost nodes at y = ny and z = nz
        // equal to the ghost node at y = 0 and z = nz
        if (z == nz-1 && y == 0 && (bc_top == 0 || bc_top == 3)) // D. or p. +z
            dev_scalarfield_z[d_vidx(x,ny,nz)] = val_z;

        if (z == nz-1 && y == 0 && (bc_top == 1 || bc_top == 2)) // N. +z
            dev_scalarfield_z[d_vidx(x,ny,nz)] = 0.0;


        // x ghost nodes at x = nx and z = nz,
        // equal to the ghost nodes at x = 0 and z = nz
        // Dirichlet, Neumann free slip or periodic +z
        if (z == nz-1 && x == 0 && (bc_bot == 0 || bc_bot == 1 || bc_bot == 3))
            dev_scalarfield_x[d_vidx(nx,y,nz)] = val_x;

        if (z == nz-1 && x == 0 && bc_bot == 2) // Neumann no slip -z
            dev_scalarfield_x[d_vidx(nx,y,nz)] = -val_x;

        // y ghost nodes at y = ny and z = nz,
        // equal to the ghost nodes at y = 0 and z = nz
        // Dirichlet, Neumann free slip or periodic +z
        if (z == nz-1 && y == 0 && (bc_bot == 0 || bc_bot == 1 || bc_bot == 3))
            dev_scalarfield_y[d_vidx(x,ny,nz)] = val_y;

        if (z == nz-1 && y == 0 && bc_bot == 2) // Neumann no slip -z
            dev_scalarfield_y[d_vidx(x,ny,nz)] = -val_y;


        // y (periodic)
        if (y == 0) {
            dev_scalarfield_x[d_vidx(x,ny,z)] = val_x;
            dev_scalarfield_y[d_vidx(x,ny,z)] = val_y;
            dev_scalarfield_z[d_vidx(x,ny,z)] = val_z;
        }
        if (y == 1) {
            dev_scalarfield_y[d_vidx(x,ny+1,z)] = val_y;
        }
        if (y == ny-1) {
            dev_scalarfield_x[d_vidx(x,-1,z)] = val_x;
            dev_scalarfield_y[d_vidx(x,-1,z)] = val_y;
            dev_scalarfield_z[d_vidx(x,-1,z)] = val_z;
        }

        // z
        if (z == 0 && bc_bot == 0) {
            dev_scalarfield_x[d_vidx(x,y,-1)] = val_y;     // Dirichlet -z
            dev_scalarfield_y[d_vidx(x,y,-1)] = val_x;     // Dirichlet -z
            dev_scalarfield_z[d_vidx(x,y,-1)] = val_z;     // Dirichlet -z
        }
        if (z == 0 && bc_bot == 1) {
            //dev_scalarfield_x[d_vidx(x,y,-1)] = val_x;   // Neumann free slip -z
            //dev_scalarfield_y[d_vidx(x,y,-1)] = val_y;   // Neumann free slip -z
            //dev_scalarfield_z[d_vidx(x,y,-1)] = val_z;   // Neumann free slip -z
            dev_scalarfield_x[d_vidx(x,y,-1)] = val_x;     // Neumann free slip -z
            dev_scalarfield_y[d_vidx(x,y,-1)] = val_y;     // Neumann free slip -z
            dev_scalarfield_z[d_vidx(x,y,-1)] = 0.0;       // Neumann free slip -z
        }
        if (z == 0 && bc_bot == 2) {
            //dev_scalarfield_x[d_vidx(x,y,-1)] = val_x;     // Neumann no slip -z
            //dev_scalarfield_y[d_vidx(x,y,-1)] = val_y;     // Neumann no slip -z
            //dev_scalarfield_z[d_vidx(x,y,-1)] = val_z;     // Neumann no slip -z
            dev_scalarfield_x[d_vidx(x,y,-1)] = -val_x;    // Neumann no slip -z
            dev_scalarfield_y[d_vidx(x,y,-1)] = -val_y;    // Neumann no slip -z
            dev_scalarfield_z[d_vidx(x,y,-1)] = 0.0;       // Neumann no slip -z
        }
        if (z == 0 && bc_bot == 3) {
            dev_scalarfield_x[d_vidx(x,y,nz)] = val_x;     // Periodic -z
            dev_scalarfield_y[d_vidx(x,y,nz)] = val_y;     // Periodic -z
            dev_scalarfield_z[d_vidx(x,y,nz)] = val_z;     // Periodic -z
        }
        if (z == 1 && bc_bot == 3) {
            dev_scalarfield_z[d_vidx(x,y,nz+1)] = val_z;   // Periodic -z
        }

        if (z == nz-1 && bc_top == 0) {
            dev_scalarfield_z[d_vidx(x,y,nz)] = val_z;     // Dirichlet +z
        }
        if (z == nz-1 && bc_top == 1) {
            //dev_scalarfield_x[d_vidx(x,y,nz)] = val_x;   // Neumann free slip +z
            //dev_scalarfield_y[d_vidx(x,y,nz)] = val_y;   // Neumann free slip +z
            //dev_scalarfield_z[d_vidx(x,y,nz)] = val_z;   // Neumann free slip +z
            //dev_scalarfield_z[d_vidx(x,y,nz+1)] = val_z; // Neumann free slip +z
            dev_scalarfield_x[d_vidx(x,y,nz)] = val_x;     // Neumann free slip +z
            dev_scalarfield_y[d_vidx(x,y,nz)] = val_y;     // Neumann free slip +z
            dev_scalarfield_z[d_vidx(x,y,nz)] = 0.0;     // Neumann free slip +z
            dev_scalarfield_z[d_vidx(x,y,nz+1)] = 0.0;   // Neumann free slip +z
        }
        if (z == nz-1 && bc_top == 2) {
            //dev_scalarfield_x[d_vidx(x,y,nz)] = val_x;     // Neumann no slip +z
            //dev_scalarfield_y[d_vidx(x,y,nz)] = val_y;     // Neumann no slip +z
            //dev_scalarfield_z[d_vidx(x,y,nz)] = val_z;     // Neumann no slip +z
            //dev_scalarfield_z[d_vidx(x,y,nz+1)] = val_z;   // Neumann no slip +z
            dev_scalarfield_x[d_vidx(x,y,nz)] = -val_x;    // Neumann no slip +z
            dev_scalarfield_y[d_vidx(x,y,nz)] = -val_y;    // Neumann no slip +z
            dev_scalarfield_z[d_vidx(x,y,nz)] = 0.0;       // Neumann no slip +z
            dev_scalarfield_z[d_vidx(x,y,nz+1)] = 0.0;     // Neumann no slip +z
        }
        if (z == nz-1 && bc_top == 3) {
            dev_scalarfield_x[d_vidx(x,y,-1)] = val_x;     // Periodic +z
            dev_scalarfield_y[d_vidx(x,y,-1)] = val_y;     // Periodic +z
            dev_scalarfield_z[d_vidx(x,y,-1)] = val_z;     // Periodic +z
        }
    }
}

// Find the porosity in each cell on the base of a sphere, centered at the cell
// center. 
__global__ void findPorositiesVelocitiesDiametersSpherical(
    const unsigned int* __restrict__ dev_cellStart,
    const unsigned int* __restrict__ dev_cellEnd,
    const Float4* __restrict__ dev_x_sorted,
    const Float4* __restrict__ dev_vel_sorted,
    Float*  __restrict__ dev_darcy_phi,
    Float*  __restrict__ dev_darcy_dphi,
    Float3* __restrict__ dev_darcy_vp_avg,
    Float*  __restrict__ dev_darcy_d_avg,
    const unsigned int iteration,
    const unsigned int np,
    const Float c_phi)
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
    //const Float R = fmin(dx, fmin(dy,dz)) * 0.5; // diameter = cell width
    const Float R = fmin(dx, fmin(dy,dz));       // diameter = 2*cell width
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
            Float4 v;
            unsigned int n = 0;

            Float3 v_avg = MAKE_FLOAT3(0.0, 0.0, 0.0);
            Float  d_avg = 0.0;

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
            /*for (int z_dim=-1; z_dim<2; ++z_dim) { // z-axis
              for (int y_dim=-1; y_dim<2; ++y_dim) { // y-axis
              for (int x_dim=-1; x_dim<2; ++x_dim) { // x-axis*/

            // Iterate over 27 neighbor cells, R = 2*cell width
            for (int z_dim=-2; z_dim<3; ++z_dim) { // z-axis
                //for (int z_dim=-1; z_dim<2; ++z_dim) { // z-axis
                for (int y_dim=-2; y_dim<3; ++y_dim) { // y-axis
                    for (int x_dim=-2; x_dim<3; ++x_dim) { // x-axis

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
                                    v  = dev_vel_sorted[i];
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
                                        v_avg += MAKE_FLOAT3(v.x, v.y, v.z);
                                        d_avg += 2.0*r;
                                        n++;
                                    }

                                    // Particle fully contained in cell sphere
                                    if (d <= R - r) {
                                        void_volume -= 4.0/3.0*M_PI*r*r*r;
                                        v_avg += MAKE_FLOAT3(v.x, v.y, v.z);
                                        d_avg += 2.0*r;
                                        n++;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (phi < 0.999) {
                v_avg /= n;
                d_avg /= n;
            }

            // Make sure that the porosity is in the interval [0.0;1.0]
            phi = fmin(1.00, fmax(0.00, void_volume/cell_volume));
            //phi = void_volume/cell_volume;

            Float dphi = phi - phi_0;
            if (iteration == 0)
                dphi = 0.0;

            // report values to stdout for debugging
            //printf("%d,%d,%d\tphi = %f dphi = %f v_avg = %f,%f,%f d_avg = %f\n",
            //       x,y,z, phi, dphi, v_avg.x, v_avg.y, v_avg.z, d_avg);

            // Save porosity, porosity change, average velocity and average diameter
            __syncthreads();
            //phi = 0.5; dphi = 0.0; // disable porosity effects
            const unsigned int cellidx = d_idx(x,y,z);
            dev_darcy_phi[cellidx]  = phi*c_phi;
            dev_darcy_dphi[cellidx] = dphi*c_phi;
            dev_darcy_vp_avg[cellidx] = v_avg;
            dev_darcy_d_avg[cellidx]  = d_avg;

#ifdef CHECK_FLUID_FINITE
            (void)checkFiniteFloat("phi", x, y, z, phi);
            (void)checkFiniteFloat("dphi", x, y, z, dphi);
            (void)checkFiniteFloat3("v_avg", x, y, z, v_avg);
            (void)checkFiniteFloat("d_avg", x, y, z, d_avg);
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
            dev_darcy_phi[cellidx]  = 1.0;
            dev_darcy_dphi[cellidx] = 0.0;

            dev_darcy_vp_avg[cellidx] = MAKE_FLOAT3(0.0, 0.0, 0.0);
            dev_darcy_d_avg[cellidx]  = 0.0;
        }
    }
}


// Find the spatial gradient in e.g. pressures per cell
// using first order central differences
__global__ void findDarcyGradientsDev(
    const Float* __restrict__ dev_scalarfield,     // in
    Float3* __restrict__ dev_vectorfield)    // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Grid sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx = d_idx(x,y,z);

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        const Float3 grad = gradient(dev_scalarfield, x, y, z, dx, dy, dz);

        // Write gradient
        __syncthreads();
        dev_vectorfield[cellidx] = grad;

#ifdef CHECK_FLUID_FINITE
        (void)checkFiniteFloat3("grad", x, y, z, grad);
#endif
    }
}

// Find and store the normalized residuals
__global__ void findDarcyNormalizedResiduals(
    const Float* __restrict__ dev_darcy_p_old,
    const Float* __restrict__ dev_darcy_p,
    Float* __restrict__ dev_darcy_norm,
    const unsigned int bc_bot,
    const unsigned int bc_top)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // 1D thread index
    const unsigned int cellidx = d_idx(x,y,z);

    // Perform the epsilon updates for all non-ghost nodes except the
    // Dirichlet boundaries at z=0 and z=nz-1.
    // Adjust z range if a boundary has the Dirichlet boundary condition.
    int z_min = 0;
    int z_max = nz-1;
    if (bc_bot == 0)
        z_min = 1;
    if (bc_top == 0)
        z_max = nz-2;

    if (x < nx && y < ny && z >= z_min && z <= z_max) {

        __syncthreads();
        const Float p = dev_darcy_p_old[cellidx];
        const Float p_new = dev_darcy_p[cellidx];

        // Find the normalized residual value. A small value is added to the
        // denominator to avoid a divide by zero.
        const Float res_norm = (e_new - e)*(e_new - e)/(e_new*e_new + 1.0e-16);

        __syncthreads();
        dev_ns_norm[cellidx] = res_norm;

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat("res_norm", x, y, z, res_norm);
#endif
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
        const Float phi = dev_darcy_phi[cellidx];

        // avoid division by zero
        if (phi > 0.9999)
            phi = 0.9999;

        const Float k = k_c*pow(phi,3)/pow(1.0 - phi, 2);

        __syncthreads();
        dev_darcy_k[cellidx] = k;
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
        const Float dx = devC_params.L[0]/nx;
        const Float dy = devC_params.L[1]/ny;
        const Float dz = devC_params.L[2]/nz;

        // calculate the divergence using first order central finite differences
        const Float div_v_p =
            (xp - xn)/dx +
            (yp - yn)/dy +
            (zp - zn)/dz;

        __syncthreads();
        dev_darcy_div_v_p[d_idx(x,y,z)] = div_v_p;
    }
}

// Find the spatial gradients of the permeability. To be used in the pressure
// diffusion term in updateDarcySolution.
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
                (k_xp - k_xn)/(2.0*dx),
                (k_yp - k_yn)/(2.0*dy),
                (k_zp - k_zn)/(2.0*dz));

        // write result
        __syncthreads();
        dev_darcy_grad_k[cellidx] = grad_k;
    }
}

__global__ void updateDarcySolution(
        const Float*  __restrict__ dev_darcy_p_old,   // in
        const Float*  __restrict__ dev_darcy_div_v_p, // in
        const Float*  __restrict__ dev_darcy_k,       // in
        const Float3* __restrict__ dev_darcy_grad_k,  // in
        const Float beta_f,                           // in
        const Float mu,                               // in
        Float* __restrict__ dev_darcy_p,              // out
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

    if (x < nx && y < ny && z < nz) {

        // 1D thread index
        const unsigned int cellidx = d_idx(x,y,z);

        // read values
        __syncthreads();
        const Float  k      = dev_darcy_k[cellidx];
        const Float3 grad_k = dev_darcy_grad_k[cellidx];

        const Float p_old_xn = dev_darcy_p_old[d_idx(x-1,y,z)];
        const Float p_old    = dev_darcy_p_old[cellidx];
        const Float p_old_xp = dev_darcy_p_old[d_idx(x+1,y,z)];
        const Float p_old_yn = dev_darcy_p_old[d_idx(x,y-1,z)];
        const Float p_old_yp = dev_darcy_p_old[d_idx(x,y+1,z)];
        const Float p_old_zn = dev_darcy_p_old[d_idx(x,y,z-1)];
        const Float p_old_zp = dev_darcy_p_old[d_idx(x,y,z+1)];

        const Float div_v_p = dev_darcy_div_v_p[cellidx];

        // find div(k*grad(p_old)). Using vector identities:
        // div(k*grad(p_old)) = k*laplace(p_old) + dot(grad(k), grad(p_old))

        // laplacian approximated by second-order central difference
        const Float laplace_p_old =
            (p_old_xp - 2.0*p_old + p_old_xn)/(dx*dx) +
            (p_old_yp - 2.0*p_old + p_old_yn)/(dy*dy) +
            (p_old_zp - 2.0*p_old + p_old_zn)/(dz*dz);

        // gradient approximated by first-order central difference
        const Float3 grad_p_old = MAKE_FLOAT3(
                (p_old_xp - p_old_xn)/(2.0*dx),
                (p_old_yp - p_old_yn)/(2.0*dy),
                (p_old_zp - p_old_zn)/(2.0*dz));

        // find new value for p
        const Float p_new = p_old
            + dt/(beta_f*phi*mu)*(k*laplace_p_old + dot(grad_k, grad_p_old))
            - dt/(beta_f*phi)*div_v_p;

        // normalized residual, avoid division by zero
        const Float res_norm = (p_new - p)*(p_new - p)/(p_new*p_new + 1.0e-16);

        // save new pressure and the residual
        __syncthreads();
        dev_darcy_p[cellidx]    = p_new;
        dev_darcy_norm[cellidx] = res_norm;
    }
}

// Print final heads and free memory
void DEM::endDarcyDev()
{
    freeDarcyMemDev();
}

// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
