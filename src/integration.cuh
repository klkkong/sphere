#ifndef INTEGRATION_CUH_
#define INTEGRATION_CUH_

// integration.cuh
// Functions responsible for temporal integration

// Second order integration scheme based on Taylor expansion of particle kinematics. 
// Kernel executed on device, and callable from host only.
__global__ void integrate(Float4* dev_x_sorted, Float4* dev_vel_sorted, // Input
        Float4* dev_angvel_sorted,
        Float4* dev_x, Float4* dev_vel, Float4* dev_angvel, // Output
        Float4* dev_force, Float4* dev_torque, Float4* dev_angpos, // Input
        Float4* dev_acc, Float4* dev_angacc,
        Float4* dev_vel0, Float4* dev_angvel0,
        Float2* dev_xysum,
        unsigned int* dev_gridParticleIndex, // Input: Sorted-Unsorted key
        unsigned int iter)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // Thread id

    if (idx < devC_np) { // Condition prevents block size error

        // Copy data to temporary arrays to avoid any potential read-after-write, 
        // write-after-read, or write-after-write hazards. 
        unsigned int orig_idx = dev_gridParticleIndex[idx];
        const Float4 force   = dev_force[orig_idx];
        const Float4 torque  = dev_torque[orig_idx];
        const Float4 angpos  = dev_angpos[orig_idx];
        const Float4 x       = dev_x_sorted[idx];
        const Float4 vel     = dev_vel_sorted[idx];
        const Float4 angvel  = dev_angvel_sorted[idx];
        Float2 xysum = dev_xysum[orig_idx];

        // Get old accelerations for three-term Taylor expansion. These values
        // don't exist in the first time step
        Float4 acc0, angacc0;
        if (iter == 0) {
            acc0 = MAKE_FLOAT4(0.0, 0.0, 0.0, 0.0);
            angacc0 = MAKE_FLOAT4(0.0, 0.0, 0.0, 0.0);
        } else {
            acc0    = dev_acc[orig_idx];
            angacc0 = dev_acc[orig_idx];
        }

        const Float  radius  = x.w;

        // New values
        Float4 x_new, vel_new, angpos_new, angvel_new;


        // Coherent read from constant memory to registers
        const Float dt = devC_dt;
        const Float3 origo = MAKE_FLOAT3(
                devC_grid.origo[0],
                devC_grid.origo[1],
                devC_grid.origo[2]); 
        const Float3 L = MAKE_FLOAT3(
                devC_grid.L[0],
                devC_grid.L[1],
                devC_grid.L[2]);

        // Particle mass
        Float m = 4.0/3.0 * PI * radius*radius*radius * devC_params.rho;

        // Find the acceleration by Newton's second law
        Float4 acc = MAKE_FLOAT4(0.0, 0.0, 0.0, 0.0);
        acc.x = force.x/m + devC_params.g[0];
        acc.y = force.y/m + devC_params.g[1];
        acc.z = force.z/m + devC_params.g[2];

        // Find the angular acceleration by Newton's second law
        // (angacc = (total moment)/Intertia, intertia = 2/5*m*r^2)
        Float4 angacc = MAKE_FLOAT4(0.0, 0.0, 0.0, 0.0);
        angacc.x = torque.x * 1.0 / (2.0/5.0 * m * radius*radius);
        angacc.y = torque.y * 1.0 / (2.0/5.0 * m * radius*radius);
        angacc.z = torque.z * 1.0 / (2.0/5.0 * m * radius*radius);

        // Modify the acceleration if the particle is marked as having a fixed
        // velocity. In that case, zero the horizontal acceleration and disable
        // gravity to counteract segregation. Particles may move in the
        // z-dimension, to allow for dilation.
        if (vel.w > 0.0f) {

            acc.x = 0.0;
            acc.y = 0.0;
            acc.z -= devC_params.g[2];

            // Zero the angular acceleration
            angacc = MAKE_FLOAT4(0.0, 0.0, 0.0, 0.0);
        }


        // Forward Euler
        // Truncation error O(dt^2) for positions and velocities
        /*x_new.x = x.x + vel.x*dt;
        x_new.y = x.y + vel.y*dt;
        x_new.z = x.z + vel.z*dt;
        x_new.w = x.w; // transfer radius

        vel_new.x = vel.x + acc.x*dt;
        vel_new.y = vel.y + acc.y*dt;
        vel_new.z = vel.z + acc.z*dt;
        vel_new.w = vel.w; // transfer fixvel

        angpos_new.x = angpos.x + angvel.x*dt;
        angpos_new.y = angpos.y + angvel.y*dt;
        angpos_new.z = angpos.z + angvel.z*dt;

        angvel_new.x = angvel.x + angacc.x*dt;
        angvel_new.y = angvel.y + angacc.y*dt;
        angvel_new.z = angvel.z + angacc.z*dt;*/

        // Two-term Taylor expansion (TY2)
        // Truncation error O(dt^3) for positions, O(dt^2) for velocities
        /*x_new.x = x.x + vel.x*dt + 0.5*acc.x*dt*dt;
        x_new.y = x.y + vel.y*dt + 0.5*acc.y*dt*dt;
        x_new.z = x.z + vel.z*dt + 0.5*acc.z*dt*dt;
        x_new.w = x.w; // transfer radius

        vel_new.x = vel.x + acc.x*dt;
        vel_new.y = vel.y + acc.y*dt;
        vel_new.z = vel.z + acc.z*dt;
        vel_new.w = vel.w; // transfer fixvel

        angpos_new.x = angpos.x + angvel.x*dt + 0.5*angacc.x*dt*dt;
        angpos_new.y = angpos.y + angvel.y*dt + 0.5*angacc.y*dt*dt;
        angpos_new.z = angpos.z + angvel.z*dt + 0.5*angacc.z*dt*dt;

        angvel_new.x = angvel.x + angacc.x*dt;
        angvel_new.y = angvel.y + angacc.y*dt;
        angvel_new.z = angvel.z + angacc.z*dt;*/

        // Three-term Taylor expansion (TY3)
        // Truncation error O(dt^4) for positions, O(dt^3) for velocities
        // Approximate acceleration change by backwards difference:
        const Float4 dacc_dt = MAKE_FLOAT4(
                (acc.x - acc0.x)/dt,
                (acc.y - acc0.y)/dt,
                (acc.z - acc0.z)/dt, 0.0);

        const Float4 dangacc_dt = MAKE_FLOAT4(
                (angacc.x - angacc0.x)/dt,
                (angacc.y - angacc0.y)/dt,
                (angacc.z - angacc0.z)/dt, 0.0);

        x_new.x = x.x + vel.x*dt + 0.5*acc.x*dt*dt + 1.0/6.0*dacc_dt.x*dt*dt*dt;
        x_new.y = x.y + vel.y*dt + 0.5*acc.y*dt*dt + 1.0/6.0*dacc_dt.y*dt*dt*dt;
        x_new.z = x.z + vel.z*dt + 0.5*acc.z*dt*dt + 1.0/6.0*dacc_dt.z*dt*dt*dt;
        x_new.w = x.w; // transfer radius

        vel_new.x = vel.x + acc.x*dt + 0.5*dacc_dt.x*dt*dt;
        vel_new.y = vel.y + acc.y*dt + 0.5*dacc_dt.y*dt*dt;
        vel_new.z = vel.z + acc.z*dt + 0.5*dacc_dt.z*dt*dt;
        vel_new.w = vel.w; // transfer fixvel

        angpos_new.x = angpos.x + angvel.x*dt + 0.5*angacc.x*dt*dt
            + 1.0/6.0*dangacc_dt.x*dt*dt*dt;
        angpos_new.y = angpos.y + angvel.y*dt + 0.5*angacc.y*dt*dt
            + 1.0/6.0*dangacc_dt.y*dt*dt*dt;
        angpos_new.z = angpos.z + angvel.z*dt + 0.5*angacc.z*dt*dt
            + 1.0/6.0*dangacc_dt.z*dt*dt*dt;

        angvel_new.x = angvel.x + angacc.x*dt + 0.5*dangacc_dt.x*dt*dt;
        angvel_new.y = angvel.y + angacc.y*dt + 0.5*dangacc_dt.y*dt*dt;
        angvel_new.z = angvel.z + angacc.z*dt + 0.5*dangacc_dt.z*dt*dt;

        // Add horizontal-displacement for this time step to the sum of
        // horizontal displacements
        xysum.x += vel.x*dt + 0.5*acc.x*dt*dt + 1.0/6.0*dacc_dt.x+dt*dt*dt;
        xysum.y += vel.y*dt + 0.5*acc.y*dt*dt + 1.0/6.0*dacc_dt.y+dt*dt*dt;


        // Move particles outside the domain across periodic boundaries
        if (devC_grid.periodic == 1) {
            if (x_new.x < origo.x)
                x_new.x += L.x;
            if (x_new.x > L.x)
                x_new.x -= L.x;
            if (ND == 3) {
                if (x_new.y < origo.y)
                    x_new.y += L.y;
                if (x_new.y > L.y)
                    x_new.y -= L.y;
            }
        } else if (devC_grid.periodic == 2) {
            if (x_new.x < origo.x)
                x_new.x += L.x;
            if (x_new.x > L.x)
                x_new.x -= L.x;
        }

        // Hold threads for coalesced write
        __syncthreads();

        // Store data in global memory at original, pre-sort positions
        dev_xysum[orig_idx]   = xysum;
        dev_acc[orig_idx]     = acc;
        dev_angacc[orig_idx]  = angacc;
        dev_angvel[orig_idx]  = angvel_new;
        dev_vel[orig_idx]     = vel_new;
        dev_angpos[orig_idx]  = angpos_new;
        dev_x[orig_idx]       = x_new;
    } 
} // End of integrate(...)


// Reduce wall force contributions from particles to a single value per wall
__global__ void summation(Float* in, Float *out)
{
    __shared__ Float cache[256];
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int cacheIdx = threadIdx.x;

    Float temp = 0.0f;
    while (idx < devC_np) {
        temp += in[idx];
        idx += blockDim.x * gridDim.x;
    }

    // Set the cache values
    cache[cacheIdx] = temp;

    __syncthreads();

    // For reductions, threadsPerBlock must be a power of two
    // because of the following code
    unsigned int i = blockDim.x/2;
    while (i != 0) {
        if (cacheIdx < i)
            cache[cacheIdx] += cache[cacheIdx + i];
        __syncthreads();
        i /= 2;
    }

    // Write sum for block to global memory
    if (cacheIdx == 0)
        out[blockIdx.x] = cache[0];
}

// Update wall positions
__global__ void integrateWalls(
        Float4* dev_walls_nx,
        Float4* dev_walls_mvfd,
        int* dev_walls_wmode,
        Float* dev_walls_force_partial,
        Float* dev_walls_vel0,
        unsigned int blocksPerGrid,
        Float t_current)
{
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x; // Thread id

    if (idx < devC_nw) { // Condition prevents block size error

        // Copy data to temporary arrays to avoid any potential
        // read-after-write, write-after-read, or write-after-write hazards. 
        Float4 w_nx   = dev_walls_nx[idx];
        Float4 w_mvfd = dev_walls_mvfd[idx];
        int wmode = dev_walls_wmode[idx];  // Wall BC, 0: fixed, 1: devs, 2: vel
        Float vel0 = dev_walls_vel0[idx];
        Float acc;

        if (wmode == 0) // Wall fixed: do nothing
            return;

        // Find the final sum of forces on wall
        w_mvfd.z = 0.0;
        for (int i=0; i<blocksPerGrid; ++i) {
            w_mvfd.z += dev_walls_force_partial[i];
        }

        Float dt = devC_dt;

        // Normal load = Deviatoric stress times wall surface area,
        // directed downwards.
        Float sigma_0 = w_mvfd.w
            + devC_params.devs_A*sin(2.0*PI*devC_params.devs_f * t_current);
        Float N = -sigma_0*devC_grid.L[0]*devC_grid.L[1];

        // Calculate resulting acceleration of wall
        // (Wall mass is stored in x component of position Float4)
        acc = (w_mvfd.z + N)/w_mvfd.x;

        // If Wall BC is controlled by velocity, it should not change
        if (wmode == 2) { 
            acc = 0.0;
        }

        //// Half-step leapfrog Verlet integration scheme ////
        
        // Update half-step velocity
        vel0 += acc * dt;

        // Update position. Second-order scheme based on Taylor expansion 
        //w_nx.w += w_mvfd.y * dt + (acc * dt*dt)/2.0;

        // Update position
        w_nx.w += vel0 * dt;

        // Update position. First-order Euler integration scheme
        //w_nx.w += w_mvfd.y * dt;

        // Update linear velocity
        //w_mvfd.y += acc * dt;
        w_mvfd.y = vel0 + 0.5 * acc * dt;

        // Store data in global memory
        dev_walls_nx[idx]   = w_nx;
        dev_walls_mvfd[idx] = w_mvfd;
        dev_walls_vel0[idx] = vel0;
    }
} // End of integrateWalls(...)


#endif
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
