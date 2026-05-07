void compute();    // advance simulation by one time step (kernels stay on GPU)
void syncToHost(); // copy final GPU state back to hPos/hVel; call once after loop
