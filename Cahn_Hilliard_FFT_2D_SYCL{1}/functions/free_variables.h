#include <stdlib.h>

void free_variables() {
    long index, gidy, i;
    long layer;

    // Free allocated memory for 3D arrays
    // Free3M(Diffusivity, NUMPHASES, NUMCOMPONENTS-1);
    Free3M(ceq, NUMPHASES, NUMPHASES);
    Free3M(cfill, NUMPHASES, NUMPHASES);
    Free3M(AtomicMobility, NUMPHASES, NUMCOMPONENTS-1);

    // Free allocated memory for FFTW arrays
    // fftw_free(phi);
    fftw_free(com);
    // fftw_free(dfdphi);
    fftw_free(dfdc);

    // Destroy FFTW plans
    fftw_destroy_plan(planF);
    fftw_destroy_plan(planB);

    // Free gridinfo fields
    index_count = layer_size * rows_x;
    for (index = 0; index < index_count; index++) {
        free_memory_fields(&gridinfo[index]);
    }

    // Free gridinfo array
    free(gridinfo);

    // Free global_max_min fields
    free(global_max_min.phi_max);
    free(global_max_min.phi_min);
    free(global_max_min.rel_change_phi);

    free(global_max_min.com_max);
    free(global_max_min.com_min);
    free(global_max_min.rel_change_com);
}
