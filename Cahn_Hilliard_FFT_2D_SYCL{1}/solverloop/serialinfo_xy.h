#include <CL/sycl.hpp>
#include <complex.h>
#include <gsl/gsl_rng.h>
#include "fftw3.h"
//#include "global_vars.h"  // Ensure this path is correct based on your project structure
//#include "functions_defs.h"

using namespace cl::sycl;

void serialinfo_xy() {
    long a;
    long k;
    long index;

    start[X] = 0;
    start[Y] = 0;
    start[Z] = 0;

    rows_x = MESH_X + 0;
    rows_y = MESH_Y + 0;
    rows_z = MESH_Z + 0;
    end[X] = rows_x - 1;
    end[Y] = rows_y - 1;
    end[Z] = rows_z - 1;

    layer_size = rows_y * rows_z;

    if (DIMENSION == 2) {
        rows_z = 1;
        start[Z] = 0;
        end[Z] = 0;
        layer_size = rows_y;
    }

    index_count = layer_size * rows_x;

    queue q;

    // Allocate memory for gridinfo
    buffer<fields, 1> gridinfo_buf(range<1>(index_count));

    // Allocate memory for global_max_min
    if (!SPINODAL) {
        buffer<double, 1> phi_max_buf(range<1>(NUMPHASES));
        buffer<double, 1> phi_min_buf(range<1>(NUMPHASES));
        buffer<double, 1> rel_change_phi_buf(range<1>(NUMPHASES));

        q.submit([&](handler& h) {
            auto phi_max = phi_max_buf.get_access<access::mode::write>(h);
            auto phi_min = phi_min_buf.get_access<access::mode::write>(h);

            h.parallel_for(range<1>(NUMPHASES), [=](id<1> i) {
                phi_max[i] = 1.0;
                phi_min[i] = 0.0;
            });
        });
    }

    buffer<double, 1> com_max_buf(range<1>(NUMCOMPONENTS - 1));
    buffer<double, 1> com_min_buf(range<1>(NUMCOMPONENTS - 1));
    buffer<double, 1> rel_change_com_buf(range<1>(NUMCOMPONENTS - 1));

    q.submit([&](handler& h) {
        auto com_max = com_max_buf.get_access<access::mode::write>(h);
        auto com_min = com_min_buf.get_access<access::mode::write>(h);

        h.parallel_for(range<1>(NUMCOMPONENTS - 1), [=](id<1> i) {
            com_max[i] = 1.0;
            com_min[i] = 1.0;
        });
    });

    q.wait();

    // Allocate memory for gridinfo fields
    q.submit([&](handler& h) {
        auto gridinfo = gridinfo_buf.get_access<access::mode::write>(h);

        h.parallel_for(range<1>(index_count), [=](id<1> index) {
            // Allocate memory for each field in gridinfo
            gridinfo[index].phia = (double*)malloc(NUMPHASES * sizeof(double));
            gridinfo[index].compi = (double*)malloc((NUMCOMPONENTS - 1) * sizeof(double));
        });
    });

    q.wait();
}
