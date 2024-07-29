#ifndef FILLING_H_
#define FILLING_H_

#include "time.h"

using namespace cl::sycl;

void fill_phase_cube(struct fill_cube fill_cube_parameters, struct fields* gridinfo, long b) {
    queue q;
    q.submit([&](handler& h) {
        h.parallel_for(range<3>(rows_x, rows_z, rows_y), [=](id<3> idx) {
            long x = idx[0];
            long z = idx[1];
            long y = idx[2];
            long index = x * layer_size + z * rows_y + y;
            long a;
            double sum;

            if (b < (NUMPHASES - 1)) {
                if ((x >= fill_cube_parameters.x_start) && (x <= fill_cube_parameters.x_end) &&
                    (y >= fill_cube_parameters.y_start) && (y <= fill_cube_parameters.y_end) &&
                    (z >= fill_cube_parameters.z_start) && (z <= fill_cube_parameters.z_end)) {

                    gridinfo[index].phia[b] = 1.00000;
                    for (a = 0; a < NUMPHASES; a++) {
                        if (b != a) {
                            gridinfo[index].phia[a] = 0.00000;
                        }
                    }
                } else {
                    if (gridinfo[index].phia[b] != 1.0000) {
                        gridinfo[index].phia[b] = 0.00000;
                    }
                }
            } else {
                sum = 0.0;
                for (a = 0; a < NUMPHASES - 1; a++) {
                    sum += gridinfo[index].phia[a];
                }
                if (sum > 1.0) {
                    printf("Wrong filling operation, will fill it with liquid\n");
                    gridinfo[index].phia[b] = 1.0;
                    for (a = 0; a < NUMPHASES - 1; a++) {
                        if (a != b) {
                            gridinfo[index].phia[a] = 0.00000;
                        }
                    }
                } else {
                    gridinfo[index].phia[b] = 1.0 - sum;
                }
            }
        });
    }).wait();
}

void fill_phase_ellipse(struct fill_ellipse fill_ellipse_parameters, struct fields* gridinfo, long b) {
    long x_center, y_center, z_center;
    double l, e, angle;
    double angle_rad = fill_ellipse_parameters.rot_angle * M_PI / 180.0;

    x_center = fill_ellipse_parameters.x_center;
    y_center = fill_ellipse_parameters.y_center;
    z_center = fill_ellipse_parameters.z_center;
    l = fill_ellipse_parameters.major_axis;
    e = fill_ellipse_parameters.eccentricity;
    angle = fill_ellipse_parameters.rot_angle;

    queue q;
    q.submit([&](handler& h) {
        h.parallel_for(range<2>(MESH_X, MESH_Y), [=](id<2> idx) {
            long x = idx[0];
            long y = idx[1];
            long gidy;
            long a;
            double x_;
            double y_;
            double sum;

            if (b < (NUMPHASES - 1)) {
                if ((double)(x - x_center) * (x - x_center) / (l * l) + 
                    ((double)(y - y_center) * (y - y_center)) / (e * e * l * l) <= 1.0) {
                    x_ = x_center + floor((double)(x - x_center) * cos(angle_rad) + 
                          (double)(y - y_center) * sin(angle_rad));
                    y_ = y_center + floor(-(double)(x - x_center) * sin(angle_rad) + 
                          (double)(y - y_center) * cos(angle_rad));
                } else {
                    x_ = x;
                    y_ = y;
                }
                gidy = x_ * MESH_Y + y_;
                if ((gidy < (MESH_X * MESH_Y)) && (gidy > 0)) {
                    if ((double)(x - x_center) * (x - x_center) / (l * l) + 
                        ((double)(y - y_center) * (y - y_center)) / (e * e * l * l) <= 1.0) {
                        gridinfo[gidy].phia[b] = 1.00000;
                        for (a = 0; a < NUMPHASES; a++) {
                            if (b != a) {
                                gridinfo[gidy].phia[a] = 0.00000;
                            }
                        }
                    } else {
                        if (gridinfo[gidy].phia[b] != 1.0000) {
                            gridinfo[gidy].phia[b] = 0.00000;
                        }
                    }
                }
            } else {
                gidy = x * MESH_Y + y;
                sum = 0.0;
                for (a = 0; a < NUMPHASES - 1; a++) {
                    sum += gridinfo[gidy].phia[a];
                }
                if (sum > 1.0) {
                    printf("Wrong filling operation, will fill it with liquid\n");
                    gridinfo[gidy].phia[b] = 1.0;
                    for (a = 0; a < NUMPHASES - 1; a++) {
                        if (a != b) {
                            gridinfo[gidy].phia[a] = 0.00000;
                        }
                    }
                } else {
                    gridinfo[gidy].phia[b] = 1.0 - sum;
                }
            }
        });
    }).wait();
}

void fill_phase_cylinder(struct fill_cylinder fill_cylinder_parameters, struct fields* gridinfo, long b) {
    long x_center, y_center, z_start, z_end;
    double radius;

    x_center = fill_cylinder_parameters.x_center;
    y_center = fill_cylinder_parameters.y_center;
    z_start = fill_cylinder_parameters.z_start;
    z_end = fill_cylinder_parameters.z_end;
    radius = fill_cylinder_parameters.radius;

    queue q;
    q.submit([&](handler& h) {
        h.parallel_for(range<3>(rows_x, rows_z, rows_y), [=](id<3> idx) {
            long x = idx[0];
            long z = idx[1];
            long y = idx[2];
            long index = x * layer_size + z * rows_y + y;
            long a;
            double sum;

            if (b < (NUMPHASES - 1)) {
                if (((x - x_center) * (x - x_center) + (y - y_center) * (y - y_center) <= radius * radius) && 
                    (z >= z_start) && (z <= z_end)) {
                    gridinfo[index].phia[b] = 1.00000;
                    for (a = 0; a < NUMPHASES; a++) {
                        if (b != a) {
                            gridinfo[index].phia[a] = 0.00000;
                        }
                    }
                } else {
                    if (gridinfo[index].phia[b] != 1.0000) {
                        gridinfo[index].phia[b] = 0.00000;
                    }
                }
            } else {
                sum = 0.0;
                for (a = 0; a < NUMPHASES - 1; a++) {
                    sum += gridinfo[index].phia[a];
                }
                if (sum > 1.0) {
                    printf("Wrong filling operation, will fill it with liquid\n");
                    gridinfo[index].phia[b] = 1.0;
                    for (a = 0; a < NUMPHASES - 1; a++) {
                        if (a != b) {
                            gridinfo[index].phia[a] = 0.00000;
                        }
                    }
                } else {
                    gridinfo[index].phia[b] = 1.0 - sum;
                }
            }
        });
    }).wait();
}

void fill_phase_sphere(struct fill_sphere fill_sphere_parameters, struct fields* gridinfo, long b) {
    long x_center, y_center, z_center;
    double radius;

    x_center = fill_sphere_parameters.x_center;
    y_center = fill_sphere_parameters.y_center;
    z_center = fill_sphere_parameters.z_center;
    radius = fill_sphere_parameters.radius;

    queue q;
    q.submit([&](handler& h) {
        h.parallel_for(range<3>(rows_x, rows_z, rows_y), [=](id<3> idx) {
            long x = idx[0];
            long z = idx[1];
            long y = idx[2];
            long index = x * layer_size + z * rows_y + y;
            long a;
            double sum;

            if (b < (NUMPHASES - 1)) {
                if (((x - x_center) * (x - x_center) + (y - y_center) * (y - y_center) + 
                     (z - z_center) * (z - z_center) <= radius * radius)) {
                    gridinfo[index].phia[b] = 1.00000;
                    for (a = 0; a < NUMPHASES; a++) {
                        if (b != a) {
                            gridinfo[index].phia[a] = 0.00000;
                        }
                    }
                } else {
                    if (gridinfo[index].phia[b] != 1.0000) {
                        gridinfo[index].phia[b] = 0.00000;
                    }
                }
            } else {
                sum = 0.0;
                for (a = 0; a < NUMPHASES - 1; a++) {
                    sum += gridinfo[index].phia[a];
                }
                if (sum > 1.0) {
                    printf("Wrong filling operation, will fill it with liquid\n");
                    gridinfo[index].phia[b] = 1.0;
                    for (a = 0; a < NUMPHASES - 1; a++) {
                        if (a != b) {
                            gridinfo[index].phia[a] = 0.00000;
                        }
                    }
                } else {
                    gridinfo[index].phia[b] = 1.0 - sum;
                }
            }
        });
    }).wait();
}

void fill_composition_cube(struct fields* gridinfo) {
    queue q;
    q.submit([&](handler& h) {
        h.parallel_for(range<3>(rows_x, rows_z, rows_y), [=](id<3> idx) {
            long x = idx[0];
            long z = idx[1];
            long y = idx[2];
            long index = x * layer_size + z * rows_y + y;
            long k;
            long b;
            double c[NUMCOMPONENTS - 1];
            double chemical_potential;
            long PHASE_FILLED = 0;

            if (!SPINODAL) {
                for (b = 0; b < NUMPHASES - 1; b++) {
                    if (gridinfo[index].phia[b] == 1.0) {
                        for (k = 0; k < NUMCOMPONENTS - 1; k++) {
                            c[k] = ceq[b][b][k];
                        }
                        for (k = 0; k < NUMCOMPONENTS - 1; k++) {
                            gridinfo[index].compi[k] = c[k];
                        }
                        PHASE_FILLED = 1;
                        break;
                    }
                }
            }
            if (!PHASE_FILLED) {
                for (k = 0; k < NUMCOMPONENTS - 1; k++) {
                    c[k] = cfill[NUMPHASES - 1][NUMPHASES - 1][k];
                }
                for (k = 0; k < NUMCOMPONENTS - 1; k++) {
                    if (SPINODAL) {
                        gridinfo[index].compi[k] = c[k] + 0.05 * (0.5 - (double)rand() / (double)RAND_MAX);
                    } else {
                        gridinfo[index].compi[k] = c[k];
                    }
                }
            }
        });
    }).wait();
}

#endif // FILLING_H_
