#include <CL/sycl.hpp>
#include <complex>
#include <gsl/gsl_rng.h>
#include <oneapi/mkl.hpp>

using namespace cl::sycl;
using namespace std::complex_literals;

double delta_kx, delta_ky, inv_denom, sum1, k2, tmp1, tmp2;
long half_MESH_X, half_MESH_Y, index_count, x, y, z, a, b;
double *kx, *ky, *P, *W;
fftw_plan planF, planB;

struct fields {
    double *compi;
    double *phia;
};

struct global_max_min_t {
    double *phi_max;
    double *phi_min;
    double *rel_change_phi;
    double *com_max;
    double *com_min;
    double *rel_change_com;
};

global_max_min_t global_max_min;
fields *gridinfo;
long MESH_X = 128, MESH_Y = 128, MESH_Z = 1, DIMENSION = 2;
double deltax = 0.1, deltay = 0.1, deltat = 0.01;
long NUMPHASES = 2, NUMCOMPONENTS = 2;
bool SPINODAL = false;
double **phi, **dfdphi, **com, **dfdc;
double **L_phi, **Kappa_phi, **Kappa_c;
double ***AtomicMobility;
double *A_fm, *B_fp;
bool tdbflag = false;
double temperature = 1000.0;

void evolve_fftw() {
    calculate_df();
    call_fftwF();
    update_Fourier_com_phi();
    call_fftwB();
    Fouriertoreal();
}

void prepfftw(queue &q) {
    long index;

    delta_kx = (2.0 * M_PI) / (MESH_X * deltax);
    delta_ky = (2.0 * M_PI) / (MESH_Y * deltay);

    kx = new double[MESH_X];
    ky = new double[MESH_Y];

    half_MESH_X = MESH_X / 2;
    half_MESH_Y = MESH_Y / 2;

    for (int i = 0; i < MESH_X; i++) {
        if (i < half_MESH_X) {
            kx[i] = i * delta_kx;
        } else {
            kx[i] = (i - MESH_X) * delta_kx;
        }
    }

    for (int j = 0; j < MESH_Y; j++) {
        if (j < half_MESH_Y) {
            ky[j] = j * delta_ky;
        } else {
            ky[j] = (j - MESH_Y) * delta_ky;
        }
    }

    P = new double[NUMPHASES - 1];
    for (int b = 0; b < NUMPHASES - 1; b++) {
        P[b] = 1.0;
    }

    W = new double[NUMPHASES - 1];

    if (!SPINODAL) {
        phi = (double**) malloc((NUMPHASES - 1) * sizeof(*phi));
        dfdphi = (double**) malloc((NUMPHASES - 1) * sizeof(*dfdphi));

        for (int b = 0; b < NUMPHASES - 1; b++) {
            phi[b] = (double*) malloc(index_count * sizeof(std::complex<double>));
            dfdphi[b] = (double*) malloc(index_count * sizeof(std::complex<double>));
        }
    }

    com = (double**) malloc((NUMCOMPONENTS - 1) * sizeof(*com));
    dfdc = (double**) malloc((NUMCOMPONENTS - 1) * sizeof(*dfdc));
    for (int a = 0; a < NUMCOMPONENTS - 1; a++) {
        com[a] = (double*) malloc(index_count * sizeof(std::complex<double>));
        dfdc[a] = (double*) malloc(index_count * sizeof(std::complex<double>));
    }

    planF = oneapi::mkl::dft::fft_plan(q, oneapi::mkl::dft::descriptor<oneapi::mkl::dft::detail::precision::DOUBLE,
                               oneapi::mkl::dft::domain::COMPLEX>(MESH_X, MESH_Y));
    planB = oneapi::mkl::dft::ifft_plan(q, oneapi::mkl::dft::descriptor<oneapi::mkl::dft::detail::precision::DOUBLE,
                               oneapi::mkl::dft::domain::COMPLEX>(MESH_X, MESH_Y));

    realtoFourier();
}

void realtoFourier() {
    long index;
    for (x = 0; x < rows_x; x++) {
        for (z = 0; z < rows_z; z++) {
            for (y = 0; y < rows_y; y++) {
                index = x * layer_size + z * rows_y + y;
                for (a = 0; a < NUMCOMPONENTS - 1; a++) {
                    com[a][index] = gridinfo[index].compi[a] + 0.0i;
                }
                if (!SPINODAL) {
                    for (b = 0; b < NUMPHASES - 1; b++) {
                        phi[b][index] = gridinfo[index].phia[b] + 0.0i;
                    }
                }
            }
        }
    }
}

void Fouriertoreal() {
    long index;
    inv_denom = 1.0 / (MESH_X * MESH_Y);

    if (!SPINODAL) {
        for (b = 0; b < NUMPHASES - 1; b++) {
            global_max_min.rel_change_phi[b] = 0.0;
        }
        for (a = 0; a < NUMCOMPONENTS - 1; a++) {
            global_max_min.rel_change_com[a] = 0.0;
        }
    }

    for (x = 0; x < rows_x; x++) {
        for (z = 0; z < rows_z; z++) {
            for (y = 0; y < rows_y; y++) {
                index = x * layer_size + z * rows_y + y;
                for (a = 0; a < NUMCOMPONENTS - 1; a++) {
                    com[a][index] = creal(com[a][index]) * inv_denom + 0.0i;
                    global_max_min.rel_change_com[a] += (creal(com[a][index]) - gridinfo[index].compi[a]) *
                                                        (creal(com[a][index]) - gridinfo[index].compi[a]);
                    gridinfo[index].compi[a] = creal(com[a][index]);
                    if (gridinfo[index].compi[a] > global_max_min.com_max[a]) {
                        global_max_min.com_max[a] = gridinfo[index].compi[a];
                    }
                    if (gridinfo[index].compi[a] < global_max_min.com_min[a]) {
                        global_max_min.com_min[a] = gridinfo[index].compi[a];
                    }
                }
                if (!SPINODAL) {
                    for (b = 0; b < NUMPHASES - 1; b++) {
                        phi[b][index] = creal(phi[b][index]) * inv_denom + 0.0i;
                        global_max_min.rel_change_phi[b] += (creal(phi[b][index]) - gridinfo[index].phia[b]) *
                                                           (creal(phi[b][index]) - gridinfo[index].phia[b]);
                        gridinfo[index].phia[b] = creal(phi[b][index]);
                        if (gridinfo[index].phia[b] > global_max_min.phi_max[b]) {
                            global_max_min.phi_max[b] = gridinfo[index].phia[b];
                        }
                        if (gridinfo[index].phia[b] < global_max_min.phi_min[b]) {
                            global_max_min.phi_min[b] = gridinfo[index].phia[b];
                        }
                    }
                }
            }
        }
    }
}

void calculate_df() {
    long index;
    double x1s;
    double Ti;
    double dgdxp;
    double dgdxm;

    for (x = 0; x < rows_x; x++) {
        for (z = 0; z < rows_z; z++) {
            for (y = 0; y < rows_y; y++) {
                index = x * layer_size + z * rows_y + y;

                if (!SPINODAL) {
                    for (b = 0; b < NUMPHASES - 1; b++) {
                        if (gridinfo[index].phia[b] < 0) {
                            W[b] = 0.0;
                        } else if (gridinfo[index].phia[b] > 1) {
                            W[b] = 1.0;
                        } else {
                            W[b] = gridinfo[index].phia[b] * gridinfo[index].phia[b] * gridinfo[index].phia[b] *
                                   (10.0 - 15.0 * gridinfo[index].phia[b] + 6.0 * gridinfo[index].phia[b] * gridinfo[index].phia[b]);
                        }
                    }

                    sum1 = 0;
                    for (b = 0; b < NUMPHASES - 1; b++) {
                        sum1 += W[b];
                    }
                }

                a = 0; // For 1 solute or 2 components only
                if (SPINODAL) {
                    if (tdbflag) {
                        x1s = gridinfo[index].compi[a];
                        dfdc[a][index] = dGSOLdX1S(temperature, x1s);
                    } else {
                        dfdc[a][index] = 2.0 * A_fm[0] * gridinfo[index].compi[a] * (1.0 - gridinfo[index].compi[a]) *
                                         (1.0 - 2.0 * gridinfo[index].compi[a]);
                    }
                } else {
                    if (tdbflag) {
                        x1s = gridinfo[index].compi[a];
                        dgdxm = dGSOL_mdX1S(temperature, x1s);
                        dgdxp = dGSOLdX1S(temperature, x1s);
                        dfdc[a][index] = (1.0 - sum1) * dgdxm + sum1 * dgdxp;
                    } else {
                        dfdc[a][index] = 2.0 * A_fm[0] * gridinfo[index].compi[a] * (1.0 - sum1) -
                                         2.0 * B_fp[0] * (1.0 - gridinfo[index].compi[a]) * sum1;
                    }
                }

                if (!SPINODAL) {
                    for (b = 0; b < NUMPHASES - 1; b++) {
                        if (gridinfo[index].phia[b] < 0 || gridinfo[index].phia[b] > 1) {
                            dWdphi = 0.0;
                        } else {
                            dWdphi = 30.0 * gridinfo[index].phia[b] * gridinfo[index].phia[b] *
                                     (1.0 - gridinfo[index].phia[b]) * (1.0 - gridinfo[index].phia[b]);
                        }
                        a = 0; // For 1 solute or 2 components
                        if (tdbflag) {
                            x1s = gridinfo[index].compi[a];
                            dgdxm = dGSOL_mdX1S(temperature, x1s);
                            dgdxp = dGSOLdX1S(temperature, x1s);
                            tmp1 = dWdphi * (-dgdxm + dgdxp);
                            tmp2 = 2.0 * P[0] * gridinfo[index].phia[b] * (1.0 - gridinfo[index].phia[b]) *
                                   (1.0 - 2.0 * gridinfo[index].phia[b]);
                            dfdphi[b][index] = tmp1 + tmp2;
                        } else {
                            tmp1 = dWdphi * (-A_fm[0] * gridinfo[index].compi[a] * gridinfo[index].compi[a] +
                                             B_fp[0] * (1.0 - gridinfo[index].compi[a]) * (1.0 - gridinfo[index].compi[a]));
                            tmp2 = 2.0 * P[0] * gridinfo[index].phia[b] * (1.0 - gridinfo[index].phia[b]) *
                                   (1.0 - 2.0 * gridinfo[index].phia[b]);
                            dfdphi[b][index] = tmp1 + tmp2;
                        }
                    }
                }
            }
        }
    }
}

void call_fftwF() {
    for (a = 0; a < NUMCOMPONENTS - 1; a++) {
        oneapi::mkl::dft::compute_forward(planF, dfdc[a]);
    }
    if (!SPINODAL) {
        for (b = 0; b < NUMPHASES - 1; b++) {
            oneapi::mkl::dft::compute_forward(planF, dfdphi[b]);
        }

        for (b = 0; b < NUMPHASES - 1; b++) {
            oneapi::mkl::dft::compute_forward(planF, phi[b]);
        }
    }
    for (a = 0; a < NUMCOMPONENTS - 1; a++) {
        oneapi::mkl::dft::compute_forward(planF, com[a]);
    }
}

void update_Fourier_com_phi() {
    long index;

    for (x = 0; x < rows_x; x++) {
        for (z = 0; z < rows_z; z++) {
            for (y = 0; y < rows_y; y++) {
                index = x * layer_size + z * rows_y + y;

                k2 = kx[x] * kx[x] + ky[y] * ky[y];
                if (!SPINODAL) {
                    for (b = 0; b < NUMPHASES - 1; b++) {
                        inv_denom = 1.0 / (1.0 + 2.0 * L_phi[0][1] * Kappa_phi[0][1] * k2 * deltat);
                        phi[b][index] = inv_denom * (phi[b][index] - dfdphi[b][index] * deltat * L_phi[0][1]);
                    }
                }

                for (a = 0; a < NUMCOMPONENTS - 1; a++) {
                    inv_denom = 1.0 / (1.0 + 2.0 * AtomicMobility[0][0][0] * Kappa_c[0][1] * k2 * k2 * deltat);
                    com[a][index] = inv_denom * (com[a][index] - dfdc[a][index] * k2 * deltat * AtomicMobility[0][0][0]);
                }
            }
        }
    }
}

void call_fftwB() {
    if (!SPINODAL) {
        for (b = 0; b < NUMPHASES - 1; b++) {
            oneapi::mkl::dft::compute_backward(planB, phi[b]);
        }
    }

    for (a = 0; a < NUMCOMPONENTS - 1; a++) {
        oneapi::mkl::dft::compute_backward(planB, com[a]);
    }
}
