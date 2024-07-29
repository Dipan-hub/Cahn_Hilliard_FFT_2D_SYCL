#ifndef GLOBAL_VARS_H_
#define GLOBAL_VARS_H_

#include <complex.h>
#include <gsl/gsl_rng.h>
#include "fftw3.h"

// Global variables
extern long MESH_X; 
extern long MESH_Y;
extern long MESH_Z;
extern long layer_size;
extern int DIMENSION;
extern double deltax;
extern double deltay;
extern double deltaz;
extern double deltat;

extern int NUMPHASES;
extern int NUMCOMPONENTS;
extern int t;

extern long ntimesteps;
extern long saveT;
extern long nsmooth;
extern long STARTTIME;
extern long RESTART;

extern double R; 
extern double V;

extern char **Components;
extern char **Phases;

extern double ***ceq;
extern double ***cfill;
//extern double ***Diffusivity;
extern double **Gamma;

extern long rows_x, rows_y, rows_z;
extern long *start, *end;
extern long *averow, *rows, *offset, *extra;

extern int ASCII;
extern long time_output;
extern int max_length;

struct max_min {
    double *phi_max;
    double *phi_min;
    double *com_max;
    double *com_min;
    double *rel_change_phi;
    double *rel_change_com;
};

extern struct max_min global_max_min;

struct fill_cube {
    long x_start;
    long x_end;
    long y_start;
    long y_end;
    long z_start; 
    long z_end;
};

extern struct fill_cube fill_cube_parameters;

struct fill_cylinder {
    long x_center;
    long y_center;
    long z_start;
    long z_end;
    double radius;
};

extern struct fill_cylinder fill_cylinder_parameters;

struct fill_ellipse {
    long x_center;
    long y_center;
    long z_center;
    double major_axis;
    double eccentricity;
    double rot_angle;
};

extern struct fill_ellipse fill_ellipse_parameters;

struct fill_sphere {
    long x_center;
    long y_center;
    long z_center;
    double radius;
};

extern struct fill_sphere fill_sphere_parameters;

struct filling_type {
    long NUMCUBES;
    long NUMCIRCLES;
    long NUMTRIANGLES;
    double volume_fraction;
    long length;
};

extern struct filling_type *filling_type_phase;

struct fields {
    double *phia;
    double *compi;
};

extern struct fields *gridinfo;

extern fftw_complex **phi; 
extern fftw_complex **dfdphi;
extern fftw_complex **com;
extern fftw_complex **dfdc;

extern fftw_plan planF;
extern fftw_plan planB;

extern int SPINODAL;
extern int tdbflag;
extern long i, j, k;
extern long x, y, z;
extern long half_MESH_X;
extern long half_MESH_Y;
extern long index_count;
extern double sum1;
extern double tmp1, tmp2;
extern double k2;
extern double inv_denom;
extern double delta_kx;
extern double delta_ky;
extern double dWdphi;
extern double *W;
extern double *kx;
extern double *ky;
extern double *P;
extern double temperature;

extern double **Kappa_phi, **Kappa_c, **L_phi;
extern double ***AtomicMobility;
extern double *A_fm, *B_fp;

extern char tdbfname[100];

#define X 0
#define Y 1
#define Z 2
#define TRUE 1

#endif // GLOBAL_VARS_H_


/*

you need to make sure all global variables and structures used in the function are declared in the header. Here is an updated version of the global_vars.h header, incorporating all necessary declarations:


////////////////////////////
#include "global_vars.h"

long MESH_X; 
long MESH_Y;
long MESH_Z;
long layer_size;
int DIMENSION;
double deltax;
double deltay;
double deltaz;
double deltat;

int NUMPHASES;
int NUMCOMPONENTS;
int t;

long ntimesteps;
long saveT;
long nsmooth;
long STARTTIME = 0;
long RESTART = 0;

double R; 
double V;

char **Components;
char **Phases;

double ***ceq;
double ***cfill;
//double ***Diffusivity;
double **Gamma;

long rows_x, rows_y, rows_z;
long *start, *end;
long *averow, *rows, *offset, *extra;

int ASCII = 0;
long time_output;
int max_length;

struct max_min global_max_min;

struct fill_cube fill_cube_parameters;
struct fill_cylinder fill_cylinder_parameters;
struct fill_ellipse fill_ellipse_parameters;
struct fill_sphere fill_sphere_parameters;
struct filling_type *filling_type_phase;

struct fields *gridinfo;

fftw_complex **phi; 
fftw_complex **dfdphi;
fftw_complex **com;
fftw_complex **dfdc;

fftw_plan planF;
fftw_plan planB;

int SPINODAL = 1;
int tdbflag = 0;
long i, j, k;
long x, y, z;
long half_MESH_X;
long half_MESH_Y;
long index_count;
double sum1;
double tmp1, tmp2;
double k2;
double inv_denom;
double delta_kx;
double delta_ky;
double dWdphi;
double *W;
double *kx;
double *ky;
double *P;
double temperature = 673.0;

double **Kappa_phi, **Kappa_c, **L_phi;
double ***AtomicMobility;
double *A_fm, *B_fp;

char tdbfname[100];
////////////////////////////////////////////
Ensure Proper Initialization:
Make sure to properly initialize dynamic memory and other resources in your main program or an initialization function to ensure everything is set up correctly before using the reading_input_parameters function.

With these changes, your reading_input_parameters function and global variables should work together seamlessly.

*/