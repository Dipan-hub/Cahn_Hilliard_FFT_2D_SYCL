#ifndef FUNCTIONS_DEFS_H_
#define FUNCTIONS_DEFS_H_

#include <cstdio>

double* MallocV(long m);
double** MallocM(long m, long n);
double*** Malloc3M(long m, long n, long k);
double**** Malloc4M(long m, long n, long k, long l);
void FreeM(double **a, long m);
void Free3M(double ***a, long m, long n);
void Free4M(double ****Mat, long m, long n, long k);
void populate_matrix(double **Mat, char *tmpstr, long NUMPHASES);
void populate_matrix3M(double ***Mat, char *tmpstr, long NUMPHASES);
void populate_diffusivity_matrix(double ***Mat, char *tmpstr, long NUMCOMPONENTS);
void PRINT_INT(char *key, int value, FILE *fp);
void PRINT_LONG(char *key, long value, FILE *fp);
void PRINT_DOUBLE(char *key, double value, FILE *fp);
void PRINT_MATRIX(char *key, double **Mat, long m, long n, FILE *fp);
void PRINT_VECTOR(char *key, double *Mat, long m, FILE *fp);
void PRINT_STRING_ARRAY(char *key, char **str, long m, FILE *fp);
void PRINT_STRING(char *key, char *str, FILE *fp);
void allocate_memory_fields(struct fields *ptr);
void free_memory_fields(struct fields *ptr);
void fill_phase_cube(struct fill_cube fill_cube_parameters, struct fields* gridinfo, long b);
void fill_phase_cylinder(struct fill_cylinder fill_cylinder_parameters, struct fields* gridinfo, long b);
void fill_phase_sphere(struct fill_sphere fill_sphere_parameters, struct fields* gridinfo, long b);
void fill_phase_ellipse(struct fill_ellipse fill_ellipse_parameters, struct fields* gridinfo, long b);
void init_propertymatrices();
void fill_composition_cube(struct fields* gridinfo);
void reading_input_parameters(char *argv[]);
void free_variables();
void read_cells_vtk_2D(FILE *fp, struct fields *gridinfo);
void readfromfile_serial2D(struct fields* gridinfo, char *argv[], long t);
void read_cells_vtk_2D_binary(FILE *fp, struct fields *gridinfo);
void readfromfile_serial2D_binary(struct fields* gridinfo, char *argv[], long t);
void populate_vector(double *Mat, char *tmpstr, long ielements);
void prepfftw();
void realtoFourier();
void Fouriertoreal();
void calculate_df();
void call_fftwF();
void update_Fourier_com_phi();
void call_fftwB();
void evolve_fftw();

#endif // FUNCTIONS_DEFS_H_
