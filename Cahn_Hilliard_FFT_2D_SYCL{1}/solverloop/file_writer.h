#ifndef FILE_WRITER_H_
#define FILE_WRITER_H_

#include <arpa/inet.h>
#include <endian.h>
#include <stdint.h>
//#include <CL/sycl.hpp>
//#include "fields.h" // Assuming this header contains the definition of the `fields` struct

//using namespace cl::sycl;

#define IS_BIG_ENDIAN     (1 == htons(1))
#define IS_LITTLE_ENDIAN  (!IS_BIG_ENDIAN)

double swap_bytes(double value) {
    double  src_num = value;
    int64_t tmp_num = htobe64(le64toh(*(int64_t*)&src_num));
    double  dst_num = *(double*)&tmp_num;
    return dst_num;
}

void writetofile_serial2D(queue &q, buffer<fields, 1> &gridinfo_buf, const char *filename, long t);
void writetofile_serial2D_binary(queue &q, buffer<fields, 1> &gridinfo_buf, const char *filename, long t);
void write_cells_vtk_2D(FILE *fp, buffer<fields, 1> &gridinfo_buf, queue &q);
void write_cells_vtk_2D_binary(FILE *fp, buffer<fields, 1> &gridinfo_buf, queue &q);
void readfromfile_serial2D(buffer<fields, 1> &gridinfo_buf, const char *filename, long t, queue &q);
void readfromfile_serial2D_binary(buffer<fields, 1> &gridinfo_buf, const char *filename, long t, queue &q);
void read_cells_vtk_2D(FILE *fp, buffer<fields, 1> &gridinfo_buf, queue &q);
void read_cells_vtk_2D_binary(FILE *fp, buffer<fields, 1> &gridinfo_buf, queue &q);

void writetofile_serial2D(queue &q, buffer<fields, 1> &gridinfo_buf, const char *filename, long t) {
    char name[1000];
    sprintf(name, "DATA/%s_%ld.vtk", filename, t);
    FILE *fp = fopen(name, "w");
    write_cells_vtk_2D(fp, gridinfo_buf, q);
    fclose(fp);
}

void writetofile_serial2D_binary(queue &q, buffer<fields, 1> &gridinfo_buf, const char *filename, long t) {
    char name[1000];
    sprintf(name, "DATA/%s_%ld.vtk", filename, t);
    FILE *fp = fopen(name, "wb");
    write_cells_vtk_2D_binary(fp, gridinfo_buf, q);
    fclose(fp);
}

void readfromfile_serial2D(buffer<fields, 1> &gridinfo_buf, const char *filename, long t, queue &q) {
    char name[1000];
    sprintf(name, "DATA/%s_%ld.vtk", filename, t);
    FILE *fp = fopen(name, "r");
    read_cells_vtk_2D(fp, gridinfo_buf, q);
    fclose(fp);
}

void readfromfile_serial2D_binary(buffer<fields, 1> &gridinfo_buf, const char *filename, long t, queue &q) {
    char name[1000];
    sprintf(name, "DATA/%s_%ld.vtk", filename, t);
    FILE *fp = fopen(name, "rb");
    read_cells_vtk_2D_binary(fp, gridinfo_buf, q);
    fclose(fp);
}

void write_cells_vtk_2D(FILE *fp, buffer<fields, 1> &gridinfo_buf, queue &q) {
    q.submit([&](handler &h) {
        auto gridinfo = gridinfo_buf.get_access<access::mode::read>(h);
        h.single_task([=]() {
            fprintf(fp, "# vtk DataFile Version 3.0\n");
            fprintf(fp, "Microsim_fields\n");
            fprintf(fp, "ASCII\n");
            fprintf(fp, "DATASET STRUCTURED_POINTS\n");
            fprintf(fp, "DIMENSIONS %ld %ld %ld\n", MESH_Y, MESH_X, 1L);
            fprintf(fp, "ORIGIN 0 0 0\n");
            fprintf(fp, "SPACING %le %le %le\n", deltax, deltay, deltaz);
            fprintf(fp, "POINT_DATA %ld\n", MESH_X * MESH_Y);

            if (!SPINODAL) {
                for (long a = 0; a < NUMPHASES - 1; a++) {
                    fprintf(fp, "SCALARS %s double 1\n", Phases[a]);
                    fprintf(fp, "LOOKUP_TABLE default\n");
                    for (long x = start[X]; x <= end[X]; x++) {
                        for (long z = start[Z]; z <= end[Z]; z++) {
                            for (long y = start[Y]; y <= end[Y]; y++) {
                                long index = x * layer_size + z * rows_y + y;
                                fprintf(fp, "%le\n", gridinfo[index].phia[a]);
                            }
                        }
                    }
                    fprintf(fp, "\n");
                }
            }

            for (long k = 0; k < NUMCOMPONENTS - 1; k++) {
                fprintf(fp, "SCALARS comp double 1\n");
                fprintf(fp, "LOOKUP_TABLE default\n");
                for (long x = start[X]; x <= end[X]; x++) {
                    for (long z = start[Z]; z <= end[Z]; z++) {
                        for (long y = start[Y]; y <= end[Y]; y++) {
                            long index = x * layer_size + z * rows_y + y;
                            fprintf(fp, "%le\n", gridinfo[index].compi[k]);
                        }
                    }
                }
                fprintf(fp, "\n");
            }
        });
    }).wait();
}

void write_cells_vtk_2D_binary(FILE *fp, buffer<fields, 1> &gridinfo_buf, queue &q) {
    q.submit([&](handler &h) {
        auto gridinfo = gridinfo_buf.get_access<access::mode::read>(h);
        h.single_task([=]() {
            fprintf(fp, "# vtk DataFile Version 3.0\n");
            fprintf(fp, "Microsim_fields\n");
            fprintf(fp, "BINARY\n");
            fprintf(fp, "DATASET STRUCTURED_POINTS\n");
            fprintf(fp, "DIMENSIONS %ld %ld %ld\n", MESH_Y, MESH_X, 1L);
            fprintf(fp, "ORIGIN 0 0 0\n");
            fprintf(fp, "SPACING %le %le %le\n", deltax, deltay, deltaz);
            fprintf(fp, "POINT_DATA %ld\n", MESH_X * MESH_Y);

            if (!SPINODAL) {
                for (long a = 0; a < NUMPHASES - 1; a++) {
                    fprintf(fp, "SCALARS %s double 1\n", Phases[a]);
                    fprintf(fp, "LOOKUP_TABLE default\n");
                    for (long x = start[X]; x <= end[X]; x++) {
                        for (long z = start[Z]; z <= end[Z]; z++) {
                            for (long y = start[Y]; y <= end[Y]; y++) {
                                long index = x * layer_size + z * rows_y + y;
                                double value = gridinfo[index].phia[a];
                                if (IS_LITTLE_ENDIAN) {
                                    value = swap_bytes(value);
                                }
                                fwrite(&value, sizeof(double), 1, fp);
                            }
                        }
                    }
                    fprintf(fp, "\n");
                }
            }

            for (long k = 0; k < NUMCOMPONENTS - 1; k++) {
                fprintf(fp, "SCALARS comp double 1\n");
                fprintf(fp, "LOOKUP_TABLE default\n");
                for (long x = start[X]; x <= end[X]; x++) {
                    for (long z = start[Z]; z <= end[Z]; z++) {
                        for (long y = start[Y]; y <= end[Y]; y++) {
                            long index = x * layer_size + z * rows_y + y;
                            double value = gridinfo[index].compi[k];
                            if (IS_LITTLE_ENDIAN) {
                                value = swap_bytes(value);
                            }
                            fwrite(&value, sizeof(double), 1, fp);
                        }
                    }
                }
                fprintf(fp, "\n");
            }
        });
    }).wait();
}

void read_cells_vtk_2D(FILE *fp, buffer<fields, 1> &gridinfo_buf, queue &q) {
    auto gridinfo = gridinfo_buf.get_access<access::mode::write>();

    long x, y, z, index;
    long a, k;
    char name[1000];
    fscanf(fp, "%*[^\n]\n");
    fscanf(fp, "%*[^\n]\n");
    fscanf(fp, "%*[^\n]\n");
    fscanf(fp, "%*[^\n]\n");
    fscanf(fp, "%*[^\n]\n");
    fscanf(fp, "%*[^\n]\n");
    fscanf(fp, "%*[^\n]\n");
    fscanf(fp, "%*[^\n]\n");

    if (!SPINODAL) {
        for (a = 0; a < NUMPHASES - 1; a++) {
            fscanf(fp, "%*[^\n]\n");
            fscanf(fp, "%*[^\n]\n");
            for (x = start[X]; x <= end[X]; x++) {
                for (z = start[Z]; z <= end[Z]; z++) {
                    for (y = start[Y]; y <= end[Y]; y++) {
                        index = x * layer_size + z * rows_y + y;
                        fscanf(fp, "%le \n", &gridinfo[index].phia[a]);
                    }
                }
            }
        }
    }

    for (k = 0; k < NUMCOMPONENTS - 1; k++) {
        fscanf(fp, "%*[^\n]\n");
        fscanf(fp, "%*[^\n]\n");
        for (x = start[X]; x <= end[X]; x++) {
            for (z = start[Z]; z <= end[Z]; z++) {
                for (y = start[Y]; y <= end[Y]; y++) {
                    index = x * layer_size + z * rows_y + y;
                    fscanf(fp, "%le \n", &gridinfo[index].compi[k]);
                }
            }
        }
    }
}

void read_cells_vtk_2D_binary(FILE *fp, buffer<fields, 1> &gridinfo_buf, queue &q) {
    auto gridinfo = gridinfo_buf.get_access<access::mode::write>();

    long x, y, z, index;
    long a, k;
    double value;

    fscanf(fp, "%*[^\n]\n");
    fscanf(fp, "%*[^\n]\n");
    fscanf(fp, "%*[^\n]\n");
    fscanf(fp, "%*[^\n]\n");
    fscanf(fp, "%*[^\n]\n");
    fscanf(fp, "%*[^\n]\n");
    fscanf(fp, "%*[^\n]\n");
    fscanf(fp, "%*[^\n]\n");

    if (!SPINODAL) {
        for (a = 0; a < NUMPHASES - 1; a++) {
            fscanf(fp, "%*[^\n]\n");
            fscanf(fp, "%*[^\n]\n");
            for (x = start[X]; x <= end[X]; x++) {
                for (z = start[Z]; z <= end[Z]; z++) {
                    for (y = start[Y]; y <= end[Y]; y++) {
                        index = x * layer_size + z * rows_y + y;
                        fread(&value, sizeof(double), 1, fp);
                        if (IS_LITTLE_ENDIAN) {
                            gridinfo[index].phia[a] = swap_bytes(value);
                        } else {
                            gridinfo[index].phia[a] = value;
                        }
                    }
                }
            }
        }
    }

    for (k = 0; k < NUMCOMPONENTS - 1; k++) {
        fscanf(fp, "%*[^\n]\n");
        fscanf(fp, "%*[^\n]\n");
        for (x = start[X]; x <= end[X]; x++) {
            for (z = start[Z]; z <= end[Z]; z++) {
                for (y = start[Y]; y <= end[Y]; y++) {
                    index = x * layer_size + z * rows_y + y;
                    fread(&value, sizeof(double), 1, fp);
                    if (IS_LITTLE_ENDIAN) {
                        gridinfo[index].compi[k] = swap_bytes(value);
                    } else {
                        gridinfo[index].compi[k] = value;
                    }
                }
            }
        }
    }
}

#endif // FILE_WRITER_H_
