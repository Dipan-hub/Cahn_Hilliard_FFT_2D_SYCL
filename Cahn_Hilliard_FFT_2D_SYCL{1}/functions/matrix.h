#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using namespace cl::sycl;

void matinvnew(double **coeffmatrix, double **inv, long size);
void multiply(double **inv, double *y, double *prod, long size);
void multiply2d(double **m1, double **m2, double **prod, long size);
void vectorsum(double *y1, double *y2, double *sum, long size);
void substituteb(double **fac, double *y, double *vec, long size);
void substitutef(double **fac, double **y1, int index, double *vec, long size);
void pivot(double **coeffmatrix, double **factor, int k, int *tag, long size);
void colswap(double **m1, double **m2, int *tag, long size);
void rowswap(double **m1, double **m2, int *tag, long size);


/////////////////////////////////////////////////////
/*
// Helper functions to allocate memory
double **MallocM(long size1, long size2) {
    double **matrix = new double*[size1];
    for (long i = 0; i < size1; ++i)
        matrix[i] = new double[size2];
    return matrix;
}

double *MallocV(long size) {
    return new double[size];
}

void FreeM(double **matrix, long size1) {
    for (long i = 0; i < size1; ++i)
        delete[] matrix[i];
    delete[] matrix;
}
*/
//////////////////////////////////////////////////////



// Sum of two vectors
void vectorsum(double *y1, double *y2, double *sum, long size) {
    for (long j = 0; j < size; ++j)
        sum[j] = y1[j] + y2[j];
}

// Multiplication of matrix and vector
void multiply(double **inv, double *y, double *prod, long size) {
    for (long i = 0; i < size; ++i) {
        double sum = 0;
        for (long j = 0; j < size; ++j)
            sum += inv[i][j] * y[j];
        prod[i] = sum;
    }
}

// Multiplication of two matrices
void multiply2d(double **m1, double **m2, double **prod, long size) {
    for (long k = 0; k < size; ++k) {
        for (long i = 0; i < size; ++i) {
            double sum = 0;
            for (long j = 0; j < size; ++j)
                sum += m1[k][j] * m2[j][i];
            prod[k][i] = sum;
        }
    }
}

// Matrix inversion using LU decomposition
void matinvnew(double **coeffmatrix, double **inv, long size) {
    std::vector<int> tag(size);
    double **factor = MallocM(size, size);
    double **inv1 = MallocM(size, size);
    double **iden = MallocM(size, size);
    double **prod = MallocM(size, size);
    double *vec1 = MallocV(size);
    double *vec = MallocV(size);

    // Making the Upper Triangular Matrix
    for (long k = 0; k < size; ++k)
        tag[k] = k;
    for (long k = 0; k < size; ++k) {
        pivot(coeffmatrix, factor, k, tag.data(), size);
        for (long i = k + 1; i < size; ++i) {
            double fact = -coeffmatrix[i][k] / coeffmatrix[k][k];
            factor[i][k] = -fact;
            for (long j = k; j < size; ++j)
                coeffmatrix[i][j] = fact * coeffmatrix[k][j] + coeffmatrix[i][j];
        }
    }
    for (long i = 0; i < size; ++i) {
        for (long j = 0; j < size; ++j) {
            if (i == j)
                factor[i][j] = 1;
            if (j > i)
                factor[i][j] = 0;
        }
    }

    // The Identity Matrix
    for (long i = 0; i < size; ++i) {
        for (long j = 0; j < size; ++j) {
            if (i == j)
                iden[i][j] = 1;
            else
                iden[i][j] = 0;
        }
    }

    // Forward and backward substitution to get the final identity matrix
    for (long i = 0; i < size; ++i) {
        substitutef(factor, iden, i, vec1, size);
        substituteb(coeffmatrix, vec1, vec, size);
        for (long j = 0; j < size; ++j)
            inv1[j][i] = vec[j];
    }

    colswap(inv1, inv, tag.data(), size);
    multiply2d(factor, coeffmatrix, prod, size);
    rowswap(prod, coeffmatrix, tag.data(), size);

    FreeM(factor, size);
    FreeM(iden, size);
    FreeM(inv1, size);
    FreeM(prod, size);
    delete[] vec1;
    delete[] vec;
}

// Back Substitution
void substituteb(double **fac, double *y, double *vec, long size) {
    vec[size - 1] = y[size - 1] / fac[size - 1][size - 1];
    for (long i = size - 2; i >= 0; --i) {
        double sum = 0;
        for (long j = i + 1; j < size; ++j)
            sum -= fac[i][j] * vec[j];
        vec[i] = (y[i] + sum) / fac[i][i];
    }
}

// Forward Substitution
void substitutef(double **fac, double **y1, int index, double *vec, long size) {
    std::vector<double> d(size);
    for (long i = 0; i < size; ++i)
        d[i] = y1[i][index];
    vec[0] = d[0];
    for (long i = 1; i < size; ++i) {
        double sum = 0;
        for (long j = 0; j < i; ++j)
            sum -= fac[i][j] * vec[j];
        vec[i] = d[i] + sum;
    }
}

// Modulus operator
double mod(double k) {
    return (k < 0) ? -k : k;
}

// Pivoting
void pivot(double **coeffmatrix, double **factor, int k, int *tag, long size) {
    double swap, big = mod(coeffmatrix[k][k]);
    int tag1 = k;
    for (long i = k + 1; i < size; ++i) {
        if (mod(coeffmatrix[i][k]) > big) {
            tag1 = i;
            big = coeffmatrix[i][k];
        }
    }
    std::swap(tag[k], tag[tag1]);

    for (long i = 0; i < size; ++i) {
        std::swap(coeffmatrix[k][i], coeffmatrix[tag1][i]);
    }
    for (long i = 0; i < k; ++i) {
        std::swap(factor[k][i], factor[tag1][i]);
    }
}

// Swapping Columns to get the final identity matrix because of the initial swapping for pivoting
void colswap(double **m1, double **m2, int *tag, long size) {
    for (long k = 0; k < size; ++k) {
        for (long j = 0; j < size; ++j) {
            for (long p = 0; p < size; ++p)
                m2[p][tag[j]] = m1[p][j];
        }
    }
}

// Switching rows
void rowswap(double **m1, double **m2, int *tag, long size) {
    for (long k = 0; k < size; ++k) {
        for (long j = 0; j < size; ++j) {
            for (long p = 0; p < size; ++p)
                m2[tag[j]][p] = m1[j][p];
        }
    }
}
