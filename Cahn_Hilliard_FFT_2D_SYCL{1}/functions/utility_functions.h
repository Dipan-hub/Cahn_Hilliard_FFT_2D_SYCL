// Memory allocation functions
double* MallocV(long m) {
    double *Mat;
    Mat = (double *)malloc(m * sizeof(*Mat));
    return Mat;
}

double** MallocM(long m, long n) {
    double **Mat;
    Mat = (double **)malloc(m * sizeof(*Mat));
    for (int i = 0; i < m; i++) {
        Mat[i] = (double *)malloc(n * sizeof(*Mat[i]));
    }
    return Mat;
}

double*** Malloc3M(long m, long n, long k) {
    double ***Mat;
    Mat = (double ***)malloc(m * sizeof(*Mat));
    for (int i = 0; i < m; i++) {
        Mat[i] = (double **)malloc(n * sizeof(*Mat[i]));
        for (int j = 0; j < n; j++) {
            Mat[i][j] = (double *)malloc(k * sizeof(*Mat[i][j]));
        }
    }
    return Mat;
}

double**** Malloc4M(long m, long n, long k, long l) {
    double ****Mat;
    Mat = (double ****)malloc(m * sizeof(*Mat));
    for (int i = 0; i < m; i++) {
        Mat[i] = (double ***)malloc(n * sizeof(*Mat[i]));
        for (int j = 0; j < n; j++) {
            Mat[i][j] = (double **)malloc(k * sizeof(*Mat[i][j]));
            for (int p = 0; p < k; p++) {
                Mat[i][j][p] = (double *)malloc(l * sizeof(*Mat[i][j][p]));
            }
        }
    }
    return Mat;
}

void FreeM(double **Mat, long m) {
    for (int i = 0; i < m; i++) {
        free(Mat[i]);
    }
    free(Mat);
    Mat = NULL;
}

void Free3M(double ***Mat, long m, long n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            free(Mat[i][j]);
        }
        free(Mat[i]);
    }
    free(Mat);
    Mat = NULL;
}

void Free4M(double ****Mat, long m, long n, long k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int l = 0; l < k; l++) {
                free(Mat[i][j][l]);
            }
            free(Mat[i][j]);
        }
        free(Mat[i]);
    }
    free(Mat);
    Mat = NULL;
}

long file_exists(const char *fname) {
    FILE *file;
    if ((file = fopen(fname, "r"))) {
        fclose(file);
        return 1;
    }
    return 0;
}

// Matrix population functions
void populate_matrix(double **Mat, char *tmpstr, long NUMPHASES) {
    char **tmp;
    char *str1, *token;
    char *saveptr1;
    
    tmp = (char**)malloc(sizeof(char*) * NUMPHASES * (NUMPHASES - 1) / 2);
    for (int i = 0; i < NUMPHASES * (NUMPHASES - 1) / 2; ++i) {
        tmp[i] = (char*)malloc(sizeof(char) * 10);
    }
    for (int i = 0, str1 = tmpstr; ; i++, str1 = NULL) {
        token = strtok_r(str1, "{,}", &saveptr1);
        if (token == NULL)
            break;
        strcpy(tmp[i], token);
    }
    
    int k = 0;
    for (int i = 0; i < NUMPHASES; i++) {
        for (int j = i + 1; j < NUMPHASES; j++) {
            Mat[i][i] = 0.0;
            Mat[i][j] = atof(tmp[k]);
            Mat[j][i] = Mat[i][j];
            k++;
        }
    }
    
    for (int i = 0; i < NUMPHASES * (NUMPHASES - 1) / 2; ++i) {
        free(tmp[i]);
    }
    free(tmp);
    tmp = NULL;
}

void populate_matrix3M(double ***Mat, char *tmpstr, long NUMPHASES) {
    char **tmp;
    char *str1, *token;
    char *saveptr1;
    
    long len = NUMPHASES * (NUMPHASES - 1) * (NUMPHASES - 2) / 6;
    tmp = (char**)malloc(sizeof(char*) * len);
    for (int i = 0; i < len; ++i) {
        tmp[i] = (char*)malloc(sizeof(char) * 10);
    }
    for (int i = 0, str1 = tmpstr; ; i++, str1 = NULL) {
        token = strtok_r(str1, "{,}", &saveptr1);
        if (token == NULL)
            break;
        strcpy(tmp[i], token);
    }
    
    int l = 0;
    for (int i = 0; i < NUMPHASES; i++) {
        for (int j = i + 1; j < NUMPHASES; j++) {
            for (int k = j + 1; k < NUMPHASES; k++) {
                Mat[i][i][i] = 0.0;
                Mat[i][j][j] = 0.0;
                Mat[i][k][k] = 0.0;
                
                Mat[i][j][k] = atof(tmp[l]);
                Mat[i][k][j] = Mat[i][j][k];
                Mat[j][i][k] = Mat[i][j][k];
                Mat[j][k][i] = Mat[i][j][k];
                Mat[k][i][j] = Mat[i][j][k];
                Mat[k][j][i] = Mat[i][j][k];
                
                l++;
            }
        }
    }
    
    for (int i = 0; i < len; ++i) {
        free(tmp[i]);
    }
    free(tmp);
    tmp = NULL;
}

void populate_diffusivity_matrix(double ***Mat, char *tmpstr, long NUMCOMPONENTS) {
    char **tmp;
    char *str1, *token;
    char *saveptr1;
    
    long len = (NUMCOMPONENTS - 1) * (NUMCOMPONENTS - 1) + 2;
    long phase;
    tmp = (char**)malloc(sizeof(char*) * len);
    for (int i = 0; i < len; ++i) {
        tmp[i] = (char*)malloc(sizeof(char) * 10);
    }
    for (int i = 0, str1 = tmpstr; ; i++, str1 = NULL) {
        token = strtok_r(str1, "{,}", &saveptr1);
        if (token == NULL)
            break;
        strcpy(tmp[i], token);
    }
    
    if (strcmp(tmp[0], "1") == 0) {
        printf("The matrix is diagonal\n");
        int l = 1;
        for (int i = 0; i < NUMCOMPONENTS - 1; i++) {
            phase = atol(tmp[1]);
            Mat[phase][i][i] = atof(tmp[1 + l]);
            l++;
        }
        for (int i = 0; i < NUMCOMPONENTS - 1; i++) {
            for (int j = i + 1; j < NUMCOMPONENTS - 1; j++) {
                Mat[phase][i][j] = 0.0;
                Mat[phase][j][i] = 0.0;
                l++;
            }
        }
    } else {
        int l = 1;
        for (int i = 0; i < NUMCOMPONENTS - 1; i++) {
            phase = atol(tmp[1]);
            Mat[phase][i][i] = atof(tmp[1 + l]);
            l++;
        }
        for (int i = 0; i < NUMCOMPONENTS - 1; i++) {
            for (int j = 0; j < NUMCOMPONENTS - 1; j++) {
                Mat[phase][i][j] = atof(tmp[1 + l]);
                l++;
            }
        }
    }
    
    for (int i = 0; i < len; ++i) {
        free(tmp[i]);
    }
    free(tmp);
    tmp = NULL;
}
void populate_A_matrix(double ***Mat, char *tmpstr, long NUMCOMPONENTS) {
  char **tmp;
  char *str1, *str2, *token;
  char *saveptr1, *saveptr2;
  
  int i,j,k,l;
  long len = (NUMCOMPONENTS-1)*(NUMCOMPONENTS-1) +1;
  long phase;
  
//   length = (NUMCOMPONENTS-1)*(NUMCOMPONENTS-1) + 2;
  tmp = (char**)malloc(sizeof(char*)*len);
  for (i = 0; i < len; ++i) {
    tmp[i] = (char*)malloc(sizeof(char)*10);
  }
  for (i = 0, str1 = tmpstr; ; i++, str1 = NULL) {
    token = strtok_r(str1, "{,}", &saveptr1);
    if (token == NULL)
        break;
    strcpy(tmp[i],token);
  }
  l=1;
  for(i=0; i < NUMCOMPONENTS-1; i++) {
    phase = atol(tmp[0]);
    Mat[phase][i][i] = atof(tmp[l]);
    l++;
  }
  for(i=0; i < NUMCOMPONENTS-1; i++) {
    for (j=i+1; j < NUMCOMPONENTS-1; j++) {
      Mat[phase][i][j] = atof(tmp[l]);
      Mat[phase][j][i] = Mat[phase][i][j];
      l++;
    }
  }
  for (i = 0; i < len; ++i) {
    free(tmp[i]);
  }
  free(tmp);
  tmp = NULL;
}

void populate_thermodynamic_matrix(double ***Mat, char *tmpstr, long NUMCOMPONENTS) {
    char **tmp;
    char *str1, *token;
    char *saveptr1;
    
    long len = (NUMCOMPONENTS - 1) + 2;
    long phase1, phase2;
    tmp = (char**)malloc(sizeof(char*) * len);
    for (int i = 0; i < len; ++i) {
        tmp[i] = (char*)malloc(sizeof(char) * 10);
    }
    for (int i = 0, str1 = tmpstr; ; i++, str1 = NULL) {
        token = strtok_r(str1, "{,}", &saveptr1);
        if (token == NULL)
            break;
        strcpy(tmp[i], token);
    }
    
    phase1 = atoi(tmp[0]);
    phase2 = atoi(tmp[1]);
    
    int l = 1;
    for (int i = 0; i < NUMCOMPONENTS - 1; i++) {
        Mat[phase1][phase2][i] = atof(tmp[l + 1]);
        l++;
    }
    
    for (int i = 0; i < len; ++i) {
        free(tmp[i]);
    }
    free(tmp);
    tmp = NULL;
}

void populate_string_array(char **string, char *tmpstr, long size) {
    char *str1, *token;
    char *saveptr1;
    
    for (int i = 0, str1 = tmpstr; ; i++, str1 = NULL) {
        token = strtok_r(str1, "{,}", &saveptr1);
        if (token == NULL)
            break;
        strcpy(string[i], token);
    }
}

// Matrix rotation functions
void get_Rotation_Matrix(double **R, double theta, int axis) {
    double costheta = cos(theta * M_PI / 180.0);
    double sintheta = sin(theta * M_PI / 180.0);
    
    if (axis == 0) {
        R[0][0] = 1.0;
        R[0][1] = 0.0;
        R[0][2] = 0.0;
        R[1][0] = 0.0;
        R[2][0] = 0.0;
        R[1][1] = costheta;
        R[1][2] = -sintheta;
        R[2][1] = sintheta;
        R[2][2] = costheta;
    }
    
    if (axis == 1) {
        R[0][0] = costheta;
        R[0][1] = 0.0;
        R[0][2] = sintheta;
        R[1][0] = 0.0;
        R[2][0] = -sintheta;
        R[1][1] = 1.0;
        R[1][2] = 0.0;
        R[2][1] = 0.0;
        R[2][2] = costheta;
    }
    
    if (axis == 2) {
        R[0][0] = costheta;
        R[0][1] = -sintheta;
        R[0][2] = 0.0;
        R[1][0] = sintheta;
        R[2][0] = 0.0;
        R[1][1] = costheta;
        R[1][2] = 0.0;
        R[2][1] = 0.0;
        R[2][2] = 1.0;
    }
}

void populate_rotation_matrix(double ****Mat, double ****Mat_Inv, char *tmpstr) {
    char **tmp;
    char *str1, *token;
    char *saveptr1;
    int i, l, j;
    
    long len = 5;
    long phase1, phase2;
    double thetax, thetay, thetaz;
    double **Rx, **Ry, **Rz, **mult;
    
    Rx = MallocM(3, 3);
    Ry = MallocM(3, 3);
    Rz = MallocM(3, 3);
    mult = MallocM(3, 3);
    
    tmp = (char**)malloc(sizeof(char*) * len);
    for (i = 0; i < len; ++i) {
        tmp[i] = (char*)malloc(sizeof(char) * 10);
    }
    
    for (i = 0, str1 = tmpstr; ; i++, str1 = NULL) {
        token = strtok_r(str1, "{,}", &saveptr1);
        if (token == NULL)
            break;
        strcpy(tmp[i], token);
    }
    
    phase1 = atol(tmp[0]);
    phase2 = atol(tmp[1]);
    
    thetax = atof(tmp[2]);
    thetay = atof(tmp[3]);
    thetaz = atof(tmp[4]);
    
    printf("phase1=%ld, phase2=%ld\n", phase1, phase2);
    
    get_Rotation_Matrix(Rx, thetax, 0);
    get_Rotation_Matrix(Ry, thetay, 1);
    get_Rotation_Matrix(Rz, thetaz, 2);
    
    multiply2d(Rx, Ry, mult, 3);
    multiply2d(mult, Rz, Mat[phase1][phase2], 3);
    
    matinvnew(Mat[phase1][phase2], Mat_Inv[phase1][phase2], 3);
    
    FreeM(Rx, 3);
    FreeM(Ry, 3);
    FreeM(Rz, 3);
    FreeM(mult, 3);
    
    for (i = 0; i < len; ++i) {
        free(tmp[i]);
    }
    free(tmp);
    tmp = NULL;
}

// Printing functions
void PRINT_INT(char *key, int value, FILE *fp) {
    fprintf(fp, "%s = %d\n", key, value);
    fprintf(fp, "\n");
}

void PRINT_LONG(char *key, long value, FILE *fp) {
    fprintf(fp, "%s = %ld\n", key, value);
    fprintf(fp, "\n");
}

void PRINT_DOUBLE(char *key, double value, FILE *fp) {
    fprintf(fp, "%s = %le\n", key, value);
    fprintf(fp, "\n");
}

void PRINT_STRING(char *key, char *str, FILE *fp) {
    fprintf(fp, "%s = %s\n", key, str);
    fprintf(fp, "\n");
}

void PRINT_MATRIX(char *key, double **Mat, long m, long n, FILE *fp) {
    fprintf(fp, "%s = [", key);
    for (long i = 0; i < m; i++) {
        for (long j = 0; j < n; j++) {
            if (j < (n - 1)) {
                fprintf(fp, "%le,", Mat[i][j]);
            } else {
                fprintf(fp, "%le", Mat[i][j]);
            }
        }
        if (i < (m - 1)) {
            fprintf(fp, "\n%*s", (int)strlen(key), "");
        } else {
            fprintf(fp, "]\n");
        }
    }
    fprintf(fp, "\n");
}

void PRINT_VECTOR(char *key, double *Mat, long m, FILE *fp) {
    fprintf(fp, "%s = [", key);
    for (long i = 0; i < m; i++) {
        if (i < (m - 1)) {
            fprintf(fp, "%le,", Mat[i]);
        } else {
            fprintf(fp, "%le]\n", Mat[i]);
        }
    }
    fprintf(fp, "\n");
}

void PRINT_STRING_ARRAY(char *key, char **str, long m, FILE *fp) {
    fprintf(fp, "%s = [", key);
    for (long i = 0; i < m; i++) {
        if (i < (m - 1)) {
            fprintf(fp, "%s,", str[i]);
        } else {
            fprintf(fp, "%s]\n", str[i]);
        }
    }
    fprintf(fp, "\n");
}

void allocate_memory_fields(struct fields *ptr) {
    if (!SPINODAL) {
        ptr->phia = (double *)malloc(NUMPHASES * sizeof(double));
    }
    ptr->compi = (double *)malloc((NUMCOMPONENTS - 1) * sizeof(double));
}

void free_memory_fields(struct fields *ptr) {
    if (!SPINODAL) {
        free(ptr->phia);
    }
    free(ptr->compi);
}

void free_memory_fields(struct fields *ptr) {
  //free(ptr->phia);
  free(ptr->compi);
}

void populate_vector(double *Mat, char *tmpstr, long ielements) {
    char **tmp;
    char *str1, *token;
    char *saveptr1;
    
    tmp = (char**)malloc(sizeof(char*) * ielements);
    for (int i = 0; i < ielements; ++i) {
        tmp[i] = (char*)malloc(sizeof(char) * 10);
    }
    for (int i = 0, str1 = tmpstr; ; i++, str1 = NULL) {
        token = strtok_r(str1, "{,}", &saveptr1);
        if (token == NULL)
            break;
        strcpy(tmp[i], token);
    }
    
    for (int i = 0; i < ielements; i++) { 
        Mat[i] = atof(tmp[i]);
    }
    
    for (int i = 0; i < ielements; ++i) {
        free(tmp[i]);
    }
    free(tmp);
    tmp = NULL;
}
