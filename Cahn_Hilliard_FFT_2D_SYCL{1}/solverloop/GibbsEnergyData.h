#include <CL/sycl.hpp>
#include "GibbsEnergyFunctions/GSol.h"
#include "GibbsEnergyFunctions/GSol.c"
#include "GibbsEnergyFunctions/GSol_m.h"
#include "GibbsEnergyFunctions/GSol_m.c"

using namespace cl::sycl;

double GSOL(queue &q, double T, double X1S);
double dGSOLdX1S(queue &q, double T, double X1S);
double GSOL_m(queue &q, double T, double X1L); 
double dGSOL_mdX1S(queue &q, double T, double X1L); 

double GSOL(queue &q, double T, double X1S) {
    double xS[2] = {X1S, 1.0 - X1S};
    double GS[1];

    buffer<double, 1> xS_buf(xS, range<1>(2));
    buffer<double, 1> GS_buf(GS, range<1>(1));

    q.submit([&](handler &h) {
        auto xS_acc = xS_buf.get_access<access::mode::read>(h);
        auto GS_acc = GS_buf.get_access<access::mode::write>(h);

        h.single_task([=]() {
            GES(T, xS_acc.get_pointer(), GS_acc.get_pointer());
        });
    }).wait();

    return GS[0];
}

double dGSOLdX1S(queue &q, double T, double X1S) {
    double xS[2] = {X1S, 1.0 - X1S};
    double dGS[2];

    buffer<double, 1> xS_buf(xS, range<1>(2));
    buffer<double, 1> dGS_buf(dGS, range<1>(2));

    q.submit([&](handler &h) {
        auto xS_acc = xS_buf.get_access<access::mode::read>(h);
        auto dGS_acc = dGS_buf.get_access<access::mode::write>(h);

        h.single_task([=]() {
            dGES(T, xS_acc.get_pointer(), dGS_acc.get_pointer());
        });
    }).wait();

    return dGS[0] - dGS[1];
}

double GSOL_m(queue &q, double T, double X1L) {
    double xL[2] = {X1L, 1.0 - X1L};
    double GL[1];

    buffer<double, 1> xL_buf(xL, range<1>(2));
    buffer<double, 1> GL_buf(GL, range<1>(1));

    q.submit([&](handler &h) {
        auto xL_acc = xL_buf.get_access<access::mode::read>(h);
        auto GL_acc = GL_buf.get_access<access::mode::write>(h);

        h.single_task([=]() {
            GEL(T, xL_acc.get_pointer(), GL_acc.get_pointer());
        });
    }).wait();

    return GL[0];
}

double dGSOL_mdX1S(queue &q, double T, double X1L) {
    double xL[2] = {X1L, 1.0 - X1L};
    double dGL[2];

    buffer<double, 1> xL_buf(xL, range<1>(2));
    buffer<double, 1> dGL_buf(dGL, range<1>(2));

    q.submit([&](handler &h) {
        auto xL_acc = xL_buf.get_access<access::mode::read>(h);
        auto dGL_acc = dGL_buf.get_access<access::mode::write>(h);

        h.single_task([=]() {
            dGEL(T, xL_acc.get_pointer(), dGL_acc.get_pointer());
        });
    }).wait();

    return dGL[0] - dGL[1];
}
