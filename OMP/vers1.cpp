#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#define N 100000
#define EPS 0.000000000000001

using namespace std;

void printVector(vector<double>& vect) {
    for (int i = 0; i < N; ++i) {
        cout << vect[i] << endl;
    }
}

void printMatrix(vector<vector<double>>& A) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            cout << A[i][j] << " ";
        }
        cout << endl;
    }
}

double getNorm(vector<double> b, vector<double>& residual) {
    double norm1 = 0, norm2 = 0;
    #pragma omp parallel for reduction(+:norm1, norm2)
    for (int i = 0; i < N; ++i) {
        norm1 += (residual[i] * residual[i]);
        norm2 += (b[i] * b[i]);
    }
    return sqrt(norm1) / sqrt(norm2);
}

vector<double> multiply(vector<double>& vect, double d) {
    vector<double> f(vect);
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        f[i] *= d;
    }
    return f;
}

vector<double> multiply(vector<vector<double>>& A, vector<double>& x) {
    vector<double> f(N);
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            f[i] += A[i][j] * x[j];
        }
    }
    return f;
}

double multiply(vector<double>& vect, vector<double>& x) {
    double f = 0;
    #pragma omp parallel for reduction(+:f)
    for (int i = 0; i < N; i++) {
        f += vect[i] * x[i];
    }
    return f;
}

void subtract(vector<double>& array1, const vector<double>& array2) {
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        array1[i] -= array2[i];
    }
}

double getTauForResidual(vector<vector<double>>& A, vector<double>& residual) {
    vector<double> Ar = multiply(A, residual);
    double Arr = multiply(Ar, residual);
    double ArAr = multiply(Ar, Ar);
    return Arr / ArAr;
}

vector<double> getResidual(vector<vector<double>>& A, vector<double>& x, vector<double>& b) {
    vector<double> Ax = multiply(A, x);
    subtract(Ax, b);
    return Ax;
}

void solveMinimalResidualMethod(vector<vector<double>>& A, vector<double>& b, vector<double>& x) {
    vector<double> residual = getResidual(A, x, b);
    while (getNorm(b, residual) < EPS) {
        double tau = getTauForResidual(A, residual);
        subtract(x, multiply(residual, tau));
        residual = getResidual(A, x, b);
    }
}

void generateData1(vector<vector<double>>& A, vector<double>& b, vector<double>& x) {
    A.resize(N, vector<double>(N, 1.0));
    b.resize(N, N + 1.0);
    x.resize(N, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        A[i][i] = 2.0;
    }
}

void generateData2(vector<vector<double>>& A, vector<double>& u, vector<double>& b, vector<double>& x) {
    A.resize(N, vector<double>(N, 1.0));
    u.resize(N);

    #pragma omp parallel for
    for (int i = 1; i < N; ++i) {
        u[i] = sin(i * (2 * M_PI) / N);
    }
    x.resize(N, 0.0);

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        A[i][i] = 2.0;
    }
    b = multiply(A, u);
}

int main() {
    vector<vector<double>> A;
    vector<double> b, x;
    generateData1(A, b, x);
    // vector<double> u;
    // generateData2(A, u, b, x);
    double startTime = omp_get_wtime();
    solveMinimalResidualMethod(A, b, x);
    double endTime = omp_get_wtime();
    cout << endTime - startTime << endl;
    //printVector(x);
    return 0;
}