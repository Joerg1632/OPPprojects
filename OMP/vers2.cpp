#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#define N 65000
#define EPS 0.000000000000001
#define PI 3.14159265358980

using namespace std;

void printVector(vector<double>& vect) {
    for (int i = 0; i < N; ++i) {
        std::cout << vect[i] << endl;
    }
}

void printMatrix(vector<vector<double>>& A) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << A[i][j] << " ";
        }
        std::cout << endl;
    }
}

void generateData1(vector<vector<double>>& A, vector<double>& b, vector<double>& x) {
    A.resize(N, vector<double>(N, 1.0));
    b.resize(N, N + 1.0);
    x.resize(N, 0.0);

    for (int i = 0; i < N; ++i) {
        A[i][i] = 2.0;
    }
}

void generateData2(vector<vector<double>>& A, vector<double>& u, vector<double>& b, vector<double>& x) {
    A.resize(N, vector<double>(N, 1.0));
    u.resize(N);
    b.resize(N);
    for (int i = 1; i < N; ++i) {
        u[i] = sin(i * (2 * M_PI) / N);
    }
    x.resize(N, 0.0);

    for (int i = 0; i < N; ++i) {
        A[i][i] = 2.0;
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            b[i] += A[i][j] * u[j];
        }
    }
}


int main() {

    vector<vector<double>> A;
    vector<double> b, x;
    double startTime = omp_get_wtime();
    generateData1(A, b, x);
    // vector<double> u;
    // generateData2(A, u, b, x);
    double normAx_b = 0, normb = 0;
    double Arr = 0, ArAr = 0, tau = 0;
    vector<double> residual(N);
    vector<double> taures(N);
    vector<double> Ar(N);
    vector<double> Ax(N);
    #pragma omp parallel
    {
        #pragma omp for reduction(+:normb)
        for (int i = 0; i < N; ++i) {
            normb += b[i] * b[i];
        }

        do
        {
            #pragma omp for
            for (int i = 0; i < N; ++i) {
                double sum = 0;
                for (int j = 0; j < N; ++j) {
                    sum += A[i][j] * x[j];
                }
                Ax[i] = sum;
            }

            #pragma omp for
            for (int i = 0; i < N; ++i) {
                residual[i] = (Ax[i] - b[i]);
            }

            #pragma omp for
            for (int i = 0; i < N; ++i) {
                double sum = 0;
                for (int j = 0; j < N; ++j) {
                    sum += A[i][j] * residual[j];
                }
                Ar[i] = sum;
            }

            #pragma omp for reduction(+:Arr)
            for (int i = 0; i < N; ++i) {
                Arr += Ar[i] * residual[i];
            }

            #pragma omp for reduction(+:ArAr)
            for (int i = 0; i < N; ++i) {
                ArAr += Ar[i] * Ar[i];
            }

            #pragma omp critical
            {
                tau = Arr / ArAr;
            }

            #pragma omp for reduction(+:ArAr) 
            for (int i = 0; i < N; ++i) {
                ArAr += residual[i] * residual[i];
            }

            #pragma omp for
            for (int i = 0; i < N; ++i) {
                taures[i] = (residual[i] * tau);
            }

        } while (sqrt(normAx_b) / sqrt(normb) < EPS * EPS);

        #pragma omp for
        for (int i = 0; i < N; i++) {
            x[i] -= residual[i];
        }

    }


    double endTime = omp_get_wtime();
    std::cout << endTime - startTime << endl;
    //printVector(x);
    return 0;
}
