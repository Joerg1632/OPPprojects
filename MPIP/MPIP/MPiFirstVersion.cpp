#include <iostream>
#include <valarray>
#include <chrono>
#include <climits>
#include <vector>
#include <mpi.h>
#include <cstring>
#include <string>

using std::chrono::high_resolution_clock;
using std::vector;
using std::string;

int amountOfProcesses;
int rank;
long double* buffer;
int* dataStarts;
int* dataLengths;

long double length(const long double* vect, const int size) {
    long double res = 0;
    for (int i = 0; i < size; i++) {
        res += vect[i] * vect[i];
    }
    return std::sqrt(res);
}

void mult(const long double* matr, const long double* vect, long double* result, const int size) {
    for (int i = 0; i < dataLengths[rank]; i++) {
        long double sum = 0;
        for (int j = 0; j < size; j++) {
            sum += matr[i * size + j] * vect[j];
        }
        result[i] = sum;
    }
}

void mult(long double* vect, const int size, const long double scalar) {
    for (int i = 0; i < size; i++) {
        vect[i] *= scalar;
    }
}

void diff(const long double* minuend, const long double* subtrahend, long double* result, const int size) {
    for (int i = 0; i < size; i++) {
        result[i] = minuend[i] - subtrahend[i];
    }
}

struct Context {
    long double* A;
    long double* x;
    long double* b;
    long double t;
    long double* multRes;
    long double* diffRes;
    const int size;
    long const double error;

    Context(const int size, const double error) : size(size), error(error) {
        A = new long double[size * dataLengths[rank]];
        x = new long double[size];
        b = new long double[size];
        t = 0.0001;
        multRes = new long double[dataLengths[rank]];
        diffRes = new long double[dataLengths[rank]];

        for (int i = 0; i < size; i++) {
            x[i] = 0;
            b[i] = size + 1;
        }
        for (int i = 0; i < dataLengths[rank]; i++) {
            for (int j = 0; j < size; j++) {
                if (dataStarts[rank] + i == j) {
                    A[i * size + j] = 2.0;
                }
                else {
                    A[i * size + j] = 1.0;
                }
            }

        }
    }

    ~Context() {
        delete[] A;
        delete[] x;
        delete[] b;
        delete[] multRes;
        delete[] diffRes;
    }
};

void gatherVector(long double* from, long double* to) {
    MPI_Allgatherv(from, dataLengths[rank], MPI_LONG_DOUBLE, to, dataLengths, dataStarts, MPI_LONG_DOUBLE, MPI_COMM_WORLD);
}

bool isCloseEnough(Context& cont) {
    mult(cont.A, cont.x, cont.multRes, cont.size);
    diff(cont.multRes, cont.b, cont.diffRes, dataLengths[rank]);
    gatherVector(cont.diffRes, buffer);
    long double res = length(buffer, cont.size) / length(cont.b, cont.size);
    return res < cont.error;
}

void next(Context& cont) {
    mult(cont.A, cont.x, cont.multRes, cont.size);
    diff(cont.multRes, cont.b, cont.diffRes, dataLengths[rank]);
    mult(cont.diffRes, dataLengths[rank], cont.t);
    diff(cont.x, cont.diffRes, cont.multRes, dataLengths[rank]);
    gatherVector(cont.multRes, cont.x);
}

void initializeRowsPerProcess(int n) {
    dataStarts = new int[n];
    dataLengths = new int[n];
    int perProcess = n / amountOfProcesses;
    int remainder = n % amountOfProcesses;
    int start = 0;
    for (int i = 0; i < remainder; i++) {
        dataStarts[i] = start;
        dataLengths[i] = perProcess + 1;
        start += perProcess + 1;
    }
    for (int i = remainder; i < amountOfProcesses; i++) {
        dataStarts[i] = start;
        dataLengths[i] = perProcess;
        start += perProcess;
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &amountOfProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int n = 10000;
    const double error = 2.5e-19;
    initializeRowsPerProcess(n);
    Context cont = Context(n, error);
    int counter = 0;
    buffer = new long double[n];
    while (!isCloseEnough(cont)) {
        next(cont);
        counter++;
    }
    delete[] dataStarts;
    delete[] dataLengths;
    delete[] buffer;
    MPI_Finalize();
}
