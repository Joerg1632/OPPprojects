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
int* dataStarts;
int* dataLengths;


void shiftData(long double* data) {
    int rankFrom = (rank + amountOfProcesses - 1) % amountOfProcesses;
    int rankTo = (rank + 1) % amountOfProcesses;
    MPI_Sendrecv_replace(data, dataLengths[0], MPI_LONG_DOUBLE,
        rankTo, 123, rankFrom, 123,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

}

long double length(long double* vect) {
    long double res = 0;
    int currRank = rank;

    for (int i = 0; i < amountOfProcesses; i++) {
        for (int j = 0; j < dataLengths[currRank]; j++) {
            res += vect[j] * vect[j];
        }


        shiftData(vect);
        currRank = (currRank + amountOfProcesses - 1) % amountOfProcesses;


    }
    return std::sqrt(res);
}

void mult(const long double* matr, long double* vect, long double* result, const int size) {

    int currRank = rank;
    for (int i = 0; i < dataLengths[rank]; i++) {
        result[i] = 0;
    }

    for (int k = 0; k < amountOfProcesses; k++) {

        int offset = dataStarts[currRank];

        for (int i = 0; i < dataLengths[rank]; i++) {
            long double sum = 0;
            for (int j = offset; j < offset + dataLengths[currRank]; j++) {
                sum += matr[i * size + j] * vect[j - offset];
            }
            result[i] += sum;
        }


        shiftData(vect);
        currRank = (currRank + amountOfProcesses - 1) % amountOfProcesses;
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
    const long double error;

    Context(const int size, const long double error) : size(size), error(error) {
        A = new long double[size * dataLengths[rank]];
        x = new long double[dataLengths[0]];
        b = new long double[dataLengths[0]];
        t = 0.0001;
        multRes = new long double[dataLengths[0]];
        diffRes = new long double[dataLengths[0]];

        for (int i = 0; i < dataLengths[rank]; i++) {
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


bool isCloseEnough(Context& cont) {
    mult(cont.A, cont.x, cont.multRes, cont.size);
    diff(cont.multRes, cont.b, cont.diffRes, dataLengths[rank]);
    long double res = length(cont.diffRes) / length(cont.b);
    return res < cont.error;
}


void next(Context& cont) {
    mult(cont.A, cont.x, cont.multRes, cont.size);
    diff(cont.multRes, cont.b, cont.diffRes, dataLengths[rank]);
    mult(cont.diffRes, dataLengths[rank], cont.t);
    diff(cont.x, cont.diffRes, cont.x, dataLengths[rank]);
}

void initializeRowsPerProcess(int n) {
    dataStarts = new int[amountOfProcesses];
    dataLengths = new int[amountOfProcesses];
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
    const long double error = 2.5e-19;
    initializeRowsPerProcess(n);
    Context cont = Context(n, error);
    int counter = 0;
    while (!isCloseEnough(cont)) {
        next(cont);
        counter++;
    }
    if (rank == 0) {
        printf("%d processes loops: %d\n", amountOfProcesses, counter);
    }
    delete[] dataStarts;
    delete[] dataLengths;
    MPI_Finalize();
}
