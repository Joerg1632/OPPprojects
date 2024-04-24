#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

#define DIMENSIONS_COUNT 2
#define X_DIM 0
#define Y_DIM 1

void initializeDimensions(int dimensions[DIMENSIONS_COUNT], int procCount, int argc, char** argv)
{
    if (argc < 3)
        MPI_Dims_create(procCount, DIMENSIONS_COUNT, dimensions);
    else
    {
        dimensions[X_DIM] = std::atoi(argv[1]);
        dimensions[Y_DIM] = std::atoi(argv[2]);

        if (dimensions[X_DIM] * dimensions[Y_DIM] != procCount)
            exit(EXIT_FAILURE);
    }
}

void initializeCommunicators(const int dimensions[DIMENSIONS_COUNT], MPI_Comm* commGrid, MPI_Comm* commRows, MPI_Comm* commColumns)
{
    int reorder = 1;
    int periods[DIMENSIONS_COUNT] = {};
    int subDimensions[DIMENSIONS_COUNT] = {};

    MPI_Cart_create(MPI_COMM_WORLD, DIMENSIONS_COUNT, dimensions, periods, reorder, commGrid);

    subDimensions[X_DIM] = false;
    subDimensions[Y_DIM] = true;
    MPI_Cart_sub(*commGrid, subDimensions, commRows);

    subDimensions[X_DIM] = true;
    subDimensions[Y_DIM] = false;
    MPI_Cart_sub(*commGrid, subDimensions, commColumns);
}

void generateMatrix(std::vector<double>& matrix, int columnCount, int leadingRow, int leadingColumn, bool onRows)
{
    for (int i = 0; i < leadingRow; ++i)
        for (int j = 0; j < leadingColumn; ++j)
            matrix[i * columnCount + j] = onRows ? i : j;
}

void splitA(const std::vector<double>& A, std::vector<double>& Ablock, int AblockSize, int n2, int coordsY, MPI_Comm commRows, MPI_Comm commColumns)
{
    if (coordsY == 0)
    {
        MPI_Scatter(A.data(), AblockSize * n2, MPI_DOUBLE, Ablock.data(), AblockSize * n2, MPI_DOUBLE, 0, commColumns);
    }

    MPI_Bcast(Ablock.data(), AblockSize * n2, MPI_DOUBLE, 0, commRows);
}

void splitB(const std::vector<double>& B, std::vector<double>& Bblock, int BblockSize, int n2, int alignedN3, int coordsX, MPI_Comm commRows, MPI_Comm commColumns)
{
    if (coordsX == 0)
    {
        MPI_Datatype columnNotResizedT;
        MPI_Datatype columnResizedT;

        MPI_Type_vector(n2, BblockSize, alignedN3, MPI_DOUBLE, &columnNotResizedT);
        MPI_Type_commit(&columnNotResizedT);

        MPI_Type_create_resized(columnNotResizedT, 0, BblockSize * sizeof(double), &columnResizedT);
        MPI_Type_commit(&columnResizedT);

        MPI_Scatter(B.data(), 1, columnResizedT, Bblock.data(), BblockSize * n2, MPI_DOUBLE, 0, commRows);

        MPI_Type_free(&columnNotResizedT);
        MPI_Type_free(&columnResizedT);
    }

    MPI_Bcast(Bblock.data(), BblockSize * n2, MPI_DOUBLE, 0, commColumns);
}

void multiply(const std::vector<double>& Ablock, const std::vector<double>& Bblock, std::vector<double>& Cblock, int AblockSize, int BblockSize, int n2)
{
    for (int i = 0; i < AblockSize; ++i)
        for (int j = 0; j < BblockSize; ++j)
            Cblock[i * BblockSize + j] = 0;

    for (int i = 0; i < AblockSize; ++i)
        for (int j = 0; j < n2; ++j)
            for (int k = 0; k < BblockSize; ++k)
                Cblock[i * BblockSize + k] += Ablock[i * n2 + j] * Bblock[j * BblockSize + k];
}

void gatherC(const std::vector<double>& Cblock, std::vector<double>& C, int AblockSize, int BblockSize, int alignedN1, int alignedN3, int procCount, MPI_Comm commGrid)
{
    MPI_Datatype notResizedRecvT;
    MPI_Datatype resizedRecvT;

    int dimsX = alignedN1 / AblockSize;
    int dimsY = alignedN3 / BblockSize;
    std::vector<int> recvCounts(procCount), displs(procCount);

    MPI_Type_vector(AblockSize, BblockSize, alignedN3, MPI_DOUBLE, &notResizedRecvT);
    MPI_Type_commit(&notResizedRecvT);

    MPI_Type_create_resized(notResizedRecvT, 0, BblockSize * sizeof(double), &resizedRecvT);
    MPI_Type_commit(&resizedRecvT);

    for (int i = 0; i < dimsX; ++i)
        for (int j = 0; j < dimsY; ++j)
        {
            recvCounts[i * dimsY + j] = 1;
            displs[i * dimsY + j] = j + i * dimsY * AblockSize;
        }

    MPI_Gatherv(Cblock.data(), AblockSize * BblockSize, MPI_DOUBLE, C.data(), recvCounts.data(), displs.data(), resizedRecvT, 0, commGrid); 
    MPI_Type_free(&notResizedRecvT);
    MPI_Type_free(&resizedRecvT);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int procRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    int procCount;
    MPI_Comm_size(MPI_COMM_WORLD, &procCount);

    int dimensions[DIMENSIONS_COUNT] = {};
    int coords[DIMENSIONS_COUNT] = {};
    double startTime;
    double finishTime;
    std::vector<double> A, B, C, Ablock, Bblock, Cblock;
    MPI_Comm commGrid;
    MPI_Comm commRows;
    MPI_Comm commColumns;

    initializeDimensions(dimensions, procCount, argc, argv);
    initializeCommunicators(dimensions, &commGrid, &commRows, &commColumns);

    MPI_Cart_coords(commGrid, procRank, DIMENSIONS_COUNT, coords);

    int n1 = 2000;
    int n2 = 1500;
    int n3 = 2500;
    int AblockSize = std::ceil(static_cast<double>(n1) / dimensions[X_DIM]);
    int BblockSize = std::ceil(static_cast<double>(n3) / dimensions[Y_DIM]);
    int alignedN1 = AblockSize * dimensions[X_DIM];
    int alignedN3 = BblockSize * dimensions[Y_DIM];

    if (coords[X_DIM] == 0 && coords[Y_DIM] == 0)
    {
        A.resize(alignedN1 * n2);
        B.resize(n2 * alignedN3);
        C.resize(alignedN1 * alignedN3);

        generateMatrix(A, n2, n1, n2, true);
        generateMatrix(B, alignedN3, n2, n3, false);
    }

    startTime = MPI_Wtime();

    Ablock.resize(AblockSize * n2);
    Bblock.resize(BblockSize * n2);
    Cblock.resize(AblockSize * BblockSize);

    splitA(A, Ablock, AblockSize, n2, coords[Y_DIM], commRows, commColumns);
    splitB(B, Bblock, BblockSize, n2, alignedN3, coords[X_DIM], commRows, commColumns);

    multiply(Ablock, Bblock, Cblock, AblockSize, BblockSize, n2);

    gatherC(Cblock, C, AblockSize, BblockSize, alignedN1, alignedN3, procCount, commGrid);

    finishTime = MPI_Wtime();

    if (coords[Y_DIM] == 0 && coords[X_DIM] == 0)
    {
        std::cout << "Time: " << finishTime - startTime << std::endl;
    }

    MPI_Comm_free(&commGrid);
    MPI_Comm_free(&commRows);
    MPI_Comm_free(&commColumns);

    MPI_Finalize();

    return EXIT_SUCCESS;
}