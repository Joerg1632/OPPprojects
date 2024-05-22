#include <mpi.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cmath>
#include <iostream>

constexpr int NUM_TASKS = 8000;

struct Task {
    int id;
    int process_id;
    int weight;
};

class TaskQueue {
public:
    TaskQueue(int capacity) : capacity(capacity), size(0), pop_index(0) {
        tasks.resize(capacity);
    }

    bool isEmpty() const {
        return size == 0;
    }

    bool isFull() const {
        return size == capacity;
    }

    bool push(const Task& task) {
        if (isFull()) {
            return false;
        }
        int push_index = (pop_index + size) % capacity;
        tasks[push_index] = task;
        size++;
        return true;
    }

    bool pop(Task& task) {
        if (isEmpty()) {
            return false;
        }
        task = tasks[pop_index];
        pop_index = (pop_index + 1) % capacity;
        size--;
        return true;
    }

private:
    std::vector<Task> tasks;
    int capacity;
    int size;
    int pop_index;
};

int num_processes;
int process_id;
int initial_task_weight_sum = 0;
static double global_result = 0;
TaskQueue* task_queue;

std::mutex queue_mutex;

void workerThread();

void initializeTasks();
void executeTasks();

int main(int argc, char* argv[]) {
    int required = MPI_THREAD_MULTIPLE;
    int provided;
    double start_time;
    double end_time;

    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided != required) {
        return EXIT_FAILURE;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    task_queue = new TaskQueue(NUM_TASKS);

    // Start worker thread
    start_time = MPI_Wtime();
    std::thread worker(workerThread);

    worker.join();
    end_time = MPI_Wtime();

    double elapsed_time = end_time - start_time;
    double max_elapsed_time = 0;
    MPI_Reduce(&elapsed_time, &max_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Print result
    MPI_Barrier(MPI_COMM_WORLD);

    printf("Summary weight %d: %d\n", process_id, initial_task_weight_sum);
    MPI_Barrier(MPI_COMM_WORLD);
    if (process_id == 0) {
        printf("Time: %lf\n", max_elapsed_time);
    }

    delete task_queue;
    MPI_Finalize();

    return EXIT_SUCCESS;
}

void initializeTasks() {
    const int TOTAL_TASK_WEIGHT = 60000000;
    int min_weight = 2 * TOTAL_TASK_WEIGHT / (NUM_TASKS * (num_processes + 1));
    int task_id = 1;

    for (int i = 0; i < NUM_TASKS; ++i) {
        Task task = { task_id, process_id, min_weight * (i % num_processes + 1) };

        if (i % num_processes == process_id) {
            task_queue->push(task);
            task_id++;
            initial_task_weight_sum += task.weight;
        }
    }
}

void executeTasks() {
    while (true) {
        Task task;

        std::unique_lock<std::mutex> lock(queue_mutex);
        if (task_queue->isEmpty()) {
            break;
        }
        task_queue->pop(task);
        lock.unlock();

        for (int i = 0; i < task.weight; ++i) {
            for (int j = 0; j < 250; ++j) {
                global_result += std::sqrt(i) * std::sqrt(j);
            }
        }
    }
}

void workerThread() {
    initializeTasks();

    // Worker start synchronization
    MPI_Barrier(MPI_COMM_WORLD);

    while (true) {
        executeTasks();

        std::unique_lock<std::mutex> lock(queue_mutex);
        if (task_queue->isEmpty()) {
            break;
        }
    }
}
