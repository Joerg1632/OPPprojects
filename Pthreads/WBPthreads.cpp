#include <mpi.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cmath>
#include <iostream>

constexpr int TASK_COUNT = 8000;

struct Task {
    int id;
    int process_id;
    int weight;
};

class TaskQueue {
public:
    TaskQueue(int capacity) : capacity(capacity), count(0), pop_index(0) {
        data.resize(capacity);
    }

    bool isEmpty() const {
        return count == 0;
    }

    bool isFull() const {
        return count == capacity;
    }

    bool push(const Task &task) {
        if (isFull()) {
            return false;
        }
        int push_index = (pop_index + count) % capacity;
        data[push_index] = task;
        count++;
        return true;
    }

    bool pop(Task &task) {
        if (isEmpty()) {
            return false;
        }
        task = data[pop_index];
        pop_index = (pop_index + 1) % capacity;
        count--;
        return true;
    }

private:
    std::vector<Task> data;
    int capacity;
    int count;
    int pop_index;
};

int process_count;
int process_id;
int proc_sum_weight = 0;
static double global_res = 0;
TaskQueue *task_queue;

std::mutex mtx;

void workerStart();

void initTasks();
void executeTasks();

int main(int argc, char *argv[]) {
    int required = MPI_THREAD_MULTIPLE;
    int provided;
    double start_time;
    double end_time;

    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided != required) {
        return EXIT_FAILURE;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    task_queue = new TaskQueue(TASK_COUNT);

    // Start worker thread
    start_time = MPI_Wtime();
    std::thread worker_thread(workerStart);

    worker_thread.join();
    end_time = MPI_Wtime();

    double time = end_time - start_time;
    double finalTime = 0;
    MPI_Reduce(&time, &finalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Print result
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "Summary weight " << process_id << " - start: " << process_start_sum_weight << ", actual: " << proc_sum_weight << "\n";
    MPI_Barrier(MPI_COMM_WORLD);
    if (process_id == 0) {
        printf("Time: %lf\n", finalTime);
    }

    delete task_queue;
    MPI_Finalize();

    return EXIT_SUCCESS;
}

void initTasks() {
    const int TOTAL_SUM_WEIGHT = 60000000;
    int min_weight = 2 * TOTAL_SUM_WEIGHT / (TASK_COUNT * (process_count + 1));
    int task_id = 1;

    for (int i = 0; i < TASK_COUNT; ++i) {
        Task task = {task_id, process_id, min_weight * (i % process_count + 1)};

        if (i % process_count == process_id) {
            task_queue->push(task);
            task_id++;
            proc_sum_weight += task.weight;
        }
    }
}

void executeTasks() {
    while (true) {
        Task task;

        if (task_queue->isEmpty()) {
            break;
        }
        task_queue->pop(task);

        for (int i = 0; i < task.weight; ++i) {
            for (int j = 0; j < 250; ++j) {
                global_res += std::sqrt(i) * std::sqrt(j);
            }
        }
    }
}

void workerStart() {
    initTasks();
    executeTasks();
}
