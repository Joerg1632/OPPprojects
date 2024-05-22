#include <mpi.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <cmath>
#include <iostream>

constexpr int NUM_TASKS = 8000;
constexpr int TOTAL_TASK_WEIGHT = 60000000;
constexpr int REQUEST_TAG = 0;
constexpr int RESPONSE_TAG = 1;
constexpr int EMPTY_TASK_RESPONSE = -1;
constexpr int TERMINATION_SIGNAL = -2;

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

int process_id, num_processes;
int initial_task_weight_sum = 0;
int completed_task_weight_sum = 0;
bool is_terminated = false;
std::unique_ptr<TaskQueue> task_queue;

std::mutex queue_mutex;
std::condition_variable worker_cv;
std::condition_variable receiver_cv;

double global_result = 0;

void initializeTasks() {
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

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (task_queue->isEmpty()) {
                break;
            }
            task_queue->pop(task);
        }

        for (int i = 0; i < task.weight; ++i) {
            for (int j = 0; j < 250; ++j) {
                global_result += std::sqrt(i) * std::sqrt(j);
            }
        }

        completed_task_weight_sum += task.weight;
    }
}

void workerThread() {
    initializeTasks();

    MPI_Barrier(MPI_COMM_WORLD);

    while (true) {
        executeTasks();

        std::unique_lock<std::mutex> lock(queue_mutex);
        while (task_queue->isEmpty() && !is_terminated) {
            receiver_cv.notify_all();
            worker_cv.wait(lock);
        }

        if (is_terminated) {
            break;
        }
    }

    std::cout << "Worker " << process_id << " finished\n";
}

void receiverThread() {
    int termination_signal = TERMINATION_SIGNAL;

    while (!is_terminated) {
        int received_tasks = 0;
        Task task;

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            while (!task_queue->isEmpty()) {
                receiver_cv.wait(lock);
            }
        }

        for (int i = 0; i < num_processes; ++i) {
            if (i == process_id) {
                continue;
            }

            MPI_Send(&process_id, 1, MPI_INT, i, REQUEST_TAG, MPI_COMM_WORLD);
            MPI_Recv(&task, sizeof(task), MPI_BYTE, i, RESPONSE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (task.id != EMPTY_TASK_RESPONSE) {
                std::unique_lock<std::mutex> lock(queue_mutex);
                task_queue->push(task);
                received_tasks++;
            }
        }

        if (received_tasks == 0) {
            std::unique_lock<std::mutex> lock(queue_mutex);
            is_terminated = true;
        }

        worker_cv.notify_all();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Send(&termination_signal, 1, MPI_INT, process_id, REQUEST_TAG, MPI_COMM_WORLD);
}
void senderThread() {
    while (true) {
        int receiving_process_id;
        Task task;

        MPI_Recv(&receiving_process_id, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (receiving_process_id == TERMINATION_SIGNAL) {
            break;
        }

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (!task_queue->isEmpty()) {
                task_queue->pop(task);
            }
            else {
                task.id = EMPTY_TASK_RESPONSE;
                task.weight = 0;
                task.process_id = process_id;
            }
        }

        MPI_Send(&task, sizeof(task), MPI_BYTE, receiving_process_id, RESPONSE_TAG, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
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

    task_queue = std::make_unique<TaskQueue>(NUM_TASKS);

    std::thread worker(workerThread);
    std::thread receiver(receiverThread);
    std::thread sender(senderThread);

    start_time = MPI_Wtime();

    worker.join();
    receiver.join();
    sender.join();

    end_time = MPI_Wtime();

    double elapsed_time = end_time - start_time;
    double max_elapsed_time = 0;
    MPI_Reduce(&elapsed_time, &max_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Process " << process_id << " - initial weight: " << initial_task_weight_sum << ", completed weight: " << completed_task_weight_sum << "\n";
    MPI_Barrier(MPI_COMM_WORLD);
    if (process_id == 0) {
        std::cout << "Elapsed time: " << max_elapsed_time << "\n";
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
