#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <iostream>
#include <functional>
#include <deque>
#include <iomanip>

#include "psolib.hpp"

typedef enum { MAP_LOCAL_REDUCE = 0, ALL_REDUCE = 1 } task_header;

/**
* Implements the message with the task to be passed between the threads.
*/
struct task_t {
	task_header header;
	long cnt;
	float eval;
	float x;
	float y;

	task_t(long n_workers): header(ALL_REDUCE), cnt(n_workers), eval(std::numeric_limits<float>::max()), x(0), y(0) {}
};

/**
* Implements a blocking queue in which two neighbour threads interact.
*/
class task_spot{
private:
	std::mutex              d_mutex;
	std::condition_variable d_condition;
	std::deque<task_t*>     d_queue;
public:
	task_spot(long n_workers) {
		this->put_task(new task_t(n_workers));
	}

	task_t* get_task(){
		std::unique_lock<std::mutex> lock(this->d_mutex);
	    this->d_condition.wait(lock, [=]{ return !this->d_queue.empty(); });
	    task_t* rc = this->d_queue.back();
	    this->d_queue.pop_back();
	    return rc;
	}

	void put_task(task_t* t){
		{
	      std::unique_lock<std::mutex> lock(this->d_mutex);
	      this->d_queue.push_front(t);
	    }
	    this->d_condition.notify_one();
	}
};


/**
* Implements the stateful node of the Ring AllReduce. It performs:
*	- MAP->LOCAL_REDUCE: updates the position of the particles of its swarm (map) and evaluates the local swarm historically best position
*		(local reduce);
*	- ALL_REDUCE: determines whether or not the evaluation of the previous swarm w.r.t. its evaluation, eventually updates it and forwards
		the ALL_REDUCE message.
*/
void worker(particle_swarm* swarm, std::function<float(float,float)> loss, long n_iterations, 
	long n_workers, task_spot* prev, task_spot* next){
	std::chrono::system_clock::time_point icompute, oreduce, ocompute, ireduce;

	long iteration_cnt = -1;	// One more iteration for the setup reduce
	while (iteration_cnt < n_iterations){
		task_t* task = prev->get_task();

		if (task->header == MAP_LOCAL_REDUCE){
			icompute = std::chrono::system_clock::now();
			oreduce = std::chrono::system_clock::now();
			if (iteration_cnt > 0){
				swarm->reduce_time += std::chrono::duration_cast<std::chrono::microseconds>(oreduce - ireduce).count();
			}
			swarm->optimize(loss);
			ocompute = std::chrono::system_clock::now();
			swarm->compute_time += std::chrono::duration_cast<std::chrono::microseconds>(ocompute - icompute).count();
			ireduce = std::chrono::system_clock::now();	
			task->header = ALL_REDUCE;
			task->cnt = n_workers;
			task->eval = swarm->sw_opt_eval;
			task->x = swarm->sw_opt_x;
			task->y = swarm->sw_opt_y;
		}
		else if (task->header == ALL_REDUCE){
			swarm->all_reduce_step(task->eval, task->x, task->y);
			task->cnt -= 1;
			if (task->cnt == 0){
				task->header = MAP_LOCAL_REDUCE;
				iteration_cnt += 1;
			}
			else {
				task->eval = swarm->sw_opt_eval;
				task->x = swarm->sw_opt_x;
				task->y = swarm->sw_opt_y;
			}
		}
		next->put_task(task);
	}
	delete prev->get_task();
}

/**
* Sticks a thread on a designed core.
*/
int place_thread(std::thread* thread, int cpu){
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(cpu, &cpuset);
	int rc = pthread_setaffinity_np(thread->native_handle(),
	                                sizeof(cpu_set_t), &cpuset);
	if (rc != 0) {
	  std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
	  return -1;
	}
    return 0;
};


int main(int argc, char * argv[]){
	if (argc < 4) {
		std::cerr << "use: " << argv[0]  << " n_particles n_iterations n_contexts\n";
        return -1;
	}
	int n_particles = atoi(argv[1]);
    int n_iterations = atoi(argv[2]);
    int contexts = atoi(argv[3]);

    auto loss = [=](float x, float y){
    	return x*x + y*y;
    };

    // PSO Hyperparameters
	float min_x = -100000;
	float min_y = -100000;
	float max_x = 100000;
	float max_y = 100000;
	float c1 = 1;
	float c2 = 1;

	// Swarms initialization
    std::vector<particle_swarm*> swarms;
    std::vector<task_spot*> spots;

    int div = n_particles / contexts;
	int mod = n_particles % contexts;

	// Assigns the swarms in a round-robin fashion to all the #contexts threads
	for(size_t i=0; i<contexts; i++){
		int curr_n_particles;
		if (mod > 0) {
			curr_n_particles = div+1;
			mod-=1;
		}
		else {
			curr_n_particles = div;
		}
		swarms.push_back(new particle_swarm(curr_n_particles, min_x, min_y, max_x, max_y, c1, c2, loss));
		spots.push_back(new task_spot(contexts));
	}

    std::vector<std::thread> workers(contexts);

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();	
	for(int i=0; i<contexts; i++) {
		workers[i] = std::thread(worker, swarms[i], loss, n_iterations, contexts, spots[i], spots[(i+1)%contexts]);

		int outcome = place_thread(&workers[i], i);
		if (outcome == -1){
			return -1;
		}
    }
    for(auto &t: workers){
    	t.join();
    }
    std::chrono::system_clock::time_point stop = std::chrono::system_clock::now();

	double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	double compute_time = 0;
	double reduce_time = 0;
	for (auto &s: swarms){
		compute_time += s->compute_time;
		reduce_time += s->reduce_time;
	}
	compute_time = compute_time/contexts;
	reduce_time = reduce_time/contexts;
	std::cout << std::fixed << std::setprecision(1) << "Total time        = " << total_time <<  " ms\t(" << total_time / n_iterations << " ms/iter)" << std::endl;
	std::cout << std::fixed << std::setprecision(1) << "Mean compute_time = " << compute_time << " us\t(" << compute_time / n_iterations << " us/iter)" << std::endl;
	std::cout << std::fixed << std::setprecision(1) << "Mean reduce_time  = " << reduce_time << " us\t(" << reduce_time / n_iterations << " us/iter)" << std::endl;

    for(int i=0; i<contexts; i++){
    	delete swarms[i];
    	delete spots[i];
    }

    return 0;
}