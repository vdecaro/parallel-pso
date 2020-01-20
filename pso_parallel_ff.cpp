#include<iostream>
#include <iomanip>
#include<vector>
#include <chrono>
#include <functional>
#include <ff/ff.hpp>
#include <ff/pipeline.hpp>

#include "psolib.hpp"

using namespace ff;

typedef enum { MAP_LOCAL_REDUCE = 0, ALL_REDUCE = 1 } task_header;

/**
* Implements the message with the task to be passed between the nodes.
*/
struct task_t {
	task_header header;
	long cnt;
	float eval;
	float x;
	float y;

	task_t(long init_cnt, float eval, float x, float y): header(ALL_REDUCE), cnt(init_cnt), eval(eval), x(x), y(y) {}
};

/**
* Implements the stateful node of the Ring AllReduce. It performs:
*	- MAP->LOCAL_REDUCE: updates the position of the particles of its swarm (map) and evaluates the local swarm historically best position
*		(local reduce);
*	- ALL_REDUCE: determines whether or not the evaluation of the previous swarm w.r.t. its evaluation, eventually updates it and forwards
		the ALL_REDUCE message.
*/
struct worker: ff_node_t<task_t> {
	particle_swarm* swarm;
	std::function<float(float,float)> loss;
	long n_iterations;
	long n_workers;
	long iteration_cnt;
	std::chrono::system_clock::time_point icompute;
	std::chrono::system_clock::time_point ocompute;
	std::chrono::system_clock::time_point ireduce;
	std::chrono::system_clock::time_point oreduce;

	worker(particle_swarm* swarm, std::function<float(float,float)> loss, long n_iterations, long n_workers): 
		swarm(swarm), loss(loss), n_iterations(n_iterations), n_workers(n_workers), iteration_cnt(-1) {}

	int svc_init() {
		ff_send_out(new task_t(n_workers-1, swarm->sw_opt_eval, swarm->sw_opt_x, swarm->sw_opt_y));
		return 0;
	}

    task_t* svc(task_t * task) { 
		if (task == NULL) {
			return GO_ON;
		}
		if (iteration_cnt < n_iterations){
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
			return task;
		}
		else {
			delete task;
			return EOS;
		}
	}
};

int main(int argc, char * argv[]) {

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
	ff_pipeline pipe;
    int div = n_particles / contexts;
	int mod = n_particles % contexts;

	// Assigns the swarms in a round-robin fashion to all the #contexts nodes of the wrapped around pipeline.
	for(int i=0; i<contexts; i++){
		int curr_n_particles;
		if (mod > 0) {
			curr_n_particles = div+1;
			mod-=1;
		}
		else {
			curr_n_particles = div;
		}
		swarms.push_back(new particle_swarm(curr_n_particles, min_x, min_y, max_x, max_y, c1, c2, loss));
		pipe.add_stage(new worker(swarms[i], loss, n_iterations, contexts));
	}
	pipe.wrap_around();
	pipe.cleanup_nodes(true);

	if (pipe.run_and_wait_end()<0) {
        error("running pipe");
        return -1;
    }

	double compute_time = 0;
	double reduce_time = 0;
	for (auto &s: swarms){
		compute_time += s->compute_time;
		reduce_time += s->reduce_time;
	}
	compute_time = compute_time/contexts;
	reduce_time = reduce_time/contexts;
	std::cout << std::fixed << std::setprecision(1) << "Total time        = " << pipe.ffTime() <<  " ms\t(" << pipe.ffTime() / n_iterations << " ms/iter)" << std::endl;
	std::cout << std::fixed << std::setprecision(1) << "Mean compute_time = " << compute_time << " us\t(" << compute_time / n_iterations << " us/iter)" << std::endl;
	std::cout << std::fixed << std::setprecision(1) << "Mean reduce_time  = " << reduce_time << " us\t(" << reduce_time / n_iterations << " us/iter)" << std::endl;

    for(int i=0; i<contexts; i++){
    	delete swarms[i];
    }

    return 0;
}