#include <functional>
#include <iomanip>
#include <iostream>
#include <chrono>

#include "psolib.hpp"

int main(int argc, char * argv[]){
	if (argc<3) {
        std::cerr << "use: " << argv[0]  << " n_particles n_iterations\n";
        return -1;
    }
    int n_particles = atoi(argv[1]);
    int n_iterations = atoi(argv[2]);

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

    particle_swarm ps(n_particles, min_x, min_y, max_x, max_y, c1, c2, loss);

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();	
    for(int i=0; i<n_iterations; i++){
    	ps.optimize(loss);
	}
	std::chrono::system_clock::time_point stop = std::chrono::system_clock::now();

	double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
	std::cout << std::fixed << std::setprecision(1) << "Total time        = " << total_time <<  " ms\t(" << total_time / n_iterations << " ms/iter)" << std::endl;
    return 0;
}