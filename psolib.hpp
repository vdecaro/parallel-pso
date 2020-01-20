#include <cstdlib>
#include <random>
#include <limits>
#include <algorithm>

/**
* Implements a generator of a uniform distribution in the interval [0, 1]
*/
class uniform_generator{
private:
	std::random_device rd;
	std::mt19937 _gen;
	std::uniform_real_distribution<float> dis; //uniform distribution between 0 and 1

public:
	uniform_generator():_gen(rd()), dis(0, 1) {}

	float gen(){
		return dis(_gen);
	}
};

/**
* Implements the particle's representation: it holds information about its position, velocity and historically best position.
*/
struct particle{
	// Position
	float x;
	float y;

	// Velocity
	float v_x;
	float v_y;

	// Historically best position
	float p_eval_opt;
	float p_opt_x;
	float p_opt_y;

	/**
	* Randomly initializes the particle's position and velocity and performs its evaluation
	*/
	particle(float min_x, float min_y, float max_x, float max_y, uniform_generator* r, std::function<float(float,float)> eval) {
		x = r->gen()*(max_x-min_x) + min_x;
		y = r->gen()*(max_y-min_y) + min_y;

		if (r->gen() >= 0.5) {
			v_x = r->gen();
		}
		else {
			v_x = -1 * r->gen();
		}

		if (r->gen() >= 0.5) {
			v_y = r->gen();
		}
		else {
			v_y = -1 * r->gen();
		}
		p_eval_opt = eval(x, y);
		p_opt_x = x;
		p_opt_y = y;
	}

	/**
	* Updates the particle's position following the PSO's rule and returns whether or not the historically best position has changed.
	*/
	bool update_and_eval(float sw_opt_x, float sw_opt_y, float min_x, float min_y, float max_x, float max_y, float c1, float c2, uniform_generator* r, std::function<float(float,float)> eval){
		v_x += c1*r->gen()*(p_opt_x - x) + c2*r->gen()+(sw_opt_x - x);
		v_y += c1*r->gen()*(p_opt_y - y) + c2*r->gen()+(sw_opt_y - y);

		x = std::max(std::min(x+v_x, max_x), min_x);
		y = std::max(std::min(y+v_y, max_y), min_y);

		float curr_eval = eval(x, y);
		bool changed = curr_eval < p_eval_opt;
		if (changed){
			p_opt_x = x;
			p_opt_y = y;
			p_eval_opt = curr_eval;
		}
		return changed;
	}

};

/**
* Implements the particle swarm's representation: it holds informations about the particles, the hyperparameters and the swarm's 
* historically best position
*/
class particle_swarm{
private:
	size_t n_particles;
	std::vector<particle> particles;

	uniform_generator r;
	float c1;
	float c2;

	float min_x;
	float min_y;
	float max_x;
	float max_y;

public:
	float sw_opt_eval;
	float sw_opt_x;
	float sw_opt_y;

	double compute_time;
	double reduce_time;

	/**
	* Randomly initializes the swarm's particles and detects the historically best position
	*/
	particle_swarm(size_t n_particles,  float min_x, float min_y, float max_x, float max_y,  float c1, float c2, 
		std::function<float(float,float)> eval): n_particles(n_particles), c1(c1), c2(c2), min_x(min_x), min_y(min_y), max_x(max_x), max_y(max_y), 
			sw_opt_eval(std::numeric_limits<float>::max()), compute_time(0), reduce_time(0) {

		for(size_t i=0; i<n_particles; i++){		
			particles.push_back(particle(min_x, min_y, max_x, max_y, &r, eval));
			particle curr = particles[i];
			if (curr.p_eval_opt < sw_opt_eval) {
				sw_opt_eval = curr.p_eval_opt;
				sw_opt_x = curr.p_opt_x;
				sw_opt_y = curr.p_opt_y;
			}
		}
	}

	/**
	* Iteratively updates the position of the particles and evaluates them with the function passed as parameters. Eventually updates
	* the swarm's historically best position.
	*/
	void optimize(std::function<float(float,float)> eval){
		float iter_sw_opt_eval = sw_opt_eval;
		float iter_sw_opt_x = sw_opt_x;
		float iter_sw_opt_y = sw_opt_y;
		
		for(size_t i=0; i<n_particles; i++){
			particle curr = particles[i];
			// Updates the position of the particle. Implements the map function.
			bool p_best_changed = curr.update_and_eval(sw_opt_x, sw_opt_y, min_x, min_y, max_x, max_y, c1, c2, &r, eval);
			
			// Historically best position changed and better than the swarm's best position. It implements the local reduce.
			if (p_best_changed && curr.p_eval_opt < iter_sw_opt_eval){	
				iter_sw_opt_eval = curr.p_eval_opt;
				iter_sw_opt_x = curr.x;
				iter_sw_opt_y = curr.y;
			}
		}
		sw_opt_eval = iter_sw_opt_eval;
		sw_opt_x = iter_sw_opt_x;
		sw_opt_y = iter_sw_opt_y;
	}

	/**
	* Implements the all reduce step w.r.t. other swarms representing subsets of the whole swarm.
	*/
	void all_reduce_step(float eval, float x, float y) {
		if (eval < sw_opt_eval){
			sw_opt_eval = eval;
			sw_opt_x = x;
			sw_opt_y = y;
		}
	}

};