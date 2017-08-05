#ifndef neuralneth
#define neuralneth
#include "layer.h"
#define ulli unsigned long long int

typedef enum init_type init_type;
enum init_type{full=0,min=1};

typedef struct neuralnet neuralnet;
struct neuralnet
{
	ulli layer_count;
	layer* use_layer;
	
	double* input;				ulli input_width;
	double* output;				ulli output_width;
	double* desired_output;
	
	double learning_rate;
	
	double*** weight;
	double*** weight_change;
	double*** weight_gradient;
	
	double* temp;
	ulli temp_size;
	layer temp_layer;
	
	int under_training;
};

void init_neuralnet(neuralnet* nn,ulli layer_count,ulli* layer_size,activation_type* acttyp,double weight_range,double learning_rate);
void min_init_neuralnet(neuralnet* nn,ulli layer_count,ulli* layer_size,activation_type* acttyp);
void delete_neuralnet(neuralnet* nn);
void print_neuralnet(neuralnet* nn);
void feedforward(neuralnet* nn);
void update_costfunction_gradient(neuralnet* nn,double (*funct)(double desired,double calculated) );
void backpropogate(neuralnet* nn);
void update_weight(neuralnet* nn);
void load_neuralnet(char* filename,neuralnet* nn,init_type it);
void store_neuralnet(char* filename,neuralnet* nn);
void training_complete(neuralnet* nn);
void get_solution(neuralnet* nn);

#endif
