#ifndef neuralneth
#define neuralneth
#include "layer.h"
#define ulli unsigned long long int

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
};

void init_neuralnet(neuralnet* nn,ulli layer_count,ulli* layer_size,activation_type* acttyp,double weight_range,double learning_rate);
void delete_neuralnet(neuralnet* nn);
void print_neuralnet(neuralnet* nn);
void feedforward(neuralnet* nn);
void update_costfunction_gradient(neuralnet* nn,double (*funct)(double desired,double calculated) );
void backpropogate(neuralnet* nn);
void update_weight(neuralnet* nn);

#endif
