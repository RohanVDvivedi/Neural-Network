#include<stdlib.h>
#include"../inc/layer.h"
#include<stdio.h>

// this source file handles methods that work on layers
// each layer is updated at once

// initializing a layer requires you to pass layer itself, its width OR the number of neurons it contains, the weight_range that initializez its biases and the type of activation function to be used for it
void init_layer(layer* l,ulli layer_width,double weight_range,activation_type act_type)
{
	l->layer_width = layer_width;
	
	l->output = ( (double*) malloc( sizeof(double) * layer_width ) );
	COPY_VECT_SCALER(l->output,0,layer_width);
	
	l->activated_output = ( (double*) malloc( sizeof(double) * layer_width ) );
	COPY_VECT_SCALER(l->activated_output,0,layer_width);
	
	l->output_gradient = ( (double*) malloc( sizeof(double) * layer_width ) );
	COPY_VECT_SCALER(l->output_gradient,0,layer_width);
	
	l->bias = ( (double*) malloc( sizeof(double) * layer_width ) );
	RANDOM_VECT( l->bias , weight_range , -weight_range , layer_width );
	
	l->bias_change = ( (double*) malloc( sizeof(double) * layer_width ) );
	COPY_VECT_SCALER(l->bias_change,0,layer_width);
	
	l->act_type = act_type;
}

// as the name suggests it frees the layer acquired memeory
void delete_layer(layer* l)
{
	free(l->output);
	free(l->activated_output);
	free(l->output_gradient);
	free(l->bias);
	free(l->bias_change);
}

// this function finds the activation value and updates the layer with its activation value
void activate_layer(layer* l)
{
	for(ulli i=0;i<l->layer_width;i++)
	{
		funct_a[l->act_type]( &(l->activated_output[i]) , &(l->output[i]) );
	}
}

// this function finds the derivative of the activation function that it used to find the output derivative
// and then multiplies it with the current output_gradient value
// this function essentially converts d costfunction / d activated_output to d costfunction / d output for the layer
void deactivate_layer(layer* l)
{
	double temp;
	for(ulli i=0;i<l->layer_width;i++)
	{
		funct_g[l->act_type]( &temp , &(l->activated_output[i]) , &(l->output[i]) );
		l->output_gradient[i] *= temp;
	}
}

// as the name suggest it prints the current layer
// usefull for debuggind the code
void print_layer(layer* l)
{
	printf("\nlayer width = \t %lld\n",l->layer_width);
	
	printf("output \t \t \t :");
	PRINT_VECT(l->output,l->layer_width);
	
	printf("\nactivated output \t :");
	PRINT_VECT(l->activated_output,l->layer_width);
	
	printf("\noutput gradient \t :");
	PRINT_VECT(l->output_gradient,l->layer_width);
	
	printf("\nbias weight \t \t :");
	PRINT_VECT(l->bias,l->layer_width);
	
	printf("\nbias weight change \t :");
	PRINT_VECT(l->bias_change,l->layer_width);	
}

void set_bias_layer(layer* l)
{
	COPY_VECT_VECT( l->output , l->bias , l->layer_width  );
}

void update_bias_layer(layer* l)
{
	COMPUTE_VECT_VECT( l->bias , l->bias , '+' , l->bias_change , l->layer_width );
}
