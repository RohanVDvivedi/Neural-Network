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
	if(weight_range!=0){RANDOM_VECT( l->bias , weight_range , -weight_range , layer_width );}
	
	l->bias_change = ( (double*) malloc( sizeof(double) * layer_width ) );
	COPY_VECT_SCALER(l->bias_change,0,layer_width);
	
	l->act_type = act_type;
}

void min_init_layer(layer* l,ulli layer_width,activation_type act_type)
{
	l->layer_width = layer_width;
	
	l->output = NULL;
	
	l->activated_output = NULL;
	
	l->output_gradient = NULL;
	
	l->bias = ( (double*) malloc( sizeof(double) * layer_width ) );
	
	l->bias_change = NULL;
	
	l->act_type = act_type;
}

// as the name suggests it frees the layer acquired memeory
void delete_layer(layer* l)
{
	if(l->output!=NULL){free(l->output);}
	if(l->activated_output!=NULL){free(l->activated_output);}
	if(l->output_gradient!=NULL){free(l->output_gradient);}
	if(l->bias!=NULL){free(l->bias);}
	if(l->bias_change!=NULL){free(l->bias_change);}
}

// as the name suggests it frees the layer acquired memeory which is no longer required
void minimize_layer(layer* l)
{
	if(l->output!=NULL){free(l->output);}							l->output = NULL;
	if(l->activated_output!=NULL){free(l->activated_output);}		l->activated_output = NULL;
	if(l->output_gradient!=NULL){free(l->output_gradient);}			l->output_gradient = NULL;
	if(l->bias_change!=NULL){free(l->bias_change);}					l->bias_change = NULL;
}

// this function finds the activation value and updates the layer with its activation value
void activate_layer(layer* l)
{
	for(ulli i=0;i<l->layer_width;i++)
	{
		funct_a[l->act_type]( &(l->activated_output[i]) , &(l->output[i]) );
	}
	
	// this switch statement encompasses any requirement of an activation that requires the values of other activations in the same layer
	switch( l->act_type )
	{
		case SOFTMAX :
		{
			double sum = 0;
			for( ulli i=0; i<l->layer_width ; i++ )
			{
				sum += l->activated_output[i];
			}
			COMPUTE_VECT_SCALER(l->activated_output,l->activated_output,'/',sum,l->layer_width);
			break;
		}
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
	printf("\nlayer width \t \t = \t %lld\n",l->layer_width);
	printf("layer activation \t = \t %s\n",funct_name[l->act_type]);
	
	if(l->output!=NULL){
	printf("output \t \t \t :");
	PRINT_VECT(l->output,l->layer_width);}
	
	if(l->activated_output!=NULL){
	printf("\nactivated output \t :");
	PRINT_VECT(l->activated_output,l->layer_width);}
	
	if(l->output_gradient!=NULL){
	printf("\noutput gradient \t :");
	PRINT_VECT(l->output_gradient,l->layer_width);}
	
	if(l->bias!=NULL){
	printf("\nbias weight \t \t :");
	PRINT_VECT(l->bias,l->layer_width);}
	
	if(l->bias_change!=NULL){
	printf("\nbias weight change \t :");
	PRINT_VECT(l->bias_change,l->layer_width);}
}

void set_bias_layer(layer* l)
{
	COPY_VECT_VECT( l->output , l->bias , l->layer_width  );
}

void update_bias_layer(layer* l)
{
	COMPUTE_VECT_VECT( l->bias , l->bias , '+' , l->bias_change , l->layer_width );
}
