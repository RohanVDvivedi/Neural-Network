#include"../inc/neuralnet.h"
#include<stdio.h>
#include<stdlib.h>
// this file has functions to initialize the neural network
// and delete all its memory at the end

// neuralnetwork is initialized by passing layer_count = number of lyaers in it
// the layer_size array that suggests the number of elements in each layer
// acttyp is an array of activation_type enum that holds what function to call for a layer activation at any time
void init_neuralnet(neuralnet* nn,ulli layer_count,ulli* layer_size,activation_type* acttyp,double weight_range,double learning_rate)
{
	nn->layer_count = layer_count;
	
	ulli maxsize = 0;
	
	nn->use_layer = ( (layer*) (malloc( layer_count * sizeof(layer) )) );
	for(ulli i=0;i<layer_count;i++)
	{
		init_layer( nn->use_layer + i , layer_size[i] , weight_range , acttyp[i] );
		maxsize = maxsize > layer_size[i] ? maxsize : layer_size[i];
	}
	
	nn->input = nn->use_layer[0].activated_output;						nn->input_width = layer_size[0];
	nn->output = nn->use_layer[ layer_count - 1 ].activated_output;		nn->output_width = layer_size[ layer_count - 1 ];
	nn->desired_output = (double*) malloc( sizeof(double) * nn->output_width );
	
	nn->learning_rate = learning_rate;
	
	// weight
	nn->weight = ( (double***) ( malloc(sizeof(double**)*(layer_count-1)) ) );
	for(ulli i=0;i<layer_count-1;i++)
	{
		nn->weight[i] = ( (double**) (malloc(sizeof(double*)*(layer_size[i]))) );
		for(ulli j=0;j<layer_size[i];j++)
		{
			nn->weight[i][j] = ( (double*) (malloc(sizeof(double)*(layer_size[i+1]))) );
			if(weight_range!=0){RANDOM_VECT( nn->weight[i][j] , weight_range , -weight_range , layer_size[i+1] );}
		}
	}
	
	// weight_gradient
	nn->weight_gradient = ( (double***) ( malloc(sizeof(double**)*(layer_count-1)) ) );
	for(ulli i=0;i<layer_count-1;i++)
	{
		nn->weight_gradient[i] = ( (double**) (malloc(sizeof(double*)*(layer_size[i]))) );
		for(ulli j=0;j<layer_size[i];j++)
		{
			nn->weight_gradient[i][j] = ( (double*) (malloc(sizeof(double)*(layer_size[i+1]))) );
			COPY_VECT_SCALER( nn->weight_gradient[i][j] , 0 , layer_size[i+1] );
		}
	}
	
	// weight_change
	nn->weight_change = ( (double***) ( malloc(sizeof(double**)*(layer_count-1)) ) );
	for(ulli i=0;i<layer_count-1;i++)
	{
		nn->weight_change[i] = ( (double**) (malloc(sizeof(double*)*(layer_size[i]))) );
		for(ulli j=0;j<layer_size[i];j++)
		{
			nn->weight_change[i][j] = ( (double*) (malloc(sizeof(double)*(layer_size[i+1]))) );
			COPY_VECT_SCALER( nn->weight_change[i][j] , 0 , layer_size[i+1] );
		}
	}
	
	
	nn->temp = (double*) malloc( sizeof(double) * maxsize );
	nn->temp_size = maxsize;
	
	nn->temp_layer.output = NULL;nn->temp_layer.activated_output = NULL;
	nn->temp_layer.bias = NULL;nn->temp_layer.bias_change = NULL;nn->temp_layer.output_gradient = NULL;
	
	nn->under_training = 1;
}

// if th neural network is to be initialized to not train , then a lot of excess memory can be discarded from the neural net and layer
// structures
// for this purpose use min_init_neuralnet to initiate a minimal form of structure
void min_init_neuralnet(neuralnet* nn,ulli layer_count,ulli* layer_size,activation_type* acttyp)
{
	nn->layer_count = layer_count;
	
	ulli maxsize = 0;
	
	nn->use_layer = ( (layer*) (malloc( layer_count * sizeof(layer) )) );
	for(ulli i=0;i<layer_count;i++)
	{
		min_init_layer( nn->use_layer + i,layer_size[i] , acttyp[i] );
		if(maxsize < layer_size[i])
		{
			maxsize = layer_size[i];
		}
	}
	
	// weight
	nn->weight = ( (double***) ( malloc(sizeof(double**)*(layer_count-1)) ) );
	for(ulli i=0;i<layer_count-1;i++)
	{
		nn->weight[i] = ( (double**) (malloc(sizeof(double*)*(layer_size[i]))) );
		for(ulli j=0;j<layer_size[i];j++)
		{
			nn->weight[i][j] = ( (double*) (malloc(sizeof(double)*(layer_size[i+1]))) );
		}
	}
	
	nn->weight_gradient = NULL;
	nn->weight_change = NULL;
	nn->desired_output = NULL;
	
	nn->temp = (double*) malloc( sizeof(double) * maxsize );
	nn->temp_size = maxsize;
	
	nn->input_width = layer_size[0];
	nn->input = (double*) ( malloc( sizeof(double) * (nn->input_width) ) );
	nn->output_width = layer_size[nn->layer_count-1];
	nn->output = (double*) ( malloc( sizeof(double) * (nn->output_width) ) );
	
	nn->temp_layer.output = (double*) malloc( sizeof(double) * nn->temp_size );
	nn->temp_layer.activated_output = (double*) malloc( sizeof(double) * nn->temp_size );
	nn->temp_layer.bias = NULL;nn->temp_layer.bias_change = NULL;nn->temp_layer.output_gradient = NULL;
	
	nn->under_training = 0;
}


void delete_neuralnet(neuralnet* nn)
{
	free(nn->temp);
	
	// weight
	for(ulli i=0;i<nn->layer_count-1;i++)
	{
		for(ulli j=0;j<nn->use_layer[i].layer_width;j++)
		{
			free(nn->weight[i][j]);
		}
		free(nn->weight[i]);
	}
	free(nn->weight);
	
	// when not under training the below data does not exist
	if( nn->under_training == 1 )
	{
	
		// weight_gradient
		if(nn->weight_gradient!=NULL){
		for(ulli i=0;i<nn->layer_count-1;i++)
		{
			for(ulli j=0;j<nn->use_layer[i].layer_width;j++)
			{
				free(nn->weight_gradient[i][j]);
			}
			free(nn->weight_gradient[i]);
		}
		free(nn->weight_gradient);}
	
		// weight_change
		if(nn->weight_change!=NULL){
		for(ulli i=0;i<nn->layer_count-1;i++)
		{
			for(ulli j=0;j<nn->use_layer[i].layer_width;j++)
			{
				free(nn->weight_change[i][j]);
			}
			free(nn->weight_change[i]);
		}
		free(nn->weight_change);}
	
		if(nn->desired_output!=NULL){free(nn->desired_output);}
	}
	else
	{
		if(nn->input!=NULL){free(nn->input);}
		if(nn->output!=NULL){free(nn->output);}	
		delete_layer(&(nn->temp_layer));
	}

	for(ulli i=0;i<nn->layer_count;i++)
	{
		delete_layer( nn->use_layer + i );
	}
	free( nn->use_layer );
}

// print the neural net variables and the contents of all of its layers
// very usefull for debugging
void print_neuralnet(neuralnet* nn)
{
	printf("\n--------------------------------------------------------------------------------\n\n");
	
	printf("printing neural net : \n");
	
	printf("layer_count \t \t : \t %lld \n",nn->layer_count);
	printf("learning_rate \t \t : \t %lf \n",nn->learning_rate);
	printf("input array size \t : \t %lld \n",nn->input_width);
	printf("output array size \t : \t %lld \n\n",nn->output_width);
	
	if(nn->desired_output!=NULL){printf("desired outputs are \t : ");PRINT_VECT(nn->desired_output,nn->output_width);}printf("\n\n");
	
	for(ulli i=0;i<nn->layer_count-1;i++)
	{
		printf("layer %lld :",i);
		print_layer(nn->use_layer + i);
		printf("\n\n");
		
		printf("weights between layers %lld - %lld\n",i,i+1);
		for(ulli j=0;j<nn->use_layer[i+1].layer_width;j++)
		{
			printf("\t");
			for(ulli k=0;k<nn->use_layer[i].layer_width;k++)
			{
				printf("%lf ",nn->weight[i][k][j]);
			}
			printf("\n");
		}
		printf("\n");
		
		if( nn->weight_gradient != NULL ){
		printf("weight_gradient between layers %lld - %lld\n",i,i+1);
		for(ulli j=0;j<nn->use_layer[i+1].layer_width;j++)
		{
			printf("\t");
			for(ulli k=0;k<nn->use_layer[i].layer_width;k++)
			{
				printf("%lf ",nn->weight_gradient[i][k][j]);
			}
			printf("\n");
		}
		printf("\n");}
		
		if(nn->weight_gradient != NULL ){
		printf("weight_change between layers %lld - %lld\n",i,i+1);
		for(ulli j=0;j<nn->use_layer[i+1].layer_width;j++)
		{
			printf("\t");
			for(ulli k=0;k<nn->use_layer[i].layer_width;k++)
			{
				printf("%lf ",nn->weight_change[i][k][j]);
			}
			printf("\n");
		}
		printf("\n");}
		
	}
	
	printf("layer %lld :",nn->layer_count-1);
	print_layer( nn->use_layer + nn->layer_count - 1 );
	printf("\n--------------------------------------------------------------------------------\n\n\n");
}
