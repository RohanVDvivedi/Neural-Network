#include "../inc/neuralnet.h"

// set values to the input of the neural net before calling this function
void feedforward(neuralnet* nn)
{
	
	for(ulli i=0;i<nn->layer_count-1;i++)
	{
		// initializa i+1 th layer to bias values of the same layer
		set_bias_layer( nn->use_layer + i + 1 );
		
		for(ulli j=0;j<nn->use_layer[i].layer_width;j++)
		{
			COMPUTE_VECT_SCALER( nn->temp , nn->weight[i][j] , '*' , nn->use_layer[i].activated_output[j] , nn->use_layer[i+1].layer_width );
			COMPUTE_VECT_VECT( nn->use_layer[i+1].output , nn->use_layer[i+1].output , '+' , nn->temp , nn->use_layer[i+1].layer_width  );
		}
		
		activate_layer( nn->use_layer + i + 1 );
		
	}
	
}

// update desired_output before calling this function
void update_costfunction_gradient(neuralnet* nn,double (*funct)(double desired,double calculated) )
{
	for(ulli i=0;i<nn->output_width;i++)
	{
		nn->use_layer[ nn->layer_count - 1 ].output_gradient[i] = funct( nn->desired_output[i] , nn->output[i] );
	}
}

// call update_costfunction_gradient before calling this function
void backpropogate(neuralnet* nn)
{

	for(ulli i=nn->layer_count-1;i>0;i--)
	{
		deactivate_layer( nn->use_layer + i );
	
		COPY_VECT_SCALER( nn->use_layer[i-1].output_gradient , 0 , nn->use_layer[i-1].layer_width );
		for(ulli j=0;j<nn->use_layer[i].layer_width;j++)
		{
			GET_MAT_VECT(nn->temp,nn->weight[i-1],j,nn->use_layer[i-1].layer_width);
			COMPUTE_VECT_SCALER( nn->temp,nn->temp,'*', nn->use_layer[i].output_gradient[j] , nn->use_layer[i-1].layer_width );
			COMPUTE_VECT_VECT(nn->use_layer[i-1].output_gradient,nn->use_layer[i-1].output_gradient,'+',nn->temp,nn->use_layer[i-1].layer_width);
		}
		
		for(ulli j=0;j<nn->use_layer[i-1].layer_width;j++)
		{
			COMPUTE_VECT_SCALER( nn->weight_gradient[i-1][j] , nn->use_layer[i].output_gradient ,'*', nn->use_layer[i-1].activated_output[j] , nn->use_layer[i].layer_width );
		}
	}
	
	// calculate and weight change and add it to current weight change
	for(ulli i=0;i<nn->layer_count-1;i++)
	{
		COMPUTE_VECT_SCALER( nn->temp , nn->use_layer[i+1].output_gradient , '*' , (-1)*nn->learning_rate , nn->use_layer[i+1].layer_width );
		COMPUTE_VECT_VECT( nn->use_layer[i+1].bias_change , nn->use_layer[i+1].bias_change , '+' , nn->temp , nn->use_layer[i+1].layer_width );
		for(ulli j=0;j<nn->use_layer[i].layer_width;j++)
		{
			COMPUTE_VECT_SCALER( nn->temp , nn->weight_gradient[i][j] , '*' , (-1)*nn->learning_rate , nn->use_layer[i+1].layer_width );
			COMPUTE_VECT_VECT( nn->weight_change[i][j] , nn->weight_change[i][j] , '+' , nn->temp , nn->use_layer[i+1].layer_width );
		}
	}
}

void update_weight(neuralnet* nn)
{
	for(ulli i=0;i<nn->layer_count-1;i++)
	{
		// update bias weights for all layers from layercount-1 to 1 except 0 i.e. the input layer
		COMPUTE_VECT_VECT( nn->use_layer[i+1].bias , nn->use_layer[i+1].bias , '+' , nn->use_layer[i+1].bias_change , nn->use_layer[i+1].layer_width );
		COPY_VECT_SCALER( nn->use_layer[i+1].bias_change , 0 , nn->use_layer[i+1].layer_width );
		
		// update the weights in between all of layers
		for(ulli j=0;j<nn->use_layer[i].layer_width;j++)
		{
			COMPUTE_VECT_VECT( nn->weight[i][j] , nn->weight[i][j] , '+' , nn->weight_change[i][j] , nn->use_layer[i+1].layer_width );
			COPY_VECT_SCALER( nn->weight_change[i][j] , 0 , nn->use_layer[i+1].layer_width );
		}
	}
}
