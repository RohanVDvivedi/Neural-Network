#include <../inc/neuralnet.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>


// using this method a neural net structure can be loaded from a file
// thus enabling us to provide a proper interface to load a structure from the memory
void load_neuralnet(char* filename,neuralnet* nn,init_type it)
{
	FILE* fp = fopen(filename,"r");
	
	fscanf(fp,"%*s %*s %*s %*s %lld",&(nn->layer_count) );
	fscanf(fp,"%*s %*s %*s %lf", &(nn->learning_rate) );
	
	ulli* layer_sizes = (ulli*) (malloc(sizeof(ulli)*nn->layer_count));
	enum activation_type* layer_act_type = (enum activation_type*) (malloc(sizeof(enum activation_type)*nn->layer_count));
	
	ulli layer_number;
	char act_type_temp[100];
	for(ulli i=0;i<nn->layer_count;i++)
	{
		fscanf(fp,"%*s %lld %*s %s %*s %*s",&layer_number,act_type_temp);
		fscanf(fp,"%lld %*s",layer_sizes + layer_number);

		ulli j = 0;
		while( strcmp( funct_name[j] , act_type_temp ) != 0 ){j++;}
		
		layer_act_type[layer_number] = ( (enum activation_type) j );
	}
	
	switch(it)
	{
		case min:
			{
				min_init_neuralnet(nn,nn->layer_count,layer_sizes,layer_act_type);
				break;
			}
		case full:
			{
				init_neuralnet(nn,nn->layer_count,layer_sizes,layer_act_type,0.0,nn->learning_rate);
				break;
			}
	}
	
	for(ulli i=0;i<nn->layer_count;i++)
	{
		fscanf(fp,"%*s %*s %*s %*s %lld %*s",&layer_number);
		for(ulli j=0;j<nn->use_layer[layer_number].layer_width;j++)
		{
			fscanf(fp,"%lf ",&(nn->use_layer[layer_number].bias[j]) );
		}
	}
			
	
	for(ulli i=0;i<nn->layer_count-1;i++)
	{
		fscanf(fp,"%*s %*s %*s %*d %*s %*d");
		for(ulli j=0;j<nn->use_layer[i+1].layer_width;j++)
		{
			for(ulli k=0;k<nn->use_layer[i].layer_width;k++)
			{
				fscanf(fp,"%lf ",&(nn->weight[i][k][j]));
			}
		}
	}
	
	free(layer_act_type);
	free(layer_sizes);
	fclose(fp);
}

// this method is used to store a neural net in to memory
// it does not store instanteous values only weights and neural net work structure is saved
// instantaneous values like weight gradients , weight changes , layer wise outputs etc are not saved
void store_neuralnet(char* filename,neuralnet* nn)
{
	FILE* fp = fopen(filename,"w");
	
	if( fp == NULL )
	{
		printf("could not open file \"%s\"\n\n",filename);
		return ;
	}

	fprintf(fp,"number of layers : %lld\n\n",nn->layer_count);
	fprintf(fp,"learning rate : %lf\n\n",nn->learning_rate);
	
	for(ulli i=0;i<nn->layer_count;i++)
	{
		fprintf(fp,"layer %lld has %s activation for %lld neurons\n",i,funct_name[nn->use_layer[i].act_type],nn->use_layer[i].layer_width);
	}
	
	for(ulli i=0;i<nn->layer_count;i++)
	{
		fprintf(fp,"bias for the layer %lld : \t ",i);
		for(ulli j=0;j<nn->use_layer[i].layer_width;j++)
		{
			fprintf(fp,"%lf ",nn->use_layer[i].bias[j]);
		}
		fprintf(fp,"\n\n");
	}
	
	fprintf(fp,"\n\n");
	
	for(ulli i=0;i<nn->layer_count-1;i++)
	{
		fprintf(fp,"weights between layers %lld - %lld\n",i,i+1);
		for(ulli j=0;j<nn->use_layer[i+1].layer_width;j++)
		{
			fprintf(fp,"\t");
			for(ulli k=0;k<nn->use_layer[i].layer_width;k++)
			{
				fprintf(fp,"%lf ",nn->weight[i][k][j]);
			}
			fprintf(fp,"\n");
		}
		fprintf(fp,"\n\n");
	}
	
	fclose(fp);
}

// once the training is complete the neural net structure can be released of its burden memory which is not going to be further used
void training_complete(neuralnet* nn)
{
	if(nn->under_training == 0)
	{
		return ;
	}
	
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
		free(nn->weight_gradient);nn->weight_gradient=NULL;}
	
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
		free(nn->weight_change);nn->weight_change=NULL;}
	
		if(nn->desired_output!=NULL){free(nn->desired_output);nn->desired_output=NULL;}

	for(ulli i=0;i<nn->layer_count;i++)
	{
		minimize_layer( nn->use_layer + i );
	}
	
	nn->input = (double*) ( malloc( sizeof(double) * (nn->input_width) ) );
	nn->output = (double*) ( malloc( sizeof(double) * (nn->output_width) ) );
	
	nn->temp_layer.output = (double*) malloc( sizeof(double) * nn->temp_size );
	nn->temp_layer.activated_output = (double*) malloc( sizeof(double) * nn->temp_size );
	nn->temp_layer.bias = NULL;nn->temp_layer.bias_change = NULL;nn->temp_layer.output_gradient = NULL;
	
	nn->under_training = 0;
}

// for you neural network that has been already trained, you can set the inputs then call this function to get the results in the output array of the structure
// this method is same as feed_forward method only difference lies in the state when it is implemented, it utilizes far lesser memory if this function is called after training is complete
void get_solution(neuralnet* nn)
{
	
	COPY_VECT_VECT( nn->temp_layer.activated_output , nn->input, nn->input_width );
	nn->temp_layer.act_type = nn->use_layer[0].act_type;
	nn->temp_layer.layer_width = nn->input_width;
	
	for(ulli i=0;i<nn->layer_count-1;i++)
	{
		COPY_VECT_VECT( nn->temp_layer.output , nn->use_layer[i + 1].bias, nn->use_layer[i + 1].layer_width );
		
		for(ulli j=0;j<nn->use_layer[i].layer_width;j++)
		{
			COMPUTE_VECT_SCALER( nn->temp , nn->weight[i][j] , '*' , nn->temp_layer.activated_output[j] , nn->use_layer[i+1].layer_width );
			COMPUTE_VECT_VECT( nn->temp_layer.output , nn->temp_layer.output , '+' , nn->temp , nn->use_layer[i+1].layer_width  );
		}
		nn->temp_layer.act_type = nn->use_layer[i+1].act_type;
		nn->temp_layer.layer_width = nn->use_layer[i+1].layer_width;
		activate_layer( &(nn->temp_layer) );
	}
	
	COPY_VECT_VECT( nn->output , nn->temp_layer.activated_output , nn->output_width );
}
