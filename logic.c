#include<stdio.h>
#include<stdlib.h>
#include<math.h>


// only include you need to make the project work
#include"./inc/neuralnet.h"


// you need to define your own cost function
// and define a method a method that implements a gradient of the cost function you want to use
// below i have defined the log gradient function that is generally used for logical functions
double costgrad(double desired,double calculated)
{
	if( calculated >= 0.99 )
	{
		calculated = 0.99;
	}
	else if( calculated <= 0.01 )
	{
		calculated = 0.01;
	}
	return ( (1-desired)/(1-calculated) ) - ( desired/calculated );
}


// main function :p
int main()
{
	// this is how you create a neural net
	neuralnet nn;
	
	// initializa neural net with 
	//		3 layers
	//		{ 2 , 3 , 1 } 			// number of neyrons in each layer
	//		{ Identity , logistic , logistic }  // the activation being used by different layers
	//		weight_range is this parameter is 4 the weights & biases are initialized randomly from -4 to 4 double values
	//		learning_rate of neuron
	init_neuralnet(&nn,3,(ulli []){2,4,1},(activation_type []){IDENTITY,LOGISTIC,LOGISTIC},4,1);
	
	// maximum sample to use from training set
	ulli maxsam = 1000;
	
	// current sample
	ulli sample = 0;
	
	int trgf = 0;
	
	double desired_output_local_array[1];
	
	while(sample < maxsam)
	{
		// take the input from user
		printf("sample number = %lld\n",sample);
		for(ulli i=0;i<nn.input_width;i++)
		{
			scanf("%lf",nn.input + i );
		}
		printf("provided input    : ");PRINT_VECT(nn.input,nn.input_width);printf("\n\n");
	
		// also take the desired value from user
		for(ulli i=0;i<nn.output_width;i++)
		{
			scanf("%lf",nn.desired_output + i );
		}
		printf("desired output    : ");PRINT_VECT(nn.desired_output,nn.output_width);printf("\n\n");
	
		// train for 90% data and then test for rest 10% data
		if(sample < ( maxsam * 90 ) / 100 ) // do training on 90% data set
		{
			printf("Training : \n");
			
			// call this functions in this sequence only for training
			// this is online training, the library can be modified for batch training, by some twitches using static update variable in update weight but i favor online training
			feedforward(&nn);						
			update_costfunction_gradient( &nn , costgrad );	// here the second parameter is the function that you created
			backpropogate( &nn );
			update_weight( &nn );
		}
		else
		{
			// now since the training is complete we can get rid of excess memory hoarded by callin the below function
			// call this function once the training is complete
			if( trgf == 0 )
			{
				training_complete(&nn);
				trgf=1;
				// once the training is complete the desired_output array is deleted
				// so we replace it with a local array just to maintain the grouping style of program
				nn.desired_output = desired_output_local_array;
			}
			
			printf("Test : \n");
			// just call get solution to find the output of the program
			get_solution(&nn);
		}
	
		// print the calculated output
		printf("calculated output : ");PRINT_VECT(nn.output,nn.output_width);printf("\n-------------------------------\n\n");
		
		sample++;
	}
	
	// to store the neuralnet on you file system if you dont want your training to be lost
	store_neuralnet("./mynn.txt",&nn);
	
	// delete neuralnet once it is done
	delete_neuralnet(&nn);
	return 0;
}
