#include <stdio.h>

#include <neuralnet.h>

int main()
{
	neuralnet nn;
	
	load_neuralnet("mynn.txt",&nn,min);
	
	print_neuralnet(&nn);
	
	for(int i=0;i<4;i++)
	{
		scanf("%lf %lf",&(nn.input[0]),&(nn.input[1]));
		get_solution(&nn);
		printf("%lf\n\n",nn.output[0]);
	}
	
	delete_neuralnet(&nn);
}
