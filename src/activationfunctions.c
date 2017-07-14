#include"../inc/activationfunctions.h"
#include<math.h>

// file consists of activation functions for a neural net
// convention is <function_name>_a for the activation function
// and <function_name>_g for the function that gives its derivative

void (*funct_a[])(double* y,double* x) = {
											logistic_a,
											tanh_a,
											identity_a,
											relu_a,
											elu_a,
											arctan_a,
											adaptlog_a
											};
void (*funct_g[])(double* dybydx,double* y,double* x) = {
											logistic_g,
											tanh_g,
											identity_g,
											relu_g,
											elu_g,
											arctan_g,
											adaptlog_g
											}; 


void logistic_a(double* y,double* x)
{
	(*y) = 1/( 1 + exp( ( (*x)*(-1) ) ) );
}
void logistic_g(double* dybydx,double* y,double* x)
{
	(*dybydx) = ( (*y) )*( 1 - (*y) );
}

void tanh_a(double* y,double* x)
{
	(*y) = tanh( (*x) );
}
void tanh_g(double* dybydx,double* y,double* x)
{
	(*dybydx) = 1 - pow( (*y) , 2 );
}

void identity_a(double* y,double* x)
{
	(*y) = (*x);
}
void identity_g(double* dybydx,double* y,double* x)
{
	(*dybydx) = 1;
}

void relu_a(double* y,double* x)
{
	(*y) = (*x) > 0 ? (*x) : 0 ;
}
void relu_g(double* dybydx,double* y,double* x)
{
	if( (*x) == 0 )
	{
		(*dybydx) = 0.5;
		return ;
	}
	(*dybydx) = (*x) > 0 ? 1 : 0 ;
}

void elu_a(double* y,double* x)
{
	(*y) = (*x) > 0 ? (*x) : exp( (*x) ) - 1 ;
}
void elu_g(double* dybydx,double* y,double* x)
{
	(*dybydx) = (*x) > 0 ? 1 : (*y) + 1 ; 
}

void arctan_a(double* y,double* x)
{
	(*y) = atan( (*x) );
}
void arctan_g(double* dybydx,double* y,double* x)
{
	(*dybydx) = 1 / ( 1 + pow( (*x) ,2) );
}

void adaptlog_a(double* y,double* x)
{
	(*y) = (*x) > 0 ? log( 1 + (*x) ) : ((-1) * log( 1 - (*x) )) ;
}
void adaptlog_g(double* dybydx,double* y,double* x)
{
	(*dybydx) = (*x) > 0 ? 1/( 1 + (*x) ) : 1/( 1 - (*x) ) ;
}
