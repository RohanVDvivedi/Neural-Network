#ifndef activationfunctionsh
#define activationfunctionsh

// xxx_a is activation function
// xxx_g is gradient function

void logistic_a(double* y,double* x);
void logistic_g(double* dybydx,double* y,double* x);

void tanh_a(double* y,double* x);
void tanh_g(double* dybydx,double* y,double* x);

void identity_a(double* y,double* x);
void identity_g(double* dybydx,double* y,double* x);

void relu_a(double* y,double* x);
void relu_g(double* dybydx,double* y,double* x);

void elu_a(double* y,double* x);
void elu_g(double* dybydx,double* y,double* x);

void arctan_a(double* y,double* x);
void arctan_g(double* dybydx,double* y,double* x);

void adaptlog_a(double* y,double* x);
void adaptlog_g(double* dybydx,double* y,double* x);

enum activation_type{LOGISTIC = 0,TANH,IDENTITY,RELU,ELU,ARCTAN,ADAPTLOG};typedef enum activation_type activation_type;

extern void (*funct_a[])(double* y,double* x);
extern void (*funct_g[])(double* dybydx,double* y,double* x); 

#endif
