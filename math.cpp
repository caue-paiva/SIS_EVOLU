#include "math.h"

// Return the arithmetic mean of an array of variable
double mean(double *data, int length){
    double total = 0;
    for(int i = 0; i < length; i++){
        total = total + data[i];
    }
    return (total/length);
}


// sum up the square of the residual 
double total_sum_of_square(double *y, int length){
     
    double total = 0;
    double residual;
    double y_mean = mean(y,length);

    for(int i = 0 ; i < length; i++){
        residual = (y[i] - y_mean);
        total = total + (residual*residual);
    }
    return total;
}


// sum up the residual of the squared errors
double residual_sum_of_square(double *y_pred, double *y_true, int length){
    double total = 0;
    double residual;

    for(int i = 0 ; i < length; i++){
        residual = (y_true[i] - y_pred[i]);
        total = total + (residual*residual);
    }
    return total;
}

// Coefficient of determination for goodness of fit of the regression
int r2(double *y_pred, double *y_true, int length){
    double sum_squared_residual = residual_sum_of_square(y_pred,y_true,length);
    double sum_squared_total = total_sum_of_square(y_true,length);
    return (1 - ((sum_squared_residual/sum_squared_total)));
}

// wrapper function around residual sum of square in order to have a nicer
// interface to calculate MSE
double mean_squared_error(double *y_pred, double *y_true, int length){
    return residual_sum_of_square(y_pred,y_true,length)/length;
}
