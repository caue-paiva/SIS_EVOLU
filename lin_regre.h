#ifndef LIN_REGRE_H
#define LIN_REGRE_H

float** allocate_x_array2(float** array, int size);
float* allocate_y_array2(float* array, int size);
float* train_use_lin_regre(float** x_array, float* y_array, int data_length, int number_predictors, int MAX_ITERATION);

#endif // LIN_REGRE_H