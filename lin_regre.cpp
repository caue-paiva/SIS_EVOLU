#include "math.h"
#include "math.cpp"
#include "string.h"
#include <cmath>
#include "lin_regre.h"

#define NUM_PRED 2

// Experiment Variables
float LEARNING_RATE = 0.01; 
float final_MSE;


float** allocate_x_array2(float **old_array, int data_len){
   float **pX = (float**) malloc(sizeof(float*)* data_len);
   
   if (pX==NULL){
      std:: cout << "failled pointer allocation";
      exit(1);
   }

    for (int i = 0; i < data_len; i++)
   {  
      pX[i] = (float*) malloc(sizeof(float) * NUM_PRED);
      if (pX[i] == NULL) {
         std::cout << "failed pointer allocation";
         exit(1);
      }
      pX[i] = old_array[i];
   }

   return pX;
}

float * allocate_y_array2(float *old_arr , int data_len){

 float * pY = (float*) malloc (sizeof(float)* data_len);
   if (pY==NULL){
      std:: cout << "failled pointer allocation";
      exit(1);
   }
   pY = old_arr;
   return pY;

}

/****************** WEIGHTS ******************/

class Weights{
    private:
        int MAX_WEIGHTS;

    public:
        float* values;
        int number_weights;

        Weights(){};

        Weights(int numberpredictor){
            number_weights = numberpredictor;
            values = (float *) std::calloc(number_weights, sizeof(float));
        };

        void update(float **X, float *y, float *y_pred, float learning_rate, int length){

            float multiplier = learning_rate/length;
            // Update each weights
            for(int i = 0; i < number_weights; i++){
                float sum = (sum_residual(X, y, y_pred, i, length));
                //printf("Sum = %f\n",sum);
                values[i] = values[i] - multiplier*sum;
            }
        }

        float sum_residual(float **X, float *y, float *y_pred, int current_predictor,  int length){
            float total = 0;
            float residual;

            for(int i = 0 ; i < length; i++){
                residual = (y_pred[i] - y[i]);
                total = total + residual*X[i][current_predictor];
            }
            return total;
        }

        // Pretty print the weights of the model
        void print_weights(){
            char function_string[1000];
            printf("Number weights = %d\n", number_weights);
            strcpy(function_string, "y = ");

            for(int i = 0; i < number_weights; i++){
                printf("Weights %d is = %f\n",i, values[i]);

                char weight[20];
                sprintf(weight,"%.2f * x%d", values[i],i);
                strcat(function_string, weight);

                if(i == number_weights-1){
                    strcat(function_string,"\n");
                }else{
                    strcat(function_string," + ");
                }
            }
            printf("%s\n",function_string);
        }
};

// Model class for Linear Regression
// Use MSE and gradient descent for optimization
// of the parameters
class LinearRegressionModel{

    // Models Variable
    float **X;
    float *y;
    int length;

    public:
        Weights weights;
        LinearRegressionModel(float **X_in, float *y_in, int length_in, int numberpredictor){
            X = X_in;
            y = y_in;
            length = length_in;
            
            weights = Weights(numberpredictor);
        }

        // Main training loop 
        void train(int max_iteration, float learning_rate){
            
            float *y_pred = (float *) std::malloc(sizeof(float)*length);

            while(max_iteration > 0){

                // Will predict the y given the weights and the Xs
                for(int i = 0; i < length; i++){
                    y_pred[i] = predict(X[i]);
                }

                weights.update(X, y, y_pred, learning_rate, length);
                
                float mse = mean_squared_error(y_pred, y, length);
               // printf("\n MSE:  %f  \n ", mse);
                if(max_iteration % 100 == 0){
                    weights.print_weights();
                    std::cout << "Iteration left: " << max_iteration << "; MSE = " << mse << "\n";
                }

                if (max_iteration == 1){
                  final_MSE = mse;
                }
                max_iteration--;
            }
            free(y_pred);
        }

        // Run the an array of predictor through the learned weight
        float predict(float *x){
            float prediction = 0;
                for(int i = 0; i < weights.number_weights; i++){
                    prediction = prediction + weights.values[i]*x[i];
                }
            return prediction;
        }
};

float* train_use_lin_regre(float** x_array2, float * y_array, int data_length, int numberpredictor, int MX_ITER){
   
   float *return_array= (float*) malloc (sizeof(float)* numberpredictor);  //this will be the array returned by the function, it will return the angular coeficient (second weight) and the RMSE
 
  // the calculation is (number_predictiors -1 [since we will not use the linear coeficient]) +1 , since we will return the RMSE, so its simply the number of predictors

   float **pX = allocate_x_array2(x_array2, data_length);
   float *pY = allocate_y_array2(y_array, data_length);

   for (int i = 0; i <data_length ; i++)
   {    printf(" \n X value loading  %f \n", x_array2[1][i]);
       // printf(" \n y value loading  %f \n", pY[i]);
   }
   

   std:: cout << "\n new data loaded \n";
   std::cout << "Making LinearRegressionModel \n";

   LinearRegressionModel linear_reg = LinearRegressionModel(pX, pY, data_length, numberpredictor);    
   linear_reg.train( MX_ITER, LEARNING_RATE);

   float* weights_values= linear_reg.weights.values;
   int number_weights= linear_reg.weights.number_weights;
   float weights_array[number_weights];

    for (int i = 1; i <number_weights  ; i++)
    {
      weights_array[i]= weights_values[i];
      printf( " \n \n final weight number %d  is %f \n \n", i, weights_array[i]);
    }

    
    for (int i = 1; i < numberpredictor; i++)
    {
      return_array[i-1] = weights_array[i];
    }
    
    return_array[numberpredictor-1] = sqrt(final_MSE);
    
   free(pY);
   
   for (int i = 0; i < data_length; i++)
   {
     free(pX[i]);
   }

   free(pX);
   
   return (return_array);
   
}
