
#include <cmath>
#include <iostream>
#include <cstring> 
#include "ma.h"
#include <random>
#include "lin_regre.h"
#include <algorithm>

// Experiment Variables

const float LEARNING_RATE = 0.05;
int number_predictors=  2;
const double Max_Num_generated = 500.0;
int  N = 10;

float final_MSE;


//max iteration is subtracted on the training, thats why its not a const or define
  
float revert__weights( float X, float YIQR, float XIQR){
    return X/(YIQR/XIQR);
}

void sort_array(float ** X, float *Y){

for (int i = 0; i < N-1; i++)
{   
    if (X[i][1] > X[i+1][1])
    {
         float temp = X[i+1][1];
         X[i+1][1] = X[i][1];
         X[i][1] = temp;
       
    }

    if (Y[i] > Y[i+1]){

        float temp2;
        temp2 =  Y[i+1];
        Y[i+1] = Y[i];
        Y[i] = temp2;
    }
   
}

}
  


float *robust_scaler(float **X, float *Y){

  int pQ1 = N/2;
  int pQ3 = N * 3 / 4;


    float XQ1 = X[pQ1][1];
    float XQ3 = X[pQ3][1];
    float YQ1 = Y[pQ1];
    float YQ3 = Y[pQ3];

    float XIQR = XQ3 - XQ1;
    float YIQR = YQ3 - YQ1;
 
   float Xmedian = X[N/2][1];
   float Ymedian= Y[N/2];
  
  float* return_array = (float *) malloc(sizeof(float)*4);
  if (return_array == NULL) {
         std::cout << "failed pointer allocation";
         exit(1);
      }

  for (int i = 0; i < N; i++)
  {
    X[i][1] = (X[i][1] - Xmedian)/XIQR;
    Y[i] = (Y[i] - Ymedian)/YIQR;
  }

    return_array[0] = Xmedian;
    return_array[1] = Ymedian;
    return_array[2] = XIQR;
    return_array[3] = YIQR;

  return return_array;
  
}

/****************** WEIGHTS ******************/

class Weights{
    private:
        int MAX_WEIGHTS;

    public:
        float* values;
        int number_weights;

        Weights(){};

        Weights(int number_predictor){
            number_weights = number_predictor;
            values = (float *) std::calloc(number_weights, sizeof(float));
             std::random_device rd;
            std::mt19937 gen(rd());
           // std::uniform_real_distribution<> dis();
            for (int i = 0; i < number_weights; ++i) {
             // values[i] = dis(gen);
    }
        };

        void update(float **X, float *y, float *y_pred, float learning_rate, int length){

            float multiplier = learning_rate/length;
            // Update each weights
            for(int i = 0; i < number_weights; i++){
                float sum = (sum_residual(X, y, y_pred, i, length));
                //printf("Gradient[%d] = %f\n", i, sum);
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
            //strcpy(function_string, "y = ");

            for(int i = 0; i < number_weights; i++){
                printf("Weights %d is = %f\n",i, values[i]);

                char weight[20];
                sprintf(weight,"%.2f * x%d", values[i],i);
               // strcat(function_string, weight);

                if(i == number_weights-1){
                   // strcat(function_string,"\n");
                }else{
                   // strcat(function_string," + ");
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
    float bias;

    public:
        Weights weights;
        LinearRegressionModel(float **X_in, float *y_in, int length_in, int number_predictor){


            X = X_in;
            y = y_in;
            length = length_in;
            bias = 0.0;
            for (int i = 0; i < N; ++i) {
       // std::cout <<  "\n loaded X " << X[i][0] << " " << X[i][1] << std::endl;
           }

           for (int i = 0; i < N; i++)
           {
           // std::cout << "\n loaded Y" << y[i] << std::endl;
           }
                      
            weights = Weights(number_predictor);
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
                float bias_gradient = 0.0;
                for (int i = 0; i < length; i++){
                   // bias_gradient += (y_pred[i] - y[i]);
                }
                bias -= learning_rate * bias_gradient / length;
                
                float mse = mean_squared_error(y_pred, y, length);
                //printf(" \n  MSE  %f \n", mse);
                if(max_iteration % 100 == 0){
                    weights.print_weights();
                    std::cout << "Iteration left: " << max_iteration << "; MSE = " << mse << " Bias = " << bias << "\n";;
                }

                // if (mse < 0.000001){
                //     printf( "good MSE reached: %lf  in %d generations \n", mse, max_iteration);
                //     break;
                // }

                if (max_iteration ==1){
                  final_MSE = mse;
                }
                max_iteration--;
            }
            free(y_pred);
        }

        // Run the an array of predictor through the learned weight
        float predict(float *x){
            float prediction = bias;
                for(int i = 0; i < weights.number_weights; i++){
                    prediction = prediction + weights.values[i]*x[i];
                }
            return prediction;
        }
};


float *normalized_Lin_Regre(float** x_array2, float * y_array){
  
    
   float *return_array= (float*) malloc (sizeof(float)* number_predictors);  //this will be the array returned by the function, it will return the angular coeficient (second weight) and the RMSE
    if (return_array ==  NULL){
        printf(" failed pointer allocation");
        exit(1);
    }

      for (int i = 0; i <N ; i++){    
        printf(" \n X value loading  %f \n", x_array2[1][i]);
        printf(" \n y value loading  %f \n", y_array[i]);
    }
   sort_array(x_array2,y_array);
   printf("sorted the array \n"); 

   for (int i = 0; i <N ; i++){    
      printf(" \n X value loading after sort  %f \n", x_array2[1][i]);
      printf(" \n y value loading  after sort  %f \n", y_array[i]);
   }

   std:: cout << "\n new data loaded \n";
   std::cout << "Making LinearRegressionModel \n";

   float *normalizer_data;
   normalizer_data = robust_scaler(x_array2,y_array);  //this function normalizes the dataset and return info from the process

   LinearRegressionModel linear_reg = LinearRegressionModel(x_array2, y_array, N, number_predictors);    
  
   linear_reg.train(MAX_ITERATION, LEARNING_RATE);

   float* weights_values= linear_reg.weights.values;
   int number_weights= linear_reg.weights.number_weights;
   float weights_array[number_weights];

    for (int i = 1; i <number_weights  ; i++)
    {
      weights_array[i] = revert__weights(weights_values[i],normalizer_data[2],normalizer_data[3]);;
      printf( " \n \n final weight number %d  is %f \n \n", i, weights_array[i]);
    }

     for (int i = 1; i < number_predictors; i++)
    {
      return_array[i-1] = weights_array[i];
    }

     return_array[number_predictors-1] = sqrt(final_MSE);

     return (return_array);
}
