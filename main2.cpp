#include "math.h"
#include "math.cpp"
#include "string.h"
#define number_predictors 2

// Experiment Variables
int MAX_ITERATION = 1000;
float LEARNING_RATE = 0.1; 
float final_MSE;

float x_array[][number_predictors] = {{1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}};
float y_array[] = {1, 2, 3, 4, 5};
const int data_length=5;
//const int number_predictors=2;

float** allocate_x_array(float old_array[][number_predictors], int data_len){
   float **pX = (float**) malloc(sizeof(float*)* data_len);
   
   if (pX==NULL){
      std:: cout << "failled pointer allocation";
      exit(1);
   }

    for (int i = 0; i < data_len; i++)
   {  
      pX[i] = (float*) malloc(sizeof(float) * number_predictors);
      if (pX[i] == NULL) {
         std::cout << "failed pointer allocation";
         exit(1);
      }
      memcpy(pX[i], old_array[i], sizeof(float) * number_predictors); 
   }

   return pX;
}

float * allocate_y_array(float *old_arr , int data_len){

 float * pY = (float*) malloc (sizeof(float)* data_len);
   if (pY==NULL){
      std:: cout << "failled pointer allocation";
      exit(1);
   }
   memcpy(pY, old_arr, sizeof(float) * data_len);
   return pY;

}

//memcpy(pY, y_array, sizeof(float) * data_length);

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
        LinearRegressionModel(float **X_in, float *y_in, int length_in, int number_predictor){
            X = X_in;
            y = y_in;
            length = length_in;
            
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
                
                float mse = mean_squared_error(y_pred, y, length);
                //printf(" \n  MSE  %f \n", mse);
                if(max_iteration % 100 == 0){
                    weights.print_weights();
                    std::cout << "Iteration left: " << max_iteration << "; MSE = " << mse << "\n";
                }

                if (max_iteration ==1){
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


int main(){
    // Variable Initialization
    float **pX = allocate_x_array(x_array, data_length);
    float *pY = allocate_y_array( y_array, data_length);
    std:: cout << "new changes updated \n";
   
    // Training
    std::cout << "Making LinearRegressionModel \n";
    LinearRegressionModel linear_reg = LinearRegressionModel(pX, pY, data_length, number_predictors);    
    std::cout << "Training \n";
    linear_reg.train(MAX_ITERATION, LEARNING_RATE);
    
    // Testing TODO: Testing is a bit clumsy right now, we could just keep a hold out of the data
    std::cout << "Testing \n";
    float X_test[2];
    X_test[0] = 1; 
    X_test[1] = 123;
    float y_test = linear_reg.predict(X_test);
    linear_reg.weights.print_weights();

    float* weights_values= linear_reg.weights.values;
    int number_weights= linear_reg.weights.number_weights;
    float weights_array[number_weights];

    for (int i = 0; i <number_weights  ; i++)
    {
      weights_array[i]= weights_values[i];
      printf( " \n \n final weight number %d  is %f \n \n", i, weights_array[i]);
    }
    

    printf("  \n the final MSE is : %f   \n", final_MSE);
    std::cout << "Testing for X0 = " << X_test[0] << ", X1 = " << X_test[1] << "\n";
    std::cout << "y = " << y_test << "\n"; 

   
}