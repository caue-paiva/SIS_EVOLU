#include "math.h"
#include "math.cpp"
#include "string.h"
#include <random>
#define number_predictors 2

// Experiment Variables
int MAX_ITERATION = 100;
double LEARNING_RATE = 0.001; 
double final_MSE;

const int N = 50;

// double x_array[][number_predictors] = {{1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}};
// double y_array[] = {1, 2, 3, 4, 5};
const int data_length=5;
//const int number_predictors=2;


void generate_dataset(double X[N][2], double Y[N], int n) {
    // Create a random device and a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1000.0);

    // Fill the first column of X with 1.0 and the second column with random numbers
    for (int i = 0; i < n; ++i) {
        X[i][0] = 1.0;
        X[i][1] = dis(gen);
    }

    // Fill the Y array based on the second column of X
    for (int i = 0; i < n; ++i) {
        if (X[i][1] <= 500.0) {
            Y[i] = X[i][1];
        } else {
            Y[i] = 1000.0 - X[i][1];
        }
    }
}

double** allocate_x_array(double old_array[][number_predictors], int data_len){
   double **pX = (double**) malloc(sizeof(double*)* data_len);
   
   if (pX==NULL){
      std:: cout << "failled pointer allocation";
      exit(1);
   }

    for (int i = 0; i < data_len; i++)
   {  
      pX[i] = (double*) malloc(sizeof(double) * number_predictors);
      if (pX[i] == NULL) {
         std::cout << "failed pointer allocation";
         exit(1);
      }
      memcpy(pX[i], old_array[i], sizeof(double) * number_predictors); 
   }

   return pX;
}

double * allocate_y_array(double *old_arr , int data_len){

 double * pY = (double*) malloc (sizeof(double)* data_len);
   if (pY==NULL){
      std:: cout << "failled pointer allocation";
      exit(1);
   }
   memcpy(pY, old_arr, sizeof(double) * data_len);
   return pY;

}

//memcpy(pY, y_array, sizeof(double) * data_length);

/****************** WEIGHTS ******************/

class Weights{
    private:
        int MAX_WEIGHTS;

    public:
        double* values;
        int number_weights;

        Weights(){};

        Weights(int number_predictor){
            number_weights = number_predictor;
            values = (double *) std::calloc(number_weights, sizeof(double));
        };

        void update(double **X, double *y, double *y_pred, double learning_rate, int length){

            double multiplier = learning_rate/length;
            // Update each weights
            for(int i = 0; i < number_weights; i++){
                double sum = (sum_residual(X, y, y_pred, i, length));
                printf("Gradient[%d] = %f\n", i, sum);
                //printf("Sum = %f\n",sum);
                values[i] = values[i] - multiplier*sum;
            }
        }

        double sum_residual(double **X, double *y, double *y_pred, int current_predictor,  int length){
            double total = 0;
            double residual;

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
    double **X;
    double *y;
    int length;

    public:
        Weights weights;
        LinearRegressionModel(double **X_in, double *y_in, int length_in, int number_predictor){


            X = X_in;
            y = y_in;
            length = length_in;
            for (int i = 0; i < N; ++i) {
        std::cout <<  "\n loaded X " << X[i][0] << " " << X[i][1] << std::endl;
           }

           for (int i = 0; i < N; i++)
           {
            std::cout << "\n loadad Y" << y[i] << std::endl;
           }
                      
            weights = Weights(number_predictor);
        }

        // Main training loop 
        void train(int max_iteration, double learning_rate){
            
            double *y_pred = (double *) std::malloc(sizeof(double)*length);

            while(max_iteration > 0){

                // Will predict the y given the weights and the Xs
                for(int i = 0; i < length; i++){
                    y_pred[i] = predict(X[i]);
                }

                weights.update(X, y, y_pred, learning_rate, length);
                
                double mse = mean_squared_error(y_pred, y, length);
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
        double predict(double *x){
            double prediction = 0;
                for(int i = 0; i < weights.number_weights; i++){
                    prediction = prediction + weights.values[i]*x[i];
                }
            return prediction;
        }
};


int main(){
    // Variable Initialization
    // double **pX = allocate_x_array(x_array, data_length);
    // double *pY = allocate_y_array( y_array, data_length);
    std:: cout << "new changes updated \n";

    double newX[N][2];
    double newY[N];

    generate_dataset(newX, newY, N);

     std::cout << "2D array (X):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << newX[i][0] << " " << newX[i][1] << std::endl;
    }

    std::cout << "\n1D array (Y):" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << newY[i] << std::endl;
    }

    // Training
    std::cout << "Making LinearRegressionModel \n";

        double **npX = new double*[N];
    for (int i = 0; i < N; ++i) {
        npX[i] = new double[2];
        npX[i][0] = newX[i][0];
        npX[i][1] = newX[i][1];
    }

    double *npY = new double[N];
    for (int i = 0; i < N; ++i) {
        npY[i] = newY[i];
    }

    LinearRegressionModel linear_reg = LinearRegressionModel(npX, npY, data_length, number_predictors);    
    std::cout << "Training \n";
    linear_reg.train(MAX_ITERATION, LEARNING_RATE);
    
    // Testing TODO: Testing is a bit clumsy right now, we could just keep a hold out of the data
    std::cout << "Testing \n";
    double X_test[2];
    X_test[0] = 1; 
    X_test[1] = 499;
    double y_test = linear_reg.predict(X_test);
    linear_reg.weights.print_weights();

    double* weights_values= linear_reg.weights.values;
    int number_weights= linear_reg.weights.number_weights;
    double weights_array[number_weights];

    for (int i = 0; i <number_weights  ; i++)
    {
      weights_array[i]= weights_values[i];
      printf( " \n \n final weight number %d  is %f \n \n", i, weights_array[i]);
    }
    

    printf("  \n the final MSE is : %f   \n", final_MSE);
    std::cout << "Testing for X0 = " << X_test[0] << ", X1 = " << X_test[1] << "\n";
    std::cout << "y = " << y_test << "\n"; 

   
}