#include "math.h"
#include "math.cpp"
#include "string.h"
#include <cmath>
#include "lin_regre.h"

#define NUM_PRED 2

// Experiment Variables
float LEARNING_RATE =0.001; 
float final_MSE;

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include "lin_regre.cpp"
#include "lin_regre.h"
#include <iostream>
#include <iomanip>


#define maxx 1000
#define TamPop 10
#define LRTRAINSIZE 10 //How many of the best individuals from past iterations are saved to train the linear regression model
#define number_predictors 2
#define MAX_ITERATION 10

float MaxMut = 5.0;
bool LR_avai = false;
float indi[TamPop+1];

float**LR_X_Array;  //array of the training data for the LR model
float *LR_Y_array;

float* global_weights; //The last item of this array is the RSE of the model

float temp_indi[TamPop+1];
float fit[TamPop+1];

float max_fit=0.0;
float media = 0.0;

int i, maxi =0;
int gen=0;


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

class Weights{  /* massive problems with gradient explosion , especially gradient[1], try to use min-max escaling to solve that*/
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
             printf(" Multiplier = %f\n", multiplier);
              const float threshold = 1.0f;
            // Update each weights
            for(int i = 0; i < number_weights; i++){
                float sum = (sum_residual(X, y, y_pred, i, length));
                 printf("Gradient[%d] = %f\n", i, sum);
               
               if (sum > threshold) {
                  sum = threshold;
                } else if (sum < -threshold) {
                   sum = -threshold;
                }
 
                values[i] = values[i] - multiplier * sum;
            }
        }

        float sum_residual(float **X, float *y, float *y_pred, int current_predictor,  int length){
            float total = 0;
            float residual;

            for(int i = 0 ; i < length; i++){
                residual = (y_pred[i] - y[i]);
                total = total + residual*X[i][current_predictor];
                printf("Residual[%d] = %f\n", i, residual);
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
            for (int i = 0; i < LRTRAINSIZE; i++)
            {
              printf (" \n loaded x %f ", X[i][1]);
              printf (" \n loaded y %f ", y[i]);
            }
            
            
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
                printf("MSE[%d] = %f\n", max_iteration, mse);
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

float* train_use_lin_regre(int data_length, int numberpredictor, int MX_ITER){
   
   float *return_array= (float*) malloc (sizeof(float)* numberpredictor);  //this will be the array returned by the function, it will return the angular coeficient (second weight) and the RMSE
 
  // the calculation is (number_predictiors -1 [since we will not use the linear coeficient]) +1 , since we will return the RMSE, so its simply the number of predictors

 

   std:: cout << "\n new data loaded \n";
   std::cout << "Making LinearRegressionModel \n";

   LinearRegressionModel linear_reg = LinearRegressionModel(LR_X_Array, LR_Y_array, data_length, numberpredictor);    
   linear_reg.train(MX_ITER, LEARNING_RATE);

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
    

   
   return (return_array);
   
}




void copy_X_array(){   
    for (int i = 0; i < LRTRAINSIZE; i++){
        LR_X_Array[i][0] = 1.0;
        indi[i] = std::round(indi[i] * 100.0) / 100.0;
        LR_X_Array[i][1] = indi[i];
       // printf( " copying X value: %f \n ",  LR_X_Array[i][1]);
    }
}

void copy_Y_array(){
     for (int i = 0; i < LRTRAINSIZE; i++){
         fit[i] = std::round(fit[i] * 100.0) / 100.0;
         LR_Y_array[i] = fit[i];
        // printf( " copying Y value: %f \n",  fit[i]);
     }
}

void calculate_weights (){ 
  
  copy_X_array();
  copy_Y_array();


  float* return_arr = train_use_lin_regre(LRTRAINSIZE , number_predictors,MAX_ITERATION); //Last index of the return_arr is the RMSE of the LR model
  
  for (int i = 0; i < number_predictors-1; i++)
  {
    float weighted_value =  (( ( (57.3) * atan(return_arr[i]) ) /90.0));

    float abs_value = fabs(weighted_value);

      if (abs_value <= 0.1 ){
            return_arr[i]= 0.5;

        } else if ( weighted_value> 0){
            weighted_value -= 0.1;
            return_arr[i] = (weighted_value);

        } else if (weighted_value < 0 ){
            weighted_value += 0.1;
            return_arr[i]= (1+ (weighted_value));

        }

   }
   for (int i = 0; i < number_predictors-1; i++){
     global_weights[i]= return_arr[i];
     printf(" \n \n CURRENT WEIGHT IS : %f  \n \n", global_weights[i]);
   }
   global_weights[number_predictors-1] = return_arr[number_predictors-1];
   LR_avai = true;
   free(return_arr);
  
}
  
void alloc_X_arr(int rows, int columns){
  LR_X_Array = (float**) malloc(sizeof(float*)* rows);
    if (LR_X_Array ==NULL){
        printf("failed allocation");
        exit(1);
        }
  
  for (int i = 0; i < rows; i++)
  {
    LR_X_Array[i] = (float *) malloc( sizeof(float)* columns);
        if (LR_X_Array[i] ==NULL){
        printf("failed allocation");
        exit(1);
    }
  }   
}

void alloc_Y_arr(int size){
    LR_Y_array = (float*) malloc(sizeof(float)* size);
     if (LR_Y_array ==NULL){
        printf("failed allocation");
        exit(1);
     }

}

void free_X_arr(int rows) {
    for (int i = 0; i < rows; ++i) {
        free(LR_X_Array[i]);
    }
    free(LR_X_Array);
}

void avalia(){
 float x;
 printf("generation %d \n", gen);

 
 for (i = 1; i <= TamPop; i++)
 {
    x = indi[i];
    float y =x;
    if (x >500){
        y = 1000-x;
    }

    fit[i] = y;
     //printf("\tFitness %d (%f)= %f\n",i,indi[i],fit[i]);
 }

}
 
void init_pop(){

 for (i = 1; i <= TamPop; i++)
 {
    indi[i] = (float) (rand() % maxx);
 }
 
}

void Sele_natu(){
 
 max_fit = fit[1];
 maxi =1;

 for (i = 2; i <= TamPop; i++){
    if (fit[i]> max_fit){
        max_fit = fit[i];
        maxi=i;
    }
 }

 for (i = 1; i <= TamPop; i++){

    if (i==maxi){
        continue;
    }

    if (!LR_avai){
     indi[i] = (indi[i]+ indi[maxi])/2.0;
    } else {
      float main_wgt = global_weights[0];
      float scd_wgt = (1-main_wgt);

      indi[i] = ((indi[maxi] * main_wgt)  + (indi[i] * scd_wgt));
    }

    indi[i] = indi[i] + ((float)   (rand()%maxx)-maxx/2)*MaxMut/100.0f;                     
      
    if(indi[i]>maxx)
			indi[i]=maxx;
	if(indi[i]<0)
			indi[i]=indi[i]*-1.0;

      } 

      printf(" indv mais adaptado: %f", indi[maxi]);
 
  } 



void torneio(){

int a, b, pai1, pai2;

max_fit= fit[1];
maxi=1;

for (i=2;i<=TamPop;i++){   // Busca pelo melhor individuo para protege-lo

        if (fit[i]>max_fit)
        {
            max_fit = fit[i];
            maxi = i;
        }
    }

for (i=1;i<=TamPop;i++){
       temp_indi[i] = indi[i];
}

for ( i = 1; i<=TamPop; i++){ //torneio
    
    if (i ==maxi){
        continue;
    }

    a = (rand()%TamPop) +1;
    b = (rand()%TamPop) +1;

    if (fit[a]> fit[b]){
        pai1=a;
    } else {
        pai1=b;
    }

    a = (rand()%TamPop) +1;
    b = (rand()%TamPop) +1;

    if (fit[a]> fit[b]){
        pai2=a;
    } else {
        pai2=b;
    }
   // calculate_weights()


    if (!LR_avai){
     indi[i] = (temp_indi[pai1] + temp_indi[pai2]);
    } else {
      float main_wgt = global_weights[0];
      float scd_wgt = (1-main_wgt);

      if (temp_indi[pai1] > temp_indi[pai2]){
        indi[i] = ((temp_indi[pai1] * main_wgt)  + (temp_indi[pai2] * scd_wgt));
       } else {
        indi[i] = ((temp_indi[pai1] * scd_wgt)  + (temp_indi[pai2] * main_wgt));
       }
 
    }
   

    indi[i] = indi[i] + (double) (((rand() %maxx - (maxx/2.0))/100.0) * MaxMut);
    }

}


void ag(){

 Sele_natu();
 avalia();
 torneio();
 calculate_weights();

 gen++;

}

int main(){
 alloc_X_arr(LRTRAINSIZE, number_predictors);
 alloc_Y_arr(LRTRAINSIZE);
 srand(time(NULL));


 

init_pop();

for (int i = 0; i < 100; i++) {  
     ag();
 }

 free_X_arr(LRTRAINSIZE);
 free(LR_Y_array);
 return 0;
}