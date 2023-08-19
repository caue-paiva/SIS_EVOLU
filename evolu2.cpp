#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstring> 
#include "ma.h"
#include <random>
#include <algorithm>

/* try to  orde the training set into another array first, then break that array on clusters of 4 ordered and train each one, then assign a cluster value to an ML angular coeficient on a pair in the indi[] array
*/

#define maxx 1000
#define TamPop 16
//How many of the best individuals from past iterations are saved to train the linear regression model
#define number_predictors 2
#define MAX_ITERATION 300  //how many iterations to train the LR model
#define Rad_to_Deegres 57.3

const int  LRTRAINSIZE = TamPop;
const int LRCLUSTERSIZE = 5;
const float LEARNING_RATE = 0.05;

int LR_RUN_COUNT = 0;

float MaxMut = 5.0;
bool LR_avai = false;

float indi[TamPop+1];   //the indi array will contain the individuals on its first  row, their fit on the second and their LR cluster assigment on the third 

float **LR_X_Array;  //array of the training data for the LR model
float **LR_X_Array2;
float *LR_Y_array;
float *LR_Y_array2;

std::pair<double, double> global_weight[TamPop+1]; //The first item of this pair is the ordered individual set, the second are their respective weights

float temp_indi[TamPop+1];
float fit[TamPop+1];

float max_fit=0.0;
float media = 0.0;

int i, maxi =0;
int gen=0;


float final_MSE;

 //max iteration is subtracted on the training, thats why its not a const or define
  
float revert__weights( float X, float YIQR, float XIQR){
    return X/(YIQR/XIQR);
}

void sort_array(float ** X, float *Y){

for (int i = 0; i < LRTRAINSIZE-1; i++)
{

for (int i = 0; i < LRTRAINSIZE-1; i++)
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

}
  
float *robust_scaler(float **X, float *Y){

   int pQ1 = LRCLUSTERSIZE/2;
   int pQ3 = LRCLUSTERSIZE * 3 / 4;


    float XQ1 = X[pQ1][1];
    float XQ3 = X[pQ3][1];
    float YQ1 = Y[pQ1];
    float YQ3 = Y[pQ3];

    float XIQR = XQ3 - XQ1;
    float YIQR = YQ3 - YQ1;
 
   float Xmedian = X[LRCLUSTERSIZE/2][1];
   float Ymedian= Y[LRCLUSTERSIZE/2];
  
  float* return_array = (float *) malloc(sizeof(float)*4);
  if (return_array == NULL) {
         std::cout << "failed pointer allocation";
         exit(1);
      }

  for (int i = 0; i < LRCLUSTERSIZE; i++)
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
        double* values;
        int number_weights;

        Weights(){};

        Weights(int number_predictor){
            number_weights = number_predictor;
            values = (double *) std::calloc(number_weights, sizeof(double));
             std::random_device rd;
             std::mt19937 gen(rd());
             std::uniform_real_distribution<> dis(-0.01, 0.01);
            for (int i = 0; i < number_weights; ++i) {
              values[i] = dis(gen);
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
            //printf("Number weights = %d\n", number_weights);
            //strcpy(function_string, "y = ");

            for(int i = 0; i < number_weights; i++){
               // printf("Weights %d is = %lf\n",i, values[i]);

               
            }
           
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
               // bias -= learning_rate * bias_gradient / length;
                
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


double* normalized_Lin_Regre(float** x_array2, float * y_array, int batch_size){
  
    double *return_array= ( double*) malloc (sizeof( double)* number_predictors);  //this will be the array returned by the function, it will return the angular coeficient (second weight) and the RMSE
    if (return_array ==  NULL){
        printf(" failed pointer allocation");
        exit(1);
    }

   
   for (int i = 0; i < batch_size-1; i++)
   {   
    if (x_array2[i][1] > x_array2[i+1][1])
    {
         float temp = x_array2[i+1][1];
         x_array2[i+1][1] = x_array2[i][1];
         x_array2[i][1] = temp;
       
    }

    if (y_array[i] > y_array[i+1]){

        float temp2;
        temp2 =  y_array[i+1];
        y_array[i+1] = y_array[i];
        y_array[i] = temp2;
    }
   
}
   printf("sorted the array \n"); 

   for (int i = 0; i <batch_size ; i++){    
      printf(" \n X value loading after sort  %f \n", x_array2[i][1]);
      printf(" \n y value loading  after sort  %f \n", y_array[i]);
   }

   std:: cout << "\n new data loaded \n";
   std::cout << "Making LinearRegressionModel \n";

   float *normalizer_data;
   normalizer_data = robust_scaler(x_array2,y_array);  //this function normalizes the dataset and return info from the process
    for (int i = 0; i <batch_size ; i++){    
      printf(" \n X value loading after normalization  %f \n", x_array2[i][1]);
      printf(" \n y value loading  after normalization  %f \n", y_array[i]);
    }

   LinearRegressionModel linear_reg = LinearRegressionModel(x_array2, y_array, batch_size, number_predictors);    
  
   linear_reg.train(MAX_ITERATION, LEARNING_RATE);

   double* weights_values= linear_reg.weights.values;
   int number_weights= linear_reg.weights.number_weights;
   double weights_array[number_weights];

    for (int i = 1; i <number_weights  ; i++){
      weights_array[i] = revert__weights(weights_values[i],normalizer_data[2],normalizer_data[3]);;
      printf( " \n \n final weight number %d  is %lf \n \n", i, weights_array[i]);
    }

    for (int i = 1; i < number_predictors; i++){
      return_array[i-1] = weights_array[i];
    }

     return_array[number_predictors-1] = sqrt(final_MSE);
     LR_RUN_COUNT++;
     return (return_array);  
}

void copy_X_array(int start, float** arr){   

    int k =0;
    for (int i = start+1; i < start+LRCLUSTERSIZE; i++){
        arr[k][0] = 1.0;
        indi[i] = std::round(indi[i] * 100.0) / 100.0;
        arr[k][1] = indi[i];
        k++;
       // printf("copying X value: %f \n ",  LR_X_Array[i][1]);
      
    }
  
}

void copy_Y_array(int start, float* arr){

    int k =0;
    for (int i = start+1; i < start+LRCLUSTERSIZE; i++){
         fit[i] = std::round(fit[i] * 100.0) / 100.0;
         arr[k] = fit[i];
         k++;
       // printf( " copying Y value: %f \n",  fit[i]);
     }
           
}

void calculate_weights (){ 
 double* return_arr;

  for (int i = 0; i < LRTRAINSIZE/LRCLUSTERSIZE; i++){   
    copy_X_array(i*LRCLUSTERSIZE, LR_X_Array);
    copy_Y_array(i*LRCLUSTERSIZE, LR_Y_array);
    copy_X_array(i*LRCLUSTERSIZE, LR_X_Array2);
    copy_Y_array(i*LRCLUSTERSIZE, LR_Y_array2);

   return_arr = normalized_Lin_Regre(LR_X_Array, LR_Y_array,LRCLUSTERSIZE);

    printf(" The angular coeficient is : %f  \n",return_arr[0]);

    double weighted_value =  ((((Rad_to_Deegres) * atan(return_arr[0])) /90.0));

    double abs_value = fabs(weighted_value);
     
      if (abs_value <= 0.1 ){
            return_arr[0]= 0.5;

        } else if ( weighted_value> 0){
            weighted_value -= 0.1;
            return_arr[0] = (weighted_value);

        } else if (weighted_value < 0 ){
            weighted_value += 0.1;
            return_arr[0] = (1+ (weighted_value));
        }
    
    for (int k = 0; k <LRCLUSTERSIZE; k++){ //error here
      
     global_weight[i*LRCLUSTERSIZE].first = LR_X_Array2[k][1];
      printf(" returned values are %f ", LR_X_Array2[k][1]);
     global_weight[i*LRCLUSTERSIZE].second= return_arr[0];
     printf(" \n \n CURRENT WEIGHT IS : %f  \n \n", global_weight[i*LRCLUSTERSIZE].second);
   }
   printf("  \n putting weights on global var");
   printf("\n LR has run %d  times ", LR_RUN_COUNT);
   LR_avai = true;

   }

}
   
float** alloc_X_arr(int rows, int columns){
    float** arr = (float**) malloc(sizeof(float*)* rows);
    if (arr ==NULL){
        printf("failed allocation");
        exit(1);
        }
  
  for (int i = 0; i < rows; i++)
  {
        arr[i] = (float *) malloc( sizeof(float)* columns);
        if (arr[i] ==NULL){
        printf("failed allocation");
        exit(1);
    }
  }

  return arr;   
}

float*  alloc_Y_arr(int size){
    float*arr = (float*) malloc(sizeof(float)* size);
     if (arr ==NULL){
        printf("failed allocation");
        exit(1);
     }
     return arr;
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

double indivi_weight_relation(float indi){
   for (int i = 0; i < LRTRAINSIZE; i++)
   {
     if (indi == global_weight[i].first){
        return  global_weight[i].second;
     }
   }

   return 0;
  
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
      float main_wgt = indivi_weight_relation(indi[i]);
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
      float main_wgt = indivi_weight_relation(indi[i]);
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
//    for (int i = 0; i < TamPop; i++)
//  {
//    printf(" os individuos sao %f ", indi[i]);
//  }

 calculate_weights();
 
 
 

 gen++;

}

int main(){
 LR_X_Array = alloc_X_arr(LRCLUSTERSIZE, number_predictors);
 LR_Y_array = alloc_Y_arr(LRCLUSTERSIZE);

 LR_X_Array2 = alloc_X_arr(LRCLUSTERSIZE, number_predictors);
 LR_Y_array2 = alloc_Y_arr(LRCLUSTERSIZE);

 srand(time(NULL));


 

init_pop();

for (int i = 0; i < 1; i++) {   
     ag();
 }

 free_X_arr(LRTRAINSIZE);
 free(LR_Y_array);
 return 0;
}
