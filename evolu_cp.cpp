#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
//#include "lin_regre.cpp"
#include "lin_regre.h"
#include <cmath>
#include <iostream>
#include <iomanip>


#define maxx 1000
#define TamPop 10
#define LRTRAINSIZE 10 //How many of the best individuals from past iterations are saved to train the linear regression model
#define number_predictors 2
#define MAX_ITERATION 1000

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

   for (int i = 0; i < LRTRAINSIZE; i++)
 {
    printf("X array %f  \n", LR_X_Array[i][1]);
 }

  float* return_arr = train_use_lin_regre(LR_X_Array, LR_Y_array, LRTRAINSIZE , number_predictors,MAX_ITERATION); //Last index of the return_arr is the RMSE of the LR model
  
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