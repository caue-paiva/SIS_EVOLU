#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <string.h>
#include <Eigen/Dense>
#include "MA_LR.h"
#include <iostream>

#define maxx 1000   //make this code work and apply the weights, then create another file where i try the sorting and batching processes for each generation or passes of generation
#define TamPop 10
#define NUMPREDI 1
#define Rad_to_Deegres 57.3

float MaxMut = 5.0;

float max_fit=0.0;
float media = 0.0;

bool WGT_ava = false;

int i, maxi =0;
int gen=0;

Eigen::VectorXd convert_1D_arr_vector(float* vec){
 
   Eigen::VectorXd retu_vector(TamPop+1);

   for (int i = 0; i < TamPop+1 ; i++)
   {
     retu_vector(i) = vec[i];
   }
   
return retu_vector;

}

Eigen::VectorXd convert_1D_arr_matrix(float *vec){
 
   Eigen::MatrixXd retu_mat(TamPop+1,1);

   for (int i = 0; i < TamPop+1 ; i++)
   {
     retu_mat(i) = vec[i];
   }
   
return retu_mat;

}

Eigen::MatrixXd convert_matrix( float** arr){
   Eigen::MatrixXd retu_mat((TamPop+1) , NUMPREDI);

   for (int i = 0; i <TamPop+1 ; i++)
   {
     for (int k = 0; k < NUMPREDI; k++)
     {
       retu_mat(i,k) = arr[i][k];
     }
     
   }  
 return retu_mat;
}

float* calc_weights(const Eigen::VectorXd &Slopes){
    float* weights_arr = (float*) malloc(sizeof(float)*NUMPREDI);
           if(weights_arr==NULL){exit(1);}
     
   for (int i = 0; i < NUMPREDI; i++)
   { 
     float slp = Slopes(i);
     printf("slope: %f",slp);
     float  weighted_value =  ((((Rad_to_Deegres) * atan(slp))));  //to calculate the weights of the crossing over, we should get the weighted value  +- 0.2 or 0.15 and then add that to the baseline 0.5, if the abs value is to low then its automatically 0
     printf(" weighted value %f \n",weighted_value );  //try dividing by 100 instead of 90 
     weighted_value /= 100;
     float abs_value = fabs(weighted_value);
     
      if (abs_value <= 0.1 ){
           weights_arr[i]= 0;

        } else if ( weighted_value> 0){
            weighted_value -= 0.15;
            weights_arr[i] = (weighted_value);

        } else if (weighted_value < 0 ){
            weighted_value += 0.15;
            weights_arr[i] = (1+ (weighted_value));
        }

   }
   WGT_ava = true;
   return weights_arr;
}


void avalia(float* indi, float* fit){
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
 
void init_pop(float*indi){

 for (i = 1; i <= TamPop; i++)
 {
    indi[i] = (float) (rand() % maxx);
 }
 
}

void Sele_natu(float*indi, float* fit){
 
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

    indi[i] = (indi[i]+ indi[maxi])/2.0;

    indi[i] = indi[i] + ((float)   (rand()%maxx)-maxx/2)*MaxMut/100.0f;                     
      
    if(indi[i]>maxx)
			indi[i]=maxx;
	if(indi[i]<0)
			indi[i]=indi[i]*-1.0;

      } 

      printf(" indv mais adaptado: %f", indi[maxi]);
 
  } 

void torneio(float*indi, float* fit, float* temp_indi){

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

    indi[i] = (temp_indi[pai1] + temp_indi[pai2])/ 2.0;

    indi[i] = indi[i] + (double) (((rand() %maxx - (maxx/2.0))/100.0) * MaxMut);
    }

}


void ag(float*indi, float* fit, float* temp_indi, float** Gene_wgth_ptr){
 
 Sele_natu(indi,fit);
 avalia(indi,fit);
 torneio(indi,fit,temp_indi);


if (gen % 5 == 0){  //novo modelo será treinado apenas a cada 5 gerações
 Eigen::MatrixXd X = convert_1D_arr_matrix(indi);
 Eigen::VectorXd Y = convert_1D_arr_vector(fit);
 Eigen::VectorXd slopes = Matrix_MSS(X,Y);
 *Gene_wgth_ptr = calc_weights(slopes);

  for (int i = 0; i < NUMPREDI; i++)
 {
   printf(" weight at the array %f \n", (*Gene_wgth_ptr[i]));
 }
 
 std::cout << " \n Here is the matrix mat: " << std::endl <<slopes << " \n"  << std::endl;


}

 gen++;

}

int main(){

 float* indi = (float*) malloc(sizeof(float)* (TamPop+1));
 float* fit = (float*) malloc(sizeof(float)* (TamPop+1));
 float* temp_indi = (float*) malloc(sizeof(float)* (TamPop+1));

 float* Gene_wgth;

 if(indi ==NULL || fit ==NULL || temp_indi ==NULL){exit(1);}

 srand(time(NULL));

 init_pop(indi);

  for (int i = 0; i < 100; i++) {  // Run the AG for 100 generations
     ag(indi,fit, temp_indi, &Gene_wgth);
 }

return 0;
}