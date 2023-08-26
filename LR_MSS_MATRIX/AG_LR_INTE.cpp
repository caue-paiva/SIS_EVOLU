#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <string.h>
#include <Eigen/Dense>
#include "MA_LR.h"
#include <iostream>
#include <vector>
#include <utility>

#define maxx 1000   //make this code work and apply the weights, then create another file where i try the sorting and batching processes for each generation or passes of generation
#define TamPop 14
#define clustersize 5
#define NUMPREDI 1
#define Rad_to_Deegres 57.3
#define NUMGEN 50

const int NumClusters =  (TamPop+1)/clustersize;
int nearzero =0;
float MaxMut = 5.0;

float max_fit=0.0;
float media = 0.0;

bool WGT_ava = false;

int i, maxi =0;
int gen=0;

std::vector <std::pair<float,float>> Indi_WGTS(15);


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
   retu_mat.setZero();
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

float* calc_weights(const Eigen::VectorXd &Slopes, const Eigen::MatrixXd &X){
    float* weights_arr = (float*) malloc(sizeof(float)*NUMPREDI);
           if(weights_arr==NULL){exit(1);}
  printf("num clusters: %d", NumClusters);
   float final_wgt_arr[NumClusters];
   
   for (int i = 0; i < NUMPREDI; i++)
   { //printf("reached the new pair code \n");
     for (int k = 0; k < NumClusters; k++)
     {
       
    // Eigen::VectorXd slp = Slopes(k);
     if (fabs(Slopes(k)) < 0.30){
       printf("slope is near 0: %f \n",Slopes(k));
       nearzero++;
     }
     
     float  weighted_value =  ((((Rad_to_Deegres) * atan(Slopes(k)))));  //to calculate the weights of the crossing over, we should get the weighted value  +- 0.2 or 0.15 and then add that to the baseline 0.5, if the abs value is to low then its automatically 0
     //printf(" weighted value %f \n",weighted_value );  //try dividing by 100 instead of 90 
     weighted_value /= 100;
     float abs_value = fabs(weighted_value);
     float final_weight;

        if (abs_value <= 0.1 ){
           final_weight= 0;

        } else if ( weighted_value> 0){
            weighted_value -= 0.15;
            final_weight = (weighted_value);

        } else if (weighted_value < 0 ){
            weighted_value += 0.15;
            final_weight = (1+ (weighted_value));
        }
     
     final_wgt_arr[k] = final_weight;
    
  }
   //std::cout << "\n" << X <<"\n";
   for (int j = 0; j < NumClusters; j++)
     {  
        int index = j*clustersize;
        float upper_limit = X.block(index, 0, clustersize, NUMPREDI).maxCoeff();
        Indi_WGTS[j] = (std::make_pair(upper_limit, final_wgt_arr[j]));

       
       // printf("\n X value: %f  value loaded on pair: %f \n", upper_limit ,Indi_WGTS[i].first );
     }

   }


   WGT_ava = true;
   return weights_arr;
}



Eigen::VectorXd cluster_data(Eigen::MatrixXd &X, Eigen::VectorXd &Y){
    if ( (TamPop+1) % clustersize  != 0 ){
        printf("failed clusterization, Tampop not divizible by cluster size");
        exit(1);
    }

   // printf("enter cluster \n");
    int cluster_count =  (TamPop+1)/clustersize;
             
    Eigen::VectorXd retu_vector(cluster_count);

    for (int i = 0; i < cluster_count ; i++)
    {    int start_index =i*clustersize;
         Eigen::VectorXd Y_cluster = Y.segment(start_index , clustersize);
         Eigen::MatrixXd X_cluster = X.block(start_index, 0, clustersize, NUMPREDI);
         Eigen::VectorXd tempVec = Matrix_MSS(X_cluster,Y_cluster);
         
         retu_vector(i) = static_cast<float> (tempVec(0));
    }
     std:: cout << retu_vector << std::endl;
   // printf("exiting cluster \n");
 return retu_vector;
}

void Sort_Matrix(Eigen::MatrixXd &X, Eigen::VectorXd &Y){

    // std::cout << "unsorted x " << X << "\n";

     // std::cout << "unsorted Y " << Y << "\n";
  
  /*Eigen::VectorXd sorted_indices = X.col(NUMPREDI-1);
  Eigen::VectorXi X_indices = Eigen::VectorXd::LinSpaced(sorted_indices.rows(), 0, sorted_indices.rows()-1).cast<int>();  //creates a vector of the indices of the X vector

  std::sort(X_indices.data(), X_indices.data() + X_indices.size(),
            [&sorted_indices](int lhs, int rhs) { return sorted_indices(lhs) < sorted_indices(rhs); });

  Eigen::MatrixXd X_sorted(X.rows(), X.cols());
  Eigen::VectorXd Y_sorted(Y.rows());

   for (int i = 0; i < X_indices.size(); ++i) {
    X_sorted.row(i) = X.row(X_indices(i));
    Y_sorted(i) = Y(X_indices(i));
  }

  X = X_sorted;
  std::cout << "sorted x " << X_sorted << "\n";

  Y = Y_sorted;

  std::cout << "sorted Y " << Y_sorted << "\n";
 */
 
  for (int i = 0; i < TamPop; i++)
  {
    for (int k = 0; k < TamPop; k++)
    {
             if(X(k, NUMPREDI-1) > X(k+1, NUMPREDI-1)){
                float temp = X(k, NUMPREDI-1);
                float tempY = Y(k);

                X(k, NUMPREDI-1) = X(k+1, NUMPREDI-1);
                Y(k) = Y(k+1);

                X(k+1,NUMPREDI-1) = temp;
                Y(k+1) = tempY;

                
             }      
    }
    
  }

     // std::cout << "sorted x " << X << "\n";

     // std::cout << "sorted Y " << Y << "\n";
  
}

void avalia(float* indi, float* fit){
 float x;
 printf("generation %d \n", gen);

 for (i = 1; i <= TamPop; i++)
 {
    x = indi[i];
    float y = (2*cos(0.039*x) + 5*sin(0.05*x) + 0.5*cos(0.01*x) + 10*sin(0.07*x) + 5*sin(0.1*x) + 5*sin(0.035*x))*10+500;

    fit[i] = y;
   // printf("\tFitness %d (%f)= %f\n",i,indi[i],fit[i]);
 }

} 
 
void init_pop(float*indi){

 for (i = 1; i <= TamPop; i++)
 {
    indi[i] = (float) (rand() % maxx);
 }
 
}

float indivi_weight_relation(float indi){
   //printf(" \n indi to be searched for wgts: %d \n",  indi);
  
   for (int i = 0; i < Indi_WGTS.size(); i++)
   { //printf(" vector of pairs of floats: %d \n", Indi_WGTS[i].first);
     if ( indi <  Indi_WGTS[i].first){
       // printf("found the correct number: \n");
        return  Indi_WGTS[i].second;
     }
   }
  // printf("size of the vector of pairs %zu \n", Indi_WGTS.size() );
   return 0;
  
}

void Sele_natu(float*indi, float* fit,float** Gene_wgth_ptr){
 
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

    if (!WGT_ava){
     indi[i] = (indi[i]+ indi[maxi])/2.0;
    } else {
        //printf(" applying wgts: \n");
      float add_wgt = indivi_weight_relation(indi[i]);
     // printf("add wgt: %f \n",add_wgt);
      float main_wgt = 0.5 + add_wgt;
      float scd_wgt = (1-main_wgt);
     // printf( "sele natu main_wgt: %f \n",main_wgt);
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

void torneio(float*indi, float* fit, float* temp_indi,float** Gene_wgth_ptr){

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

   if (!WGT_ava){
     indi[i] = (temp_indi[pai1] + temp_indi[pai2]);
    } else {
        float add_wgt = indivi_weight_relation(indi[i]);
     //  printf("add wgt: %f \n",add_wgt);
      float main_wgt = 0.5 + add_wgt;
      float scd_wgt = (1-main_wgt);
     // printf( "torn 2 main_wgt: %f \n",main_wgt);
     // printf( "torn 2 second_wgt: %f \n",scd_wgt);
      if (temp_indi[pai1] > temp_indi[pai2]){
        indi[i] = ((temp_indi[pai1] * main_wgt)  + (temp_indi[pai2] * scd_wgt));
       } else {
        indi[i] = ((temp_indi[pai1] * scd_wgt)  + (temp_indi[pai2] * main_wgt));
       }
 
    }
   

    indi[i] = indi[i] + (double) (((rand() %maxx - (maxx/2.0))/100.0) * MaxMut);
    
    }

}


void ag(float*indi, float* fit, float* temp_indi, float** Gene_wgth_ptr){
 if(WGT_ava){
   for (int i = 0; i < TamPop; i++)
   {
     //printf(" array of indi %f , their representation on the pairs %f \n",indi[i], Indi_WGTS[i].first);
   } 
 }

 Sele_natu(indi,fit,Gene_wgth_ptr);
 avalia(indi,fit);
 torneio(indi,fit,temp_indi, Gene_wgth_ptr);


if (gen % 5 == 0 && gen != 0){  //novo modelo será treinado apenas a cada 5 gerações
 Eigen::MatrixXd X = convert_1D_arr_matrix(indi);
 Eigen::MatrixXd sortedX = convert_1D_arr_matrix(indi);

 Eigen::VectorXd Y = convert_1D_arr_vector(fit);
 Eigen::VectorXd parsedY = convert_1D_arr_vector(fit);

 Sort_Matrix(sortedX,parsedY);

  Eigen::VectorXd slopes = cluster_data(sortedX,parsedY);
 //Eigen::VectorXd slopes = Matrix_MSS(X,Y);

 *Gene_wgth_ptr = calc_weights(slopes,sortedX);

   /* for (int i = 0; i < TamPop; i++)
   {
     printf(" X value: %f  value loaded on pair: %f \n",sortedX(i, NUMPREDI-1),Indi_WGTS[i].first );
   } */

 
 
 //std::cout << " \n Here is the matrix mat: " << std::endl << slopes << " \n"  << std::endl;


}

/*if(WGT_ava){
    for (int i = 0; i < (TamPop+1)/clustersize ; i += clustersize)
 {
   printf(" weights at the vector of pairs %f \n", Indi_WGTS[i].second);
 }

}*/

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

  for (int i = 0; i < NUMGEN; i++) {  // Run the AG for 100 generations
     ag(indi,fit, temp_indi, &Gene_wgth);
 }

 printf("number of nearzeros: %d ", nearzero);

return 0;
}