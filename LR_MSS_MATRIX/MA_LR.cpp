#include <stdio.h>
#include <stdlib.h>
#include <Eigen/Dense>
#include "MA_LR.h"




/*void generate_dataset(Eigen::MatrixXd &X, Eigen::VectorXd &Y, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 500.0);
    
    X = Eigen::MatrixXd::Ones(n, NUMPRED); // Initialize X with ones
    Y = Eigen::VectorXd::Zero(n); // Initialize Y with zeros
    
    for (int i = 0; i < n; ++i) {
        double random_value = dis(gen);
        X(i, 0) = random_value;
        Y(i) = random_value <= 500.0 ? random_value *1 : 1000.0 - random_value;
    }
} */


Eigen::VectorXd Matrix_MSS(const Eigen::MatrixXd &X , const Eigen::VectorXd &Y){
   
    Eigen::MatrixXd X_transpo = X.transpose();

    Eigen::MatrixXd Temp =  (X_transpo * X);

    if (Temp.determinant() == 0){
        printf("matrix cant be inverted \n");
        exit(1);
    }

    Eigen::MatrixXd X_inverted = Temp.inverse();

    Eigen::MatrixXd X_Y = X_transpo * Y;

    Eigen::VectorXd Final_ma = X_inverted * X_Y;

    return Final_ma;


}



/*int main(){
    // Variable Initialization

    Eigen::MatrixXd X;
    Eigen::VectorXd Y;

    Eigen::MatrixXd Pred;



    generate_dataset(X, Y, DATASIZE);

    Pred = Matrix_MSS(X,Y);

    std::cout << "Here is the matrix mat:" << std::endl <<Pred << std::endl;
    
    return 0;
}*/