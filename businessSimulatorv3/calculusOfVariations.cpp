/*
File handles the calculation of coefficients and parameters that 
parameterize the solution to the continuous-time discounted cash-flow problem.

TODO: 
1. Refactor some aspects of businessSimluatorv3.cpp into calculusOfVariations.cpp.

*/


#include "calculusOfVariations.h"
#include "businessSimulatorv3.h"
#include <math.h>   
#include "pch.h"
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>


using namespace std;
using namespace Eigen;
using Eigen::MatrixXd;

float getLambdaWithIndex(vector<float> lambdas, int j) {
    return lambdas[j - 1];
}


void addCoefficientValue(vector<float>& lambdaICoefficients, int coeffIndex, float coeff) {
    lambdaICoefficients[coeffIndex - 1] += coeff;
}

vector<float> calculateLambdaCoefficientsFromTerm0(int T, int N) {

    vector<float> lambdaICoefficients(N, 0);
    for (int i = 1; i < T - 1; i++) {
        for (int j = 1; j < i + 1; j++) {
            addCoefficientValue(lambdaICoefficients, j, 1); 
        }
        addCoefficientValue(lambdaICoefficients, i + 1, 0.5);
    }

    return lambdaICoefficients;
}

vector<float> calculateLambdaCoefficientsFromTerm1(int T, int N) {
    vector<float> lambdaICoefficients(N, 0);
    if (T == 1) {
        addCoefficientValue(lambdaICoefficients, 1, 1.0 / 6.0);
    }
    else if (T > 1) {
        addCoefficientValue(lambdaICoefficients, 1, 0.5);
    }
    else {}
    return lambdaICoefficients;
}

vector<float> calculateLambdaCoefficientsFromTerm2(int T, int N) {
    vector<float> lambdaICoefficients(N, 0);
    if (T > 1) {
        for (int i = 1; i < T; i++) {
            addCoefficientValue(lambdaICoefficients, i, 0.5);
        }
        addCoefficientValue(lambdaICoefficients, T, 1.0 / 6.0);
        return lambdaICoefficients;
    }
    else {
        return lambdaICoefficients;
    }
}

std::tuple<MatrixXd, MatrixXd> interpolateWithIntegralConstraints(vector<float> cashFlows, float c0, float cN) {

    //printf("Interpolating Integral Constraints \n");

    auto N = cashFlows.size();
    auto A = MatrixXd(N + 2, N + 2);
    auto b = MatrixXd(N + 2, 1);

    

    for (int t = 1; t < N + 1; t++) {
        auto C_T = cashFlows[t - 1];
        b(t - 1, 0) = C_T;       
        auto term0LambdaCoeffs = calculateLambdaCoefficientsFromTerm0(t, N);
        auto term1LambdaCoeffs = calculateLambdaCoefficientsFromTerm1(t, N);
        auto term2LambdaCoeffs = calculateLambdaCoefficientsFromTerm2(t, N);

        vector<float> lambdaICoefficients(N, 0);
        for (int i = 0; i < N; i++) {
            lambdaICoefficients[i] += term0LambdaCoeffs[i] + term1LambdaCoeffs[i] + term2LambdaCoeffs[i];
        }
      
        for (int i = 0; i < N; i++) {
            A(t - 1, i) = lambdaICoefficients[i];
        }

        A(t - 1, N) = 1;
        A(t - 1, N + 1) = 0.5 * (pow(t, 2) - pow(t - 1, 2));

    }

    
    //Need to calculate the Nth equation. 
    //Calculate the c(t) on the basis of achieving the c_0 at T=0
    A(N, N) = 1;
    A(N, N + 1) = 0;
    //printf("Nth Equation Calculation \n");

    //N+1 Equation
    //Calculate the c(t) on the basis of achieving the c_N value at T=N;
    ///Need to calculate the values for the next row. 
    vector<float> lambdaCoeffValues(N, 0);
    for (int i = 1; i < N; i++) {
        for (int j = 1; j < i + 1; j++) {
            lambdaCoeffValues[j - 1] += 1;
        }
        lambdaCoeffValues[i] += 1 / 2;
    }
    lambdaCoeffValues[0] += 1 / 2;
        
    for (int i = 0; i < N; i++) {
        A(N + 1, i) = lambdaCoeffValues[i];
    }

    A(N + 1, N) = 1;
    A(N + 1, N + 1) = N;

    //Set endpoints equal to zero.
    b(N, 0) = c0;
    b(N + 1, 0) = cN;

    //Return pair of A (discretization matrix) and b. 
    auto r = std::tuple<Eigen::MatrixXd, Eigen::MatrixXd>(A, b);

    return r;
}


double functionTerm1(double t, vector<double> lambdas) {
    double val = 0.0;
    //Need to cast to int. 
    //Need to use floor in C++ 
    for (int i = 1; i < int(floor(t)); i++) {
        for (int j = 1; j < i + 1; j++) {
            val += lambdas[j - 1];
        }
        val += lambdas[i] / 2.0;
    }
    return val;
}

double functionTerm2(double t, vector<double> lambdas) {
    return std::min(pow(t, 2.0), 1.0) * lambdas[0] / 2.0;
}


double functionTerm3(double t, vector<double> lambdas) {
    if (t >= 1) {
        double val = 0.0;
        for (int i = 1; i < floor(t) + 1; i++){
            val += lambdas[i - 1];
        }
        val += 0.5 * lambdas[ceil(t) - 1] * (t - floor(t)); 
        val = val * (t - floor(t));
        return val; 
    }
    else {
        return 0.0;
    }
}


std::tuple<vector<double>, float, float> calculateParametersRepresentingF(int x0, int xT, vector<float> cashFlows) {

    //printf("Started Calculation of Parameters Representing F \n");

    float c0 = 0.0;
    float cN = 0.0;
    auto r = interpolateWithIntegralConstraints(cashFlows, c0, cN);

    Eigen::MatrixXd A = std::get<0>(r);
    Eigen:MatrixXd b = std::get<1>(r);

    //printf("Starting Linear Solve For Parameters \n");
    VectorXd x = A.lu().solve(b);
    //printf("Finished Solve For Parameters \n");

    //printf("Getting Lambdas \n");    
    
    float c_0 = x(x.size() - 2, 0);
    float c_1 = x(x.size() - 1, 0);
    //printf("Finished Getting Lambdas \n");

    vector<double> lambdasVec;

    try {        
        for (int j = 0; j < (x.size() - 2); j++) {
            //printf("Placing x on back of lambdasVec with index %i while the length of x is %i \n", j, x.size());
            lambdasVec.emplace_back(x(j, 0));
            //printf("Passing placing x on back of lambdasVec \n"); 
        }
    }
    catch (...) {
        string errorMessage = string("Placing x on back of lambdasVec ");         
        std::string fullErrorMessage = errorMessage + "\n"s;
        printf(fullErrorMessage.c_str());
        fclose(stdout);
    }
    
    //printf("Finsihded Copying Lambdas \n");
    auto result = std::tuple<vector<double>, float, float>(lambdasVec, c_0, c_1);


    //printf("Finished Calculation of Parameters Representing F \n");
    return result;
}


