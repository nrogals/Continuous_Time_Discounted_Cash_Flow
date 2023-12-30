#pragma once
#include <vector>
#include <string>   
#include <iostream> 
#include <Eigen/Core>
#include <Eigen/Dense>
#include <tuple>
#include <stdio.h>


using namespace std;
using namespace Eigen;
using namespace System;

//The question here is:
//why are the different exports not making it to the dll file


//extern "C" __declspec(dllexport) double calculatedParameteriziedTimeFunction(double* lambdas, int lengthOfLambdas, double A, double B, double t);
//extern "C" __declspec(dllexport) std::tuple<vector<double>, float, float> calculateParametersRepresentingF(int x0, int xT, vector<float> cashFlows);


double functionTerm1(double t, vector<double> lambdas);

double functionTerm2(double t, vector<double> lambdas);

double functionTerm3(double t, vector<double> lambdas);

std::tuple<vector<double>, float, float> calculateParametersRepresentingF(int x0, int xT, vector<float> cashFlows);