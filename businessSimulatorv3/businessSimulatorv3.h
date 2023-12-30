#pragma once
using namespace System;


extern "C" __declspec(dllexport) double calculatedParameteriziedTimeFunction(double const* lambdas, int lambdasLen, double A, double B, double t);


//TODO:
//Implement each of the following. 



//Define function for calculating the discounted cash-flow. 
extern "C" __declspec(dllexport) double calculateDiscountedCashFlowWithCalculusOfVariations(const double* cashFlows, int cashFlowLength, int beginTime, int endTime, double waccDiscount);

//Define function for calculating the discounted cash-flow. 
extern "C" __declspec(dllexport) double calculateDiscountedCashFlowWithTraditional(const double* cashFlows, int cashFlowLength, int beginTime, int endTime, double waccDiscount);


//Define function for calculating the instantaneous cash-flow at time t. 
extern "C" __declspec(dllexport) double calculateInstantaneousCashFlowAtT(const double* cashFlows, int cashFlowLength, int beginTime, int endTime);


extern "C" __declspec(dllexport) double calculateAccuracyOfMethods(double t);



