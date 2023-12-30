#include "pch.h"
#include "businessSimulatorv3.h"
#include "calculusOfVariations.h"
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <chrono>
#include <random>
#include <numeric>      
#include <tuple>
#include <unordered_map>
#include <assert.h> 
#include <stdio.h>      /* printf */
#include <math.h>       /* pow */
#include <utility>      // std::pair

//Need to add boost. 
//Some example headers are: 
//Header <boost/numeric/odeint/integrate/integrate.hpp>

//TODO: Check why the warnings are on? 


using namespace std;
using namespace System; 

#include <stdio.h>      /* printf */
#include <math.h>       /* pow */
#include <iostream>
#include <boost/lambda/lambda.hpp>
#include <boost/numeric/odeint/integrate/integrate.hpp>
#include <boost/numeric/odeint/stepper/runge_kutta4.hpp>
#include <boost/numeric/odeint/integrate/integrate_const.hpp>
#include <cstdio>
#include <stdio.h>

const char* filePath = "C:\\Users\\Nate Rogalskyj\\Desktop\\Business_Simulator_Logs\\businessSimulatorOutput.txt";


enum Period { quarter, year };

int uniformGen()
{
	std::random_device rd;  // Will be used to obtain a seed for the random number engine
	std::mt19937 t(rd()); // Standard mersenne_twister_engine seeded with rd()
	std::uniform_real_distribution<> dis(-1000, 1000);	
	return dis(t);
}

int binomialTreeGen()
{
	static int i = 0;
	std::random_device rd;  // Will be used to obtain a seed for the random number engine
	std::mt19937 t(rd()); // Standard mersenne_twister_engine seeded with rd()	
		
	std::discrete_distribution<int> trinomial {1,1,1};
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);

	int trinomialOutcome = trinomial(generator) - 1;
	i += trinomialOutcome;
	
	return i;
}


class Simulator {

private: 
	int simulationNumber = 0; 
	unordered_map<int, vector<std::pair<std::string, float>>> simulationNumberToReturn;

public:
	std::tuple<vector<float>, vector<float>> simulationBussinessCashFlows(int, Period, float);
	void simulateBusiness(int numberOfPeriods, Period p, float waccPerPeriod);	
	std::vector<std::pair<string, double>> calculateRelativeDifference(vector<std::pair<string, double>> inferredDiscountedCashFlows, int simulationNumber);
};


float calculateDailyPeriodWACC(Period p, float WACC) {
	
	float waccCompoundedOnPeriod; 
	if (p == quarter) {
		//On average, there are 91 days in a quarter. This average 
		//was arrived at by taking 365.0/4 = 91.25.
		waccCompoundedOnPeriod = pow((1 + WACC), (1.0 / 91.0));
	}
	else if (p == year) {
		//On average there are 365 or 366 days in a year. 
		waccCompoundedOnPeriod = pow((1 + WACC), (1.0 / 365.0));

	}
	else {
		throw gcnew NotImplementedException();
	}

	return waccCompoundedOnPeriod;
}


float calculateDailyDCF(vector<float> dailyCashFlows, float dailyDiscountRate) {
	
	double discountedCashFlow = 0.0;
	for (int j = 0; j < dailyCashFlows.size(); j++) {
		discountedCashFlow += dailyCashFlows[j] / pow(dailyDiscountRate, j + 1); 
	}

	return discountedCashFlow;
}

double calculatedParameteriziedTimeFunction(const double* lambdas, int lambdasLen, double A, double B, double t) {

	//TODO: Check edge case appropiately for t. 
	//TODO: Complete error handling for appropiate t. 


	double retVal = 0.0;
	try {
		vector<double> lambdasVec(lambdasLen, 0.0);
		for (int i = 0; i < lambdasLen; i++) {
			double val = *(lambdas + i);
			lambdasVec[i] = val;
		}		

		retVal = functionTerm1(t, lambdasVec) + functionTerm2(t, lambdasVec) + functionTerm3(t, lambdasVec) + B * t + A;
		
	}
	catch (const std::exception& e){
		cout << "Error is the following " << e.what();
	}	
	
	return retVal;
}

int getNumberOfSubDivisionsInPeriod(Period p) {

	int numSubDivisionsInPeriod = 0;

	if (p == quarter) {
		numSubDivisionsInPeriod = 90;
	}
	else if (p == year) {
		numSubDivisionsInPeriod = 365;
	}
	else {
		throw gcnew NotImplementedException();
	}

	return numSubDivisionsInPeriod;

}


//Function simulates the cash-flows over a period of time using two different modes. 
//Places the simulation number and the return value into global class dictionary. 
//Provides a tuple of simulated cash-flows one from the Uniform Cash-flow generator
//and the other from the binary tree generator to the client for inference. 
std::tuple<vector<float>, vector<float>> Simulator::simulationBussinessCashFlows(int numberOfPeriods, Period p, float waccPerPeriod) {

	int numSubDivisionsInPeriod = getNumberOfSubDivisionsInPeriod(p); 

	vector<float> vUniformGen(numberOfPeriods * numSubDivisionsInPeriod);
	std::generate(vUniformGen.begin(), vUniformGen.end(), uniformGen);

	vector<float> vBinomialTreeGen(numberOfPeriods * numSubDivisionsInPeriod);
	std::generate(vBinomialTreeGen.begin(), vBinomialTreeGen.end(), binomialTreeGen);

	vector<float> cPeriodsUniformGen(numberOfPeriods, 0);
	vector<float> cPeriodsBinomialTreeGen(numberOfPeriods, 0);

	//printf("Calculating Quarterly Accumulation %d \n", simulationNumber);
	for (int j = 0; j < numberOfPeriods; j++) {
		float init = 0.0;
		auto uniformGenCI = std::accumulate(vUniformGen.begin() + j * numSubDivisionsInPeriod, vUniformGen.begin() + (j + 1) * numSubDivisionsInPeriod, init);
		auto binomialTreeGenCI = std::accumulate(vBinomialTreeGen.begin() + j * numSubDivisionsInPeriod, vBinomialTreeGen.begin() + (j + 1) * numSubDivisionsInPeriod, init);
		cPeriodsUniformGen[j] = uniformGenCI;
		cPeriodsBinomialTreeGen[j] = binomialTreeGenCI;
	}

	int dayNumber = 0;
	for (auto i : vUniformGen) {
		//printf("Daily Accumulation: %f and Day Number %i \n", i, dayNumber);
		dayNumber += 1; 
	}

	int quarter = 0;
	for (auto i : cPeriodsUniformGen) {
		//printf("Quarterly Aggegration and its Accumulation: %f \n", i, quarter);
		quarter += 1;
	}

	float dailyDiscountRate = calculateDailyPeriodWACC(p, waccPerPeriod);
	//printf("Calculating Daily DCF with the Daily Discount Rate %f and WACC Per Period is %f \n", dailyDiscountRate, waccPerPeriod);
	std::tuple<vector<float>, vector<float>> r = std::make_tuple(cPeriodsUniformGen, cPeriodsBinomialTreeGen);
	float uniformRet = calculateDailyDCF(vUniformGen, dailyDiscountRate);
	float binomialRet = calculateDailyDCF(vBinomialTreeGen, dailyDiscountRate);

	//printf("True Daily Uniform Discounted Cash Flow Value %f and True Daily Binary Discounted Cash Flow %f \n", uniformRet, binomialRet);

	//printf("Unifrom and Binomial Generators \n");
	vector<std::pair<std::string, float>> ret; 
	ret.emplace_back(std::make_pair("Uniform", uniformRet));
	ret.emplace_back("Binomial", binomialRet);
	simulationNumberToReturn.insert(std::pair<int, vector<std::pair<std::string, float>>>(simulationNumber, ret));

	return r;

}

double* convertVectorToPtr(vector<double> v) {

	double* vPtr = (double*) malloc(v.size() * sizeof(double));
	for (int i = 0; i < v.size(); i++) {
		vPtr[i] = v[i];
	}

	return vPtr;
}


vector<double> convertFromFloatVectorToDoubleVector(vector<float> x) {
	vector<double> xDouble; 
	for (auto x_i : x) {
		xDouble.emplace_back(float(x_i));
	}
	return xDouble;
}


void Simulator::simulateBusiness(int numberOfPeriods, Period p, float waccPerPeriod) {
				
	int numberOfSimulations = 1000; 
	for (int i = 0; i < numberOfSimulations; i++) {	
		printf("************************* Simulation %d ************************* \n", simulationNumber);
		//Creates the simulation in background. 
		auto simulation = Simulator::simulationBussinessCashFlows(numberOfPeriods, p, waccPerPeriod);	
		
		auto fVector = std::get<0>(simulation);
		auto pVector = std::get<1>(simulation);		

		vector<double> fVectorDouble = convertFromFloatVectorToDoubleVector(fVector);
		vector<double> pVectorDouble = convertFromFloatVectorToDoubleVector(pVector);

		const double* fPtr = convertVectorToPtr(fVectorDouble);
		const double* pPtr = convertVectorToPtr(pVectorDouble);
		
		//Refactor this so that the calculation method is an input to the simulator. 
		//After refactoring, there should be a better boundary between the calculation method,
		//I provide and the simulation that is done. Development of this boundary between inference method 
		//and simulation is done for integrity's sake.
		//printf("Calculating Inferred DCF %d \n");
		auto discountRate = 1 + waccPerPeriod;
		auto cV1DCF = calculateDiscountedCashFlowWithCalculusOfVariations(fPtr, fVector.size(), 0, fVector.size(), discountRate);
		auto cV2DCF = calculateDiscountedCashFlowWithCalculusOfVariations(pPtr, fVector.size(), 0, fVector.size(), discountRate);

		auto t1DCF = calculateDiscountedCashFlowWithTraditional(fPtr, pVector.size(), 0, pVector.size(), discountRate);
		auto t2DCF = calculateDiscountedCashFlowWithTraditional(pPtr, pVector.size(), 0, pVector.size(), discountRate);

		//printf("With Uniform Cash Flow Generation, the traditional Discounted Cash Flow is %f Calculus Of Variations Cash Flow is %f \n", t1DCF, cV1DCF);
		//printf("With Binary Cash Flow Generation, the traditional Discounted Cash Flow is %f Calculus Of Variations Cash Flow is %f \n", t2DCF, cV2DCF);

		//printf("Calculating Traditional and Calculus of Variations Errors \n");
		//Calculate Calculus of Variations Valuation Errors
		vector<std::pair<string, double>> inferredDiscountedCashFlowsForCalcVariations;
		inferredDiscountedCashFlowsForCalcVariations.emplace_back(std::pair <string, double>("Uniform", cV1DCF));
		inferredDiscountedCashFlowsForCalcVariations.emplace_back(std::pair <string, double>("Binomial", cV2DCF));
		auto calcVariationsRelErrors = Simulator::calculateRelativeDifference(inferredDiscountedCashFlowsForCalcVariations, simulationNumber);

		//Calculate Traditional Valuation Errors
		vector<std::pair<string, double>> inferredDiscountedCashFlowsForTraditional;
		inferredDiscountedCashFlowsForTraditional.emplace_back(std::pair <string, double>("Uniform", t1DCF));
		inferredDiscountedCashFlowsForTraditional.emplace_back(std::pair <string, double>("Binomial", t2DCF));
		auto traditionalRelErrors = Simulator::calculateRelativeDifference(inferredDiscountedCashFlowsForTraditional, simulationNumber);		
		
		//printf("Simulation Number %d \n", simulationNumber);
		printf("For Calculus Of Variations Valuation Method, Uniform Error is %f, Binomial Error is %f \n", std::get<1>(calcVariationsRelErrors[0]), std::get<1>(calcVariationsRelErrors[1]));
		printf("For Traditional Valuation Method, Uniform Error is %f, Binomial Error is %f \n", std::get<1>(traditionalRelErrors[0]), std::get<1>(traditionalRelErrors[1]));

		simulationNumber += 1;

		delete fPtr;
		delete pPtr;


	}	

	return ;
}

vector<std::pair<string, double>> Simulator::calculateRelativeDifference(vector<std::pair<string, double>> inferredDiscountedCashFlows, int simulationNumber){
		
	vector<std::pair<std::string, float>> actualDCF = Simulator::simulationNumberToReturn[simulationNumber];
	vector<std::pair<string, double>> relErrorsForDCF;
	for (int i = 0; i < inferredDiscountedCashFlows.size(); i++) {		
		auto actualPair = actualDCF[i]; 
		auto inferredPair = inferredDiscountedCashFlows[i];

		if (actualPair.first == inferredPair.first) {
			relErrorsForDCF.emplace_back(std::pair<string, float>(actualPair.first, ((inferredPair.second - actualPair.second) / actualPair.second)));
		}
		else {
			relErrorsForDCF.emplace_back(std::pair<string, float>(actualPair.first, nan("1")));
		}

	}

	return relErrorsForDCF;
}

class inferenceEngine {
	protected: 
		int numberOfTimesCalled = 0; 

	public: 		
		virtual float inferDiscountedCashFlowVals(vector<float> cashFlows, float rateOfReturn) = 0;	
};



float discountingEvaluatedF(vector<std::pair<float, float>> fEvaluated, float waccDiscount) {

	vector<float> discountingVector(fEvaluated.size(), 0);	
	int i = 0; 
	for (std::pair<float, float> tAndF : fEvaluated) {
		discountingVector[i] = std::exp(waccDiscount * tAndF.first);		
		i += 1;
	}

	vector<float> fVals(fEvaluated.size(), 0);
	for (int j = 0; fEvaluated.size(); j++) {
		fVals[j] = fEvaluated[j].second;
	}

	return std::inner_product(discountingVector.begin(), discountingVector.end(), fVals.begin(), 0);
}




class calculusOfVariationsInferenceEngine : inferenceEngine {

public: 	

	virtual float inferDiscountedCashFlowVals(vector<float> cashFlows, float rateOfReturn) override {		

		int x0 = 0; 
		int xT = cashFlows.size();
		int numIntervals = 100;		
		std::tuple<vector<double>, float, float> r = calculateParametersRepresentingF(x0, xT, cashFlows);
		//Extract parameters

		vector<double> lambdas; 
		float A; 
		float B; 

		std::tie(lambdas, A, B) = r; 

		
		float discounteddCashFlow = 0.1f;
		//float discountedCashFlow = discountingFunction(, rateOfReturn);
		return discounteddCashFlow;
	}

};


/*
Need to implement the below functions.								Completion Status
1. calculateDiscountedCashFlowWithTraditional						COMPLETED, NEEDS TESTING
2. calculateDiscountedCashFlowWithCalculusOfVariations				COMPLETED, NEEDS TESTING
3. calculateAccuracyTraditionalDcfMethod							COMPLTED,  NEEDS TESTING 
4. calculateAccuracyCaculusOfVariationsDcfMethod					COMPLETED, NEEDS TESTING

*/

vector<double> formVectorFromPointer(const double* ptr, int l) {

	vector<double> vec;
	for (int i = 0; i < l; i++) {
		float val = *(ptr + i);
		vec.emplace_back(val);
	}

	return vec;
}


/* The type of container used to hold the state vector */
typedef std::vector< double > state_type;

//Need to implement a system that allows for the integration of f(x)
//to yield the continuously disounted cash-flow. 

//Problem:			
//continuous DCF = integral(low = 0, high = T) c(t) * e^-rt dt
// 
//Equivalent differential problem is: df/dt = c(t) * e^-rt,
//f(T) = integral(low=0, high = T) c(t) e^-rt dt + f(0). 
// 
//I take f(0) = 0. 
// 
//where r = ln(WACC) where WACC > 1 and represents the 
//the growth rate of money. Tesla's WACC is 1.16, while XOM's WACC is 1.08.

double* globalLambdasPtr = nullptr;
int globalLambdasLen = -1; 
double globalA = -1; 
double globalB = -1; 
double globalWacc = 0.0;

boost::numeric::odeint::runge_kutta4< state_type > stepper;


void discountedCashFlowDifferential(const state_type& x, state_type& dxdt, const double t)
{	

	double c_t = calculatedParameteriziedTimeFunction(globalLambdasPtr, globalLambdasLen, globalA, globalB, t);
	dxdt[0] = c_t * exp(t * - log(globalWacc));
}

//Create a log in C# 

double calculateDiscountedCashFlowWithCalculusOfVariations(const double* cashFlows, int cashFlowLength, int beginTime, int endTime, double waccDiscount) {


	//TODO, need to work on setting up a log for this function.
	//I need to get the path for the file in c++
	//C:\Users\Nate Rogalskyj\Desktop\Business_Simulator_Logs	

	//printf("Form Vector Below: ");
	vector<double> cashFlowsVec = formVectorFromPointer(cashFlows, cashFlowLength);
	vector<float> cashFlowVec; 
	for (auto c : cashFlowsVec) {
		cashFlowVec.emplace_back((float)c);
	}

	std::tuple<vector<double>, float, float> t = calculateParametersRepresentingF(beginTime, endTime, cashFlowVec);
	auto lambdaParameters = std::get<0>(t);
	auto c_0 = std::get<1>(t);
	auto c_1 = std::get<2>(t);

	//printf("Calculate Parameters Representing F ");	

	//Set up global parameters to be used in the discountedCashFlowDifferential.
	globalA = c_0; 
	globalB = c_1;
	globalLambdasLen = lambdaParameters.size();
	globalWacc = waccDiscount;
	
	//printf("Malloc Memory for \n");
	
	//TODO: Need to free the malloc.
	globalLambdasPtr = (double*) malloc(globalLambdasLen * sizeof(double));
	for (int i = 0; i < globalLambdasLen; i++) {
		globalLambdasPtr[i] = lambdaParameters[i];
	}	
	
	vector<double> startState = {0.0};
	double start_time = 0.0; 
	double end_time = (double) endTime;
	double dt = 0.01;
	//printf("Running ODE Solver \n");
	auto r = boost::numeric::odeint::integrate_const(stepper, discountedCashFlowDifferential, startState, start_time, end_time, dt);
	//printf("ODE Solver Finished \n");

	free(globalLambdasPtr);

	//printf("This sentence is redirected to a file. \n");
	//qfclose(stdout);

	auto val = startState[0];	                                                                                                                            
	return val;
};


double calculateDiscountedCashFlowWithTraditional(const double* cashFlows, int cashFlowLength, int beginTime, int endTime, double waccDiscount) {
	
	double standardDiscountedCashFlow = 0.0; 
	vector<double> cashFlowVec = formVectorFromPointer(cashFlows, cashFlowLength);
	for (int i = 0; i < cashFlowVec.size(); i++) {
		standardDiscountedCashFlow += cashFlowVec[i] / pow(waccDiscount, i + 1);
	}

	return standardDiscountedCashFlow;
}


//Define function for calculating the instantaneous cash-flow at time t. 
double calculateInstantaneousCashFlowAtT(const double* cashFlows, int cashFlowLength, int beginTime, int endTime) {
	return 0.0;
}

//Define function for calculating the accuracy of Calculus Of Variations under simulation. 
//extern "C" __declspec(dllexport) void calculateAccuracyOfMethods();
double calculateAccuracyOfMethods(double t) {	

	//freopen(filePath, "wb", stdout);
	printf("Creating Simulator \n");
	auto simulator = new Simulator();
	int numberOfPeriods = 40;
	Period p = quarter;
	//TODO: Replace with different number that represents a more 
	//accurate discount rate per period in the future.
	//It should be set to 0.0 for checking purposes.
	float waccPerPeriod = 0.05;
	printf("Entering Simulator \n");



	simulator->simulateBusiness(numberOfPeriods, p, waccPerPeriod);
	delete simulator;

	fclose(stdout);

	return 0.0;
}

