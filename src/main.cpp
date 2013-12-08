/* 
 * File:   main.cpp
 * Author: hittesdm
 *
 * Created on September 27, 2012, 5:45 PM
 */

#include <cstdlib>
#include <iostream>
#include <ql/quantlib.hpp>
#define BOOST_AUTO_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/detail/lightweight_test.hpp>
#include <boost/math/distributions.hpp>
#include <boost/function.hpp>
#include <boost/math/distributions.hpp>
#include <boost/format.hpp>
#include <functional>
#include <numeric>
#include <fstream>
#include <utility>
#include <boost/assign/std/vector.hpp>

namespace {

    using namespace QuantLib;
    //_____________________________________
    
	/** 
    * Introducing QuantLib: Calculating Future Value
    */
    
    BOOST_AUTO_TEST_CASE(testCalculateLoanPayment) {
        typedef FixedRateCoupon Loan;
        Natural lengthOfLoan = 365;
        Real loanAmount = 100.0;
        Rate rate = .05;
        Date today = Date::todaysDate();
        Date paymentDate = today + lengthOfLoan;
        Loan loan(paymentDate, loanAmount, rate, ActualActual(), today, paymentDate);
        Real payment = loanAmount + loan.accruedAmount(paymentDate);
        std::cout << "Payment due in " << lengthOfLoan << " days on loan amount of $" <<
                loanAmount << " at annual rate of " << rate * 100 << "% is: $" << payment << std::endl;
	}
     
    /**
     * Introducing QuantLib: Calculating Present Value 
     */
	BOOST_AUTO_TEST_CASE(testCalculatePresentValue) {
		Leg cashFlows;
        Date date = Date::todaysDate();
        cashFlows.push_back(boost::shared_ptr<CashFlow>(new SimpleCashFlow(105.0,
                date + 365)));
        Rate rate = .05;
        Real npv = CashFlows::npv(cashFlows, InterestRate(rate, ActualActual(),
                Compounded, Annual),true);
        std::cout << "Net Present Value (NPV) of cash flows is: " << npv << std::endl;

	}
    
    /**
     * Introducing QuantLib: Bond Pricing and Interest Rates
     */
    BOOST_AUTO_TEST_CASE(testCalculateBondPrice) {

        Leg cashFlows;
        Date date = Date::todaysDate();

        cashFlows.push_back(boost::shared_ptr<CashFlow > (new SimpleCashFlow(5.0, date + 365)));
        cashFlows.push_back(boost::shared_ptr<CashFlow > (new SimpleCashFlow(5.0, date + 2 * 365)));
        cashFlows.push_back(boost::shared_ptr<CashFlow > (new SimpleCashFlow(105.0, date + 3 * 365)));
        Rate rate = .03;
        Real npv = CashFlows::npv(cashFlows, InterestRate(rate, ActualActual(ActualActual::Bond), Compounded, Annual), true);
        std::cout << "Price of 3 year bond with annual coupons is: " << npv << std::endl;

    }
	
    /**
     * Introducing QuantLib: Interest Rate Conversions
     */
    BOOST_AUTO_TEST_CASE(testInterestRateConversions) {
        //annual/effective rate
        Rate annualRate = .05;

        //5% rate compounded annually
        InterestRate effectiveRate(annualRate, ActualActual(), Compounded, Annual);
        std::cout << "Rate with annual compounding is: " << effectiveRate.rate() << std::endl;

        //what is the equivalent semi-annual one year rate?
        InterestRate semiAnnualCompoundingOneYearRate = effectiveRate.equivalentRate(Compounded, Semiannual, 1);
        std::cout << "Equivalent one year semi-annually compounded rate is: " << semiAnnualCompoundingOneYearRate.rate() << std::endl;

        //what is the equivalent 1 year rate if compounded continuously?
        InterestRate continuousOneYearRate = effectiveRate.equivalentRate(Continuous, Annual, 1);
        std::cout << "Equivalent one year continuously compounded rate is: " << continuousOneYearRate.rate() << std::endl;
    }

    /**
     * Introducing QuantLib: The Term Structure of Interest Rates
     */
    BOOST_AUTO_TEST_CASE(testPriceBondWithFlatTermStructure) {
        Calendar calendar = UnitedStates(UnitedStates::GovernmentBond);
        const Natural settlementDays = 3;
        Date today = Date::todaysDate();
        Date issueDate = today;
        Date terminationDate = issueDate + Period(3, Years);
        //Rate rate = .09;
        Rate rate = .03;
        //InterestRate couponRate(.10, ActualActual(ActualActual::Bond), Compounded, Annual);
        InterestRate couponRate(.05, ActualActual(ActualActual::Bond), Compounded, Annual);
        Real faceValue = 100.0;
        //std::vector<InterestRate> coupons(2, couponRate);
        std::vector<InterestRate> coupons(3, couponRate);
        Schedule schedule(issueDate, terminationDate, Period(Annual), calendar,
                Unadjusted, Unadjusted, DateGeneration::Backward, false);
        FixedRateBond fixedRateBond(settlementDays, faceValue, schedule, coupons);
        boost::shared_ptr<YieldTermStructure> flatForwardRates(new FlatForward(issueDate, rate, ActualActual(ActualActual::Bond), Compounded, Annual));
        Handle<YieldTermStructure> flatTermStructure(flatForwardRates);
        boost::shared_ptr<PricingEngine> bondEngine(new DiscountingBondEngine(flatTermStructure));

        fixedRateBond.setPricingEngine(bondEngine);
        Real npv = fixedRateBond.NPV();
        std::cout << "NPV of bond is: " << npv << std::endl;
    }

    /**
     * Introducing QuantLib: Internal Rate of Return 
     */
    class IRRSolver {
    public:

        explicit IRRSolver(const Leg& cashFlows, Real npv) : _cashFlows(cashFlows), _npv(npv) {
        };

        Real operator() (const Rate& rate) const {
            InterestRate interestRate(rate, ActualActual(ActualActual::Bond), Compounded, Annual);
            return CashFlows::npv(_cashFlows, interestRate, false) - _npv;
        }

    private:
        const Real _npv;
        const Leg& _cashFlows;

    };

    BOOST_AUTO_TEST_CASE(testCalculateBondYieldToMaturity) {
        Calendar calendar = UnitedStates(UnitedStates::GovernmentBond);
        const Natural settlementDays = 3;
        Date today = Date::todaysDate();
        Date issueDate = today;
        Date terminationDate = issueDate + Period(3, Years);
        Rate rate = .03;

        InterestRate couponRate(.05, ActualActual(ActualActual::Bond), Compounded, Annual);
        Real faceValue = 100.0;
        std::vector<InterestRate> coupons(3, couponRate);
        Schedule schedule(issueDate, terminationDate, Period(Annual), calendar,
                Unadjusted, Unadjusted, DateGeneration::Backward, false);
        FixedRateBond fixedRateBond(settlementDays, faceValue, schedule, coupons);
        boost::shared_ptr<YieldTermStructure> flatForwardRates(new FlatForward(issueDate,
                rate, ActualActual(ActualActual::Bond), Compounded, Annual));
        Handle<YieldTermStructure> flatTermStructure(flatForwardRates);
        boost::shared_ptr<PricingEngine> bondEngine(new DiscountingBondEngine(flatTermStructure));

        fixedRateBond.setPricingEngine(bondEngine);
        Real npv = fixedRateBond.NPV();
        std::cout << "NPV of bond is: " << npv << std::endl;

        //solve for yield to maturity using bisection solver
        Bisection bisection;
        Real accuracy = 0.00000001, guess = .10;
        Real min = .0025, max = .15;

        //invoke bisection solver with IRRSolver functor 
        Real irr = bisection.solve(IRRSolver(fixedRateBond.cashflows(), npv),
                accuracy, guess, min, max);

        std::cout << "Bond yield to maturity (IRR) is: " << irr << std::endl;

        //invoke bisection solver with C++ 11 lambda expression

        irr = bisection.solve([&] (const Rate & rate) {
            return CashFlows::npv(fixedRateBond.cashflows(), InterestRate(rate,
            ActualActual(ActualActual::Bond), Compounded, Annual), false) - npv;
        },
        accuracy, guess, min, max);

        std::cout << "Bond yield to maturity (IRR) is: " << irr << std::endl;

        irr = fixedRateBond.yield(ActualActual(ActualActual::Bond), Compounded, Annual);

        std::cout << "Bond yield to maturity (IRR) is: " << irr << std::endl;
    }

    /**
     * Introducing QuantLib: Duration and Convexity 
     */
    BOOST_AUTO_TEST_CASE(testCalculateBondDurationAndConvexity) {

        Calendar calendar = UnitedStates(UnitedStates::GovernmentBond);
        const Natural settlementDays = 3;
        Date today = Date::todaysDate();
        Date issueDate = today;
        Date terminationDate = issueDate + Period(3, Years);
        Rate rate = .03;

        InterestRate couponRate(.05, ActualActual(ActualActual::Bond), Compounded, Annual);
        Real faceValue = 100.0;
        std::vector<InterestRate> coupons(3, couponRate);
        Schedule schedule(issueDate, terminationDate, Period(Annual), calendar,
                Unadjusted, Unadjusted, DateGeneration::Backward, false);
        FixedRateBond fixedRateBond(settlementDays, faceValue, schedule, coupons);
        boost::shared_ptr<YieldTermStructure> flatForwardRates(new FlatForward(issueDate,
                rate, ActualActual(ActualActual::Bond), Compounded, Annual));
        Handle<YieldTermStructure> flatTermStructure(flatForwardRates);
        boost::shared_ptr<PricingEngine> bondEngine(new DiscountingBondEngine(flatTermStructure));
        fixedRateBond.setPricingEngine(bondEngine);

        //calculate bond price
        Real price = fixedRateBond.NPV();
        std::cout << "Bond price: " << price << std::endl;

        //calculate yield to maturity (YTM)/internal rate of return (IRR)
        Real ytm = fixedRateBond.yield(ActualActual(ActualActual::Bond), Compounded, Annual);
        std::cout << "yield to maturity: " << ytm << std::endl;

        //calculate Macaulay duration
        InterestRate yield(ytm, ActualActual(ActualActual::Bond), Compounded, Annual);
        Time macDuration = BondFunctions::duration(fixedRateBond, yield, Duration::Macaulay, today);
        std::cout << "Macaulay duration: " << macDuration << std::endl;

        //calculate modified duration
        Time modDuration = BondFunctions::duration(fixedRateBond, yield, Duration::Modified, today);
        std::cout << "Modified duration: " << modDuration << std::endl;

        //calculate convexity
        Real convexity = BondFunctions::convexity(fixedRateBond, yield, today);
        std::cout << "Convexity: " << convexity << std::endl;

        //estimate new bond price for an increase in interest rate of 1% using modified duration
        Real priceDuration = price + price * (-modDuration * .01);
        std::cout << boost::format("Estimated bond price using only duration (rate up .01): %.2f ") %
                priceDuration << std::endl;

        //estimate new bond price for an increase in interest rate of 1% using duration and convexity
        Real priceConvexity = price + price * (-modDuration * .01 + (.5 * convexity * std::pow(.01, 2)));
        std::cout << boost::format("Estimated bond price using duration and convexity (rate up .01): %.2f") %
                priceConvexity << std::endl;

        //calculate dollar value of a basis point (DVO1) 
        Real DVO1 = std::abs(BondFunctions::basisPointValue(fixedRateBond, yield, today));
        std::cout << boost::format("Dollar value of a basis point (DVO1): %.4f") % DVO1 << std::endl;

    }

    /**
     * Introducing QuantLib: Pricing Futures and Forward Contracts 
     */
    BOOST_AUTO_TEST_CASE(testCalculateForwardPriceOfFixedRateBond) {

        Calendar calendar = UnitedStates(UnitedStates::GovernmentBond);
        const Natural settlementDays = 1;
        Date today = Date::todaysDate();
        Date bondIssueDate = calendar.adjust(today, ModifiedFollowing);
        Date bondMaturityDate = bondIssueDate + Period(3, Years);
        Rate rate = .03;
		Settings::instance().evaluationDate() = bondIssueDate;

		//coupon schedule
        std::vector<Rate> coupons(1, .05);
       
		//fixed rate bond
        Real faceValue = 100.0;
		boost::shared_ptr<FixedRateBond> fixedRateBondPtr(new FixedRateBond(settlementDays, calendar, 
			faceValue, bondIssueDate, bondMaturityDate, Period(Annual), coupons, ActualActual(ActualActual::Bond)));
        boost::shared_ptr<YieldTermStructure> flatForwardRates(new FlatForward(bondIssueDate,
                rate, ActualActual(ActualActual::Bond), Compounded, Annual));
        RelinkableHandle<YieldTermStructure> flatTermStructure(flatForwardRates);
        boost::shared_ptr<PricingEngine> bondEngine(new DiscountingBondEngine(flatTermStructure));
        fixedRateBondPtr->setPricingEngine(bondEngine);

        //calculate bond price
        Real bondPrice = fixedRateBondPtr->NPV();
        std::cout << "Bond price: " << bondPrice << std::endl;

		//print cash flows using C++ 11 range-based for loop
		for (boost::shared_ptr<CashFlow> cashFlow : fixedRateBondPtr->cashflows()) {
			std::cout << boost::format("Cash flow: %s, %.2f") % cashFlow->date() % cashFlow->amount() << std::endl; 
		}
		
		//forward maturity tenor 
		Date forwardMaturityDate = bondIssueDate + Period(15, Months); 
		Natural daysToMaturityOfForwardContract = (forwardMaturityDate - bondIssueDate) - settlementDays;
		std::cout << boost::format("Expiration of forward contract: %s") % forwardMaturityDate << std::endl;
		std::cout << boost::format("Days to maturity of forward contract: %i") % daysToMaturityOfForwardContract << std::endl;
		
		//forward contract on a fixed rate bond - calculate future value of the bond using annualized rate, not continuous
		Real income = (fixedRateBondPtr->nextCouponRate(bondIssueDate) * faceValue);
		const Real strike = bondPrice * (1 + rate * daysToMaturityOfForwardContract/365) - income; 
		std::cout << boost::format("Strike price of forward contract is: %.2f") % strike << std::endl;

		//forward contract on a fixed rate bond
		FixedRateBondForward fixedRateBondForward(bondIssueDate, forwardMaturityDate, Position::Type::Long, strike, settlementDays,
				ActualActual(ActualActual::Bond), calendar, ModifiedFollowing, fixedRateBondPtr, flatTermStructure, flatTermStructure);

		//calculate forward price of bond 
		Real forwardPrice = fixedRateBondForward.NPV();
		std::cout << boost::format("Bond forward contract value: %.2f") % forwardPrice << std::endl;

		//evaluate forward contract by shocking interest rates +/- 1%
        boost::shared_ptr<YieldTermStructure> flatForwardRatesUpOnePercent(new FlatForward(bondIssueDate,
                rate + .01, ActualActual(ActualActual::Bond), Compounded, Annual));
        
		flatTermStructure.linkTo(flatForwardRatesUpOnePercent);
		
		//recalculate forward price of bond 
		std::cout << boost::format("Bond forward contract value (rates up 1 percent): %.2f") % fixedRateBondForward.NPV() << std::endl;

		boost::shared_ptr<YieldTermStructure> flatForwardRatesDownOnePercent(new FlatForward(bondIssueDate,
                rate - .01, ActualActual(ActualActual::Bond), Compounded, Annual));

		flatTermStructure.linkTo(flatForwardRatesDownOnePercent);
		
		//recalculate forward price of bond 
		std::cout << boost::format("Bond forward contract value (rates down 1 percent): %.2f") % fixedRateBondForward.NPV() << std::endl;
		
	}

    /**
     * Introducing QuantLib: Linear Optimization
     */

	class PortfolioAllocationCostFunction: public CostFunction {
    
    public:
       Real value(const Array& x) const {
           QL_REQUIRE(x.size()==2, "Two bonds in portfolio!");
           return -1 * (4*x[0] + 3*x[1]); //mult by -1 to maximize otherwise will minimize 
       }
       
       Disposable<Array> values(const Array& x) const {
           QL_REQUIRE(x.size()==2, "Two bonds in portfolio!");
           Array values(1);
           values[0] = value(x);
       }
    }; 

   	class PortfolioAllocationConstraints : public Constraint {
    public:
		PortfolioAllocationConstraints(const std::vector<std::function<bool (const Array&)> >& expressions)  
        	: Constraint(boost::shared_ptr<Constraint::Impl>(new PortfolioAllocationConstraints::Impl(expressions))) {}
         
    private:

        //constraint implementation
        class Impl : public Constraint::Impl {
        public:
            Impl(const std::vector<std::function<bool (const Array&)> >& expressions) :
            	expressions_(expressions) {}
                
			bool test(const Array& x) const {
                for (auto iter = expressions_.begin(); iter < expressions_.end(); ++iter) {
                  	if (!(*iter)(x)) {
						return false;
					}
                }

                //will only get here if all constraints satisfied
                return true;
            }
        private:
         	const std::vector<std::function<bool (const Array&)> > expressions_;        
        };

    }; 
    
    BOOST_AUTO_TEST_CASE(testLinearOptimization) {
		PortfolioAllocationCostFunction portfolioAllocationCostFunction;	
        std::vector<std::function<bool (const Array&)> >  constraints(3);

        //constraints implemented as C++ 11 lambda expressions
        constraints[0] = [] (const Array& x) { return (x[0] + x[1] <= 100.0);}; 
        constraints[1] = [] (const Array& x) { return ((2 * x[0] + x[1]) / 100.0 <= 1.5);}; 
        constraints[2] = [] (const Array& x) { return ((3 * x[0] + 4 * x[1]) / 100.0 <= 3.6);}; 

		//instantiate constraints
		PositiveConstraint greaterThanZeroConstraint;
		PortfolioAllocationConstraints portfolioAllocationConstraints(constraints);
		CompositeConstraint allConstraints(portfolioAllocationConstraints, greaterThanZeroConstraint);

        //end criteria 
		Size maxIterations = 1000;
		Size minStatIterations = 10;
		Real rootEpsilon = 1e-8;
		Real functionEpsilon = 1e-9;
		Real gradientEpsilon = 1e-5;

		EndCriteria endCriteria(maxIterations, minStatIterations, rootEpsilon, 
			functionEpsilon, gradientEpsilon);

		Problem bondAllocationProblem(portfolioAllocationCostFunction, allConstraints, Array(2, 1));
		Simplex solver(.1);
       
        EndCriteria::Type solution = solver.minimize(bondAllocationProblem, endCriteria);
		std::cout << boost::format("Simplex solution type: %s") % solution << std::endl;
        
        const Array& results = bondAllocationProblem.currentValue();
		std::cout << boost::format("Allocate %.2f percent to bond 1 and %.2f percent to bond 2.") % results[0] % results[1] << std::endl;  
			

    }

    /**
     * Introducing QuantLib: The Efficient Frontier 
     */
    
	double calculatePortfolioReturn(double proportionA, double expectedReturnA, double expectedReturnB) {
		return proportionA * expectedReturnA + (1-proportionA) * expectedReturnB;
	}

	Volatility calculatePortfolioRisk(double proportionA, Volatility volatilityA, Volatility volatilityB, double covarianceAB) {
		return std::sqrt(std::pow(proportionA,2) * std::pow(volatilityA,2) + std::pow(1-proportionA,2) * 
			std::pow(volatilityB,2) + (2 * proportionA * (1-proportionA) * covarianceAB)); 	
	}

    BOOST_AUTO_TEST_CASE(testEfficientFrontier) {

        Matrix covarianceMatrix(4,4);
        
        //row 1
        covarianceMatrix[0][0] = .1; //AAPL-AAPL
        covarianceMatrix[0][1] = .03; //AAPL-IBM
        covarianceMatrix[0][2] = -.08; //AAPL-ORCL
        covarianceMatrix[0][3] = .05; //AAPL-GOOG
        //row 2
        covarianceMatrix[1][0] = .03; //IBM-AAPL
        covarianceMatrix[1][1] = .20; //IBM-IBM 
        covarianceMatrix[1][2] = .02; //IBM-ORCL
        covarianceMatrix[1][3] = .03; //IBM-GOOG
		//row 3
        covarianceMatrix[2][0] = -.08; //ORCL-AAPL 
        covarianceMatrix[2][1] = .02; //ORCL-IBM
        covarianceMatrix[2][2] = .3; //ORCL-ORCL
        covarianceMatrix[2][3] = .2; //ORCL-GOOG
       	//row 4
        covarianceMatrix[3][0] = .05; //GOOG-AAPL
        covarianceMatrix[3][1] = .03; //GOOG-IBM
        covarianceMatrix[3][2] = .2; //GOOG-ORCL
        covarianceMatrix[3][3] = .9; //GOOG-GOOG

        std::cout << "Covariance matrix of returns: " << std::endl;
        std::cout << covarianceMatrix << std::endl;
      
       	//portfolio return vector         
		Matrix portfolioReturnVector(4,1);
        portfolioReturnVector[0][0] = .08; //AAPL
        portfolioReturnVector[1][0] = .09; //IBM
        portfolioReturnVector[2][0] = .10; //ORCL
        portfolioReturnVector[3][0] = .11; //GOOG
        
		std::cout << "Portfolio return vector" << std::endl;
        std::cout << portfolioReturnVector << std::endl;
		
        //constant 
        Rate c = .05;
        
        //portfolio return vector minus constant rate
        Matrix portfolioReturnVectorMinusC(4,1);
		for (int i=0; i<4; ++i) {
    		portfolioReturnVectorMinusC[i][0] = portfolioReturnVector[i][0] - c;
		}

		std::cout << boost::format("Portfolio return vector minus constant rate (c = %f)") % c << std::endl;
		std::cout << portfolioReturnVectorMinusC << std::endl;

        //inverse of covariance matrix
        const Matrix& inverseOfCovarMatrix = inverse(covarianceMatrix);
		
		//z vectors
      	const Matrix& portfolioAz = inverseOfCovarMatrix * portfolioReturnVector;
		std::cout << "Portfolio A z vector" << std::endl;
		std::cout << portfolioAz << std::endl;
		double sumOfPortfolioAz = 0.0;
		std::for_each(portfolioAz.begin(), portfolioAz.end(), [&](Real n) {
			sumOfPortfolioAz += n;	
		});
				
		const Matrix& portfolioBz = inverseOfCovarMatrix * portfolioReturnVectorMinusC;
		std::cout << "Portfolio B z vector" << std::endl;
		std::cout << portfolioBz << std::endl;
		double sumOfPortfolioBz = 0.0;
		std::for_each(portfolioBz.begin(), portfolioBz.end(), [&](Real n) {
			sumOfPortfolioBz += n;	
		});

		//portfolio weights
		Matrix weightsPortfolioA(4,1);
		for (int i=0; i<4; ++i) {
			weightsPortfolioA[i][0] = portfolioAz[i][0]/sumOfPortfolioAz;
		}		

		std::cout << "Portfolio A weights" << std::endl;
		std::cout << weightsPortfolioA << std::endl;

		Matrix weightsPortfolioB(4,1);
		for (int i=0; i<4; ++i) {
			weightsPortfolioB[i][0] = portfolioBz[i][0]/sumOfPortfolioBz;
		}		

		std::cout << "Portfolio B weights" << std::endl;
		std::cout << weightsPortfolioB << std::endl;
		
		//portfolio risk and return
		const Matrix& expectedReturnPortfolioAMatrix = transpose(weightsPortfolioA) * portfolioReturnVector;	
		double expectedReturnPortfolioA = expectedReturnPortfolioAMatrix[0][0];
		const Matrix& variancePortfolioAMatrix =  transpose(weightsPortfolioA) * covarianceMatrix * weightsPortfolioA;
		double variancePortfolioA = variancePortfolioAMatrix[0][0];
		double stdDeviationPortfolioA = std::sqrt(variancePortfolioA);
		std::cout << boost::format("Portfolio A expected return: %f") % expectedReturnPortfolioA << std::endl;
		std::cout << boost::format("Portfolio A variance: %f") % variancePortfolioA << std::endl;
       	std::cout << boost::format("Portfolio A standard deviation: %f") % stdDeviationPortfolioA << std::endl;
		
		const Matrix& expectedReturnPortfolioBMatrix = transpose(weightsPortfolioB) * portfolioReturnVector;	
		double expectedReturnPortfolioB = expectedReturnPortfolioBMatrix[0][0];
		const Matrix& variancePortfolioBMatrix =  transpose(weightsPortfolioB) * covarianceMatrix * weightsPortfolioB;
		double variancePortfolioB = variancePortfolioBMatrix[0][0];
		double stdDeviationPortfolioB = std::sqrt(variancePortfolioB);
		std::cout << boost::format("Portfolio B expected return: %f") % expectedReturnPortfolioB << std::endl;
		std::cout << boost::format("Portfolio B variance: %f") % variancePortfolioB << std::endl;
       	std::cout << boost::format("Portfolio B standard deviation: %f") % stdDeviationPortfolioB << std::endl;

		//covariance and correlation of returns
		const Matrix& covarianceABMatrix = transpose(weightsPortfolioA) * covarianceMatrix * weightsPortfolioB;
		double covarianceAB = covarianceABMatrix[0][0];
		double correlationAB = covarianceAB/(stdDeviationPortfolioA * stdDeviationPortfolioB);
		std::cout << boost::format("Covariance of portfolio A and B: %f") % covarianceAB << std::endl;
		std::cout << boost::format("Correlation of portfolio A and B: %f") % correlationAB << std::endl;

		//generate envelope set of portfolios
		double startingProportion = -.40;
		double increment = .10;
		std::map<double, std::pair<Volatility,double> > mapOfProportionToRiskAndReturn;
		std::map<Volatility, double> mapOfVolatilityToReturn;
		for (int i=0; i<40; ++i) {
			double proportionA = startingProportion + i*increment;
			Volatility riskEF = calculatePortfolioRisk(proportionA, stdDeviationPortfolioA, stdDeviationPortfolioB, covarianceAB);
			double returnEF = calculatePortfolioReturn(proportionA, expectedReturnPortfolioA, expectedReturnPortfolioB);
			mapOfProportionToRiskAndReturn[proportionA] = std::make_pair(riskEF, returnEF);
			mapOfVolatilityToReturn[riskEF] = returnEF;
		}

		//write data to a file
		std::ofstream envelopeSetFile;
		envelopeSetFile.open("/tmp/envelope.dat",std::ios::out);
		for (std::map<double, std::pair<Volatility,double> >::const_iterator i=mapOfProportionToRiskAndReturn.begin(); i != mapOfProportionToRiskAndReturn.end(); ++i) {
			envelopeSetFile << boost::format("%f %f %f") % i->first % i->second.first % i->second.second << std::endl;
		}
		envelopeSetFile.close();

		//find minimum risk portfolio on efficient frontier
		std::pair<Volatility,double> minimumVariancePortolioRiskAndReturn = *mapOfVolatilityToReturn.begin(); 
		Volatility minimumRisk = minimumVariancePortolioRiskAndReturn.first; 
		double maximumReturn = minimumVariancePortolioRiskAndReturn.second;
		std::cout << boost::format("Maximum portfolio return for risk of %f is %f") % minimumRisk % maximumReturn << std::endl;
		
		//generate efficient frontier
		std::map<Volatility, double> efficientFrontier;
		for (std::map<double, std::pair<Volatility,double> >::const_iterator i=mapOfProportionToRiskAndReturn.begin(); i != mapOfProportionToRiskAndReturn.end(); ++i) {
			efficientFrontier[i->second.first] = i->second.second;
			if (i->second.first == minimumRisk) break;
		}

		//write efficient frontier to file
		std::ofstream efFile;
		efFile.open("/tmp/ef.dat", std::ios::out);
		for (std::map<Volatility, double>::const_iterator i=efficientFrontier.begin(); i != efficientFrontier.end(); ++i) {
			efFile << boost::format("%f %f") % i->first % i->second << std::endl;
		}
		efFile.close();

		//plot with gnuplot using commands below Run 'gnuplot' then type in: 
		/*
        set key top left
		set key box
		set xlabel "Volatility"
		set ylabel "Expected Return"
		plot '/tmp/envelope.dat' using 2:3 with linespoints title "Feasible Set", '/tmp/ef.dat' using 1:2  w points pointtype 5 t "Efficient Frontier"
		*/
		
    }

    /**
     * Introducing QuantLib: Portfolio Optimization 
     */
    
    class ThetaCostFunction: public CostFunction {
    	 
    public:
		ThetaCostFunction(const Matrix& covarianceMatrix, 
				const Matrix& returnMatrix) : covarianceMatrix_(covarianceMatrix),
				returnMatrix_(returnMatrix) {
		}
		
    	Real value(const Array& proportions) const {
    		QL_REQUIRE(proportions.size()==3, "Four assets in portfolio!");
            Array allProportions(4);
            allProportions[0] = proportions[0];
            allProportions[1] = proportions[1];
            allProportions[2] = proportions[2];
            allProportions[3] = 1 - (proportions[0] + proportions[1] + proportions[2]);
			return -1 * ((portfolioMean(allProportions) - c_)/portfolioStdDeviation(allProportions));	
       	}
       
		Disposable<Array> values(const Array& proportions) const {
        	QL_REQUIRE(proportions.size()==3, "Four assets in portfolio!");
           	Array values(1);
		   	values[0] = value(proportions);
           	return values;
       	}
       
       	void setC(Real c) { c_ = c; }
       	Real getC() const { return c_;}

		Real portfolioMean(const Array& proportions) const {
			Real portfolioMean = (proportions * returnMatrix_)[0];
            //std::cout << boost::format("Portfolio mean: %.4f") % portfolioMean << std::endl;
            return portfolioMean;
		}

		Real portfolioStdDeviation(const Array& proportions) const {
            Matrix matrixProportions(4,1);
			for (size_t row = 0; row < 4; ++row) {
				matrixProportions[row][0] = proportions[row];
			}
			
			const Matrix& portfolioVarianceMatrix = transpose(matrixProportions) * covarianceMatrix_ * matrixProportions;
			Real portfolioVariance = portfolioVarianceMatrix[0][0];
			Real stdDeviation = std::sqrt(portfolioVariance);
            //std::cout << boost::format("Portfolio standard deviation: %.4f") % stdDeviation << std::endl;
            return stdDeviation;
		}
        
	private:	
        const Matrix& covarianceMatrix_;
		const Matrix& returnMatrix_;
		Real c_;
    };


    BOOST_AUTO_TEST_CASE(testNoShortSales) {
        
        Matrix covarianceMatrix(4,4);
        
        //row 1
        covarianceMatrix[0][0] = .1; //AAPL-AAPL
        covarianceMatrix[0][1] = .03; //AAPL-IBM
        covarianceMatrix[0][2] = -.08; //AAPL-ORCL
        covarianceMatrix[0][3] = .05; //AAPL-GOOG
        //row 2
        covarianceMatrix[1][0] = .03; //IBM-AAPL
        covarianceMatrix[1][1] = .20; //IBM-IBM 
        covarianceMatrix[1][2] = .02; //IBM-ORCL
        covarianceMatrix[1][3] = .03; //IBM-GOOG
		//row 3
        covarianceMatrix[2][0] = -.08; //ORCL-AAPL 
        covarianceMatrix[2][1] = .02; //ORCL-IBM
        covarianceMatrix[2][2] = .30; //ORCL-ORCL
        covarianceMatrix[2][3] = .2; //ORCL-GOOG
       	//row 4
        covarianceMatrix[3][0] = .05; //GOOG-AAPL
        covarianceMatrix[3][1] = .03; //GOOG-IBM
        covarianceMatrix[3][2] = .2; //GOOG-ORCL
        covarianceMatrix[3][3] = .9; //GOOG-GOOG

        std::cout << "Covariance matrix of returns: " << std::endl;
        std::cout << covarianceMatrix << std::endl;
      
       	//portfolio return vector         
		Matrix portfolioReturnVector(4,1);
        portfolioReturnVector[0][0] = .08; //AAPL
        portfolioReturnVector[1][0] = .09; //IBM
        portfolioReturnVector[2][0] = .10; //ORCL
        portfolioReturnVector[3][0] = .11; //GOOG
      
        std::cout << "Portfolio return vector" << std::endl;
        std::cout << portfolioReturnVector << std::endl;
        
        //constraints
		std::vector<std::function<bool (const Array&)> >  noShortSalesConstraints(2);

        //constraints implemented as C++ 11 lambda expressions
        noShortSalesConstraints[0] = [] (const Array& x) {Real x4 = 1.0 - ( x[0] + x[1] + x[2]);  return (x[0] >= 0.0 && x[1] >= 0.0 && x[2] >= 0.0 && x4 >= 0.0);}; 
        noShortSalesConstraints[1] = [] (const Array& x) { Real x4 = 1.0 - ( x[0] + x[1] + x[2]); return 1.0 - (x[0] + x[1] + x[2] + x4) < 1e-9;};
		
		//instantiate constraints
		PortfolioAllocationConstraints noShortSalesPortfolioConstraints(noShortSalesConstraints);

		Size maxIterations = 100000;
		Size minStatIterations = 100;
		Real rootEpsilon = 1e-9;
		Real functionEpsilon = 1e-9;
		Real gradientNormEpsilon = 1e-9;
		EndCriteria endCriteria (maxIterations, minStatIterations, 
			rootEpsilon, functionEpsilon, gradientNormEpsilon);	
        
        std::map<Rate, std::pair<Volatility, Real> > mapOfStdDeviationToMeanNoShortSales;
       
		Rate startingC = -.035;
		Real increment = .005;
		
        for (int i = 0; i < 40; ++i) {
            
            Rate c = startingC + (i * increment);  
		
         	ThetaCostFunction thetaCostFunction(covarianceMatrix, portfolioReturnVector);
			thetaCostFunction.setC(c);
            Problem efficientFrontierNoShortSalesProblem(thetaCostFunction, noShortSalesPortfolioConstraints, Array(3, .2500));
			Simplex solver(.01);
            
			EndCriteria::Type noShortSalesSolution = solver.minimize(efficientFrontierNoShortSalesProblem, endCriteria);

			std::cout << boost::format("Solution type: %s") % noShortSalesSolution << std::endl;
        
    	    const Array& results = efficientFrontierNoShortSalesProblem.currentValue();
            Array proportions(4);
            proportions[0] = results[0];
            proportions[1] = results[1];
            proportions[2] = results[2];
            proportions[3] = 1.0 - (results[0] + results[1] + results[2]);
            std::cout << boost::format("Constant (c): %.4f") % thetaCostFunction.getC() << std::endl;
			std::cout << boost::format("AAPL weighting: %.4f") % proportions[0] << std::endl;  
			std::cout << boost::format("IBM weighting: %.4f") % proportions[1] << std::endl;  
			std::cout << boost::format("ORCL weighting: %.4f") % proportions[2] << std::endl;  
			std::cout << boost::format("GOOG weighting: %.4f") % proportions[3] << std::endl;
			std::cout << boost::format("Theta: %.4f") % (-1 * efficientFrontierNoShortSalesProblem.functionValue()) << std::endl;
            Real portfolioMean = thetaCostFunction.portfolioMean(proportions); 
            std::cout << boost::format("Portfolio mean: %.4f") % portfolioMean << std::endl;
            Volatility portfolioStdDeviation = thetaCostFunction.portfolioStdDeviation(proportions); 
            std::cout << boost::format("Portfolio standard deviation: %.4f") % portfolioStdDeviation << std::endl;
            mapOfStdDeviationToMeanNoShortSales[c] = std::make_pair(portfolioStdDeviation, portfolioMean); 
            std::cout << "------------------------------------" << std::endl;
        }
		
        //write efficient frontier with no short sales to file
		std::ofstream noShortSalesFile;
		noShortSalesFile.open("/tmp/noshortsales.dat", std::ios::out);
		for (std::map<Rate, std::pair<Volatility, Real> >::const_iterator i=mapOfStdDeviationToMeanNoShortSales.begin(); i != mapOfStdDeviationToMeanNoShortSales.end(); ++i) {
			noShortSalesFile << boost::format("%f %f %f") % i->first % i->second.first % i->second.second << std::endl;
		}
		noShortSalesFile.close();
		
		//plot with gnuplot using commands below Run 'gnuplot' then type in: 
		/*
         set terminal png
         set output "/tmp/noshortsales.png"
         set key top left
		 set key box
		 set xlabel "Volatility"
		 set ylabel "Expected Return"
		 plot '/tmp/noshortsales.dat' using 2:3 w linespoints title "No Short Sales"
		*/
    }
    
  	/** 
     * Introducing QuantLib: Portfolio Optimization (NOT PUBLISHED TO BLOG) 
     */
    
    BOOST_AUTO_TEST_CASE(testNoShortSalesWithPositionLimits) {
        
        Matrix covarianceMatrix(4,4);
        
        //row 1
        covarianceMatrix[0][0] = .1; //AAPL-AAPL
        covarianceMatrix[0][1] = .03; //AAPL-IBM
        covarianceMatrix[0][2] = -.08; //AAPL-ORCL
        covarianceMatrix[0][3] = .05; //AAPL-GOOG
        //row 2
        covarianceMatrix[1][0] = .03; //IBM-AAPL
        covarianceMatrix[1][1] = .20; //IBM-IBM 
        covarianceMatrix[1][2] = .02; //IBM-ORCL
        covarianceMatrix[1][3] = .03; //IBM-GOOG
		//row 3
        covarianceMatrix[2][0] = -.08; //ORCL-AAPL 
        covarianceMatrix[2][1] = .02; //ORCL-IBM
        covarianceMatrix[2][2] = .30; //ORCL-ORCL
        covarianceMatrix[2][3] = .2; //ORCL-GOOG
       	//row 4
        covarianceMatrix[3][0] = .05; //GOOG-AAPL
        covarianceMatrix[3][1] = .03; //GOOG-IBM
        covarianceMatrix[3][2] = .2; //GOOG-ORCL
        covarianceMatrix[3][3] = .9; //GOOG-GOOG

        std::cout << "Covariance matrix of returns: " << std::endl;
        std::cout << covarianceMatrix << std::endl;
      
       	//portfolio return vector         
		Matrix portfolioReturnVector(4,1);
        portfolioReturnVector[0][0] = .08; //AAPL
        portfolioReturnVector[1][0] = .09; //IBM
        portfolioReturnVector[2][0] = .10; //ORCL
        portfolioReturnVector[3][0] = .11; //GOOG
      
        std::cout << "Portfolio return vector" << std::endl;
        std::cout << portfolioReturnVector << std::endl;
        
        //constraints
		std::vector<std::function<bool (const Array&)> >  noShortSalesConstraints(3);

        //constraints implemented as C++ 11 lambda expressions
        noShortSalesConstraints[0] = [] (const Array& x) {Real x4 = 1.0 - ( x[0] + x[1] + x[2]);  return (x[0] >= 0.05 && x[1] >= 0.05 && x[2] > 0.05 && x4 > 0.05);}; 
        noShortSalesConstraints[1] = [] (const Array& x) { Real x4 = 1.0 - ( x[0] + x[1] + x[2]); return 1.0 - (x[0] + x[1] + x[2] + x4) < 1e-9;};
        noShortSalesConstraints[2] = [] (const Array& x) { Real x4 = 1.0 - ( x[0] + x[1] + x[2]); return (x[0] <= .50 && x[1] <= .50 && x[2] <= .50 && x4 <= .50);};
		
		//instantiate constraints
		PortfolioAllocationConstraints noShortSalesPortfolioConstraints(noShortSalesConstraints);

		Size maxIterations = 100000;
		Size minStatIterations = 100;
		Real rootEpsilon = 1e-9;
		Real functionEpsilon = 1e-9;
		Real gradientNormEpsilon = 1e-9;
		EndCriteria endCriteria (maxIterations, minStatIterations, 
			rootEpsilon, functionEpsilon, gradientNormEpsilon);	

        std::map<Rate, std::pair<Volatility, Real> > mapOfStdDeviationToMeanNoShortSales;
       
		Rate startingC = -.035;
		Real increment = .005;
		
        for (int i = 0; i < 40; ++i) {
            
            Rate c = startingC + (i * increment);  
		
         	ThetaCostFunction thetaCostFunction(covarianceMatrix, portfolioReturnVector);
			thetaCostFunction.setC(c);
            Problem efficientFrontierNoShortSalesProblem(thetaCostFunction, noShortSalesPortfolioConstraints, Array(3, .2500));
			Simplex solver(.01);
            
			EndCriteria::Type noShortSalesSolution = solver.minimize(efficientFrontierNoShortSalesProblem, endCriteria);

			std::cout << boost::format("Solution type: %s") % noShortSalesSolution << std::endl;
        
    	    const Array& results = efficientFrontierNoShortSalesProblem.currentValue();
            Array proportions(4);
            proportions[0] = results[0];
            proportions[1] = results[1];
            proportions[2] = results[2];
            proportions[3] = 1.0 - (results[0] + results[1] + results[2]);
            std::cout << boost::format("Constant (c): %.4f") % thetaCostFunction.getC() << std::endl;
			std::cout << boost::format("AAPL weighting: %.4f") % proportions[0] << std::endl;  
			std::cout << boost::format("IBM weighting: %.4f") % proportions[1] << std::endl;  
			std::cout << boost::format("ORCL weighting: %.4f") % proportions[2] << std::endl;  
			std::cout << boost::format("GOOG weighting: %.4f") % proportions[3] << std::endl;
			std::cout << boost::format("Theta: %.4f") % (-1 * efficientFrontierNoShortSalesProblem.functionValue()) << std::endl;
            Real portfolioMean = thetaCostFunction.portfolioMean(proportions); 
            std::cout << boost::format("Portfolio mean: %.4f") % portfolioMean << std::endl;
            Volatility portfolioStdDeviation = thetaCostFunction.portfolioStdDeviation(proportions); 
            std::cout << boost::format("Portfolio standard deviation: %.4f") % portfolioStdDeviation << std::endl;
            mapOfStdDeviationToMeanNoShortSales[c] = std::make_pair(portfolioStdDeviation, portfolioMean); 
            std::cout << "------------------------------------" << std::endl;
        }
		
        //write efficient frontier with no short sales to file
		std::ofstream noShortSalesFile;
		noShortSalesFile.open("/tmp/positionlimits.dat", std::ios::out);
		for (std::map<Rate, std::pair<Volatility, Real> >::const_iterator i=mapOfStdDeviationToMeanNoShortSales.begin(); i != mapOfStdDeviationToMeanNoShortSales.end(); ++i) {
			noShortSalesFile << boost::format("%f %f %f") % i->first % i->second.first % i->second.second << std::endl;
		}
		noShortSalesFile.close();
		
		//plot with gnuplot using commands below Run 'gnuplot' then type in: 
		/*
         set terminal png
         set output "/tmp/positionlimits.png"
         set key top left
		 set key box
		 set xlabel "Volatility"
		 set ylabel "Expected Return"
		 plot '/tmp/noshortsales.dat' using 2:3 w linespoints title "No Short Sales", "/tmp/positionlimits.dat" using 2:3 w linespoints title "Position Limits" 
		*/
    }

    /**
     * Introducing QuantLib: Option Theory and Pricing 
     */
    
    Real expectedValueCallPayoff(Real spot, Real strike,
            Rate r, Volatility sigma, Time t, Real x) {
        Real mean = log(spot)+(r - 0.5 * sigma * sigma) * t;
        Real stdDev = sigma * sqrt(t);
        boost::math::lognormal d(mean, stdDev);
        return PlainVanillaPayoff(Option::Type::Call, strike)(x) * boost::math::pdf(d, x);
    }

    BOOST_AUTO_TEST_CASE(testPriceCallOption) {
        Real spot = 100.0; //current price of the underlying stock
        Rate r = 0.03; //risk-free rate
        Time t = 0.5; //half a year
        Volatility vol = 0.20; //estimated volatility of underlying
        Real strike = 110.0;

        //b need not be infinity, but can just be a large number that is highly improbable
        Real a = strike, b = strike * 10.0;

        boost::function < Real(Real) > ptrToExpectedValueCallPayoff =
                boost::bind(&expectedValueCallPayoff, spot, strike, r, vol, t, _1);

        Real absAcc = 0.00001;
        Size maxEval = 1000;
        SimpsonIntegral numInt(absAcc, maxEval);

        /* numerically integrate the call option payoff function from a to b and
         * calculate the present value using the risk free rate as the discount
         * factor
         */
        Real callOptionValue = numInt(ptrToExpectedValueCallPayoff, a, b) * exp(-r * t);

        std::cout << "Call option value is: " << callOptionValue << std::endl;
    }

    /**
     * Introducing QuantLib: Black-Scholes and the Greeks 
     */
    
    BOOST_AUTO_TEST_CASE(testBlackScholes) {
        Real strike = 110.0;
        Real timeToMaturity = .50; //years
        Real spot = 100.0;
        Rate riskFree = .03;
        Rate dividendYield = 0.0;
        Volatility sigma = .20;
        Real vol = sigma * std::sqrt(timeToMaturity);
        DiscountFactor growth = std::exp(-dividendYield * timeToMaturity);
        DiscountFactor discount = std::exp(-riskFree * timeToMaturity);
        boost::shared_ptr<PlainVanillaPayoff> vanillaCallPayoff = boost::shared_ptr<PlainVanillaPayoff>(new PlainVanillaPayoff(Option::Type::Call, strike));
       
        BlackScholesCalculator bsCalculator(vanillaCallPayoff, spot, growth, vol, discount);
        std::cout << boost::format("Value of 110.0 call is %.4f") % bsCalculator.value() << std::endl;
        std::cout << boost::format("Delta of 110.0 call is %.4f") % bsCalculator.delta() << std::endl;
        std::cout << boost::format("Gamma of 110.0 call is %.4f") % bsCalculator.gamma() << std::endl;
        std::cout << boost::format("Vega of 110.0 call is %.4f") % (bsCalculator.vega(timeToMaturity)/100) << std::endl;
        std::cout << boost::format("Theta of 110.0 call is %.4f") % (bsCalculator.thetaPerDay(timeToMaturity)) << std::endl;

        Real changeInSpot = 1.0;
        BlackScholesCalculator bsCalculatorSpotUpOneDollar(Option::Type::Call, strike, spot + changeInSpot, growth, vol, discount);
        std::cout << boost::format("Value of 110.0 call (spot up $%d) is %.4f") % changeInSpot % bsCalculatorSpotUpOneDollar.value() << std::endl;
        std::cout << boost::format("Value of 110.0 call (spot up $%d) estimated from delta is %.4f") % changeInSpot % (bsCalculator.value() +
                bsCalculator.delta() * changeInSpot) << std::endl;
        
        std::cout << boost::format("Value of 110.0 call (spot up $%d) estimated from delta and gamma is %.4f") % changeInSpot % (bsCalculator.value() + 
                (bsCalculator.delta() * changeInSpot) + (.5 * bsCalculator.gamma() * changeInSpot)) << std::endl;

        Real changeInSigma = .01;
        BlackScholesCalculator bsCalculatorSigmaUpOnePoint(Option::Type::Call, strike, spot, growth, (sigma + changeInSigma) * std::sqrt(timeToMaturity) , discount);
        std::cout << boost::format("Value of 110.0 call (sigma up %.2f) is %.4f") % changeInSigma % bsCalculatorSigmaUpOnePoint.value() << std::endl;

        std::cout << boost::format("Value of 110.0 call (sigma up %.2f) estimated from vega) is %.4f") % changeInSigma % (bsCalculator.value() + 
                (bsCalculator.vega(timeToMaturity)/100)) << std::endl;
	}

	/**
     * Introducing QuantLib: Implied Volatility
     */    

    BOOST_AUTO_TEST_CASE(testBlackScholesImpliedVolatility) {

        Real strike = 110.0;
        Real timeToMaturity = .50; //years
        Real spot = 100.0;
        Rate riskFree = .03;
        Rate dividendYield = 0.0;
        DiscountFactor growth = std::exp(-dividendYield * timeToMaturity);
        DiscountFactor discount = std::exp(-riskFree * timeToMaturity);
        boost::shared_ptr<PlainVanillaPayoff> vanillaCallPayoff =
                boost::shared_ptr<PlainVanillaPayoff > (new PlainVanillaPayoff(Option::Type::Call, strike));

        Bisection bisection;
        Real accuracy = 0.000001, guess = .30;
        Real min = .05, max = .40;
        Real price = 2.6119;

        Volatility sigma = bisection.solve([&](const Volatility & sigma) {
            Real stdDev = sigma * std::sqrt(timeToMaturity);
            BlackScholesCalculator bsCalculator(vanillaCallPayoff, spot, growth, stdDev, discount);
            return bsCalculator.value() - price;
        }, accuracy, guess, min, max);

        std::cout << boost::format("Implied volatility of %f call is %.4f") % strike % sigma << std::endl;

    }

    class StrikeInfo {
    public:
        typedef std::pair<SimpleQuote, SimpleQuote> BidAsk;

        StrikeInfo(Option::Type optionType, const BidAsk& bidAsk, Real strike) :
        _payoff(new PlainVanillaPayoff(optionType, strike)), _bidAsk(bidAsk),
        _impliedVol(0.0) {
            //std::cout << "In constructor" << std::endl;
        }

        ~StrikeInfo() {
            //std::cout << "In destructor" << std::endl;
        }

        //copy constructor

        StrikeInfo(const StrikeInfo& that)
        : _payoff(new PlainVanillaPayoff(that.getPayoff().optionType(), that.getPayoff().strike())),
        _bidAsk(that.getBidAsk()), _impliedVol(that.getImpliedVol()) {
            //std::cout << "In copy constructor" << std::endl;
        }

        //assignment operator - implements copy-and-swap idiom

        StrikeInfo& operator=(StrikeInfo that) {
            //std::cout << "In assignment operator" << std::endl;
            swap(*this, that);
        }

        //swap

        friend void swap(StrikeInfo& first, StrikeInfo& second) {
            using std::swap;
            first._payoff.swap(second._payoff);
            std::swap(first._impliedVol, second._impliedVol);
            std::swap(first._bidAsk, second._bidAsk);
        }

        const StrikedTypePayoff& getPayoff() const {
            return *_payoff;
        }

        const BidAsk& getBidAsk() const {
            return _bidAsk;
        }

        const Volatility& getImpliedVol() const {
            return _impliedVol;
        }

        void setImpliedVol(Volatility impliedVol) {
            _impliedVol = impliedVol;
        }

		Real getStrike() { return _payoff->strike(); }


    private:

        boost::scoped_ptr<StrikedTypePayoff> _payoff;
        Volatility _impliedVol;
        BidAsk _bidAsk;

    };
	
    BOOST_AUTO_TEST_CASE(testESFuturesImpliedVolatility) {

		using namespace boost::posix_time;
    	using namespace boost::gregorian;
		
		ActualActual actualActual;
		Settings::instance().evaluationDate() = Date(26, Month::August, 2013);
		Date expiration(20, Month::September, 2013);
		
		Time timeToMaturity = actualActual.yearFraction(Settings::instance().evaluationDate(), expiration);
		ptime quoteTime(from_iso_string("20130826T143000"));
		time_duration timeOfDayDuration = quoteTime.time_of_day();
		timeToMaturity += (timeOfDayDuration.hours() + timeOfDayDuration.minutes()/60.0)/(24.0 * 365.0);
		std::cout << boost::format("Time to maturity: %.6f") % timeToMaturity << std::endl;
        Real forwardBid = 1656.00; 
        Real forwardAsk = 1656.25;
        Rate riskFree = .00273;  //interpolated LIBOR rate (between EDU3 and EDV3)
        DiscountFactor discount = std::exp(-riskFree * timeToMaturity);

		//calculate implied volatilities for OTM put options
		std::vector<StrikeInfo> putOptions;
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(7.75, 8.00), 1600));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(8.50, 9.00), 1605));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(9.25, 9.75), 1610));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(10.25, 10.75), 1615));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(11.25, 11.75), 1620));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(12.50, 12.75), 1625));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(13.75, 14.00), 1630));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(15.00, 15.50), 1635));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(16.50, 17.00), 1640));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(18.00, 18.50), 1645));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(20.00, 20.25), 1650));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(21.75, 22.25), 1655));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(24.00, 24.25), 1660));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(26.25, 26.75), 1665));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(28.75, 29.25), 1670));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(31.25, 32.25), 1675));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(34.25, 35.25), 1680));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(37.25, 38.25), 1685));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(40.75, 41.75), 1690));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(44.25, 45.25), 1695));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(47.50, 49.75), 1700));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(51.50, 53.75), 1705));
		putOptions.push_back(StrikeInfo(Option::Type::Put, std::make_pair(55.75, 58.00), 1710));
        	
		for (StrikeInfo& putOption: putOptions) {
			StrikeInfo::BidAsk bidAsk = putOption.getBidAsk();
			Real price = (bidAsk.first.value() + bidAsk.second.value())/2.0;
			const StrikedTypePayoff& payoff = putOption.getPayoff();
			if (payoff(forwardAsk) > 0) continue; //skip ITM options
        	Bisection bisection;
        	Real accuracy = 0.000001, guess = .20;
        	Real min = .05, max = .40;
        	Volatility sigma = bisection.solve([&](const Volatility & sigma) {
            	Real stdDev = sigma * std::sqrt(timeToMaturity);
            	BlackCalculator blackCalculator(payoff.optionType(), payoff.strike(), forwardAsk, stdDev, discount);
            	return blackCalculator.value() - price;
        		}, accuracy, guess, min, max);

			putOption.setImpliedVol(sigma);
        	std::cout << boost::format("IV of %f put is %.4f") % putOption.getStrike() % sigma << std::endl;
		}

		//calculate implied volatilities for OTM call options
		std::vector<StrikeInfo> callOptions;
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(63.00, 65.25), 1600));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(59.25, 60.25), 1605));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(55.25, 56.25), 1610));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(51.00, 52.00), 1615));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(47.00, 48.00), 1620));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(43.25, 44.25), 1625));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(39.50, 40.50), 1630));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(35.75, 36.75), 1635));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(32.25, 33.25), 1640));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(29.25, 29.75), 1645));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(26.00, 26.25), 1650));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(23.00, 23.50), 1655));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(20.00, 20.50), 1660));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(17.25, 17.75), 1665));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(14.75, 15.25), 1670));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(12.50, 13.00), 1675));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(10.50, 11.00), 1680));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(8.75, 9.25), 1685));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(7.00, 7.50), 1690));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(5.75, 6.00), 1695));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(4.70, 4.80), 1700));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(3.70, 3.85), 1705));
		callOptions.push_back(StrikeInfo(Option::Type::Call, std::make_pair(2.90, 3.05), 1710));
        	
		for (StrikeInfo& callOption: callOptions) {
			StrikeInfo::BidAsk bidAsk = callOption.getBidAsk();
			Real price = (bidAsk.first.value() + bidAsk.second.value())/2.0;
			const StrikedTypePayoff& payoff = callOption.getPayoff();
			if (payoff(forwardBid) > 0) continue; //skip ITM options
        	Bisection bisection;
        	Real accuracy = 0.000001, guess = .20;
        	Real min = .05, max = .40;
        	Volatility sigma = bisection.solve([&](const Volatility & sigma) {
            	Real stdDev = sigma * std::sqrt(timeToMaturity);
            	BlackCalculator blackCalculator(payoff.optionType(), payoff.strike(), forwardBid, stdDev, discount);
            	return blackCalculator.value() - price;
        		}, accuracy, guess, min, max);

			callOption.setImpliedVol(sigma);

        	std::cout << boost::format("IV of %f call is %.4f") % callOption.getStrike() % sigma << std::endl;
		}

		//write strike and IV to file for each option
		std::ofstream ivFile;
		ivFile.open("/tmp/iv.dat", std::ios::out);

		//write OTM put IVs
		for (StrikeInfo& putOption: putOptions) {
			if (putOption.getImpliedVol() > 0.0) {
				ivFile << boost::format("%f %f") % putOption.getStrike() % putOption.getImpliedVol() << std::endl; 
			}
		}

		//write OTM call IVs		
		for (StrikeInfo& callOption: callOptions) {
			if (callOption.getImpliedVol() > 0.0) {
				ivFile << boost::format("%f %f") % callOption.getStrike() % callOption.getImpliedVol() << std::endl; 
			}
		}
		ivFile.close();

		//plot with gnuplot using commands below Run 'gnuplot' then type in: 
		/*
         set terminal png
         set output "/tmp/volsmile.png"
         set key top center
		 set key box
		 set xlabel "Strike"
		 set ylabel "Volatility"
		 plot '/tmp/iv.dat' using 1:2 w linespoints title "ES Volatility Smile"
		*/
    }
   
    /**
     * Introducing QuantLib: The Volatility Surface 
     */
    
    BOOST_AUTO_TEST_CASE(testVolatilitySurface) {
		
        using namespace boost::posix_time;
    	using namespace boost::gregorian;
        using namespace boost::assign;

        std::vector<Real> strikes;
		strikes += 1650.0, 1660.0, 1670.0, 1675.0, 1680.0;
        
        std::vector<Date> expirations;
		expirations += Date(20, Month::Dec, 2013), Date(17, Month::Jan, 2014), Date(21, Month::Mar, 2014),
			Date(20, Month::Jun, 2014), Date(19, Month::Sep, 2014);

        Matrix volMatrix(strikes.size(), expirations.size());
		
		//1650 - Dec, Jan, Mar, Jun, Sep
		volMatrix[0][0] = .15640;
		volMatrix[0][1] = .15433;
		volMatrix[0][2] = .16079;
		volMatrix[0][3] = .16394;
		volMatrix[0][4] = .17383;

		//1660 - Dec, Jan, Mar, Jun, Sep
		volMatrix[1][0] = .15343;
		volMatrix[1][1] = .15240;
		volMatrix[1][2] = .15804;
		volMatrix[1][3] = .16255;
		volMatrix[1][4] = .17303;
		
		//1670 - Dec, Jan, Mar, Jun, Sep
		volMatrix[2][0] = .15128;
		volMatrix[2][1] = .14888;
		volMatrix[2][2] = .15512;
		volMatrix[2][3] = .15944;
		volMatrix[2][4] = .17038;

		//1675 - Dec, Jan, Mar, Jun, Sep
		volMatrix[3][0] = .14798;
		volMatrix[3][1] = .14906;
		volMatrix[3][2] = .15522;
		volMatrix[3][3] = .16171;
		volMatrix[3][4] = .16156;

		//1680 - Dec, Jan, Mar, Jun, Sep
		volMatrix[4][0] = .14580;
		volMatrix[4][1] = .14576;
		volMatrix[4][2] = .15364;
		volMatrix[4][3] = .16037;
		volMatrix[4][4] = .16042;
	
        Date evaluationDate(30, Month::Sep, 2013);
		Settings::instance().evaluationDate() = evaluationDate;
		Calendar calendar = UnitedStates(UnitedStates::NYSE);
        DayCounter dayCounter = ActualActual(); 
		BlackVarianceSurface volatilitySurface(Settings::instance().evaluationDate(), calendar, expirations, strikes, volMatrix, dayCounter);		
    	volatilitySurface.enableExtrapolation(true);

		std::cout << "Using standard bilinear interpolation..." << std::endl;		
		Real dec1650Vol = volatilitySurface.blackVol(expirations[0], 1650.0, true);
		std::cout << boost::format("Dec13 1650.0 volatility: %f") % dec1650Vol << std::endl;
		
		Real dec1655Vol = volatilitySurface.blackVol(expirations[0], 1655.0, true);
		std::cout << boost::format("Dec13 1655.0 volatility (interpolated): %f") % dec1655Vol << std::endl;
		
		Real dec1685Vol = volatilitySurface.blackVol(expirations[0], 1685.0, true);
		std::cout << boost::format("Dec13 1685.0 volatility (interpolated): %f") % dec1685Vol << std::endl;
		
		Real jun1655Vol = volatilitySurface.blackVol(expirations[3], 1655.0, true);
		std::cout << boost::format("Jun14 1655.0 volatility (interpolated): %f") % jun1655Vol << std::endl;
		
		Real sep1680Vol = volatilitySurface.blackVol(expirations[4], 1680.0, true);
		std::cout << boost::format("Sep14 1680.0 volatility: %f") % sep1680Vol << std::endl;
	
		//change interpolator to bicubic splines
        volatilitySurface.setInterpolation<Bicubic>();
  
		std::cout << "Using bicubic spline interpolation..." << std::endl;
		dec1650Vol = volatilitySurface.blackVol(expirations[0], 1650.0, true);
		std::cout << boost::format("Dec13 1650.0 volatility: %f") % dec1650Vol << std::endl;
		
		dec1655Vol = volatilitySurface.blackVol(expirations[0], 1655.0, true);
		std::cout << boost::format("Dec13 1655.0 volatility (interpolated): %f") % dec1655Vol << std::endl;
		
		dec1685Vol = volatilitySurface.blackVol(expirations[0], 1685.0, true);
		std::cout << boost::format("Dec13 1685.0 volatility (interpolated): %f") % dec1685Vol << std::endl;
		
		jun1655Vol = volatilitySurface.blackVol(expirations[3], 1655.0, true);
		std::cout << boost::format("Jun14 1655.0 volatility (interpolated): %f") % jun1655Vol << std::endl;
		
		sep1680Vol = volatilitySurface.blackVol(expirations[4], 1680.0, true);
		std::cout << boost::format("Sep14 1680.0 volatility: %f") % sep1680Vol << std::endl; 
       
        //write out data points for gnuplot contour plot
        std::ofstream volSurfaceFile;
        volSurfaceFile.open("/home/mick/Documents/blog/volsurface/volsurface.dat", std::ios::out);

        for (Date expiration: expirations) {
        	for (Real strike = strikes[0] - 5.0; strike <= strikes[4] + 5.0; ++strike) {
               Real volatility = volatilitySurface.blackVol(expiration, strike, true);
               volSurfaceFile << boost::format("%f %f %f") % strike % dayCounter.dayCount(Settings::instance().evaluationDate(), expiration) % volatility << std::endl;
            }
        }

        volSurfaceFile.close();
	}

    /**
     * Introducing QuantLib: American Option Pricing With Dividends 
     */

	boost::shared_ptr<ZeroCurve> bootstrapDividendCurve(const Date& evaluationDate, const Date& expiration, const Date& exDivDate, Real underlyingPrice, Real annualDividend) {
        
		UnitedStates calendar(UnitedStates::NYSE);
		Settings::instance().evaluationDate() = evaluationDate;
		Real settlementDays = 2.0;
		
        Real dividendDiscountDays = (expiration - evaluationDate) + settlementDays;
		std::cout << boost::format("Dividend discounting days: %d") % dividendDiscountDays << std::endl;
		Rate dividendYield = (annualDividend/underlyingPrice) * dividendDiscountDays/365;
		
		// ex div dates and yields
       	std::vector<Date> exDivDates;
        std::vector<Rate> dividendYields;
       
		//last ex div date and yield
		exDivDates.push_back(calendar.advance(exDivDate, Period(-3, Months), ModifiedPreceding, true));
		dividendYields.push_back(dividendYield);
		
		//currently announced ex div date and yield
		exDivDates.push_back(exDivDate);
		dividendYields.push_back(dividendYield);
	
		//next ex div date (projected) and yield
		Date projectedNextExDivDate = calendar.advance(exDivDate, Period(3, Months), ModifiedPreceding, true); 
		std::cout << boost::format("Next projected ex div date for INTC: %s") % projectedNextExDivDate << std::endl;
		exDivDates.push_back(projectedNextExDivDate);
		dividendYields.push_back(dividendYield);
		
        return boost::shared_ptr<ZeroCurve>(new ZeroCurve(exDivDates, dividendYields, ActualActual(), calendar));
		
	}

	boost::shared_ptr<YieldTermStructure> bootstrapLiborZeroCurve(const Date& evaluationDate) {
		
		using namespace boost::assign;

        //bootstrap from USD LIBOR rates;
        IborIndex libor = USDLiborON();  
		const Calendar& calendar = libor.fixingCalendar();
		const Date& settlement = calendar.advance(evaluationDate, 2, Days);
        const DayCounter& dayCounter = libor.dayCounter();       
		Settings::instance().evaluationDate() = settlement;
	
        //rates obtained from http://www.global-rates.com/interest-rates/libor/libor.aspx 
		Rate overnight = .10490/100.0;
		Rate oneWeek = .12925/100.0;
		Rate oneMonth = .16750/100.0;
		Rate twoMonths = .20700/100.0;
		Rate threeMonths = .23810/100.0;
		Rate sixMonths = .35140/100.0;
		Rate twelveMonths = .58410/100.0;
                
    	std::vector<boost::shared_ptr<RateHelper>> liborRates;
		liborRates += boost::shared_ptr<RateHelper>(new DepositRateHelper(overnight,
			boost::shared_ptr<IborIndex>(new USDLiborON()))); 
		liborRates += boost::shared_ptr<RateHelper>(new DepositRateHelper(oneWeek,
			boost::shared_ptr<IborIndex>(new USDLibor(Period(1, Weeks)))));
		liborRates += boost::shared_ptr<RateHelper>(new DepositRateHelper(oneMonth,
			boost::shared_ptr<IborIndex>(new USDLibor(Period(1, Months)))));
		liborRates += boost::shared_ptr<RateHelper>(new DepositRateHelper(twoMonths,
			boost::shared_ptr<IborIndex>(new USDLibor(Period(2, Months)))));
		liborRates += boost::shared_ptr<RateHelper>(new DepositRateHelper(threeMonths,
			boost::shared_ptr<IborIndex>(new USDLibor(Period(3, Months)))));
		liborRates += boost::shared_ptr<RateHelper>(new DepositRateHelper(sixMonths,
			boost::shared_ptr<IborIndex>(new USDLibor(Period(6, Months)))));
		liborRates += boost::shared_ptr<RateHelper>(new DepositRateHelper(twelveMonths,
			boost::shared_ptr<IborIndex>(new USDLibor(Period(12, Months)))));
        
        //use cubic interpolation
		boost::shared_ptr<YieldTermStructure> yieldCurve = 
				boost::shared_ptr<YieldTermStructure>(new PiecewiseYieldCurve<ZeroYield, Cubic>(settlement, liborRates, dayCounter));

		return yieldCurve;	
    }
   
	boost::shared_ptr<BlackVolTermStructure> bootstrapVolatilityCurve(const Date& evaluationDate, const std::vector<Real>& strikes, 
            const std::vector<Volatility>& vols, const Date& expiration) {

        Calendar calendar = UnitedStates(UnitedStates::NYSE);
        
        std::vector<Date> expirations;
		expirations.push_back(expiration);

        Matrix volMatrix(strikes.size(), 1);
		
		//implied volatilities from Interactive Brokers
        for (int i=0; i< vols.size(); ++i) {
		    volMatrix[i][0] = vols[i];
        }
		
		return boost::shared_ptr<BlackVolTermStructure>(new BlackVarianceSurface(evaluationDate, calendar, expirations, strikes, volMatrix, Actual365Fixed()));		
	}

    BOOST_AUTO_TEST_CASE(testAmericanOptionPricingWithDividends) {

        using namespace boost::assign;
        
		//set up calendar/dates
        Calendar calendar = UnitedStates(UnitedStates::NYSE);
        Date today(15, Nov, 2013);
		Real settlementDays = 2;
		Date settlement = calendar.advance(today, settlementDays, Days);
        Settings::instance().evaluationDate() = today;
        
		//define options to price
		Option::Type type(Option::Call);
      	Real underlying = 24.52;
       
        std::vector<Real> strikes;
        strikes += 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0;
        
        std::vector<Volatility> vols;
        vols += .23356, .21369, .20657, .20128, .19917, .19978, .20117;
               
     	Date expiration(21, Feb, 2014);
        Date exDivDate(5, Feb, 2014);
        Real annualDividend = .90;
        
		//build yield term structure from LIBOR rates 
		Handle<YieldTermStructure> yieldTermStructure(bootstrapLiborZeroCurve(today));
		//Handle<YieldTermStructure> yieldTermStructure(boost::shared_ptr<YieldTermStructure>(new FlatForward(settlement, .2565/100.0, Actual365Fixed())));
		
		//build dividend term structure
		Handle<YieldTermStructure> dividendTermStructure(bootstrapDividendCurve(today, expiration, exDivDate, underlying, annualDividend));
        //Real dividendDiscountDays = (expiration - today) + 1;
		//Handle<YieldTermStructure> dividendTermStructure(boost::shared_ptr<YieldTermStructure>(new FlatForward(settlement, (annualDividend/underlying) * dividendDiscountDays/365.0 , Actual365Fixed())));

		//build vol term structure 
		Handle<BlackVolTermStructure> volatilityTermStructure(bootstrapVolatilityCurve(today, strikes, vols, expiration));
		//Handle<BlackVolTermStructure> volatilityTermStructure(boost::shared_ptr<BlackVolTermStructure>(new BlackConstantVol(settlement, calendar, 23.48/100.0, Actual365Fixed())));
		//instantiate BSM process
		Handle<Quote> underlyingH(boost::shared_ptr<Quote>(new SimpleQuote(underlying)));
		boost::shared_ptr<BlackScholesMertonProcess> bsmProcess(new BlackScholesMertonProcess(underlyingH, dividendTermStructure, yieldTermStructure, volatilityTermStructure));
	
        //instantiate pricing engine
        boost::shared_ptr<PricingEngine> pricingEngine(new FDAmericanEngine<CrankNicolson>(bsmProcess, 1000, 999));
        
       	//price the options
        boost::shared_ptr<Exercise> americanExercise(new AmericanExercise(settlement, expiration));
        for (Real strike: strikes) {
			boost::shared_ptr<StrikedTypePayoff> payoff(new PlainVanillaPayoff(type, strike));
			VanillaOption americanOption(payoff, americanExercise);
			americanOption.setPricingEngine(pricingEngine);
			Real tv = americanOption.NPV();
			std::cout << boost::format("Intel %s %.2f %s value is: %.2f") % expiration % strike % type % tv  << std::endl;
			std::cout << boost::format("Delta: %.4f") % americanOption.delta() << std::endl;
			std::cout << boost::format("Gamma: %.4f") % americanOption.gamma() << std::endl;
        }
	}

}

	BOOST_AUTO_TEST_CASE(testGeometricBrownianMotion) {

		Real startingPrice = 20.16; //closing price for INTC on 12/7/2012
		Real mu = .2312;  //one year historical annual return
		Volatility sigma = 0.2116; //one year historical volatility
		Size timeSteps = 255;  //trading days in a year
		Time length = 1;  //one year
        const boost::shared_ptr<StochasticProcess>& gbm = 
			boost::shared_ptr<StochasticProcess>(new GeometricBrownianMotionProcess(startingPrice, mu, sigma));

		//generate normally distributed random numbers from uniform distribution using Box-Muller transformation
		BigInteger seed = SeedGenerator::instance().get();
		typedef BoxMullerGaussianRng<MersenneTwisterUniformRng> MersenneBoxMuller;
		MersenneTwisterUniformRng mersenneRng(seed);
		MersenneBoxMuller boxMullerRng(mersenneRng);
		RandomSequenceGenerator<MersenneBoxMuller> gsg(timeSteps, boxMullerRng);
		PathGenerator<RandomSequenceGenerator<MersenneBoxMuller> > gbmPathGenerator(gbm, length, timeSteps, gsg, false);
	
		const Path& samplePath = gbmPathGenerator.next().value;
		
		std::ofstream gbmFile;
		gbmFile.open("/home/mick/Documents/blog/geometric-brownian-motion/gbm.dat", std::ios::out);
		for (Size i=0; i<timeSteps; ++i) {
			gbmFile << boost::format("%d %.4f") % i % samplePath.at(i) << std::endl;
		}
		gbmFile.close();	
		
    }