using Microsoft.SqlServer.Server;
using System.Linq;

namespace AiDotNet.Helpers;

public static class StatisticsHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public static T CalculateMedian(Vector<T> values)
    {
        var sortedValues = values.ToArray();
        Array.Sort(sortedValues);
        int n = sortedValues.Length;
        if (n % 2 == 0)
        {
            return NumOps.Divide(NumOps.Add(sortedValues[n / 2 - 1], sortedValues[n / 2]), NumOps.FromDouble(2));
        }

        return sortedValues[n / 2];
    }

    public static T CalculateMeanAbsoluteDeviation(Vector<T> values, T median)
    {
        return NumOps.Divide(values.Select(x => NumOps.Abs(NumOps.Subtract(x, median))).Aggregate(NumOps.Zero, NumOps.Add),NumOps.FromDouble(values.Length));
    }

    public static T CalculateVariance(Vector<T> values, T mean)
    {
        T sumOfSquares = values.Select(x => NumOps.Square(NumOps.Subtract(x, mean))).Aggregate(NumOps.Zero, NumOps.Add);

        return NumOps.Divide(sumOfSquares, NumOps.FromDouble(values.Length - 1));
    }

    public static T CalculateStandardDeviation(Vector<T> values)
    {
        return NumOps.Sqrt(CalculateVariance(values, values.Average()));
    }

    public static T CalculateMeanSquaredError(Vector<T> actualValues, Vector<T> predictedValues)
    {
        T sumOfSquaredErrors = actualValues.Zip(predictedValues, (a, p) => NumOps.Square(NumOps.Subtract(a, p)))
                                           .Aggregate(NumOps.Zero, NumOps.Add);
        return NumOps.Divide(sumOfSquaredErrors, NumOps.FromDouble(actualValues.Length));
    }

    public static T CalculateRootMeanSquaredError(Vector<T> actualValues, Vector<T> predictedValues)
    {
        return NumOps.Sqrt(CalculateMeanSquaredError(actualValues, predictedValues));
    }

    public static T CalculateMeanAbsoluteError(Vector<T> actualValues, Vector<T> predictedValues)
    {
        T sumOfAbsoluteErrors = actualValues.Zip(predictedValues, (a, p) => NumOps.Abs(NumOps.Subtract(a, p))).Aggregate(NumOps.Zero, NumOps.Add);

        return NumOps.Divide(sumOfAbsoluteErrors, NumOps.FromDouble(actualValues.Length));
    }

    public static T CalculateR2(Vector<T> actualValues, Vector<T> predictedValues)
    {
        T actualMean = CalculateMean(actualValues);
        T totalSumSquares = actualValues.Select(a => NumOps.Square(NumOps.Subtract(a, actualMean)))
                                        .Aggregate(NumOps.Zero, NumOps.Add);
        T residualSumSquares = actualValues.Zip(predictedValues, (a, p) => NumOps.Square(NumOps.Subtract(a, p)))
                                           .Aggregate(NumOps.Zero, NumOps.Add);

        return NumOps.Subtract(NumOps.One, NumOps.Divide(residualSumSquares, totalSumSquares));
    }

    public static T CalculateMean(Vector<T> values)
    {
        return NumOps.Divide(values.Aggregate(NumOps.Zero, NumOps.Add), NumOps.FromDouble(values.Length));
    }

    public static T CalculateAdjustedR2(T r2, int n, int p)
    {
        T nMinusOne = NumOps.FromDouble(n - 1);
        T nMinusPMinusOne = NumOps.FromDouble(n - p - 1);
        T oneMinusR2 = NumOps.Subtract(NumOps.One, r2);

        return NumOps.Subtract(NumOps.One, NumOps.Multiply(oneMinusR2, NumOps.Divide(nMinusOne, nMinusPMinusOne)));
    }

    public static T CalculateExplainedVarianceScore(Vector<T> actual, Vector<T> predicted)
    {
        T varActual = actual.Variance();
        T varError = actual.Subtract(predicted).Variance();

        return NumOps.Subtract(NumOps.One, NumOps.Divide(varError, varActual));
    }

    public static T CalculateNormalCDF(T mean, T stdDev, T x)
    {
        if (NumOps.LessThan(stdDev, NumOps.Zero) || NumOps.Equals(stdDev, NumOps.Zero)) return NumOps.Zero;
        T sqrt2 = NumOps.Sqrt(NumOps.FromDouble(2.0));
        T argument = NumOps.Divide(NumOps.Subtract(x, mean), NumOps.Multiply(stdDev, sqrt2));

        return NumOps.Divide(NumOps.Add(NumOps.One, Erf(argument)), NumOps.FromDouble(2.0));
    }

    private static T Erf(T x)
    {
        T t = NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Abs(x))));

        T result = NumOps.Subtract(NumOps.One, NumOps.Multiply(t, NumOps.Exp(NumOps.Negate(NumOps.Square(x)))));
        result = NumOps.Multiply(result, NumOps.Exp(NumOps.Negate(NumOps.Add(
            NumOps.FromDouble(1.26551223),
            NumOps.Multiply(t, NumOps.Add(
                NumOps.FromDouble(1.00002368),
                NumOps.Multiply(t, NumOps.Add(
                    NumOps.FromDouble(0.37409196),
                    NumOps.Multiply(t, NumOps.Add(
                        NumOps.FromDouble(0.09678418),
                        NumOps.Multiply(t, NumOps.Add(
                            NumOps.FromDouble(-0.18628806),
                            NumOps.Multiply(t, NumOps.Add(
                                NumOps.FromDouble(0.27886807),
                                NumOps.Multiply(t, NumOps.FromDouble(-1.13520398))
                            ))
                        ))
                    ))
                ))
            ))
        ))));

        return NumOps.GreaterThan(x, NumOps.Zero) ? result : NumOps.Negate(result);
    }

    public static T CalculateNormalPDF(T mean, T stdDev, T x)
    {
        if (NumOps.LessThan(stdDev, NumOps.Zero) || NumOps.Equals(stdDev, NumOps.Zero)) return NumOps.Zero;

        var num = NumOps.Divide(NumOps.Subtract(x, mean), stdDev);
        var numSquared = NumOps.Multiply(num, num);
        var exponent = NumOps.Multiply(NumOps.FromDouble(-0.5), numSquared);

        var numerator = NumOps.Exp(exponent);
        var denominator = NumOps.Multiply(NumOps.Sqrt(NumOps.FromDouble(2 * Math.PI)), stdDev);

        return NumOps.Divide(numerator, denominator);
    }

    public static DistributionFitResult<T> DetermineBestFitDistribution(Vector<T> values)
    {
        // Test Normal Distribution
        var result = FitNormalDistribution(values);
        T bestGoodnessOfFit = result.GoodnessOfFit;
        result.DistributionType = DistributionType.Normal;

        // Test Laplace Distribution
        var laplaceFit = FitLaplaceDistribution(values);
        if (NumOps.LessThan(laplaceFit.GoodnessOfFit, bestGoodnessOfFit))
        {
            bestGoodnessOfFit = laplaceFit.GoodnessOfFit;
            result = laplaceFit;
            result.DistributionType = DistributionType.Laplace;
        }

        // Test Student's t-Distribution
        var studentFit = FitStudentDistribution(values);
        if (NumOps.LessThan(studentFit.GoodnessOfFit, bestGoodnessOfFit))
        {
            bestGoodnessOfFit = studentFit.GoodnessOfFit;
            result = studentFit;
            result.DistributionType = DistributionType.Student;
        }

        // Test Log-Normal Distribution
        var logNormalFit = FitLogNormalDistribution(values);
        if (NumOps.LessThan(logNormalFit.GoodnessOfFit, bestGoodnessOfFit))
        {
            bestGoodnessOfFit = logNormalFit.GoodnessOfFit;
            result = logNormalFit;
            result.DistributionType = DistributionType.LogNormal;
        }

        // Test Exponential Distribution
        var exponentialFit = FitExponentialDistribution(values);
        if (NumOps.LessThan(exponentialFit.GoodnessOfFit, bestGoodnessOfFit))
        {
            bestGoodnessOfFit = exponentialFit.GoodnessOfFit;
            result = exponentialFit;
            result.DistributionType = DistributionType.Exponential;
        }

        // Test Weibull Distribution
        var weibullFit = FitWeibullDistribution(values);
        if (NumOps.LessThan(weibullFit.GoodnessOfFit, bestGoodnessOfFit))
        {
            result = weibullFit;
            result.DistributionType = DistributionType.Weibull;
        }

        return result;
    }

    private static DistributionFitResult<T> FitLaplaceDistribution(Vector<T> values)
    {
        T median = CalculateMedian(values);
        T mad = CalculateMeanAbsoluteDeviation(values, median);
        T goodnessOfFit = CalculateGoodnessOfFit(values, x => CalculateLaplacePDF(median, mad, x));

        return new DistributionFitResult<T>
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, T>
            {
                { "Median", median },
                { "MAD", mad }
            }
        };
    }

    private static DistributionFitResult<T> FitLogNormalDistribution(Vector<T> values)
    {
        var logSample = values.Transform(x => NumOps.Log(x));
        T mu = CalculateMean(logSample);
        T sigma = CalculateStandardDeviation(logSample);
        T goodnessOfFit = CalculateGoodnessOfFit(values, x => CalculateLogNormalPDF(mu, sigma, x));

        return new DistributionFitResult<T>
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, T>
            {
                { "Mu", mu },
                { "Sigma", sigma }
            }
        };
    }

    public static T CalculateChiSquarePDF(int degreesOfFreedom, T x)
    {
        T halfDf = NumOps.Divide(NumOps.FromDouble(degreesOfFreedom), NumOps.FromDouble(2.0));
        T term1 = NumOps.Power(x, NumOps.Subtract(halfDf, NumOps.One));
        T term2 = NumOps.Exp(NumOps.Negate(NumOps.Divide(x, NumOps.FromDouble(2.0))));
        T denominator = NumOps.Multiply(NumOps.Power(NumOps.FromDouble(2.0), halfDf), Gamma(halfDf));

        return NumOps.Divide(NumOps.Multiply(term1, term2), denominator);
    }

    public static T Digamma(T x)
    {
        // Approximation of the digamma function
        T result = NumOps.Zero;
        T eight = NumOps.FromDouble(8);

        for (int i = 0; i < 100 && NumOps.LessThan(x, eight) || NumOps.Equals(x, eight); i++)
        {
            result = NumOps.Subtract(result, NumOps.Divide(NumOps.One, x));
            x = NumOps.Add(x, NumOps.One);
        }

        if (NumOps.LessThan(x, eight) || NumOps.Equals(x, eight)) return result;

        T invX = NumOps.Divide(NumOps.One, x);
        T invX2 = NumOps.Square(invX);

        return NumOps.Subtract(
            NumOps.Subtract(
                NumOps.Log(x),
                NumOps.Multiply(NumOps.FromDouble(0.5), invX)
            ),
            NumOps.Multiply(
                invX2,
                NumOps.Subtract(
                    NumOps.FromDouble(1.0 / 12),
                    NumOps.Multiply(
                        invX2,
                        NumOps.Subtract(
                            NumOps.FromDouble(1.0 / 120),
                            NumOps.Multiply(invX2, NumOps.FromDouble(1.0 / 252))
                        )
                    )
                )
            )
        );
    }

    public static T LogGamma(T x)
    {
        // Lanczos approximation for log of Gamma function
        T[] c = { NumOps.FromDouble(76.18009172947146), NumOps.FromDouble(-86.50532032941677), NumOps.FromDouble(24.01409824083091),
                  NumOps.FromDouble(-1.231739572450155), NumOps.FromDouble(0.1208650973866179e-2), NumOps.FromDouble(-0.5395239384953e-5) };
        T sum = NumOps.FromDouble(0.99999999999980993);
        for (int i = 0; i < 6; i++)
            sum = NumOps.Add(sum, NumOps.Divide(c[i], NumOps.Add(NumOps.Add(x, NumOps.FromDouble(i)), NumOps.One)));

        return NumOps.Add(
            NumOps.Subtract(
                NumOps.Multiply(
                    NumOps.Add(x, NumOps.FromDouble(0.5)),
                    NumOps.Log(NumOps.Add(x, NumOps.FromDouble(5.5)))
                ),
                NumOps.Add(x, NumOps.FromDouble(5.5))
            ),
            NumOps.Log(NumOps.Multiply(NumOps.FromDouble(2.5066282746310005), NumOps.Divide(sum, x)))
        );
    }

    public static T Gamma(T x)
    {
        return NumOps.Exp(LogGamma(x));
    }

    public static (T Shape, T Scale) EstimateWeibullParameters(Vector<T> values)
    {
        // Method of moments estimation for Weibull parameters
        T mean = values.Average();
        T variance = CalculateVariance(values, mean);

        // Initial guess for shape parameter
        T shape = NumOps.Sqrt(NumOps.Divide(NumOps.Multiply(NumOps.FromDouble(Math.PI), NumOps.FromDouble(Math.PI)), 
                                      NumOps.Multiply(NumOps.FromDouble(6), variance)));

        // Newton-Raphson method to refine shape estimate
        for (int i = 0; i < 10; i++)
        {
            T g = Gamma(NumOps.Add(NumOps.One, NumOps.Divide(NumOps.One, shape)));
            T g2 = Gamma(NumOps.Add(NumOps.One, NumOps.Divide(NumOps.FromDouble(2), shape)));
            T f = NumOps.Subtract(NumOps.Subtract(NumOps.Divide(g2, NumOps.Multiply(g, g)), NumOps.One), 
                               NumOps.Divide(variance, NumOps.Multiply(mean, mean)));
            T fPrime = NumOps.Subtract(
                NumOps.Multiply(NumOps.FromDouble(2), 
                    NumOps.Divide(NumOps.Subtract(
                        NumOps.Divide(Digamma(NumOps.Add(NumOps.One, NumOps.Divide(NumOps.FromDouble(2), shape))), shape),
                        NumOps.Divide(Digamma(NumOps.Add(NumOps.One, NumOps.Divide(NumOps.One, shape))), shape)),
                    NumOps.Divide(g2, NumOps.Multiply(g, g)))),
                NumOps.Multiply(NumOps.FromDouble(2), 
                    NumOps.Multiply(NumOps.Subtract(NumOps.Divide(g2, NumOps.Multiply(g, g)), NumOps.One),
                                 NumOps.Divide(Digamma(NumOps.Add(NumOps.One, NumOps.Divide(NumOps.One, shape))), shape))));

            shape = NumOps.Subtract(shape, NumOps.Divide(f, fPrime));

            if (NumOps.LessThan(NumOps.Abs(f), NumOps.FromDouble(1e-6)))
                break;
        }

        T scale = NumOps.Divide(mean, Gamma(NumOps.Add(NumOps.One, NumOps.Divide(NumOps.One, shape))));

        return (shape, scale);
    }

    public static T CalculateInverseExponentialCDF(T lambda, T probability)
    {
        if (NumOps.LessThanOrEquals(probability, NumOps.Zero) || NumOps.GreaterThanOrEquals(probability, NumOps.One))
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");

        return NumOps.Divide(NumOps.Negate(NumOps.Log(NumOps.Subtract(NumOps.One, probability))), lambda);
    }

    public static (T LowerBound, T UpperBound) CalculateWeibullCredibleIntervals(Vector<T> sample, T lowerProbability, T upperProbability)
    {
        // Estimate Weibull parameters
        var (shape, scale) = EstimateWeibullParameters(sample);

        // Calculate credible intervals
        T lowerBound = NumOps.Multiply(scale, NumOps.Power(NumOps.Negate(NumOps.Log(NumOps.Subtract(NumOps.One, lowerProbability))), 
                                                     NumOps.Divide(NumOps.One, shape)));
        T upperBound = NumOps.Multiply(scale, NumOps.Power(NumOps.Negate(NumOps.Log(NumOps.Subtract(NumOps.One, upperProbability))), 
                                                     NumOps.Divide(NumOps.One, shape)));

        return (lowerBound, upperBound);
    }

    public static T CalculateInverseNormalCDF(T probability)
    {
        // Approximation of inverse normal CDF
        if (NumOps.LessThanOrEquals(probability, NumOps.Zero) || NumOps.GreaterThanOrEquals(probability, NumOps.One))
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");

        T t = NumOps.Sqrt(NumOps.Multiply(NumOps.FromDouble(-2), NumOps.Log(NumOps.LessThan(probability, NumOps.FromDouble(0.5)) ? probability : 
            NumOps.Subtract(NumOps.One, probability))));
        T c0 = NumOps.FromDouble(2.515517);
        T c1 = NumOps.FromDouble(0.802853);
        T c2 = NumOps.FromDouble(0.010328);
        T d1 = NumOps.FromDouble(1.432788);
        T d2 = NumOps.FromDouble(0.189269);
        T d3 = NumOps.FromDouble(0.001308);

        T x = NumOps.Subtract(t, NumOps.Divide(
            NumOps.Add(c0, NumOps.Add(NumOps.Multiply(c1, t), NumOps.Multiply(c2, NumOps.Multiply(t, t)))),
            NumOps.Add(NumOps.One, NumOps.Add(NumOps.Multiply(d1, t), 
                NumOps.Add(NumOps.Multiply(d2, NumOps.Multiply(t, t)), 
                    NumOps.Multiply(d3, NumOps.Multiply(t, NumOps.Multiply(t, t))))))));

        return NumOps.LessThan(probability, NumOps.FromDouble(0.5)) ? NumOps.Negate(x) : x;
    }

    public static T CalculateInverseChiSquareCDF(int degreesOfFreedom, T probability)
    {
        if (NumOps.LessThanOrEquals(probability, NumOps.Zero) || NumOps.GreaterThanOrEquals(probability, NumOps.One))
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");
        if (degreesOfFreedom <= 0)
            throw new ArgumentOutOfRangeException(nameof(degreesOfFreedom), "Degrees of freedom must be positive");

        // Initial guess
        T x = NumOps.Multiply(
            NumOps.FromDouble(degreesOfFreedom),
            NumOps.Power(
                NumOps.Add(
                    NumOps.Subtract(
                        NumOps.One,
                        NumOps.Divide(NumOps.FromDouble(2), NumOps.FromDouble(9 * degreesOfFreedom))
                    ),
                    NumOps.Multiply(
                        NumOps.Sqrt(NumOps.Divide(NumOps.FromDouble(2), NumOps.FromDouble(9 * degreesOfFreedom))),
                        CalculateInverseNormalCDF(probability)
                    )
                ),
                NumOps.FromDouble(3)
            )
        );

        // Newton-Raphson method for refinement
        const int maxIterations = 20;
        T epsilon = NumOps.FromDouble(1e-8);
        for (int i = 0; i < maxIterations; i++)
        {
            T fx = NumOps.Subtract(CalculateChiSquareCDF(degreesOfFreedom, x), probability);
            T dfx = CalculateChiSquarePDF(degreesOfFreedom, x);

            T delta = NumOps.Divide(fx, dfx);
            x = NumOps.Subtract(x, delta);

            if (NumOps.LessThan(NumOps.Abs(delta), epsilon))
                break;
        }

        return x;
    }

    public static T CalculateChiSquareCDF(int degreesOfFreedom, T x)
    {
        return IncompleteGamma(NumOps.Divide(NumOps.FromDouble(degreesOfFreedom), NumOps.FromDouble(2)), NumOps.Divide(x, NumOps.FromDouble(2)));
    }

    public static T IncompleteGamma(T a, T x)
    {
        const int maxIterations = 100;
        T epsilon = NumOps.FromDouble(1e-8);

        T sum = NumOps.Zero;
        T term = NumOps.Divide(NumOps.One, a);
        for (int i = 0; i < maxIterations; i++)
        {
            sum = NumOps.Add(sum, term);
            term = NumOps.Multiply(term, NumOps.Divide(x, NumOps.Add(a, NumOps.FromDouble(i + 1))));
            if (NumOps.LessThan(term, epsilon))
                break;
        }

        return NumOps.Multiply(
            sum,
            NumOps.Exp(NumOps.Subtract(
                NumOps.Subtract(NumOps.Multiply(a, NumOps.Log(x)), x),
                LogGamma(a)
            ))
        );
    }

    public static T CalculateInverseNormalCDF(T mean, T stdDev, T probability)
    {
        return NumOps.Add(mean, NumOps.Multiply(stdDev, CalculateInverseNormalCDF(probability)));
    }

    public static T CalculateInverseStudentTCDF(int degreesOfFreedom, T probability)
    {
        if (NumOps.LessThanOrEquals(probability, NumOps.Zero) || NumOps.GreaterThanOrEquals(probability, NumOps.One))
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");

        T x = CalculateInverseNormalCDF(probability);
        T y = NumOps.Square(x);

        T a = NumOps.Divide(NumOps.Add(y, NumOps.One), NumOps.FromDouble(4));
        T b = NumOps.Divide(NumOps.Add(NumOps.Add(NumOps.Multiply(NumOps.FromDouble(5), y), NumOps.FromDouble(16)), NumOps.Multiply(y, NumOps.FromDouble(3))), NumOps.FromDouble(96));
        T c = NumOps.Divide(NumOps.Subtract(NumOps.Add(NumOps.Multiply(NumOps.Add(NumOps.Multiply(NumOps.FromDouble(3), y), NumOps.FromDouble(19)), y), NumOps.FromDouble(17)), NumOps.FromDouble(15)), NumOps.FromDouble(384));
        T d = NumOps.Divide(NumOps.Subtract(NumOps.Subtract(NumOps.Add(NumOps.Multiply(NumOps.Add(NumOps.Multiply(NumOps.FromDouble(79), y), NumOps.FromDouble(776)), y), NumOps.FromDouble(1482)), NumOps.Multiply(y, NumOps.FromDouble(1920))), NumOps.FromDouble(945)), NumOps.FromDouble(92160));

        T t = NumOps.Multiply(x, NumOps.Add(NumOps.One, NumOps.Divide(NumOps.Add(a, NumOps.Divide(NumOps.Add(b, NumOps.Divide(NumOps.Add(c, NumOps.Divide(d, NumOps.FromDouble(degreesOfFreedom))), NumOps.FromDouble(degreesOfFreedom))), NumOps.FromDouble(degreesOfFreedom))), NumOps.FromDouble(degreesOfFreedom))));

        if (degreesOfFreedom <= 2)
        {
            // Additional refinement for low degrees of freedom
            T sign = NumOps.GreaterThan(x, NumOps.Zero) ? NumOps.One : NumOps.Negate(NumOps.One);
            T innerTerm = NumOps.Subtract(NumOps.Power(probability, NumOps.Divide(NumOps.FromDouble(-2.0), NumOps.FromDouble(degreesOfFreedom))), NumOps.One);
            t = NumOps.Multiply(sign, NumOps.Sqrt(NumOps.Multiply(NumOps.FromDouble(degreesOfFreedom), innerTerm)));
        }

        return t;
    }

    public static T CalculateInverseLaplaceCDF(T median, T mad, T probability)
    {
        T half = NumOps.FromDouble(0.5);
        T sign = NumOps.GreaterThan(probability, half) ? NumOps.One : NumOps.Negate(NumOps.One);
        return NumOps.Subtract(median, NumOps.Multiply(NumOps.Multiply(mad, sign), NumOps.Log(NumOps.Subtract(NumOps.One, 
            NumOps.Multiply(NumOps.FromDouble(2), NumOps.Abs(NumOps.Subtract(probability, half)))))));
    }

    public static (T LowerBound, T UpperBound) CalculateCredibleIntervals(Vector<T> values, T confidenceLevel, DistributionType distributionType)
    {
        T lowerProbability = NumOps.Divide(NumOps.Subtract(NumOps.One, confidenceLevel), NumOps.FromDouble(2)); // 0.025 for 95% CI
        T upperProbability = NumOps.Subtract(NumOps.One, lowerProbability); // 0.975 for 95% CI
        (var mean, var stdDev) = CalculateMeanAndStandardDeviation(values);
        var median = CalculateMedian(values);
        var mad = CalculateMeanAbsoluteDeviation(values, median);

        return distributionType switch
        {
            DistributionType.Normal => (
                CalculateInverseNormalCDF(mean, stdDev, lowerProbability),
                CalculateInverseNormalCDF(mean, stdDev, upperProbability)
            ),
            DistributionType.Laplace => (
                CalculateInverseLaplaceCDF(median, mad, lowerProbability),
                CalculateInverseLaplaceCDF(median, mad, upperProbability)
            ),
            DistributionType.Student => (
                CalculateInverseStudentTCDF(values.Length - 1, lowerProbability),
                CalculateInverseStudentTCDF(values.Length - 1, upperProbability)
            ),
            DistributionType.LogNormal => (
                NumOps.Exp(CalculateInverseNormalCDF(
                    NumOps.Subtract(NumOps.Log(mean), NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Log(NumOps.Add(NumOps.One, NumOps.Divide(NumOps.Square(stdDev), NumOps.Square(mean)))))),
                    NumOps.Sqrt(NumOps.Log(NumOps.Add(NumOps.One, NumOps.Divide(NumOps.Square(stdDev), NumOps.Square(mean))))),
                    lowerProbability
                )),
                NumOps.Exp(CalculateInverseNormalCDF(
                    NumOps.Subtract(NumOps.Log(mean), NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Log(NumOps.Add(NumOps.One, NumOps.Divide(NumOps.Square(stdDev), NumOps.Square(mean)))))),
                    NumOps.Sqrt(NumOps.Log(NumOps.Add(NumOps.One, NumOps.Divide(NumOps.Square(stdDev), NumOps.Square(mean))))),
                    upperProbability
                ))
            ),
            DistributionType.Exponential => (
                CalculateInverseExponentialCDF(NumOps.Divide(NumOps.One, mean), lowerProbability),
                CalculateInverseExponentialCDF(NumOps.Divide(NumOps.One, mean), upperProbability)
            ),
            DistributionType.Weibull => CalculateWeibullCredibleIntervals(values, lowerProbability, upperProbability),
            _ => throw new ArgumentException("Invalid distribution type"),
        };
    }

    public static (T LowerBound, T UpperBound) CalculateWeibullConfidenceIntervals(Vector<T> values, T confidenceLevel)
    {
        const int bootstrapSamples = 1000;
        var rng = new Random();
        var estimates = new List<(T Shape, T Scale)>();

        for (int i = 0; i < bootstrapSamples; i++)
        {
            var bootstrapSample = new Vector<T>(values.Length);
            for (int j = 0; j < values.Length; j++)
            {
                bootstrapSample[j] = values[rng.Next(values.Length)];
            }
            estimates.Add(EstimateWeibullParameters(bootstrapSample));
        }

        var sortedShapes = estimates.Select(e => e.Shape).OrderBy(s => s).ToList();
        var sortedScales = estimates.Select(e => e.Scale).OrderBy(s => s).ToList();

        T halfConfidenceLevel = NumOps.Divide(confidenceLevel, NumOps.FromDouble(2));
        T oneMinusHalfConfidenceLevel = NumOps.Subtract(NumOps.One, halfConfidenceLevel);
        T onePlusHalfConfidenceLevel = NumOps.Add(NumOps.One, halfConfidenceLevel);

        T lowerIndexT = NumOps.Multiply(NumOps.FromDouble(bootstrapSamples), oneMinusHalfConfidenceLevel);
        int lowerIndex = Convert.ToInt32(NumOps.Round(lowerIndexT));

        T upperIndexT = NumOps.Multiply(NumOps.FromDouble(bootstrapSamples), onePlusHalfConfidenceLevel);
        int upperIndex = Convert.ToInt32(NumOps.Round(upperIndexT));

        return (NumOps.Multiply(sortedShapes[lowerIndex], sortedScales[lowerIndex]),
                NumOps.Multiply(sortedShapes[upperIndex], sortedScales[upperIndex]));
    }

    public static T CalculateExponentialPDF(T lambda, T x)
    {
        if (NumOps.LessThanOrEquals(lambda, NumOps.Zero) || NumOps.LessThan(x, NumOps.Zero)) return NumOps.Zero;
        return NumOps.Multiply(lambda, NumOps.Exp(NumOps.Negate(NumOps.Multiply(lambda, x))));
    }

    public static T CalculateWeibullPDF(T k, T lambda, T x)
    {
        if (NumOps.LessThanOrEquals(k, NumOps.Zero) || NumOps.LessThanOrEquals(lambda, NumOps.Zero) || NumOps.LessThan(x, NumOps.Zero)) return NumOps.Zero;
        return NumOps.Multiply(
            NumOps.Divide(k, lambda),
            NumOps.Multiply(
                NumOps.Power(NumOps.Divide(x, lambda), NumOps.Subtract(k, NumOps.One)),
                NumOps.Exp(NumOps.Negate(NumOps.Power(NumOps.Divide(x, lambda), k)))
            )
        );
    }

    public static T CalculateLogNormalPDF(T mu, T sigma, T x)
    {
        if (NumOps.LessThanOrEquals(x, NumOps.Zero) || NumOps.LessThanOrEquals(sigma, NumOps.Zero)) return NumOps.Zero;
        var logX = NumOps.Log(x);
        var twoSigmaSquared = NumOps.Multiply(NumOps.FromDouble(2), NumOps.Square(sigma));
        return NumOps.Divide(
            NumOps.Exp(NumOps.Negate(NumOps.Divide(NumOps.Square(NumOps.Subtract(logX, mu)), twoSigmaSquared))),
            NumOps.Multiply(x, NumOps.Multiply(sigma, NumOps.Sqrt(NumOps.FromDouble(2 * Math.PI))))
        );
    }

    public static T CalculateLaplacePDF(T median, T mad, T x)
    {
        return NumOps.Divide(
            NumOps.Exp(NumOps.Negate(NumOps.Divide(NumOps.Abs(NumOps.Subtract(x, median)), mad))),
            NumOps.Multiply(NumOps.FromDouble(2), mad)
        );
    }

    public static T CalculateGoodnessOfFit(Vector<T> values, Func<T, T> pdfFunction)
    {
        T logLikelihood = NumOps.Zero;
        foreach (var value in values)
        {
            logLikelihood = NumOps.Add(logLikelihood, NumOps.Log(pdfFunction(value)));
        }

        return NumOps.Negate(logLikelihood);
    }

    private static DistributionFitResult<T> FitNormalDistribution(Vector<T> values)
    {
        (var mean, var stdDev) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(values);
        T goodnessOfFit = CalculateGoodnessOfFit(values, x => CalculateNormalPDF(mean, stdDev, x));

        return new DistributionFitResult<T>
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, T>
            {
                { "Mean", mean },
                { "StandardDeviation", stdDev }
            }
        };
    }

    public static (T Mean, T StandardDeviation) CalculateMeanAndStandardDeviation(Vector<T> values)
    {
        if (values.Length == 0)
        {
            return (NumOps.Zero, NumOps.Zero);
        }

        T sum = NumOps.Zero;
        T sumOfSquares = NumOps.Zero;
        int count = values.Length;

        for (int i = 0; i < count; i++)
        {
            sum = NumOps.Add(sum, values[i]);
            sumOfSquares = NumOps.Add(sumOfSquares, NumOps.Square(values[i]));
        }

        T mean = NumOps.Divide(sum, NumOps.FromDouble(count));
        T variance = NumOps.Subtract(
            NumOps.Divide(sumOfSquares, NumOps.FromDouble(count)),
            NumOps.Square(mean)
        );

        T standardDeviation = NumOps.Sqrt(variance);

        return (mean, standardDeviation);
    }

    public static T CalculateStudentPDF(T x, T mean, T stdDev, int df)
    {
        T t = NumOps.Divide(NumOps.Subtract(x, mean), stdDev);

        T numerator = Gamma(NumOps.Divide(NumOps.Add(NumOps.FromDouble(df), NumOps.One), NumOps.FromDouble(2.0)));
        T denominator = NumOps.Multiply(
            NumOps.Sqrt(NumOps.Multiply(NumOps.FromDouble(df), NumOps.FromDouble(Math.PI))),
            NumOps.Multiply(
                Gamma(NumOps.Divide(NumOps.FromDouble(df), NumOps.FromDouble(2.0))),
                NumOps.Power(
                    NumOps.Add(NumOps.One, NumOps.Divide(NumOps.Multiply(t, t), NumOps.FromDouble(df))),
                    NumOps.Divide(NumOps.Add(NumOps.FromDouble(df), NumOps.One), NumOps.FromDouble(2.0))
                )
            )
        );

        return NumOps.Divide(numerator, denominator);
    }

    private static (T min, T max) CalculateMinMax(Vector<T> sample)
    {
        if (sample.Length == 0) return (NumOps.Zero, NumOps.Zero);

        T min = sample[0], max = sample[0];
        for (int i = 1; i < sample.Length; i++)
        {
            if (NumOps.LessThan(sample[i], min)) min = sample[i];
            if (NumOps.GreaterThan(sample[i], max)) max = sample[i];
        }

        return (min, max);
    }

    private static DistributionFitResult<T> FitWeibullDistribution(Vector<T> values)
    {
        // Initial guess for k and lambda
        T k = NumOps.One;
        T lambda = StatisticsHelper<T>.CalculateMean(values);

        // Implement a simple optimization algorithm here to find best k and lambda
        // This is a placeholder and should be replaced with a proper optimization method
        for (int i = 0; i < 100; i++)
        {
            k = NumOps.Add(k, NumOps.FromDouble(0.1));
            lambda = NumOps.Power(StatisticsHelper<T>.CalculateMean(values.Transform(x => NumOps.Power(x, k))), NumOps.Divide(NumOps.One, k));
        }

        T goodnessOfFit = CalculateGoodnessOfFit(values, x => CalculateWeibullPDF(k, lambda, x));

        return new DistributionFitResult<T>
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, T>
            {
                { "K", k },
                { "Lambda", lambda }
            }
        };
    }

    private static DistributionFitResult<T> FitExponentialDistribution(Vector<T> sample)
    {
        T lambda = NumOps.Divide(NumOps.One, StatisticsHelper<T>.CalculateMean(sample));

        T goodnessOfFit = CalculateGoodnessOfFit(sample, x => CalculateExponentialPDF(lambda, x));

        return new DistributionFitResult<T>
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, T>
            {
                { "Lambda", lambda }
            }
        };
    }

    private static DistributionFitResult<T> FitStudentDistribution(Vector<T> values)
    {
        var df = values.Length - 1;
        (var mean, var stdDev) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(values);
        T goodnessOfFit = CalculateGoodnessOfFit(values, x => CalculateStudentPDF(x, mean, stdDev, df));

        return new DistributionFitResult<T>
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, T>
            {
                { "DegreesOfFreedom", NumOps.FromDouble(df) },
                { "Mean", mean },
                { "StandardDeviation", stdDev }
            }
        };
    }

    private static T Erfc(T x)
    {
        return NumOps.Subtract(NumOps.One, Erf(x));
    }

    public static (T LowerBound, T UpperBound) CalculateConfidenceIntervals(Vector<T> values, T confidenceLevel, DistributionType distributionType)
    {
        (var mean, var stdDev) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(values);
        var median = StatisticsHelper<T>.CalculateMedian(values);
        var mad = StatisticsHelper<T>.CalculateMeanAbsoluteDeviation(values, median);
        T lowerBound, upperBound;

        switch (distributionType)
        {
            case DistributionType.Normal:
                var zScore = CalculateInverseNormalCDF(NumOps.Subtract(NumOps.One, NumOps.Divide(NumOps.Subtract(NumOps.One, confidenceLevel), NumOps.FromDouble(2))));
                var marginOfError = NumOps.Multiply(zScore, NumOps.Divide(stdDev, NumOps.Sqrt(NumOps.FromDouble(values.Length))));
                lowerBound = NumOps.Subtract(mean, marginOfError);
                upperBound = NumOps.Add(mean, marginOfError);
                break;
            case DistributionType.Laplace:
                var laplaceValue = CalculateInverseLaplaceCDF(median, mad, confidenceLevel);
                lowerBound = NumOps.Subtract(NumOps.Multiply(NumOps.FromDouble(2), median), laplaceValue);
                upperBound = laplaceValue;
                break;
            case DistributionType.Student:
                var tValue = CalculateInverseStudentTCDF(values.Length - 1, NumOps.Subtract(NumOps.One, NumOps.Divide(NumOps.Subtract(NumOps.One, confidenceLevel), NumOps.FromDouble(2))));
                var tMarginOfError = NumOps.Multiply(tValue, NumOps.Divide(stdDev, NumOps.Sqrt(NumOps.FromDouble(values.Length))));
                lowerBound = NumOps.Subtract(mean, tMarginOfError);
                upperBound = NumOps.Add(mean, tMarginOfError);
                break;
            case DistributionType.LogNormal:
                var logSample = values.Transform(x => NumOps.Log(x));
                (var logMean, var logStdDev) = StatisticsHelper<T>.CalculateMeanAndStandardDeviation(logSample);
                var logZScore = CalculateInverseNormalCDF(NumOps.Subtract(NumOps.One, NumOps.Divide(NumOps.Subtract(NumOps.One, confidenceLevel), NumOps.FromDouble(2))));
                lowerBound = NumOps.Exp(NumOps.Subtract(logMean, NumOps.Multiply(logZScore, NumOps.Divide(logStdDev, NumOps.Sqrt(NumOps.FromDouble(values.Length))))));
                upperBound = NumOps.Exp(NumOps.Add(logMean, NumOps.Multiply(logZScore, NumOps.Divide(logStdDev, NumOps.Sqrt(NumOps.FromDouble(values.Length))))));
                break;
            case DistributionType.Exponential:
                var lambda = NumOps.Divide(NumOps.One, mean);
                var chiSquareLower = CalculateInverseChiSquareCDF(
                    Convert.ToInt32(NumOps.Multiply(NumOps.FromDouble(2), NumOps.FromDouble(values.Length))), 
                    NumOps.Divide(NumOps.Subtract(NumOps.One, confidenceLevel), NumOps.FromDouble(2)));

                var chiSquareUpper = CalculateInverseChiSquareCDF(
                    Convert.ToInt32(NumOps.Multiply(NumOps.FromDouble(2), NumOps.FromDouble(values.Length))), 
                    NumOps.Subtract(NumOps.One, NumOps.Divide(NumOps.Subtract(NumOps.One, confidenceLevel), NumOps.FromDouble(2))));
                lowerBound = NumOps.Multiply(NumOps.Divide(NumOps.Multiply(NumOps.FromDouble(2), NumOps.FromDouble(values.Length)), chiSquareUpper), 
                    NumOps.Divide(NumOps.One, lambda));
                upperBound = NumOps.Multiply(NumOps.Divide(NumOps.Multiply(NumOps.FromDouble(2), NumOps.FromDouble(values.Length)), chiSquareLower), 
                    NumOps.Divide(NumOps.One, lambda));
                break;
            case DistributionType.Weibull:
                // For Weibull, we'll use a bootstrap method to estimate confidence intervals
                (lowerBound, upperBound) = CalculateWeibullConfidenceIntervals(values, confidenceLevel);
                break;
            default:
                throw new ArgumentException("Invalid distribution type");
        }

        return (lowerBound, upperBound);
    }

    public static T CalculatePearsonCorrelation(Vector<T> x, Vector<T> y)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var n = numOps.FromDouble(x.Length);

        var sumX = x.Aggregate(numOps.Zero, numOps.Add);
        var sumY = y.Aggregate(numOps.Zero, numOps.Add);
        var sumXY = x.Zip(y, numOps.Multiply).Aggregate(numOps.Zero, numOps.Add);
        var sumXSquare = x.Select(numOps.Square).Aggregate(numOps.Zero, numOps.Add);
        var sumYSquare = y.Select(numOps.Square).Aggregate(numOps.Zero, numOps.Add);

        var numerator = numOps.Subtract(numOps.Multiply(n, sumXY), numOps.Multiply(sumX, sumY));
        var denominatorX = numOps.Sqrt(numOps.Subtract(numOps.Multiply(n, sumXSquare), numOps.Square(sumX)));
        var denominatorY = numOps.Sqrt(numOps.Subtract(numOps.Multiply(n, sumYSquare), numOps.Square(sumY)));

        return numOps.Divide(numerator, numOps.Multiply(denominatorX, denominatorY));
    }

    public static List<T> CalculateLearningCurve(Vector<T> yActual, Vector<T> yPredicted, int steps)
    {
        var learningCurveList = new List<T>();
        var stepSize = yActual.Length / steps;

        for (int i = 1; i <= steps; i++)
        {
            var subsetSize = i * stepSize;
            var subsetActual = new Vector<T>(yActual.Take(subsetSize));
            var subsetPredicted = new Vector<T>(yPredicted.Take(subsetSize));

            var r2 = CalculateR2(subsetActual, subsetPredicted);
            learningCurveList.Add(r2);
        }

        return learningCurveList;
    }

    public static (T LowerInterval, T UpperInterval) CalculatePredictionIntervals(Vector<T> actual, Vector<T> predicted, T confidenceLevel)
    {
        int n = actual.Length;
        T sumSquaredErrors = NumOps.Zero;
        T meanPredicted = NumOps.Zero;

        for (int i = 0; i < n; i++)
        {
            T error = NumOps.Subtract(actual[i], predicted[i]);
            sumSquaredErrors = NumOps.Add(sumSquaredErrors, NumOps.Multiply(error, error));
            meanPredicted = NumOps.Add(meanPredicted, predicted[i]);
        }

        meanPredicted = NumOps.Divide(meanPredicted, NumOps.FromDouble(n));
        T mse = NumOps.Divide(sumSquaredErrors, NumOps.FromDouble(n - 2));
        T standardError = NumOps.Sqrt(mse);

        T tValue = CalculateTValue(n - 2, confidenceLevel);
        T margin = NumOps.Multiply(tValue, standardError);

        return (NumOps.Subtract(meanPredicted, margin), NumOps.Add(meanPredicted, margin));
    }

    public static T CalculatePredictionIntervalCoverage(Vector<T> actual, Vector<T> predicted, T lowerInterval, T upperInterval)
    {
        int covered = 0;
        int n = actual.Length;

        for (int i = 0; i < n; i++)
        {
            if (NumOps.GreaterThanOrEquals(actual[i], lowerInterval) && NumOps.LessThanOrEquals(actual[i], upperInterval))
            {
                covered++;
            }
        }

        return NumOps.Divide(NumOps.FromDouble(covered), NumOps.FromDouble(n));
    }

    public static T CalculateMeanPredictionError(Vector<T> actual, Vector<T> predicted)
    {
        int n = actual.Length;
        T sumError = NumOps.Zero;

        for (int i = 0; i < n; i++)
        {
            sumError = NumOps.Add(sumError, NumOps.Abs(NumOps.Subtract(actual[i], predicted[i])));
        }

        return NumOps.Divide(sumError, NumOps.FromDouble(n));
    }

    public static T CalculateMedianPredictionError(Vector<T> actual, Vector<T> predicted)
    {
        int n = actual.Length;
        var absoluteErrors = new T[n];

        for (int i = 0; i < n; i++)
        {
            absoluteErrors[i] = NumOps.Abs(NumOps.Subtract(actual[i], predicted[i]));
        }

        Array.Sort(absoluteErrors, (a, b) => NumOps.LessThan(a, b) ? -1 : (NumOps.Equals(a, b) ? 0 : 1));

        if (n % 2 == 0)
        {
            int middleIndex = n / 2;
            return NumOps.Divide(NumOps.Add(absoluteErrors[middleIndex - 1], absoluteErrors[middleIndex]), NumOps.FromDouble(2));
        }
        else
        {
            return absoluteErrors[n / 2];
        }
    }

    public static T CalculateTValue(int degreesOfFreedom, T confidenceLevel)
    {
        // We'll use the inverse of the Student's t-distribution
        // to calculate the t-value for the given confidence level
        T alpha = NumOps.Divide(NumOps.Subtract(NumOps.One, confidenceLevel), NumOps.FromDouble(2));
        return CalculateInverseStudentTCDF(degreesOfFreedom, NumOps.Subtract(NumOps.One, alpha));
    }

    public static (T FirstQuantile, T ThirdQuantile) CalculateQuantiles(Vector<T> data)
    {
        var sortedData = data.OrderBy(x => x).ToArray();
        int n = sortedData.Length;

        T Q1 = CalculateQuantile(sortedData, NumOps.FromDouble(0.25));
        T Q3 = CalculateQuantile(sortedData, NumOps.FromDouble(0.75));

        return (Q1, Q3);
    }

    public static T CalculateQuantile(T[] sortedData, T quantile)
    {
        int n = sortedData.Length;
        T position = NumOps.Multiply(NumOps.FromDouble(n - 1), quantile);
        int index = Convert.ToInt32(NumOps.Round(position));
        T fraction = NumOps.Subtract(position, NumOps.FromDouble(index));

        if (index + 1 < n)
            return NumOps.Add(
                NumOps.Multiply(sortedData[index], NumOps.Subtract(NumOps.One, fraction)),
                NumOps.Multiply(sortedData[index + 1], fraction)
            );
        else
            return sortedData[index];
    }

    public static (T skewness, T kurtosis) CalculateSkewnessAndKurtosis(Vector<T> sample, T mean, T stdDev, int n)
    {
        T skewnessSum = NumOps.Zero, kurtosisSum = NumOps.Zero;
        for (int i = 0; i < n; i++)
        {
            T diff = NumOps.Divide(NumOps.Subtract(sample[i], mean), stdDev);
            T diff3 = NumOps.Multiply(NumOps.Multiply(diff, diff), diff);
            skewnessSum = NumOps.Add(skewnessSum, diff3);
            kurtosisSum = NumOps.Add(kurtosisSum, NumOps.Multiply(diff3, diff));
        }

        T skewness = n > 2 ? NumOps.Divide(skewnessSum, NumOps.Multiply(NumOps.FromDouble(n - 1), NumOps.FromDouble(n - 2))) : NumOps.Zero;
        T kurtosis = n > 3 ? 
            NumOps.Subtract(
                NumOps.Multiply(
                    NumOps.Divide(kurtosisSum, NumOps.Multiply(NumOps.Multiply(NumOps.FromDouble(n - 1), NumOps.FromDouble(n - 2)), NumOps.FromDouble(n - 3))),
                    NumOps.Multiply(NumOps.FromDouble(n), NumOps.FromDouble(n + 1))
                ),
                NumOps.Divide(
                    NumOps.Multiply(NumOps.FromDouble(3), NumOps.Square(NumOps.FromDouble(n - 1))),
                    NumOps.Multiply(NumOps.FromDouble(n - 2), NumOps.FromDouble(n - 3))
                )
            ) : NumOps.Zero;

        return (skewness, kurtosis);
    }

    public static (T Lower, T Upper) CalculateToleranceInterval(Vector<T> actual, Vector<T> predicted, T confidenceLevel)
    {
        int n = actual.Length;
        T mean = predicted.Average();
        T stdDev = CalculateStandardDeviation(predicted);
        T factor = NumOps.FromDouble(Math.Sqrt(1 + (1.0 / n)));
        T tValue = CalculateTValue(n - 1, confidenceLevel);
        T margin = NumOps.Multiply(tValue, NumOps.Multiply(stdDev, factor));

        return (NumOps.Subtract(mean, margin), NumOps.Add(mean, margin));
    }

    public static (T Lower, T Upper) CalculateForecastInterval(Vector<T> actual, Vector<T> predicted, T confidenceLevel)
    {
        int n = actual.Length;
        T mean = predicted.Average();
        T mse = CalculateMeanSquaredError(actual, predicted);
        T factor = NumOps.FromDouble(Math.Sqrt(1 + (1.0 / n)));
        T tValue = CalculateTValue(n - 1, confidenceLevel);
        T margin = NumOps.Multiply(tValue, NumOps.Multiply(NumOps.Sqrt(mse), factor));

        return (NumOps.Subtract(mean, margin), NumOps.Add(mean, margin));
    }

    public static List<(T Quantile, T Lower, T Upper)> CalculateQuantileIntervals(Vector<T> actual, Vector<T> predicted, T[] quantiles)
    {
        var result = new List<(T Quantile, T Lower, T Upper)>();
        var sortedPredictions = new Vector<T>([.. predicted.OrderBy(x => x)]);

        foreach (var q in quantiles)
        {
            T lowerQuantile = CalculateQuantile(sortedPredictions, NumOps.Subtract(q, NumOps.FromDouble(0.025)));
            T upperQuantile = CalculateQuantile(sortedPredictions, NumOps.Add(q, NumOps.FromDouble(0.025)));
            result.Add((q, lowerQuantile, upperQuantile));
        }

        return result;
    }

    public static (T Lower, T Upper) CalculateBootstrapInterval(Vector<T> actual, Vector<T> predicted, T confidenceLevel)
    {
        int n = actual.Length;
        int bootstrapSamples = 1000;
        var bootstrapMeans = new List<T>();

        Random random = new();
        for (int i = 0; i < bootstrapSamples; i++)
        {
            var sample = new Vector<T>(n);
            for (int j = 0; j < n; j++)
            {
                int index = random.Next(n);
                sample[j] = predicted[index];
            }
            bootstrapMeans.Add(sample.Average());
        }

        bootstrapMeans.Sort();
        int lowerIndex = Convert.ToInt32(NumOps.Divide(NumOps.Multiply(confidenceLevel, NumOps.FromDouble(bootstrapSamples)), NumOps.FromDouble(2)));
        int upperIndex = bootstrapSamples - lowerIndex - 1;

        return (bootstrapMeans[lowerIndex], bootstrapMeans[upperIndex]);
    }

    public static (T Lower, T Upper) CalculateSimultaneousPredictionInterval(Vector<T> actual, Vector<T> predicted, T confidenceLevel)
    {
        int n = actual.Length;
        T mean = predicted.Average();
        T mse = CalculateMeanSquaredError(actual, predicted);
        T factor = NumOps.Sqrt(NumOps.Multiply(NumOps.FromDouble(2), confidenceLevel));
        T margin = NumOps.Multiply(factor, NumOps.Sqrt(mse));

        return (NumOps.Subtract(mean, margin), NumOps.Add(mean, margin));
    }

    public static (T Lower, T Upper) CalculateJackknifeInterval(Vector<T> actual, Vector<T> predicted)
    {
        int n = actual.Length;
        var jackknifeSamples = new List<T>();

        for (int i = 0; i < n; i++)
        {
            var sample = new Vector<T>(n - 1);
            int index = 0;
            for (int j = 0; j < n; j++)
            {
                if (j != i)
                {
                    sample[index++] = predicted[j];
                }
            }
            jackknifeSamples.Add(sample.Average());
        }

        T jackknifeEstimate = new Vector<T>([.. jackknifeSamples]).Average();
        T jackknifeStdError = CalculateStandardDeviation(new Vector<T>([.. jackknifeSamples]));
        T tValue = CalculateTValue(n - 1, NumOps.FromDouble(0.95));
        T margin = NumOps.Multiply(tValue, jackknifeStdError);

        return (NumOps.Subtract(jackknifeEstimate, margin), NumOps.Add(jackknifeEstimate, margin));
    }

    public static (T Lower, T Upper) CalculatePercentileInterval(Vector<T> predicted, T confidenceLevel)
    {
        var sortedPredictions = new Vector<T>([.. predicted.OrderBy(x => x)]);
        int n = sortedPredictions.Length;
        T alpha = NumOps.Subtract(NumOps.One, confidenceLevel);
        int lowerIndex = Convert.ToInt32(NumOps.Divide(NumOps.Multiply(alpha, NumOps.FromDouble(n)), NumOps.FromDouble(2.0)));
        int upperIndex = n - lowerIndex - 1;

        return (sortedPredictions[lowerIndex], sortedPredictions[upperIndex]);
    }

    public static T CalculateRSS(Vector<T> actual, Vector<T> predicted)
    {
        return actual.Subtract(predicted).Select(x => NumOps.Square(x)).Aggregate(NumOps.Zero, NumOps.Add);
    }

    public static T CalculateMAE(Vector<T> actual, Vector<T> predicted)
    {
        return NumOps.Divide(actual.Subtract(predicted).Select(NumOps.Abs).Aggregate(NumOps.Zero, NumOps.Add), NumOps.FromDouble(actual.Length));
    }

    public static T CalculateMSE(Vector<T> actual, Vector<T> predicted)
    {
        return NumOps.Divide(actual.Subtract(predicted).Select(x => NumOps.Square(x)).Aggregate(NumOps.Zero, NumOps.Add), NumOps.FromDouble(actual.Length));
    }

    public static T CalculateMAPE(Vector<T> actual, Vector<T> predicted)
    {
        var mape = actual.Zip(predicted, (a, p) => NumOps.Abs(NumOps.Divide(NumOps.Subtract(a, p), a)))
                            .Where(x => !NumOps.Equals(x, NumOps.Zero))
                            .Aggregate(NumOps.Zero, NumOps.Add);
        return NumOps.Multiply(NumOps.Divide(mape, NumOps.FromDouble(actual.Length)), NumOps.FromDouble(100));
    }

    public static T CalculateMedianAbsoluteError(Vector<T> actual, Vector<T> predicted)
    {
        var absoluteErrors = actual.Subtract(predicted).Select(NumOps.Abs).OrderBy(x => x).ToArray();
        int n = absoluteErrors.Length;
        return n % 2 == 0
            ? NumOps.Divide(NumOps.Add(absoluteErrors[n / 2 - 1], absoluteErrors[n / 2]), NumOps.FromDouble(2))
            : absoluteErrors[n / 2];
    }

    public static T CalculateMaxError(Vector<T> actual, Vector<T> predicted)
    {
        return actual.Subtract(predicted).Select(NumOps.Abs).Max();
    }

    public static T CalculateSampleStandardError(Vector<T> actual, Vector<T> predicted, int numberOfParameters)
    {
        T mse = CalculateMSE(actual, predicted);
        int degreesOfFreedom = actual.Length - numberOfParameters;
        return NumOps.Sqrt(NumOps.Divide(mse, NumOps.FromDouble(degreesOfFreedom)));
    }

    public static T CalculatePopulationStandardError(Vector<T> actual, Vector<T> predicted)
    {
        return NumOps.Sqrt(CalculateMSE(actual, predicted));
    }

    public static T CalculateMeanBiasError(Vector<T> actual, Vector<T> predicted)
    {
        return NumOps.Divide(actual.Subtract(predicted).Aggregate(NumOps.Zero, NumOps.Add), NumOps.FromDouble(actual.Length));
    }

    public static T CalculateTheilUStatistic(Vector<T> actual, Vector<T> predicted)
    {
        T numerator = NumOps.Sqrt(CalculateMSE(actual, predicted));
        T denominatorActual = NumOps.Sqrt(NumOps.Divide(actual.Select(x => NumOps.Square(x)).Aggregate(NumOps.Zero, NumOps.Add), NumOps.FromDouble(actual.Length)));
        T denominatorPredicted = NumOps.Sqrt(NumOps.Divide(predicted.Select(x => NumOps.Square(x)).Aggregate(NumOps.Zero, NumOps.Add), NumOps.FromDouble(predicted.Length)));

        return NumOps.Divide(numerator, NumOps.Add(denominatorActual, denominatorPredicted));
    }

    public static T CalculateDurbinWatsonStatistic(Vector<T> actual, Vector<T> predicted)
    {
        var errors = actual.Subtract(predicted);
        return CalculateDurbinWatsonStatistic([.. errors]);
    }

    public static T CalculateDurbinWatsonStatistic(List<T> residualList)
    {
        T sumSquaredDifferences = NumOps.Zero;
        T sumSquaredErrors = NumOps.Zero;

        for (int i = 1; i < residualList.Count; i++)
        {
            sumSquaredDifferences = NumOps.Add(sumSquaredDifferences, NumOps.Square(NumOps.Subtract(residualList[i], residualList[i - 1])));
            sumSquaredErrors = NumOps.Add(sumSquaredErrors, NumOps.Square(residualList[i]));
        }
        sumSquaredErrors = NumOps.Add(sumSquaredErrors, NumOps.Square(residualList[0]));

        return NumOps.Divide(sumSquaredDifferences, sumSquaredErrors);
    }

    public static T CalculateAICAlternative(int sampleSize, int parameterSize, T rss)
    {
        if (sampleSize <= 0 || NumOps.LessThanOrEquals(rss, NumOps.Zero)) return NumOps.Zero;
        T logData = NumOps.Divide(rss, NumOps.FromDouble(sampleSize));

        return NumOps.Add(NumOps.Multiply(NumOps.FromDouble(sampleSize), NumOps.Log(logData)), NumOps.Multiply(NumOps.FromDouble(2), NumOps.FromDouble(parameterSize)));
    }

    public static T CalculateAIC(int sampleSize, int parameterSize, T rss)
    {
        if (sampleSize <= 0 || NumOps.LessThanOrEquals(rss, NumOps.Zero)) return NumOps.Zero;
        T logData = NumOps.Multiply(NumOps.FromDouble(2 * Math.PI), NumOps.Divide(rss, NumOps.FromDouble(sampleSize)));

        return NumOps.Add(NumOps.Multiply(NumOps.FromDouble(2), NumOps.FromDouble(parameterSize)),
                            NumOps.Multiply(NumOps.FromDouble(sampleSize), NumOps.Add(NumOps.Log(logData), NumOps.One)));
    }

    public static T CalculateBIC(int sampleSize, int parameterSize, T rss)
    {
        if (sampleSize <= 0 || NumOps.LessThanOrEquals(rss, NumOps.Zero)) return NumOps.Zero;
        T logData = NumOps.Divide(rss, NumOps.FromDouble(sampleSize));

        return NumOps.Add(NumOps.Multiply(NumOps.FromDouble(sampleSize), NumOps.Log(logData)),
                            NumOps.Multiply(NumOps.FromDouble(parameterSize), NumOps.Log(NumOps.FromDouble(sampleSize))));
    }

    public static T CalculateAccuracy(Vector<T> actual, Vector<T> predicted)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var correctPredictions = numOps.Zero;
        var totalPredictions = numOps.FromDouble(actual.Length);

        for (int i = 0; i < actual.Length; i++)
        {
            if (numOps.Equals(actual[i], predicted[i]))
            {
                correctPredictions = numOps.Add(correctPredictions, numOps.One);
            }
        }

        return numOps.Divide(correctPredictions, totalPredictions);
    }

    public static T CalculateAccuracy(Vector<T> actual, Vector<T> predicted, PredictionType predictionType, T? tolerance = default)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var correctPredictions = numOps.Zero;
        var totalPredictions = numOps.FromDouble(actual.Length);
        tolerance ??= numOps.FromDouble(0.05); // default of 5%

        for (int i = 0; i < actual.Length; i++)
        {
            if (predictionType == PredictionType.Binary)
            {
                if (numOps.Equals(actual[i], predicted[i]))
                {
                    correctPredictions = numOps.Add(correctPredictions, numOps.One);
                }
            }
            else // Regression
            {
                var difference = numOps.Abs(numOps.Subtract(actual[i], predicted[i]));
                var threshold = numOps.Multiply(actual[i], tolerance);
                if (numOps.LessThanOrEquals(difference, threshold))
                {
                    correctPredictions = numOps.Add(correctPredictions, numOps.One);
                }
            }
        }

        return numOps.Divide(correctPredictions, totalPredictions);
    }

    public static (T Precision, T Recall, T F1Score) CalculatePrecisionRecallF1(Vector<T> actual, Vector<T> predicted, PredictionType predictionType, T? threshold = default)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var truePositives = numOps.Zero;
        var falsePositives = numOps.Zero;
        var falseNegatives = numOps.Zero;
        threshold ??= numOps.FromDouble(0.1); // default of 10%

        for (int i = 0; i < actual.Length; i++)
        {
            if (predictionType == PredictionType.Binary)
            {
                if (numOps.Equals(predicted[i], numOps.One))
                {
                    if (numOps.Equals(actual[i], numOps.One))
                    {
                        truePositives = numOps.Add(truePositives, numOps.One);
                    }
                    else
                    {
                        falsePositives = numOps.Add(falsePositives, numOps.One);
                    }
                }
                else if (numOps.Equals(actual[i], numOps.One))
                {
                    falseNegatives = numOps.Add(falseNegatives, numOps.One);
                }
            }
            else // Regression
            {
                var difference = numOps.Abs(numOps.Subtract(actual[i], predicted[i]));
                if (numOps.LessThanOrEquals(difference, threshold))
                {
                    truePositives = numOps.Add(truePositives, numOps.One);
                }
                else if (numOps.GreaterThan(predicted[i], actual[i]))
                {
                    falsePositives = numOps.Add(falsePositives, numOps.One);
                }
                else
                {
                    falseNegatives = numOps.Add(falseNegatives, numOps.One);
                }
            }
        }

        var precision = numOps.Divide(truePositives, numOps.Add(truePositives, falsePositives));
        var recall = numOps.Divide(truePositives, numOps.Add(truePositives, falseNegatives));
        var f1Score = CalculateF1Score(precision, recall);

        return (precision, recall, f1Score);
    }

    public static T CalculateF1Score(T precision, T recall)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var numerator = numOps.Multiply(numOps.FromDouble(2), numOps.Multiply(precision, recall));
        var denominator = numOps.Add(precision, recall);
        return numOps.Equals(denominator, numOps.Zero) ? numOps.Zero : numOps.Divide(numerator, denominator);
    }

    public static Matrix<T> CalculateCorrelationMatrix(Matrix<T> features, ModelStatsOptions options)
    {
        int featureCount = features.Columns;
        var correlationMatrix = new Matrix<T>(featureCount, featureCount, NumOps);

        for (int i = 0; i < featureCount; i++)
        {
            for (int j = 0; j < featureCount; j++)
            {
                if (i == j)
                {
                    correlationMatrix[i, j] = NumOps.One;
                }
                else
                {
                    Vector<T> vectorI = features.GetColumn(i);
                    Vector<T> vectorJ = features.GetColumn(j);

                    T correlation = CalculatePearsonCorrelation(vectorI, vectorJ);
                    correlationMatrix[i, j] = correlation;

                    // Check for multicollinearity
                    if (NumOps.GreaterThan(NumOps.Abs(correlation), NumOps.FromDouble(options.MulticollinearityThreshold)))
                    {
                        // You might want to log this or handle it in some way
                        Console.WriteLine($"High correlation detected between features {i} and {j}: {correlation}");
                    }
                }
            }
        }

        return correlationMatrix;
    }

    public static List<T> CalculateVIF(Matrix<T> correlationMatrix, ModelStatsOptions options)
    {
        var vifValues = new List<T>();

        for (int i = 0; i < correlationMatrix.Rows; i++)
        {
            var subMatrix = correlationMatrix.RemoveRow(i).RemoveColumn(i);
            var inverseSubMatrix = subMatrix.Inverse();
            var rSquared = NumOps.Subtract(NumOps.One, NumOps.Divide(NumOps.One, inverseSubMatrix[0, 0]));
            var vif = NumOps.Divide(NumOps.One, NumOps.Subtract(NumOps.One, rSquared));
            vifValues.Add(vif);

            // Check if VIF exceeds the maximum allowed value
            if (NumOps.GreaterThan(vif, NumOps.FromDouble(options.MaxVIF)))
            {
                // You might want to log this or handle it in some way
                Console.WriteLine($"High VIF detected for feature {i}: {vif}");
            }
        }

        return vifValues;
    }

    public static T CalculateConditionNumber(Matrix<T> matrix, ModelStatsOptions options)
    {
        return options.ConditionNumberMethod switch
        {
            ConditionNumberMethod.SVD => CalculateConditionNumberSVD(matrix),
            ConditionNumberMethod.L1Norm => CalculateConditionNumberL1Norm(matrix),
            ConditionNumberMethod.InfinityNorm => CalculateConditionNumberLInfNorm(matrix),
            ConditionNumberMethod.PowerIteration => CalculateConditionNumberPowerIteration(matrix),
            _ => throw new ArgumentException("Unsupported condition number calculation method", nameof(options))
        };
    }

    private static T CalculateConditionNumberSVD(Matrix<T> matrix)
    {
        var svd = new SvdDecomposition<T>(matrix);
        var singularValues = svd.S;

        if (singularValues.Length == 0)
        {
            return NumOps.Zero;
        }

        T maxSingularValue = singularValues.Max();
        T minSingularValue = singularValues.Min();

        if (NumOps.Equals(minSingularValue, NumOps.Zero))
        {
            return NumOps.MaxValue;
        }

        return NumOps.Divide(maxSingularValue, minSingularValue);
    }

    private static T CalculateConditionNumberL1Norm(Matrix<T> matrix)
    {
        T normA = MatrixL1Norm(matrix);
        Matrix<T> inverseMatrix = matrix.Inverse();
        T normAInverse = MatrixL1Norm(inverseMatrix);

        return NumOps.Multiply(normA, normAInverse);
    }

    private static T CalculateConditionNumberLInfNorm(Matrix<T> matrix)
    {
        T normA = MatrixInfinityNorm(matrix);
        Matrix<T> inverseMatrix = matrix.Inverse();
        T normAInverse = MatrixInfinityNorm(inverseMatrix);

        return NumOps.Multiply(normA, normAInverse);
    }

    private static T CalculateConditionNumberPowerIteration(Matrix<T> matrix, int maxIterations = 100, T? tolerance = default)
    {
        tolerance ??= NumOps.FromDouble(1e-10);

        T largestEigenvalue = PowerIteration(matrix, maxIterations, tolerance);
        T smallestEigenvalue = PowerIteration(matrix.Inverse(), maxIterations, tolerance);

        return NumOps.Divide(largestEigenvalue, smallestEigenvalue);
    }

    private static T MatrixL1Norm(Matrix<T> matrix)
    {
        T maxColumnSum = NumOps.Zero;
        for (int j = 0; j < matrix.Columns; j++)
        {
            T columnSum = NumOps.Zero;
            for (int i = 0; i < matrix.Rows; i++)
            {
                columnSum = NumOps.Add(columnSum, NumOps.Abs(matrix[i, j]));
            }
            maxColumnSum = NumOps.GreaterThan(maxColumnSum, columnSum) ? maxColumnSum : columnSum;
        }

        return maxColumnSum;
    }

    private static T MatrixInfinityNorm(Matrix<T> matrix)
    {
        T maxRowSum = NumOps.Zero;
        for (int i = 0; i < matrix.Rows; i++)
        {
            T rowSum = NumOps.Zero;
            for (int j = 0; j < matrix.Columns; j++)
            {
                rowSum = NumOps.Add(rowSum, NumOps.Abs(matrix[i, j]));
            }
            maxRowSum = NumOps.GreaterThan(maxRowSum, rowSum) ? maxRowSum : rowSum;
        }

        return maxRowSum;
    }

    private static T PowerIteration(Matrix<T> matrix, int maxIterations, T tolerance)
    {
        Vector<T> v = Vector<T>.CreateRandom(matrix.Rows);
        T eigenvalue = NumOps.Zero;

        for (int i = 0; i < maxIterations; i++)
        {
            Vector<T> Av = matrix * v;
            T newEigenvalue = v.DotProduct(Av);

            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(newEigenvalue, eigenvalue)), tolerance))
            {
                return newEigenvalue;
            }

            eigenvalue = newEigenvalue;
            v = Av.Normalize();
        }

        return eigenvalue;
    }

    public static T CalculateDIC(ModelStats<T> modelStats)
    {
        // DIC = D(θ̄) + 2pD
        // where D(θ̄) is the deviance at the posterior mean, and pD is the effective number of parameters
        var _numOps = MathHelper.GetNumericOperations<T>();
        var devianceAtPosteriorMean = _numOps.Multiply(_numOps.FromDouble(-2), modelStats.LogLikelihood);
        var effectiveNumberOfParameters = modelStats.EffectiveNumberOfParameters;

        return _numOps.Add(devianceAtPosteriorMean, _numOps.Multiply(_numOps.FromDouble(2), effectiveNumberOfParameters));
    }

    public static T CalculateWAIC(ModelStats<T> modelStats)
    {
        // WAIC = -2 * (lppd - pWAIC)
        // where lppd is the log pointwise predictive density, and pWAIC is the effective number of parameters
        var _numOps = MathHelper.GetNumericOperations<T>();
        var lppd = modelStats.LogPointwisePredictiveDensity;
        var pWAIC = modelStats.EffectiveNumberOfParameters;

        return _numOps.Multiply(_numOps.FromDouble(-2), _numOps.Subtract(lppd, pWAIC));
    }

    public static T CalculateLOO(ModelStats<T> modelStats)
    {
        // LOO = -2 * (Σ log(p(yi | y-i)))
        // where p(yi | y-i) is the leave-one-out predictive density for the i-th observation
        var _numOps = MathHelper.GetNumericOperations<T>();
        var looSum = modelStats.LeaveOneOutPredictiveDensities.Aggregate(_numOps.Zero,
            (acc, density) => _numOps.Add(acc, _numOps.Log(density))
        );

        return _numOps.Multiply(_numOps.FromDouble(-2), looSum);
    }

    public static T CalculatePosteriorPredictiveCheck(ModelStats<T> modelStats)
    {
        // Calculate the proportion of posterior predictive samples that are more extreme than the observed data
        var _numOps = MathHelper.GetNumericOperations<T>();
        var observedStatistic = modelStats.ObservedTestStatistic;
        var posteriorPredictiveSamples = modelStats.PosteriorPredictiveSamples;
        var moreExtremeSamples = posteriorPredictiveSamples.Count(sample => _numOps.GreaterThan(sample, observedStatistic));

        return _numOps.Divide(_numOps.FromDouble(moreExtremeSamples), _numOps.FromDouble(posteriorPredictiveSamples.Count));
    }

    public static T CalculateBayesFactor(ModelStats<T> modelStats)
    {
        // Bayes Factor = P(D|M1) / P(D|M2)
        // where P(D|M1) is the marginal likelihood of the current model and P(D|M2) is the marginal likelihood of a reference model
        var _numOps = MathHelper.GetNumericOperations<T>();
        var currentModelMarginalLikelihood = modelStats.MarginalLikelihood;
        var referenceModelMarginalLikelihood = modelStats.ReferenceModelMarginalLikelihood;

        return _numOps.Divide(currentModelMarginalLikelihood, referenceModelMarginalLikelihood);
    }

    public static T CalculateLikelihood(T actual, T predicted)
    {
        var _numOps = MathHelper.GetNumericOperations<T>();
        T residual = _numOps.Subtract(actual, predicted);

        return _numOps.Exp(_numOps.Multiply(_numOps.FromDouble(-0.5), _numOps.Multiply(residual, residual)));
    }

    public static IEnumerable<T> GeneratePosteriorPredictiveSamples(Matrix<T> features, Vector<T> coefficients, int numSamples)
    {
        var _numOps = MathHelper.GetNumericOperations<T>();
        var random = new Random();
        var samples = new List<T>();

        for (int i = 0; i < numSamples; i++)
        {
            var predictedValues = features.Multiply(coefficients);
            var noise = Vector<T>.CreateRandom(predictedValues.Length);
            samples.Add(CalculateObservedTestStatistic(predictedValues, predictedValues.Add(noise)));
        }

        return samples;
    }

    public static List<T> CalculateLeaveOneOutPredictiveDensities(Matrix<T> features, Vector<T> actualValues, Func<Matrix<T>, Vector<T>, Vector<T>> modelFitFunction)
    {
        var _numOps = MathHelper.GetNumericOperations<T>();
        var looPredictiveDensities = new List<T>();

        for (int i = 0; i < features.Rows; i++)
        {
            var trainingFeatures = features.RemoveRow(i);
            var trainingValues = actualValues.RemoveAt(i);
            var coefficients = modelFitFunction(trainingFeatures, trainingValues);
            var predictedValue = features.GetRow(i).DotProduct(coefficients);
            looPredictiveDensities.Add(CalculateLikelihood(actualValues[i], predictedValue));
        }

        return looPredictiveDensities;
    }

    public static T CalculateLogPointwisePredictiveDensity(Vector<T> actualValues, Vector<T> predictedValues)
    {
        var _numOps = MathHelper.GetNumericOperations<T>();
        T lppd = _numOps.Zero;

        for (int i = 0; i < actualValues.Length; i++)
        {
            T likelihood = CalculateLikelihood(actualValues[i], predictedValues[i]);
            lppd = _numOps.Add(lppd, _numOps.Log(likelihood));
        }

        return lppd;
    }

    public static T CalculateLogLikelihood(Vector<T> actualValues, Vector<T> predictedValues)
    {
        var _numOps = MathHelper.GetNumericOperations<T>();
        T logLikelihood = _numOps.Zero;

        for (int i = 0; i < actualValues.Length; i++)
        {
            T residual = _numOps.Subtract(actualValues[i], predictedValues[i]);
            logLikelihood = _numOps.Add(logLikelihood, _numOps.Log(_numOps.Abs(residual)));
        }

        return _numOps.Multiply(_numOps.FromDouble(-0.5), logLikelihood);
    }

    public static T CalculateEffectiveNumberOfParameters(Matrix<T> features, Vector<T> coefficients)
    {
        // Calculate the hat matrix (H = X(X'X)^(-1)X')
        var transposeFeatures = features.Transpose();
        var inverseMatrix = (transposeFeatures * features).Inverse();
        var hatMatrix = features * (inverseMatrix * transposeFeatures);
    
        // The effective number of parameters is the trace of the hat matrix
        return hatMatrix.Diagonal().Sum();
    }

    public static T CalculateObservedTestStatistic(Vector<T> actualValues, Vector<T> predictedValues, TestStatisticType testType = TestStatisticType.ChiSquare)
    {
        var _numOps = MathHelper.GetNumericOperations<T>();
        
        switch (testType)
        {
            case TestStatisticType.ChiSquare:
                // Chi-square test statistic
                T chiSquare = _numOps.Zero;
                for (int i = 0; i < actualValues.Length; i++)
                {
                    T residual = _numOps.Subtract(actualValues[i], predictedValues[i]);
                    chiSquare = _numOps.Add(chiSquare, _numOps.Divide(_numOps.Multiply(residual, residual), predictedValues[i]));
                }
                return chiSquare;

            case TestStatisticType.FTest:
                // F-test statistic
                T sst = CalculateTotalSumOfSquares(actualValues);
                T sse = CalculateResidualSumOfSquares(actualValues, predictedValues);
                int dfModel = predictedValues.Length - 1;
                int dfResidual = actualValues.Length - predictedValues.Length;
                
                T msModel = _numOps.Divide(_numOps.Subtract(sst, sse), _numOps.FromDouble(dfModel));
                T msResidual = _numOps.Divide(sse, _numOps.FromDouble(dfResidual));
                
                return _numOps.Divide(msModel, msResidual);

            default:
                throw new ArgumentException("Unsupported test statistic type");
        }
    }

    public static T CalculateMarginalLikelihood(Vector<T> actualValues, Vector<T> predictedValues, int numParameters)
    {
        var _numOps = MathHelper.GetNumericOperations<T>();
        
        // Calculate log-likelihood
        T logLikelihood = CalculateLogLikelihood(actualValues, predictedValues);
        
        // Calculate BIC (Bayesian Information Criterion)
        T n = _numOps.FromDouble(actualValues.Length);
        T k = _numOps.FromDouble(numParameters);
        T bic = _numOps.Multiply(_numOps.FromDouble(-2), logLikelihood);
        bic = _numOps.Add(bic, _numOps.Multiply(k, _numOps.Log(n)));
        
        // Approximate marginal likelihood using BIC
        return _numOps.Exp(_numOps.Multiply(_numOps.FromDouble(-0.5), bic));
    }

    public static T CalculateTotalSumOfSquares(Vector<T> values)
    {
        T mean = values.Mean();
        var meanVector = Vector<T>.CreateDefault(values.Length, mean);
        var differences = values.Subtract(meanVector);

        return differences.DotProduct(differences);
    }

    public static T CalculateResidualSumOfSquares(Vector<T> actual, Vector<T> predicted)
    {
        var residuals = actual.Subtract(predicted);
        return residuals.DotProduct(residuals);
    }

    public static List<T> CalculatePosteriorPredictiveSamples(Vector<T> actual, Vector<T> predicted, int featureCount, int numSamples = 1000)
    {
        var _numOps = MathHelper.GetNumericOperations<T>();
        var n = actual.Length;
        var rss = CalculateResidualSumOfSquares(actual, predicted);
        var sigma2 = _numOps.Divide(rss, _numOps.FromDouble(n - featureCount));
        var standardError = _numOps.Sqrt(sigma2);

        var random = new Random();
        var samples = new List<T>(numSamples);

        for (int i = 0; i < numSamples; i++)
        {
            var sample = _numOps.Zero;
            for (int j = 0; j < n; j++)
            {
                var noise = _numOps.Multiply(standardError, _numOps.FromDouble(random.NextGaussian()));
                sample = _numOps.Add(sample, _numOps.Add(predicted[j], noise));
            }
            samples.Add(_numOps.Divide(sample, _numOps.FromDouble(n)));
        }

        return samples;
    }

    public static T CalculateReferenceModelMarginalLikelihood(Vector<T> actual)
    {
        var _numOps = MathHelper.GetNumericOperations<T>();
        var n = actual.Length;
        var mean = actual.Mean();
        var variance = _numOps.Divide(CalculateTotalSumOfSquares(actual), _numOps.FromDouble(n - 1));

        // Calculate log marginal likelihood for the reference model (intercept-only model)
        var logML = _numOps.Multiply(_numOps.FromDouble(-0.5), _numOps.FromDouble(n * Math.Log(2 * Math.PI)));
        logML = _numOps.Subtract(logML, _numOps.Multiply(_numOps.FromDouble(0.5 * n), _numOps.Log(variance)));
        logML = _numOps.Subtract(logML, _numOps.Divide(_numOps.FromDouble(n - 1), _numOps.FromDouble(2)));

        return _numOps.Exp(logML);
    }
}