global using AiDotNet.Extensions;

namespace AiDotNet.Models;

public class BasicStats
{
    public double Mean { get; private set; }
    public double Variance { get; private set; }
    public double StandardDeviation { get; private set; }
    public double Skewness { get; private set; }
    public double Kurtosis { get; private set; }
    public double Min { get; private set; }
    public double Max { get; private set; }
    public int N { get; private set; }
    public double Median { get; private set; }
    public double UpperConfidenceLevel { get; private set; }
    public double LowerConfidenceLevel { get; private set; }
    public double UpperCredibleLevel { get; private set; }
    public double LowerCredibleLevel { get; private set; }
    public DistributionFitResult BestDistributionFit { get; private set; } = new();
    public double FirstQuartile { get; private set; }
    public double ThirdQuartile { get; private set; }
    public double InterquartileRange { get; private set; }
    public double R2 { get; private set; }
    public double AdjustedR2 { get; private set; }
    public double ExplainedVarianceScore { get; private set; }
    public double MAD { get; private set; }

    public BasicStats(Vector<double> actual, Vector<double> predicted, int numberOfParameters, double confidenceLevel = 0.95)
    {
        CalculateStats(actual, predicted, confidenceLevel, numberOfParameters);
    }

    public static BasicStats Empty()
    {
        return new BasicStats(Vector<double>.Empty(), Vector<double>.Empty(), 0, 0);
    }

    private void CalculateStats(Vector<double> actual, Vector<double> predicted, double confidenceLevel, int numberOfParameters)
    {
        N = actual.Length;
        Mean = actual.Mean();
        Variance = actual.Variance();
        StandardDeviation = Math.Sqrt(Variance);
        (Skewness, Kurtosis) = CalculateSkewnessAndKurtosis(actual, Mean, StandardDeviation, N);
        Min = actual.Min();
        Max = actual.Max();
        Median = CalculateMedian(actual);
        (FirstQuartile, ThirdQuartile) = CalculateQuartiles(actual);
        InterquartileRange = ThirdQuartile - FirstQuartile;
        R2 = CalculateR2(actual, predicted);
        AdjustedR2 = CalculateAdjustedR2(R2, N, numberOfParameters);
        MAD = CalculateMeanAbsoluteDeviation(predicted, Median);

        BestDistributionFit = DetermineBestFitDistribution(predicted);
        var bestDistributionType = BestDistributionFit.BestFitDistribution;

        (UpperConfidenceLevel, LowerConfidenceLevel) = CalculateConfidenceIntervals(predicted, confidenceLevel, bestDistributionType);
        (UpperCredibleLevel, LowerCredibleLevel) = CalculateCredibleIntervals(predicted, confidenceLevel, bestDistributionType);

        ExplainedVarianceScore = CalculateExplainedVarianceScore(actual, predicted);
    }

    private (double FirstQuartile, double ThirdQuartile) CalculateQuartiles(Vector<double> data)
    {
        var sortedData = data.OrderBy(x => x).ToArray();
        int n = sortedData.Length;

        double Q1 = CalculateQuartile(sortedData, 0.25);
        double Q3 = CalculateQuartile(sortedData, 0.75);

        return (Q1, Q3);
    }

    private double CalculateQuartile(double[] sortedData, double quartile)
    {
        int n = sortedData.Length;
        double position = (n - 1) * quartile;
        int index = (int)Math.Floor(position);
        double fraction = position - index;

        if (index + 1 < n)
            return sortedData[index] * (1 - fraction) + sortedData[index + 1] * fraction;
        else
            return sortedData[index];
    }

    public (double, double) CalculateStandardDeviation(Vector<double> sample)
    {
        var mean = sample.Average();
        var sum = 0.0;
        for (int i = 0; i < sample.Length; i++)
        {
            sum += Math.Pow(sample[i] - mean, 2);
        }

        return (mean, Math.Sqrt(sum / (sample.Length - 1)));
    }

    private static double CalculateGoodnessOfFit(Vector<double> sample, Func<double, double> pdfFunction)
    {
        // Using negative log-likelihood as a measure of goodness of fit
        double logLikelihood = 0;
        foreach (var value in sample)
        {
            logLikelihood += Math.Log(pdfFunction(value));
        }

        return -logLikelihood;
    }

    public double CalculateNormalCDF(double mean, double stdDev, double x)
    {
        if (stdDev <= 0) return 0;
        return 0.5 * (1 + Erf((x - mean) / (stdDev * Math.Sqrt(2))));
    }

    private double Erf(double x)
    {
        // Constants
        double a1 =  0.254829592;
        double a2 = -0.284496736;
        double a3 =  1.421413741;
        double a4 = -1.453152027;
        double a5 =  1.061405429;
        double p  =  0.3275911;

        // Save the sign of x
        int sign = (x < 0) ? -1 : 1;
        x = Math.Abs(x);

        // A&S formula 7.1.26
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);

        return sign * y;
    }

    private double Erfc(double x)
    {
        return 1 - Erf(x);
    }

    public double CalculateNormalPDF(double mean, double stdDev, double x)
    {
        if (stdDev <= 0) return 0;
        var num = (x - mean) / stdDev;

        return Math.Exp(-0.5 * num * num) / (Math.Sqrt(2 * Math.PI) * stdDev);
    }

    private DistributionFitResult FitNormalDistribution(Vector<double> sample)
    {
        (var mean, var stdDev) = CalculateStandardDeviation(sample);
        double goodnessOfFit = CalculateGoodnessOfFit(sample, x => CalculateNormalPDF(mean, stdDev, x));

        return new DistributionFitResult
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, double>
            {
                { "Mean", mean },
                { "StandardDeviation", stdDev }
            }
        };
    }

    public double CalculateStudentPDF(double x, double mean, double stdDev, int df)
    {
        double t = (x - mean) / stdDev;

        return Gamma((df + 1) / 2.0) / (Math.Sqrt(df * Math.PI) * Gamma(df / 2.0) * Math.Pow(1 + t * t / df, (df + 1) / 2.0));
    }

    public double Gamma(double x)
    {
        return Math.Exp(LogGamma(x));
    }

    private static double LogGamma(double x)
    {
        // Lanczos approximation for log of Gamma function
        double[] c = {76.18009172947146, -86.50532032941677, 24.01409824083091,
                      -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5};
        double sum = 0.99999999999980993;
        for (int i = 0; i < 6; i++)
            sum += c[i] / (x + i + 1);

        return (x + 0.5) * Math.Log(x + 5.5) - (x + 5.5) + Math.Log(2.5066282746310005 * sum / x);
    }

    public double CalculateLaplacePDF(double median, double mad, double x) => 1 / (2 * mad) * Math.Exp(-Math.Abs(x - median) / mad);

    public double CalculateMeanAbsoluteDeviation(Vector<double> sample, double median)
    {
        return sample.Select(x => Math.Abs(x - median)).Average();
    }

    private DistributionFitResult FitLaplaceDistribution(Vector<double> sample)
    {
        double median = CalculateMedian(sample);
        double mad = CalculateMeanAbsoluteDeviation(sample, median);
        double goodnessOfFit = CalculateGoodnessOfFit(sample, x => CalculateLaplacePDF(median, mad, x));

        return new DistributionFitResult
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, double>
            {
                { "Median", median },
                { "MAD", mad }
            }
        };
    }

    private DistributionFitResult FitStudentDistribution(Vector<double> sample)
    {
        int df = sample.Length - 1;
        (var mean, var stdDev) = CalculateStandardDeviation(sample);
        double goodnessOfFit = CalculateGoodnessOfFit(sample, x => CalculateStudentPDF(x, mean, stdDev, df));

        return new DistributionFitResult
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, double>
            {
                { "DegreesOfFreedom", df },
                { "Mean", mean },
                { "StandardDeviation", stdDev }
            }
        };
    }

    public DistributionFitResult DetermineBestFitDistribution(Vector<double> sample)
    {
        // Test Normal Distribution
        var result = FitNormalDistribution(sample);
        double bestGoodnessOfFit = result.GoodnessOfFit;
        result.BestFitDistribution = DistributionType.Normal;

        // Test Laplace Distribution
        var laplaceFit = FitLaplaceDistribution(sample);
        if (laplaceFit.GoodnessOfFit < bestGoodnessOfFit)
        {
            bestGoodnessOfFit = laplaceFit.GoodnessOfFit;
            result = laplaceFit;
            result.BestFitDistribution = DistributionType.Laplace;
        }

        // Test Student's t-Distribution
        var studentFit = FitStudentDistribution(sample);
        if (studentFit.GoodnessOfFit < bestGoodnessOfFit)
        {
            bestGoodnessOfFit = studentFit.GoodnessOfFit;
            result = studentFit;
            result.BestFitDistribution = DistributionType.Student;
        }

        // Test Log-Normal Distribution
        var logNormalFit = FitLogNormalDistribution(sample);
        if (logNormalFit.GoodnessOfFit < bestGoodnessOfFit)
        {
            bestGoodnessOfFit = logNormalFit.GoodnessOfFit;
            result = logNormalFit;
            result.BestFitDistribution = DistributionType.LogNormal;
        }

        // Test Exponential Distribution
        var exponentialFit = FitExponentialDistribution(sample);
        if (exponentialFit.GoodnessOfFit < bestGoodnessOfFit)
        {
            bestGoodnessOfFit = exponentialFit.GoodnessOfFit;
            result = exponentialFit;
            result.BestFitDistribution = DistributionType.Exponential;
        }

        // Test Weibull Distribution
        var weibullFit = FitWeibullDistribution(sample);
        if (weibullFit.GoodnessOfFit < bestGoodnessOfFit)
        {
            result = weibullFit;
            result.BestFitDistribution = DistributionType.Weibull;
        }

        return result;
    }

    public double CalculateExponentialPDF(double lambda, double x)
    {
        if (lambda <= 0 || x < 0) return 0;
        return lambda * Math.Exp(-lambda * x);
    }

    private DistributionFitResult FitExponentialDistribution(Vector<double> sample)
    {
        double lambda = 1 / sample.ToArray().Average();

        double goodnessOfFit = CalculateGoodnessOfFit(sample, x => CalculateExponentialPDF(lambda, x));

        return new DistributionFitResult
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, double>
            {
                { "Lambda", lambda }
            }
        };
    }

    public double CalculateWeibullPDF(double k, double lambda, double x)
    {
        if (k <= 0 || lambda <= 0 || x < 0) return 0;
        return k / lambda * Math.Pow(x / lambda, k - 1) * Math.Exp(-Math.Pow(x / lambda, k));
    }

    private DistributionFitResult FitWeibullDistribution(Vector<double> sample)
    {
        // Initial guess for k and lambda
        double k = 1.0;
        double lambda = sample.ToArray().Average();

        // Implement a simple optimization algorithm here to find best k and lambda
        // This is a placeholder and should be replaced with a proper optimization method
        for (int i = 0; i < 100; i++)
        {
            k += 0.1;
            lambda = sample.ToArray().Select(x => Math.Pow(x, k)).Average();
            lambda = Math.Pow(lambda, 1 / k);
        }

        double goodnessOfFit = CalculateGoodnessOfFit(sample, x => CalculateWeibullPDF(k, lambda, x));

        return new DistributionFitResult
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, double>
            {
                { "K", k },
                { "Lambda", lambda }
            }
        };
    }

    public double CalculateLogNormalPDF(double mu, double sigma, double x)
    {
        if (x <= 0 || sigma <= 0) return 0;
        var logX = Math.Log(x);
        return Math.Exp(-Math.Pow(logX - mu, 2) / (2 * sigma * sigma)) / (x * sigma * Math.Sqrt(2 * Math.PI));
    }

    private DistributionFitResult FitLogNormalDistribution(Vector<double> sample)
    {
        var logSample = sample.ToArray().Select(x => Math.Log(x)).ToArray();
        double mu = logSample.Average();
        double sigma = Math.Sqrt(logSample.Select(x => Math.Pow(x - mu, 2)).Sum() / (logSample.Length - 1));

        double goodnessOfFit = CalculateGoodnessOfFit(sample, x => CalculateLogNormalPDF(mu, sigma, x));

        return new DistributionFitResult
        {
            GoodnessOfFit = goodnessOfFit,
            Parameters = new Dictionary<string, double>
            {
                { "Mu", mu },
                { "Sigma", sigma }
            }
        };
    }

    private static (double min, double max) CalculateMinMax(Vector<double> sample)
    {
        if (sample.Length == 0) return (double.NaN, double.NaN);

        double min = sample[0], max = sample[0];
        for (int i = 1; i < sample.Length; i++)
        {
            if (sample[i] < min) min = sample[i];
            if (sample[i] > max) max = sample[i];
        }

        return (min, max);
    }

    private static (double skewness, double kurtosis) CalculateSkewnessAndKurtosis(Vector<double> sample, double mean, double stdDev, int n)
    {
        double skewnessSum = 0, kurtosisSum = 0;
        for (int i = 0; i < n; i++)
        {
            var diff = (sample[i] - mean) / stdDev;
            var diff3 = diff * diff * diff;
            skewnessSum += diff3;
            kurtosisSum += diff3 * diff;
        }

        var skewness = n > 2 ? skewnessSum / ((n - 1) * (n - 2)) : 0;
        var kurtosis = n > 3 ? kurtosisSum / ((n - 1) * (n - 2) * (n - 3)) * n * (n + 1) - 3 * (n - 1) * (n - 1) / ((n - 2) * (n - 3)) : 0;

        return (skewness, kurtosis);
    }

    public double CalculateInverseNormalCDF(double mean, double stdDev, double probability)
    {
        return mean + stdDev * CalculateInverseNormalCDF(probability);
    }

    public (double LowerBound, double UpperBound) CalculateCredibleIntervals(Vector<double> sample, 
        double confidenceLevel, DistributionType distributionType)
    {
        double lowerProbability = (1 - confidenceLevel) / 2; // 0.025 for 95% CI
        double upperProbability = 1 - lowerProbability; // 0.975 for 95% CI
        (var mean, var stdDev) = CalculateStandardDeviation(sample);
        var median = CalculateMedian(sample);
        var mad = CalculateMeanAbsoluteDeviation(sample, median);

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
                CalculateInverseStudentTCDF(sample.Length - 1, lowerProbability),
                CalculateInverseStudentTCDF(sample.Length - 1, upperProbability)
            ),
            DistributionType.LogNormal => (
                Math.Exp(CalculateInverseNormalCDF(Math.Log(mean) - 0.5 * Math.Log(1 + stdDev * stdDev / (mean * mean)), 
                    Math.Sqrt(Math.Log(1 + stdDev * stdDev / (mean * mean))), lowerProbability)),
                Math.Exp(CalculateInverseNormalCDF(Math.Log(mean) - 0.5 * Math.Log(1 + stdDev * stdDev / (mean * mean)), 
                    Math.Sqrt(Math.Log(1 + stdDev * stdDev / (mean * mean))), upperProbability))
            ),
            DistributionType.Exponential => (
                CalculateInverseExponentialCDF(1 / mean, lowerProbability),
                CalculateInverseExponentialCDF(1 / mean, upperProbability)
            ),
            DistributionType.Weibull => CalculateWeibullCredibleIntervals(sample, lowerProbability, upperProbability),
            _ => throw new ArgumentException("Invalid distribution type"),
        };
    }

    public double CalculateInverseNormalCDF(double probability)
    {
        // Approximation of inverse normal CDF
        if (probability <= 0 || probability >= 1)
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");

        double t = Math.Sqrt(-2 * Math.Log(probability < 0.5 ? probability : 1 - probability));
        double c0 = 2.515517;
        double c1 = 0.802853;
        double c2 = 0.010328;
        double d1 = 1.432788;
        double d2 = 0.189269;
        double d3 = 0.001308;

        double x = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);

        return probability < 0.5 ? -x : x;
    }

    public double CalculateInverseStudentTCDF(int degreesOfFreedom, double probability)
    {
        // This is an approximation and might not be as accurate as a dedicated statistical library
        if (probability <= 0 || probability >= 1)
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");

        double x = CalculateInverseNormalCDF(probability);
        double y = x * x;

        double a = (y + 1) / 4;
        double b = ((5 * y + 16) * y + 3) / 96;
        double c = (((3 * y + 19) * y + 17) * y - 15) / 384;
        double d = ((((79 * y + 776) * y + 1482) * y - 1920) * y - 945) / 92160;

        double t = x * (1 + (a + (b + (c + d / degreesOfFreedom) / degreesOfFreedom) / degreesOfFreedom) / degreesOfFreedom);

        if (degreesOfFreedom <= 2)
        {
            // Additional refinement for low degrees of freedom
            t = Math.Sign(x) * Math.Sqrt(degreesOfFreedom * (Math.Pow(probability, -2.0 / degreesOfFreedom) - 1));
        }

        return t;
    }

    public double CalculateInverseLaplaceCDF(double median, double mad, double probability) =>
        median - mad * Math.Sign(probability - 0.5) * Math.Log(1 - 2 * Math.Abs(probability - 0.5));

    public (double LowerBound, double UpperBound) CalculateConfidenceIntervals(Vector<double> sample, 
    double confidenceLevel, DistributionType distributionType)
    {
        (var mean, var stdDev) = CalculateStandardDeviation(sample);
        var median = CalculateMedian(sample);
        var mad = CalculateMeanAbsoluteDeviation(sample, median);
        double lowerBound, upperBound;

        switch (distributionType)
        {
            case DistributionType.Normal:
                var zScore = CalculateInverseNormalCDF(1 - (1 - confidenceLevel) / 2);
                var marginOfError = zScore * (stdDev / Math.Sqrt(sample.Length));
                lowerBound = mean - marginOfError;
                upperBound = mean + marginOfError;
                break;
            case DistributionType.Laplace:
                var laplaceValue = CalculateInverseLaplaceCDF(median, mad, confidenceLevel);
                lowerBound = 2 * median - laplaceValue;
                upperBound = laplaceValue;
                break;
            case DistributionType.Student:
                var tValue = CalculateInverseStudentTCDF(sample.Length - 1, 1 - (1 - confidenceLevel) / 2);
                var tMarginOfError = tValue * (stdDev / Math.Sqrt(sample.Length));
                lowerBound = mean - tMarginOfError;
                upperBound = mean + tMarginOfError;
                break;
            case DistributionType.LogNormal:
                var sampleArray = sample.ToArray();
                (var logMean, var logStdDev) = CalculateStandardDeviation(sample.Transform(x => Math.Log(x)));
                var logZScore = CalculateInverseNormalCDF(1 - (1 - confidenceLevel) / 2);
                lowerBound = Math.Exp(logMean - logZScore * logStdDev / Math.Sqrt(sample.Length));
                upperBound = Math.Exp(logMean + logZScore * logStdDev / Math.Sqrt(sample.Length));
                break;
            case DistributionType.Exponential:
                var lambda = 1 / mean;
                var chiSquareLower = CalculateInverseChiSquareCDF(2 * sample.Length, (1 - confidenceLevel) / 2);
                var chiSquareUpper = CalculateInverseChiSquareCDF(2 * sample.Length, 1 - (1 - confidenceLevel) / 2);
                lowerBound = 2 * sample.Length / chiSquareUpper * (1 / lambda);
                upperBound = 2 * sample.Length / chiSquareLower * (1 / lambda);
                break;
            case DistributionType.Weibull:
                // For Weibull, we'll use a bootstrap method to estimate confidence intervals
                (lowerBound, upperBound) = CalculateWeibullConfidenceIntervals(sample, confidenceLevel);
                break;
            default:
                throw new ArgumentException("Invalid distribution type");
        }

        return (lowerBound, upperBound);
    }

    private double CalculateChiSquareCDF(int degreesOfFreedom, double x)
    {
        return IncompleteGamma(degreesOfFreedom / 2.0, x / 2.0);
    }

    private double IncompleteGamma(double a, double x)
    {
        // Implementation of incomplete gamma function
        // This is a simplified version and might not be accurate for all inputs
        const int maxIterations = 100;
        const double epsilon = 1e-8;

        double sum = 0;
        double term = 1.0 / a;
        for (int i = 0; i < maxIterations; i++)
        {
            sum += term;
            term *= x / (a + i + 1);
            if (term < epsilon)
                break;
        }

        return sum * Math.Exp(-x + a * Math.Log(x) - LogGamma(a));
    }

    private double CalculateChiSquarePDF(int degreesOfFreedom, double x)
    {
        return Math.Pow(x, degreesOfFreedom / 2.0 - 1) * Math.Exp(-x / 2.0) / (Math.Pow(2, degreesOfFreedom / 2.0) * Gamma(degreesOfFreedom / 2.0));
    }

    public double CalculateInverseChiSquareCDF(int degreesOfFreedom, double probability)
    {
        if (probability <= 0 || probability >= 1)
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");
        if (degreesOfFreedom <= 0)
            throw new ArgumentOutOfRangeException(nameof(degreesOfFreedom), "Degrees of freedom must be positive");

        // Initial guess
        double x = degreesOfFreedom * Math.Pow(1 - 2.0 / (9 * degreesOfFreedom) + Math.Sqrt(2.0 / (9 * degreesOfFreedom)) * CalculateInverseNormalCDF(probability), 3);

        // Newton-Raphson method for refinement
        const int maxIterations = 20;
        const double epsilon = 1e-8;
        for (int i = 0; i < maxIterations; i++)
        {
            double fx = CalculateChiSquareCDF(degreesOfFreedom, x) - probability;
            double dfx = CalculateChiSquarePDF(degreesOfFreedom, x);

            double delta = fx / dfx;
            x -= delta;

            if (Math.Abs(delta) < epsilon)
                break;
        }

        return x;
    }

    public double Digamma(double x)
    {
        // Approximation of the digamma function
        double result = 0;
        for (int i = 0; i < 100 && x <= 8; i++)
        {
            result -= 1 / x;
            x += 1;
        }
        if (x <= 8) return result;
        double invX = 1 / x;
        double invX2 = invX * invX;

        return Math.Log(x) - 0.5 * invX - invX2 * ((1.0 / 12) - invX2 * ((1.0 / 120) - invX2 * (1.0 / 252)));
    }

    private (double Shape, double Scale) EstimateWeibullParameters(Vector<double> sample)
    {
        // Method of moments estimation for Weibull parameters
        double mean = sample.Average();
        double variance = CalculateVariance(sample, mean);

        // Initial guess for shape parameter
        double shape = Math.Sqrt(Math.PI * Math.PI / (6 * variance));

        // Newton-Raphson method to refine shape estimate
        for (int i = 0; i < 10; i++)
        {
            double g = Gamma(1 + 1 / shape);
            double g2 = Gamma(1 + 2 / shape);
            double f = g2 / (g * g) - 1 - variance / (mean * mean);
            double fPrime = 2 * (Digamma(1 + 2 / shape) / shape - Digamma(1 + 1 / shape) / shape) * g2 / (g * g)
                            - 2 * (g2 / (g * g) - 1) * Digamma(1 + 1 / shape) / shape;

            shape = shape - f / fPrime;

            if (Math.Abs(f) < 1e-6)
                break;
        }

        double scale = mean / Gamma(1 + 1 / shape);

        return (shape, scale);
    }

    public (double LowerBound, double UpperBound) CalculateWeibullCredibleIntervals(
        Vector<double> sample, double lowerProbability, double upperProbability)
    {
        // Estimate Weibull parameters
        var (shape, scale) = EstimateWeibullParameters(sample);

        // Calculate credible intervals
        double lowerBound = scale * Math.Pow(-Math.Log(1 - lowerProbability), 1 / shape);
        double upperBound = scale * Math.Pow(-Math.Log(1 - upperProbability), 1 / shape);

        return (lowerBound, upperBound);
    }

    public (double LowerBound, double UpperBound) CalculateWeibullConfidenceIntervals(Vector<double> sample, double confidenceLevel)
    {
        const int bootstrapSamples = 1000;
        var rng = new Random();
        var estimates = new List<(double Shape, double Scale)>();

        for (int i = 0; i < bootstrapSamples; i++)
        {
            var bootstrapSample = new Vector<double>(sample.Length);
            for (int j = 0; j < sample.Length; j++)
            {
                bootstrapSample[j] = sample[rng.Next(sample.Length)];
            }
            estimates.Add(EstimateWeibullParameters(bootstrapSample));
        }

        var sortedShapes = estimates.Select(e => e.Shape).OrderBy(s => s).ToList();
        var sortedScales = estimates.Select(e => e.Scale).OrderBy(s => s).ToList();

        int lowerIndex = (int)(bootstrapSamples * (1 - confidenceLevel) / 2);
        int upperIndex = (int)(bootstrapSamples * (1 + confidenceLevel) / 2);

        return (sortedShapes[lowerIndex] * sortedScales[lowerIndex], sortedShapes[upperIndex] * sortedScales[upperIndex]);
    }

    public double CalculateInverseExponentialCDF(double lambda, double probability)
    {
        if (probability <= 0 || probability >= 1)
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be between 0 and 1");

        return -Math.Log(1 - probability) / lambda;
    }

    public double CalculateVariance(Vector<double> sample, double mean)
    {
        double sumSquaredDifferences = 0;

        for (int i = 0; i < sample.Length; i++)
        {
            double difference = sample[i] - mean;
            sumSquaredDifferences += difference * difference;
        }

        return sumSquaredDifferences / (sample.Length - 1);
    }

    private double CalculateMedian(Vector<double> vector)
    {
        var sortedVector = vector.ToArray();
        Array.Sort(sortedVector);
        int n = sortedVector.Length;
        if (n % 2 == 0)
        {
            return (sortedVector[n / 2 - 1] + sortedVector[n / 2]) / 2;
        }

        return sortedVector[n / 2];
    }

    private double CalculateR2(Vector<double> actual, Vector<double> predicted)
    {
        double meanActual = actual.Mean();
        double ssTotal = actual.Subtract(meanActual).Transform(x => x * x).Sum();
        double ssResidual = actual.Subtract(predicted).Transform(x => x * x).Sum();

        return 1 - (ssResidual / ssTotal);
    }

    private double CalculateAdjustedR2(double r2, int n, int p)
    {
        return 1 - ((1 - r2) * (n - 1) / (n - p - 1));
    }

    private double CalculateExplainedVarianceScore(Vector<double> actual, Vector<double> predicted)
    {
        double varActual = actual.Variance();
        double varError = actual.Subtract(predicted).Variance();
        return 1 - (varError / varActual);
    }
}