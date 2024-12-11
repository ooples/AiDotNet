public class ErrorStats
{
    public double MAE { get; private set; }
    public double MSE { get; private set; }
    public double RMSE { get; private set; }
    public double MAPE { get; private set; }
    public double MeanBiasError { get; private set; }
    public double MedianAbsoluteError { get; private set; }
    public double MaxError { get; private set; }
    public double TheilUStatistic { get; private set; }
    public double DurbinWatsonStatistic { get; private set; }
    public double SampleStandardError { get; private set; }
    public double PopulationStandardError { get; private set; }
    public double AIC { get; private set; }
    public double BIC { get; private set; }
    public double AICAlt { get; private set; }
    public double RSS { get; private set; }
    public List<double> ErrorList { get; private set; } = [];


    public ErrorStats(Vector<double> actual, Vector<double> predicted, int numberOfParameters)
    {
        CalculateErrorStats(actual, predicted, numberOfParameters);
    }

    public static ErrorStats Empty()
    {
        return new ErrorStats(Vector<double>.Empty(), Vector<double>.Empty(), 0, 0);
    }

    private void CalculateErrorStats(Vector<double> actual, Vector<double> predicted, int numberOfParameters)
    {
        int n = actual.Length;

        // Calculate basic error metrics
        MAE = CalculateMAE(actual, predicted);
        RSS = CalculateRSS(actual, predicted);
        MSE = CalculateMSE(actual, predicted);
        RMSE = Math.Sqrt(MSE);
        MAPE = CalculateMAPE(actual, predicted);
        MedianAbsoluteError = CalculateMedianAbsoluteError(actual, predicted);
        MaxError = CalculateMaxError(actual, predicted);

        // Calculate standard errors
        SampleStandardError = CalculateSampleStandardError(actual, predicted, numberOfParameters);
        PopulationStandardError = CalculatePopulationStandardError(actual, predicted);

        // Calculate bias and autocorrelation metrics
        MeanBiasError = CalculateMeanBiasError(actual, predicted);
        TheilUStatistic = CalculateTheilUStatistic(actual, predicted);
        DurbinWatsonStatistic = CalculateDurbinWatsonStatistic(actual, predicted);

        // Calculate information criteria
        AIC = CalculateAIC(n, numberOfParameters, RSS);
        BIC = CalculateBIC(n, numberOfParameters, RSS);
        AICAlt = CalculateAICAlternative(n, numberOfParameters, RSS);

        // Populate error list
        ErrorList = [.. actual.Subtract(predicted)];
    }

    private double CalculateRSS(Vector<double> actual, Vector<double> predicted)
    {
        return actual.Subtract(predicted).Select(x => x * x).Sum();
    }

    private double CalculateMAE(Vector<double> actual, Vector<double> predicted)
    {
        return actual.Subtract(predicted).Select(Math.Abs).Average();
    }

    private double CalculateMSE(Vector<double> actual, Vector<double> predicted)
    {
        return actual.Subtract(predicted).Select(x => x * x).Average();
    }

    private double CalculateMAPE(Vector<double> actual, Vector<double> predicted)
    {
        return actual.Zip(predicted, (a, p) => Math.Abs((a - p) / a))
                     .Where(x => !double.IsInfinity(x) && !double.IsNaN(x))
                     .Average() * 100;
    }

    private double CalculateMedianAbsoluteError(Vector<double> actual, Vector<double> predicted)
    {
        var absoluteErrors = actual.Subtract(predicted).Select(Math.Abs).OrderBy(x => x).ToArray();
        int n = absoluteErrors.Length;
        return n % 2 == 0
            ? (absoluteErrors[n / 2 - 1] + absoluteErrors[n / 2]) / 2
            : absoluteErrors[n / 2];
    }

    private double CalculateMaxError(Vector<double> actual, Vector<double> predicted)
    {
        return actual.Subtract(predicted).Select(Math.Abs).Max();
    }

    private double CalculateSampleStandardError(Vector<double> actual, Vector<double> predicted, int numberOfParameters)
    {
        double mse = CalculateMSE(actual, predicted);
        int degreesOfFreedom = actual.Length - numberOfParameters;
        return Math.Sqrt(mse / degreesOfFreedom);
    }

    private double CalculatePopulationStandardError(Vector<double> actual, Vector<double> predicted)
    {
        return Math.Sqrt(CalculateMSE(actual, predicted));
    }

    private double CalculateMeanBiasError(Vector<double> actual, Vector<double> predicted)
    {
        return actual.Subtract(predicted).Average();
    }

    private double CalculateTheilUStatistic(Vector<double> actual, Vector<double> predicted)
    {
        double numerator = Math.Sqrt(CalculateMSE(actual, predicted));
        double denominatorActual = Math.Sqrt(actual.Select(x => x * x).Average());
        double denominatorPredicted = Math.Sqrt(predicted.Select(x => x * x).Average());
        return numerator / (denominatorActual + denominatorPredicted);
    }

    private double CalculateDurbinWatsonStatistic(Vector<double> actual, Vector<double> predicted)
    {
        var errors = actual.Subtract(predicted);
        double sumSquaredDifferences = 0;
        double sumSquaredErrors = 0;

        for (int i = 1; i < errors.Length; i++)
        {
            sumSquaredDifferences += Math.Pow(errors[i] - errors[i - 1], 2);
            sumSquaredErrors += Math.Pow(errors[i], 2);
        }
        sumSquaredErrors += Math.Pow(errors[0], 2);

        return sumSquaredDifferences / sumSquaredErrors;
    }

    public double CalculateAICAlternative(int sampleSize, int parameterSize, double rss)
    {
        if (sampleSize <= 0 || rss <= 0) return 0;
        double logData = rss / sampleSize;
        return (sampleSize * Math.Log(logData)) + (2 * parameterSize);
    }

    public double CalculateAIC(int sampleSize, int parameterSize, double rss)
    {
        if (sampleSize <= 0 || rss <= 0) return 0;
        double logData = 2 * Math.PI * (rss / sampleSize);
        return (2 * parameterSize) + (sampleSize * (Math.Log(logData) + 1));
    }

    public double CalculateBIC(int sampleSize, int parameterSize, double rss)
    {
        if (sampleSize <= 0 || rss <= 0) return 0;
        double logData = rss / sampleSize;
        return (sampleSize * Math.Log(logData)) + (parameterSize * Math.Log(sampleSize));
    }
}