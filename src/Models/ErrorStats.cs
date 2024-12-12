using AiDotNet.Interfaces;
using AiDotNet.Helpers;
namespace AiDotNet.Models;

public class ErrorStats<T>
{
    private readonly INumericOperations<T> NumOps;

    public T MAE { get; private set; }
    public T MSE { get; private set; }
    public T RMSE { get; private set; }
    public T MAPE { get; private set; }
    public T MeanBiasError { get; private set; }
    public T MedianAbsoluteError { get; private set; }
    public T MaxError { get; private set; }
    public T TheilUStatistic { get; private set; }
    public T DurbinWatsonStatistic { get; private set; }
    public T SampleStandardError { get; private set; }
    public T PopulationStandardError { get; private set; }
    public T AIC { get; private set; }
    public T BIC { get; private set; }
    public T AICAlt { get; private set; }
    public T RSS { get; private set; }
    public List<T> ErrorList { get; private set; } = new List<T>();

    public ErrorStats(Vector<T> actual, Vector<T> predicted, int numberOfParameters)
    {
        NumOps = MathHelper.GetNumericOperations<T>();

        // Initialize all variables to zero
        MAE = NumOps.Zero;
        MSE = NumOps.Zero;
        RMSE = NumOps.Zero;
        MAPE = NumOps.Zero;
        MeanBiasError = NumOps.Zero;
        MedianAbsoluteError = NumOps.Zero;
        MaxError = NumOps.Zero;
        TheilUStatistic = NumOps.Zero;
        DurbinWatsonStatistic = NumOps.Zero;
        SampleStandardError = NumOps.Zero;
        PopulationStandardError = NumOps.Zero;
        AIC = NumOps.Zero;
        BIC = NumOps.Zero;
        AICAlt = NumOps.Zero;
        RSS = NumOps.Zero;

        ErrorList = [];

        CalculateErrorStats(actual, predicted, numberOfParameters);
    }

    public static ErrorStats<T> Empty()
    {
        return new ErrorStats<T>(Vector<T>.Empty(), Vector<T>.Empty(), 0);
    }

    private void CalculateErrorStats(Vector<T> actual, Vector<T> predicted, int numberOfParameters)
    {
        int n = actual.Length;

        // Calculate basic error metrics
        MAE = CalculateMAE(actual, predicted);
        RSS = CalculateRSS(actual, predicted);
        MSE = CalculateMSE(actual, predicted);
        RMSE = NumOps.Sqrt(MSE);
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
        ErrorList = actual.Subtract(predicted).ToList();
    }

    private T CalculateRSS(Vector<T> actual, Vector<T> predicted)
    {
        return actual.Subtract(predicted).Select(x => NumOps.Square(x)).Aggregate(NumOps.Zero, NumOps.Add);
    }

    private T CalculateMAE(Vector<T> actual, Vector<T> predicted)
    {
        return NumOps.Divide(actual.Subtract(predicted).Select(NumOps.Abs).Aggregate(NumOps.Zero, NumOps.Add), NumOps.FromDouble(actual.Length));
    }

    private T CalculateMSE(Vector<T> actual, Vector<T> predicted)
    {
        return NumOps.Divide(actual.Subtract(predicted).Select(x => NumOps.Square(x)).Aggregate(NumOps.Zero, NumOps.Add), NumOps.FromDouble(actual.Length));
    }

    private T CalculateMAPE(Vector<T> actual, Vector<T> predicted)
    {
        var mape = actual.Zip(predicted, (a, p) => NumOps.Abs(NumOps.Divide(NumOps.Subtract(a, p), a)))
                         .Where(x => !NumOps.Equals(x, NumOps.Zero))
                         .Aggregate(NumOps.Zero, NumOps.Add);
        return NumOps.Multiply(NumOps.Divide(mape, NumOps.FromDouble(actual.Length)), NumOps.FromDouble(100));
    }

    private T CalculateMedianAbsoluteError(Vector<T> actual, Vector<T> predicted)
    {
        var absoluteErrors = actual.Subtract(predicted).Select(NumOps.Abs).OrderBy(x => x).ToArray();
        int n = absoluteErrors.Length;
        return n % 2 == 0
            ? NumOps.Divide(NumOps.Add(absoluteErrors[n / 2 - 1], absoluteErrors[n / 2]), NumOps.FromDouble(2))
            : absoluteErrors[n / 2];
    }

    private T CalculateMaxError(Vector<T> actual, Vector<T> predicted)
    {
        return actual.Subtract(predicted).Select(NumOps.Abs).Max();
    }

    private T CalculateSampleStandardError(Vector<T> actual, Vector<T> predicted, int numberOfParameters)
    {
        T mse = CalculateMSE(actual, predicted);
        int degreesOfFreedom = actual.Length - numberOfParameters;
        return NumOps.Sqrt(NumOps.Divide(mse, NumOps.FromDouble(degreesOfFreedom)));
    }

    private T CalculatePopulationStandardError(Vector<T> actual, Vector<T> predicted)
    {
        return NumOps.Sqrt(CalculateMSE(actual, predicted));
    }

    private T CalculateMeanBiasError(Vector<T> actual, Vector<T> predicted)
    {
        return NumOps.Divide(actual.Subtract(predicted).Aggregate(NumOps.Zero, NumOps.Add), NumOps.FromDouble(actual.Length));
    }

    private T CalculateTheilUStatistic(Vector<T> actual, Vector<T> predicted)
    {
        T numerator = NumOps.Sqrt(CalculateMSE(actual, predicted));
        T denominatorActual = NumOps.Sqrt(NumOps.Divide(actual.Select(x => NumOps.Square(x)).Aggregate(NumOps.Zero, NumOps.Add), NumOps.FromDouble(actual.Length)));
        T denominatorPredicted = NumOps.Sqrt(NumOps.Divide(predicted.Select(x => NumOps.Square(x)).Aggregate(NumOps.Zero, NumOps.Add), NumOps.FromDouble(predicted.Length)));
        return NumOps.Divide(numerator, NumOps.Add(denominatorActual, denominatorPredicted));
    }

    private T CalculateDurbinWatsonStatistic(Vector<T> actual, Vector<T> predicted)
    {
        var errors = actual.Subtract(predicted);
        T sumSquaredDifferences = NumOps.Zero;
        T sumSquaredErrors = NumOps.Zero;

        for (int i = 1; i < errors.Length; i++)
        {
            sumSquaredDifferences = NumOps.Add(sumSquaredDifferences, NumOps.Square(NumOps.Subtract(errors[i], errors[i - 1])));
            sumSquaredErrors = NumOps.Add(sumSquaredErrors, NumOps.Square(errors[i]));
        }
        sumSquaredErrors = NumOps.Add(sumSquaredErrors, NumOps.Square(errors[0]));

        return NumOps.Divide(sumSquaredDifferences, sumSquaredErrors);
    }

    public T CalculateAICAlternative(int sampleSize, int parameterSize, T rss)
    {
        if (sampleSize <= 0 || NumOps.LessThanOrEquals(rss, NumOps.Zero)) return NumOps.Zero;
        T logData = NumOps.Divide(rss, NumOps.FromDouble(sampleSize));
        return NumOps.Add(NumOps.Multiply(NumOps.FromDouble(sampleSize), NumOps.Log(logData)), NumOps.Multiply(NumOps.FromDouble(2), NumOps.FromDouble(parameterSize)));
    }

    public T CalculateAIC(int sampleSize, int parameterSize, T rss)
    {
        if (sampleSize <= 0 || NumOps.LessThanOrEquals(rss, NumOps.Zero)) return NumOps.Zero;
        T logData = NumOps.Multiply(NumOps.FromDouble(2 * Math.PI), NumOps.Divide(rss, NumOps.FromDouble(sampleSize)));
        return NumOps.Add(NumOps.Multiply(NumOps.FromDouble(2), NumOps.FromDouble(parameterSize)), 
                          NumOps.Multiply(NumOps.FromDouble(sampleSize), NumOps.Add(NumOps.Log(logData), NumOps.One)));
    }

    public T CalculateBIC(int sampleSize, int parameterSize, T rss)
    {
        if (sampleSize <= 0 || NumOps.LessThanOrEquals(rss, NumOps.Zero)) return NumOps.Zero;
        T logData = NumOps.Divide(rss, NumOps.FromDouble(sampleSize));
        return NumOps.Add(NumOps.Multiply(NumOps.FromDouble(sampleSize), NumOps.Log(logData)), 
                          NumOps.Multiply(NumOps.FromDouble(parameterSize), NumOps.Log(NumOps.FromDouble(sampleSize))));
    }
}