namespace AiDotNet.Statistics;

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

    internal ErrorStats(ErrorStatsInputs<T> inputs)
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

        CalculateErrorStats(inputs.Actual, inputs.Predicted, inputs.FeatureCount);
    }

    public static ErrorStats<T> Empty()
    {
        return new ErrorStats<T>(new());
    }

    private void CalculateErrorStats(Vector<T> actual, Vector<T> predicted, int numberOfParameters)
    {
        int n = actual.Length;

        // Calculate basic error metrics
        MAE = StatisticsHelper<T>.CalculateMeanAbsoluteError(actual, predicted);
        RSS = StatisticsHelper<T>.CalculateResidualSumOfSquares(actual, predicted);
        MSE = StatisticsHelper<T>.CalculateMeanSquaredError(actual, predicted);
        RMSE = NumOps.Sqrt(MSE);
        MAPE = StatisticsHelper<T>.CalculateMeanAbsolutePercentageError(actual, predicted);
        MedianAbsoluteError = StatisticsHelper<T>.CalculateMedianAbsoluteError(actual, predicted);
        MaxError = StatisticsHelper<T>.CalculateMaxError(actual, predicted);

        // Calculate standard errors
        SampleStandardError = StatisticsHelper<T>.CalculateSampleStandardError(actual, predicted, numberOfParameters);
        PopulationStandardError = StatisticsHelper<T>.CalculatePopulationStandardError(actual, predicted);

        // Calculate bias and autocorrelation metrics
        MeanBiasError = StatisticsHelper<T>.CalculateMeanBiasError(actual, predicted);
        TheilUStatistic = StatisticsHelper<T>.CalculateTheilUStatistic(actual, predicted);
        DurbinWatsonStatistic = StatisticsHelper<T>.CalculateDurbinWatsonStatistic(actual, predicted);

        // Calculate information criteria
        AIC = StatisticsHelper<T>.CalculateAIC(n, numberOfParameters, RSS);
        BIC = StatisticsHelper<T>.CalculateBIC(n, numberOfParameters, RSS);
        AICAlt = StatisticsHelper<T>.CalculateAICAlternative(n, numberOfParameters, RSS);

        // Populate error list
        ErrorList = [.. actual.Subtract(predicted)];
    }
}