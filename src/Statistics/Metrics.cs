namespace AiDotNet.Statistics;

/// <summary>
/// Metrics data to help evaluate the performance of a model by comparing the predicted values to the actual values.
/// Predicted values are taken from the out of sample (oos) data only.
/// </summary>
public sealed class Metrics : IMetrics
{
    /// <summary>
    /// Mean squared error (MSE) is the average of the squared differences between the predicted values and the actual values.
    /// </summary>
    public double MeanSquaredError { get; }

    /// <summary>
    /// Root mean squared error (RMSE) is the square root of the mean squared error.
    /// </summary>
    public double RootMeanSquaredError { get; }

    /// <summary>
    /// Adjusted R2 is the coefficient of determination (R2) adjusted for the number of independent variables in the model.
    /// </summary>
    public double AdjustedR2 { get; }

    /// <summary>
    /// R2 is the coefficient of determination which is a measure of how well the predictions fit the actual values.
    /// </summary>
    public double R2 { get; }

    /// <summary>
    /// The standard error of the predictions.
    /// </summary>
    public double PredictionsStandardError { get; }

    /// <summary>
    /// The standard deviation of the predictions.
    /// </summary>
    public double PredictionsStandardDeviation { get; }

    /// <summary>
    /// The standard error of the average predictions.
    /// </summary>
    public double AverageStandardError { get; }

    /// <summary>
    /// The standard deviation of the average predictions.
    /// </summary>
    public double AverageStandardDeviation { get; }

    /// <summary>
    /// The degrees of freedom is the number of independent values that can vary in the data sample.
    /// </summary>
    public int DegreesOfFreedom { get; }
    
    public double Quartile1Value { get; private set; }
    public double Quartile2Value { get; private set; }
    public double Quartile3Value { get; private set; }

    private double OosPredictionsAvg { get; }
    private double OosActualValuesAvg { get; }
    private int ParamsCount { get; }
    private int SampleSize { get; }
    private double ResidualSumOfSquares { get; set; }
    private double TotalSumOfSquares { get; set; }
    private IQuartile Quartile { get; set; }

    public Metrics(double[] oosPredictions, double[] oosActualValues, int paramCount, IQuartile? quartile)
    {
        OosPredictionsAvg = oosPredictions.Average();
        OosActualValuesAvg = oosActualValues.Average();
        ParamsCount = paramCount;
        SampleSize = oosPredictions.Length;
        DegreesOfFreedom = CalculateDegreesOfFreedom();
        var (residualSquaresSum, totalSquaresSum, r2) = 
            MetricsHelper.CalculateR2(oosPredictions, oosActualValues, OosActualValuesAvg, SampleSize);
        ResidualSumOfSquares = residualSquaresSum;
        TotalSumOfSquares = totalSquaresSum;
        R2 = r2;
        AdjustedR2 = CalculateAdjustedR2(R2);
        AverageStandardDeviation = CalculateAverageStandardDeviation();
        PredictionsStandardDeviation = CalculatePredictionStandardDeviation();
        AverageStandardError = CalculateAverageStandardError();
        PredictionsStandardError = CalculatePredictionStandardError();
        MeanSquaredError = CalculateMeanSquaredError();
        RootMeanSquaredError = CalculateRootMeanSquaredError();

        Quartile = quartile ?? new StandardQuartile();
        var sortedOosPredictions = oosPredictions.DeepCopySort();
        var (q1Value, q2Value, q3Value) = Quartile.FindQuartiles(sortedOosPredictions);
        Quartile1Value = q1Value;
        Quartile2Value = q2Value;
        Quartile3Value = q3Value;
    }

    public Metrics(double[][] oosPredictions, double[][] oosActualValues, int paramCount, IQuartile? quartile)
    {
        OosPredictionsAvg = oosPredictions.Average();
        OosActualValuesAvg = oosActualValues.Average();
        ParamsCount = paramCount;
        SampleSize = oosPredictions.Length;
        DegreesOfFreedom = CalculateDegreesOfFreedom();
        var r2Result = MetricsHelper.CalculateR2(oosPredictions, oosActualValues, OosActualValuesAvg, SampleSize);
        ResidualSumOfSquares = r2Result.residualSquaresSum;
        TotalSumOfSquares = r2Result.totalSquaresSum;
        R2 = r2Result.r2;
        AdjustedR2 = CalculateAdjustedR2(R2);
        AverageStandardDeviation = CalculateAverageStandardDeviation();
        PredictionsStandardDeviation = CalculatePredictionStandardDeviation();
        AverageStandardError = CalculateAverageStandardError();
        PredictionsStandardError = CalculatePredictionStandardError();
        MeanSquaredError = CalculateMeanSquaredError();
        RootMeanSquaredError = CalculateRootMeanSquaredError();

        Quartile = quartile ?? new StandardQuartile();
        // sort oos predictions here
        var sortedOosPredictions = oosPredictions.DeepCopySort();
        var (q1Value, q2Value, q3Value) = Quartile.FindQuartiles(sortedOosPredictions);
        Quartile1Value = q1Value;
        Quartile2Value = q2Value;
        Quartile3Value = q3Value;
    }

    internal override double CalculateMeanSquaredError()
    {
        return ResidualSumOfSquares / SampleSize;
    }

    internal override double CalculateRootMeanSquaredError()
    {
        return MeanSquaredError >= 0 ? Math.Sqrt(MeanSquaredError) : 0;
    }

    internal override double CalculateAdjustedR2(double r2)
    {
        return SampleSize != 1 && DegreesOfFreedom != 1 ? 1 - (1 - Math.Pow(r2, 2)) * (SampleSize - 1) / (DegreesOfFreedom - 1) : 0;
    }

    internal override double CalculateAverageStandardError()
    {
        return AverageStandardDeviation / Math.Sqrt(SampleSize);
    }

    internal override double CalculatePredictionStandardError()
    {
        return PredictionsStandardDeviation / Math.Sqrt(SampleSize);
    }

    private static double CalculateStandardDeviation(double avgSumSquares)
    {
        return avgSumSquares >= 0 ? Math.Sqrt(avgSumSquares) : 0;
    }

    internal override double CalculateAverageStandardDeviation()

    {
        var avgSumSquares = TotalSumOfSquares / SampleSize;

        return CalculateStandardDeviation(avgSumSquares);
    }

    internal override double CalculatePredictionStandardDeviation()
    {
        var avgSumSquares = ResidualSumOfSquares / SampleSize;

        return CalculateStandardDeviation(avgSumSquares);
    }

    internal override int CalculateDegreesOfFreedom()
    {
        return SampleSize - ParamsCount;
    }
}