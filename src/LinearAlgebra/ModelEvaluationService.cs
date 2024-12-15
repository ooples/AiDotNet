namespace AiDotNet.LinearAlgebra;

/*
public class ModelEvaluationService<T> : IModelEvaluator<T>
{
    private readonly int _numberOfParameters;

    public ModelEvaluationService(int numberOfParameters)
    {
        _numberOfParameters = numberOfParameters;
    }

    public ModelEvaluationResult<T> EvaluateModel(
        Vector<T> actualTrain, Vector<T> predictedTrain,
        Vector<T> actualVal, Vector<T> predictedVal,
        Vector<T> actualTest, Vector<T> predictedTest)
    {
        var result = new ModelEvaluationResult<T>
        {
            TrainingMetrics = CalculateMetrics(actualTrain, predictedTrain),
            ValidationMetrics = CalculateMetrics(actualVal, predictedVal),
            TestMetrics = CalculateMetrics(actualTest, predictedTest)
        };

        return result;
    }

    public Dictionary<string, T> CalculateMetrics(Vector<T> actual, Vector<T> predicted)
    {
        var basicStats = new BasicStats<T>();
        basicStats.Calculate(actual, predicted);
        var errorStats = new ErrorStats<T>(actual, predicted, _numberOfParameters);

        return CombineMetrics(basicStats, errorStats);
    }

    private Dictionary<string, T> CombineMetrics(BasicStats<T> basicStats, ErrorStats<T> errorStats)
    {
        var combinedMetrics = new Dictionary<string, T>
        {
            ["Mean"] = basicStats.Mean,
            ["Median"] = basicStats.Median,
            ["StandardDeviation"] = basicStats.StandardDeviation,
            ["Variance"] = basicStats.Variance,
            ["Skewness"] = basicStats.Skewness,
            ["Kurtosis"] = basicStats.Kurtosis,
            ["MAE"] = errorStats.MAE,
            ["MSE"] = errorStats.MSE,
            ["RMSE"] = errorStats.RMSE,
            ["R2"] = errorStats.RSquared,
            ["AdjustedR2"] = errorStats.AdjustedRSquared,
            ["MAPE"] = errorStats.MAPE
        };

        return combinedMetrics;
    }
}
*/