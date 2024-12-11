namespace AiDotNet.LinearAlgebra;

public class ModelEvaluationService : IModelEvaluator
{
    public ModelEvaluationResult EvaluateModel(
        Vector<double> actualTrain, Vector<double> predictedTrain,
        Vector<double> actualVal, Vector<double> predictedVal,
        Vector<double> actualTest, Vector<double> predictedTest)
    {
        var result = new ModelEvaluationResult
        {
            TrainingMetrics = CalculateMetrics(actualTrain, predictedTrain),
            ValidationMetrics = CalculateMetrics(actualVal, predictedVal),
            TestMetrics = CalculateMetrics(actualTest, predictedTest)
        };

        return result;
    }

    public Dictionary<string, double> CalculateMetrics(Vector<double> actual, Vector<double> predicted)
    {
        var basicStats = new BasicStats(actual, predicted);
        var errorStats = new ErrorStats(actual, predicted);

        return CombineMetrics(basicStats, errorStats);
    }

    private Dictionary<string, double> CombineMetrics(BasicStats basicStats, ErrorStats errorStats)
    {
        var combinedMetrics = new Dictionary<string, double>
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
            ["R2"] = errorStats.R2,
            ["AdjustedR2"] = errorStats.AdjustedR2,
            ["MAPE"] = errorStats.MAPE
        };

        return combinedMetrics;
    }
}