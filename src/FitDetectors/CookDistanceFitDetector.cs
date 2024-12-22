using AiDotNet.Models.Options;

namespace AiDotNet.FitDetectors;

public class CookDistanceFitDetector<T> : FitDetectorBase<T>
{
    private readonly CookDistanceFitDetectorOptions _options;

    public CookDistanceFitDetector(CookDistanceFitDetectorOptions? options = null)
    {
        _options = options ?? new CookDistanceFitDetectorOptions();
    }

    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T> evaluationData)
    {
        var cookDistances = CalculateCookDistances(evaluationData);
        var fitType = DetermineFitType(cookDistances);
        var confidenceLevel = CalculateConfidenceLevel(evaluationData);
        var recommendations = GenerateRecommendations(fitType, evaluationData);

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "CookDistances", cookDistances }
            }
        };
    }

    protected override FitType DetermineFitType(ModelEvaluationData<T> evaluationData)
    {
        var cookDistances = CalculateCookDistances(evaluationData);
        return DetermineFitType(cookDistances);
    }

    private FitType DetermineFitType(Vector<T> cookDistances)
    {
        var influentialPointsCount = cookDistances.Count(d => _numOps.GreaterThan(d, _numOps.FromDouble(_options.InfluentialThreshold)));
        var influentialRatio = _numOps.Divide(_numOps.FromDouble(influentialPointsCount), _numOps.FromDouble(cookDistances.Length));

        if (_numOps.GreaterThan(influentialRatio, _numOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (_numOps.LessThan(influentialRatio, _numOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else
        {
            return FitType.GoodFit;
        }
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var cookDistances = CalculateCookDistances(evaluationData);
        var influentialPointsCount = cookDistances.Count(d => _numOps.GreaterThan(d, _numOps.FromDouble(_options.InfluentialThreshold)));
        var influentialRatio = _numOps.Divide(_numOps.FromDouble(influentialPointsCount), _numOps.FromDouble(cookDistances.Length));

        // Normalize confidence level to be between 0 and 1
        return _numOps.Subtract(_numOps.One, influentialRatio);
    }

    private Vector<T> CalculateCookDistances(ModelEvaluationData<T> evaluationData)
    {
        var X = evaluationData.ModelStats.FeatureMatrix;
        var y = evaluationData.ModelStats.Actual;
        var yPredicted = evaluationData.ModelStats.Model?.Predict(X) ?? Vector<T>.Empty();
        var residuals = y.Subtract(yPredicted);

        var n = X.Rows;
        var p = X.Columns;
        var hatMatrix = X.Multiply(X.Transpose().Multiply(X).Inverse()).Multiply(X.Transpose());
        var mse = _numOps.Divide(residuals.DotProduct(residuals), _numOps.FromDouble(n - p));

        var cookDistances = new Vector<T>(n);
        for (int i = 0; i < n; i++)
        {
            var hii = hatMatrix[i, i];
            var ri = residuals[i];
            var di = _numOps.Divide(_numOps.Multiply(ri, ri), _numOps.Multiply(_numOps.FromDouble(p), mse));
            di = _numOps.Multiply(di, _numOps.Divide(_numOps.Multiply(hii, hii), _numOps.Multiply(_numOps.Subtract(_numOps.One, hii), _numOps.Subtract(_numOps.One, hii))));
            cookDistances[i] = di;
        }

        return cookDistances;
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var cookDistances = CalculateCookDistances(evaluationData);
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.Overfit:
                recommendations.Add("The model may be overfitting. Consider the following:");
                recommendations.Add("1. Investigate and potentially remove highly influential points.");
                recommendations.Add("2. Increase regularization strength.");
                recommendations.Add("3. Simplify the model by reducing the number of features or model complexity.");
                break;

            case FitType.Underfit:
                recommendations.Add("The model may be underfitting. Consider the following:");
                recommendations.Add("1. Increase model complexity or add more relevant features.");
                recommendations.Add("2. Reduce regularization strength.");
                recommendations.Add("3. Investigate if important predictors are missing from the model.");
                break;

            case FitType.GoodFit:
                recommendations.Add("The model appears to have a good fit. Consider the following:");
                recommendations.Add("1. Validate the model on new, unseen data.");
                recommendations.Add("2. Monitor model performance over time for potential drift.");
                recommendations.Add("3. Consider fine-tuning hyperparameters for potential improvements.");
                break;
        }

        recommendations.Add("Top 5 most influential points (index: Cook's Distance):");
        var sortedIndices = cookDistances.Select((value, index) => new { Value = value, Index = index })
                                         .OrderByDescending(x => x.Value)
                                         .Take(5)
                                         .ToList();
        foreach (var item in sortedIndices)
        {
            recommendations.Add($"   - {item.Index}: {item.Value}");
        }

        return recommendations;
    }
}