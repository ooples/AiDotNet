using AiDotNet.Models.Options;
using AiDotNet.Models.Results;

namespace AiDotNet.FitDetectors;

public class FeatureImportanceFitDetector<T> : FitDetectorBase<T>
{
    private readonly FeatureImportanceFitDetectorOptions _options;
    private readonly Random _random;

    public FeatureImportanceFitDetector(FeatureImportanceFitDetectorOptions? options = null)
    {
        _options = options ?? new FeatureImportanceFitDetectorOptions();
        _random = new Random(_options.RandomSeed);
    }

    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T> evaluationData)
    {
        var fitType = DetermineFitType(evaluationData);
        var confidenceLevel = CalculateConfidenceLevel(evaluationData);
        var recommendations = GenerateRecommendations(fitType, evaluationData);

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations
        };
    }

    protected override FitType DetermineFitType(ModelEvaluationData<T> evaluationData)
    {
        var featureImportances = CalculateFeatureImportances(evaluationData);
        var averageImportance =featureImportances.Average();
        var importanceStdDev = StatisticsHelper<T>.CalculateStandardDeviation(featureImportances);
        var featureCorrelations = CalculateFeatureCorrelations(evaluationData.ModelStats.FeatureMatrix);

        if (_numOps.GreaterThan(averageImportance, _numOps.FromDouble(_options.HighImportanceThreshold)) &&
            _numOps.LessThan(importanceStdDev, _numOps.FromDouble(_options.LowVarianceThreshold)))
        {
            return FitType.GoodFit;
        }
        else if (_numOps.GreaterThan(averageImportance, _numOps.FromDouble(_options.HighImportanceThreshold)) &&
                 _numOps.GreaterThan(importanceStdDev, _numOps.FromDouble(_options.HighVarianceThreshold)))
        {
            return FitType.Overfit;
        }
        else if (_numOps.LessThan(averageImportance, _numOps.FromDouble(_options.LowImportanceThreshold)) ||
                 AreFeaturesMostlyUncorrelated(featureCorrelations))
        {
            return FitType.Underfit;
        }
        else
        {
            return FitType.Unstable;
        }
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var featureImportances = CalculateFeatureImportances(evaluationData);
        var averageImportance = featureImportances.Average();
        var importanceStdDev = StatisticsHelper<T>.CalculateStandardDeviation(featureImportances);
        var featureCorrelations = CalculateFeatureCorrelations(evaluationData.ModelStats.FeatureMatrix);

        var importanceFactor = _numOps.Divide(averageImportance, _numOps.FromDouble(_options.HighImportanceThreshold));
        var varianceFactor = _numOps.Divide(_numOps.FromDouble(_options.LowVarianceThreshold), _numOps.Add(_numOps.One, importanceStdDev));
        var correlationFactor = _numOps.Subtract(_numOps.FromDouble(1), AverageAbsoluteCorrelation(featureCorrelations));

        return _numOps.Multiply(_numOps.Multiply(importanceFactor, varianceFactor), correlationFactor);
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();
        var featureImportances = CalculateFeatureImportances(evaluationData);

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model appears to be well-fitted based on feature importances.");
                recommendations.Add("Consider fine-tuning hyperparameters for potential further improvements.");
                break;
            case FitType.Overfit:
                recommendations.Add("The model may be overfitting. Consider the following:");
                recommendations.Add("1. Implement regularization techniques.");
                recommendations.Add("2. Reduce model complexity or use simpler models.");
                recommendations.Add("3. Collect more training data if possible.");
                break;
            case FitType.Underfit:
                recommendations.Add("The model may be underfitting. Consider the following:");
                recommendations.Add("1. Increase model complexity or use more sophisticated models.");
                recommendations.Add("2. Feature engineering to create more informative features.");
                recommendations.Add("3. Collect more relevant features if possible.");
                break;
            case FitType.Unstable:
                recommendations.Add("The model fit is unstable. Consider the following:");
                recommendations.Add("1. Analyze feature correlations and remove highly correlated features.");
                recommendations.Add("2. Use feature selection techniques to identify the most relevant features.");
                recommendations.Add("3. Implement cross-validation to ensure model stability.");
                break;
        }

        recommendations.Add("Top 3 most important features:");
        var topFeatures = featureImportances
            .Select((importance, index) => new { Importance = importance, Index = index })
            .OrderByDescending(x => x.Importance)
            .Take(3);

        foreach (var feature in topFeatures)
        {
            recommendations.Add($"- Feature {feature.Index}: Importance = {feature.Importance:F4}");
        }

        return recommendations;
    }

    private Vector<T> CalculateFeatureImportances(ModelEvaluationData<T> evaluationData)
    {
        var baselineError = CalculateError(evaluationData.ModelStats.Actual, evaluationData.ModelStats.Predicted);
        var featureImportances = new Vector<T>(evaluationData.ModelStats.FeatureMatrix.Columns);

        for (int i = 0; i < evaluationData.ModelStats.FeatureMatrix.Columns; i++)
        {
            var permutedFeature = PermuteFeature(evaluationData.ModelStats.FeatureMatrix.GetColumn(i));
            var permutedMatrix = evaluationData.ModelStats.FeatureMatrix.Copy();
            permutedMatrix.SetColumn(i, permutedFeature);

            var permutedPredictions = evaluationData.ModelStats.Model?.Predict(permutedMatrix) ?? Vector<T>.Empty();
            var permutedError = CalculateError(evaluationData.ModelStats.Actual, permutedPredictions);

            featureImportances[i] = _numOps.Subtract(permutedError, baselineError);
        }

        return featureImportances;
    }

    private T CalculateError(Vector<T> actual, Vector<T> predicted)
    {
        return StatisticsHelper<T>.CalculateMeanSquaredError(actual, predicted);
    }

    private Vector<T> PermuteFeature(Vector<T> feature)
    {
        var permutedFeature = feature.Copy();
        int n = permutedFeature.Length;

        for (int i = n - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            T temp = permutedFeature[i];
            permutedFeature[i] = permutedFeature[j];
            permutedFeature[j] = temp;
        }

        return permutedFeature;
    }

    private Matrix<T> CalculateFeatureCorrelations(Matrix<T> features)
    {
        int numFeatures = features.Columns;
        var correlations = new Matrix<T>(numFeatures, numFeatures);

        for (int i = 0; i < numFeatures; i++)
        {
            for (int j = i; j < numFeatures; j++)
            {
                var correlation = StatisticsHelper<T>.CalculatePearsonCorrelation(features.GetColumn(i), features.GetColumn(j));
                correlations[i, j] = correlation;
                correlations[j, i] = correlation;
            }
        }

        return correlations;
    }

    private bool AreFeaturesMostlyUncorrelated(Matrix<T> correlations)
    {
        int numFeatures = correlations.Rows;
        int uncorrelatedCount = 0;

        for (int i = 0; i < numFeatures; i++)
        {
            for (int j = i + 1; j < numFeatures; j++)
            {
                if (_numOps.LessThan(_numOps.Abs(correlations[i, j]), _numOps.FromDouble(_options.CorrelationThreshold)))
                {
                    uncorrelatedCount++;
                }
            }
        }

        int totalPairs = (numFeatures * (numFeatures - 1)) / 2;
        return (double)uncorrelatedCount / totalPairs > _options.UncorrelatedRatioThreshold;
    }

    private T AverageAbsoluteCorrelation(Matrix<T> correlations)
    {
        int numFeatures = correlations.Rows;
        T sum = _numOps.Zero;
        int count = 0;

        for (int i = 0; i < numFeatures; i++)
        {
            for (int j = i + 1; j < numFeatures; j++)
            {
                sum = _numOps.Add(sum, _numOps.Abs(correlations[i, j]));
                count++;
            }
        }

        return _numOps.Divide(sum, _numOps.FromDouble(count));
    }
}