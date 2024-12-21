namespace AiDotNet.FitDetectors;

public class ShapleyValueFitDetector<T> : FitDetectorBase<T>
{
    private readonly ShapleyValueFitDetectorOptions _options;
    private readonly Random _random;

    public ShapleyValueFitDetector(ShapleyValueFitDetectorOptions options)
    {
        _options = options ?? new ShapleyValueFitDetectorOptions();
        _random = new Random();
    }

    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T> evaluationData)
    {
        var features = GetFeatures(evaluationData);
        var shapleyValues = CalculateShapleyValues(evaluationData, features);
        var fitType = DetermineFitType(shapleyValues);
        var confidenceLevel = CalculateConfidenceLevel(evaluationData);
        var recommendations = GenerateRecommendations(fitType, shapleyValues);
        var shapleyValuesStrings = shapleyValues.ToDictionary(
            kvp => kvp.Key,
            kvp => kvp.Value
        );

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ShapleyValues", shapleyValuesStrings }
            }
        };
    }

    protected override FitType DetermineFitType(ModelEvaluationData<T> evaluationData)
    {
        var features = GetFeatures(evaluationData);
        var shapleyValues = CalculateShapleyValues(evaluationData, features);
        return DetermineFitType(shapleyValues);
    }

    private FitType DetermineFitType(Dictionary<string, T> shapleyValues)
    {
        var sortedValues = shapleyValues.OrderByDescending(kv => kv.Value).ToList();
        var totalImportance = sortedValues.Aggregate(_numOps.Zero, (acc, kv) => _numOps.Add(acc, kv.Value));
        var cumulativeImportance = _numOps.Zero;
        var featureCount = 0;

        foreach (var kv in sortedValues)
        {
            cumulativeImportance = _numOps.Add(cumulativeImportance, kv.Value);
            featureCount++;

            if (_numOps.GreaterThanOrEquals(
                _numOps.Divide(cumulativeImportance, totalImportance),
                _numOps.FromDouble(_options.ImportanceThreshold)))
            {
                break;
            }
        }

        var importantFeatureRatio = _numOps.Divide(_numOps.FromDouble(featureCount), _numOps.FromDouble(shapleyValues.Count));

        if (_numOps.LessThanOrEquals(importantFeatureRatio, _numOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (_numOps.GreaterThanOrEquals(importantFeatureRatio, _numOps.FromDouble(_options.UnderfitThreshold)))
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
        var features = GetFeatures(evaluationData);
        var shapleyValues = CalculateShapleyValues(evaluationData, features);
        var sortedValues = shapleyValues.Values.OrderByDescending(v => v).ToList();
        var totalImportance = sortedValues.Aggregate(_numOps.Zero, (acc, v) => _numOps.Add(acc, v));
        var cumulativeImportance = _numOps.Zero;
        var featureCount = 0;

        foreach (var value in sortedValues)
        {
            cumulativeImportance = _numOps.Add(cumulativeImportance, value);
            featureCount++;

            if (_numOps.GreaterThanOrEquals(
                _numOps.Divide(cumulativeImportance, totalImportance),
                _numOps.FromDouble(_options.ImportanceThreshold)))
            {
                break;
            }
        }

        return _numOps.Subtract(
            _numOps.One,
            _numOps.Divide(
                _numOps.FromDouble(featureCount),
                _numOps.FromDouble(shapleyValues.Count)
            )
        );
    }

    private Dictionary<string, T> CalculateShapleyValues(ModelEvaluationData<T> evaluationData, List<string> features)
    {
        var shapleyValues = new Dictionary<string, T>();
        var n = features.Count;

        foreach (var feature in features)
        {
            T shapleyValue = _numOps.Zero;

            for (int i = 0; i < _options.MonteCarloSamples; i++)
            {
                var permutation = features.OrderBy(x => _random.Next()).ToList();
                var index = permutation.IndexOf(feature);
                var withFeature = new HashSet<string>(permutation.Take(index + 1));
                var withoutFeature = new HashSet<string>(permutation.Take(index));

                var marginalContribution = _numOps.Subtract(
                    CalculatePerformance(evaluationData, withFeature),
                    CalculatePerformance(evaluationData, withoutFeature));

                shapleyValue = _numOps.Add(shapleyValue, marginalContribution);
            }

            shapleyValues[feature] = _numOps.Divide(shapleyValue, _numOps.FromDouble(_options.MonteCarloSamples));
        }

        return shapleyValues;
    }

    private List<string> GetFeatures(ModelEvaluationData<T> evaluationData)
    {
        return evaluationData.ModelStats.FeatureNames;
    }

    private T CalculatePerformance(ModelEvaluationData<T> evaluationData, HashSet<string> features)
    {
        var subsetFeatures = evaluationData.ModelStats.FeatureValues
            .Where(kv => features.Contains(kv.Key))
            .ToDictionary(kv => kv.Key, kv => kv.Value);

        var featureMatrix = CreateFeatureMatrix(subsetFeatures);
        var predictions = evaluationData.ModelStats.Model?.Predict(featureMatrix) ?? Vector<T>.Empty();
        return StatisticsHelper<T>.CalculateR2(evaluationData.ModelStats.Actual, predictions);
    }

    private Matrix<T> CreateFeatureMatrix(Dictionary<string, Vector<T>> features)
    {
        int rowCount = features.First().Value.Length;
        int colCount = features.Count;
        var matrix = new Matrix<T>(rowCount, colCount);

        int colIndex = 0;
        foreach (var feature in features.Values)
        {
            for (int i = 0; i < rowCount; i++)
            {
                matrix[i, colIndex] = feature[i];
            }
            colIndex++;
        }

        return matrix;
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var features = GetFeatures(evaluationData);
        var shapleyValues = CalculateShapleyValues(evaluationData, features);
        return GenerateRecommendations(fitType, shapleyValues);
    }

    private List<string> GenerateRecommendations(FitType fitType, Dictionary<string, T> shapleyValues)
    {
        var recommendations = new List<string>();
        var sortedFeatures = shapleyValues.OrderByDescending(kv => kv.Value).ToList();

        switch (fitType)
        {
            case FitType.Overfit:
                recommendations.Add("The model may be overfitting. Consider the following:");
                recommendations.Add("1. Increase regularization strength.");
                recommendations.Add("2. Reduce model complexity.");
                recommendations.Add("3. Gather more training data.");
                recommendations.Add("4. Consider removing less important features:");
                for (int i = sortedFeatures.Count - 1; i >= Math.Max(0, sortedFeatures.Count - 5); i--)
                {
                    recommendations.Add($"   - {sortedFeatures[i].Key}");
                }
                break;

            case FitType.Underfit:
                recommendations.Add("The model may be underfitting. Consider the following:");
                recommendations.Add("1. Increase model complexity.");
                recommendations.Add("2. Reduce regularization strength.");
                recommendations.Add("3. Add more relevant features.");
                recommendations.Add("4. Engineer new features based on domain knowledge.");
                break;

            case FitType.GoodFit:
                recommendations.Add("The model appears to have a good fit. Consider the following:");
                recommendations.Add("1. Fine-tune hyperparameters for potential improvements.");
                recommendations.Add("2. Validate the model on new, unseen data.");
                recommendations.Add("3. Monitor model performance over time for potential drift.");
                break;
        }

        recommendations.Add("Top 5 most important features:");
        for (int i = 0; i < Math.Min(5, sortedFeatures.Count); i++)
        {
            recommendations.Add($"   - {sortedFeatures[i].Key}");
        }

        return recommendations;
    }
}