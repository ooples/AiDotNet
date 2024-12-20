namespace AiDotNet.FitDetectors;

public class BootstrapFitDetector<T> : FitDetectorBase<T>
{
    private readonly BootstrapFitDetectorOptions _options;
    private readonly Random _random;

    public BootstrapFitDetector(BootstrapFitDetectorOptions? options = null)
    {
        _options = options ?? new BootstrapFitDetectorOptions();
        _random = new Random();
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
        var bootstrapResults = PerformBootstrap(evaluationData.TrainingPredictionStats, evaluationData.ValidationPredictionStats, evaluationData.TestPredictionStats);

        var meanTrainingR2 = new Vector<T>(bootstrapResults.Select(r => r.TrainingR2)).Average();
        var meanValidationR2 = new Vector<T>(bootstrapResults.Select(r => r.ValidationR2)).Average();
        var meanTestR2 = new Vector<T>(bootstrapResults.Select(r => r.TestR2)).Average();

        var r2Difference = _numOps.Abs(_numOps.Subtract(meanTrainingR2, meanValidationR2));

        if (_numOps.GreaterThan(meanTrainingR2, _numOps.FromDouble(_options.GoodFitThreshold)) &&
            _numOps.GreaterThan(meanValidationR2, _numOps.FromDouble(_options.GoodFitThreshold)) &&
            _numOps.GreaterThan(meanTestR2, _numOps.FromDouble(_options.GoodFitThreshold)))
        {
            return FitType.Good;
        }
        else if (_numOps.GreaterThan(r2Difference, _numOps.FromDouble(_options.OverfitThreshold)) &&
                 _numOps.GreaterThan(meanTrainingR2, meanValidationR2))
        {
            return FitType.Overfit;
        }
        else if (_numOps.LessThan(meanTrainingR2, _numOps.FromDouble(_options.UnderfitThreshold)) &&
                 _numOps.LessThan(meanValidationR2, _numOps.FromDouble(_options.UnderfitThreshold)) &&
                 _numOps.LessThan(meanTestR2, _numOps.FromDouble(_options.UnderfitThreshold)))
        {
            return FitType.Underfit;
        }
        else if (_numOps.GreaterThan(r2Difference, _numOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.HighVariance;
        }
        else
        {
            return FitType.Unstable;
        }
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var bootstrapResults = PerformBootstrap(evaluationData.TrainingPredictionStats, evaluationData.ValidationPredictionStats, evaluationData.TestPredictionStats);

        var r2Differences = bootstrapResults.Select(r => _numOps.Abs(_numOps.Subtract(r.TrainingR2, r.ValidationR2))).ToList();
        r2Differences.Sort();

        int lowerIndex = (int)Math.Floor((1 - _options.ConfidenceInterval) / 2 * _options.NumberOfBootstraps);
        int upperIndex = (int)Math.Ceiling((1 + _options.ConfidenceInterval) / 2 * _options.NumberOfBootstraps) - 1;

        var confidenceInterval = _numOps.Subtract(r2Differences[upperIndex], r2Differences[lowerIndex]);
        var confidenceLevel = _numOps.Subtract(_numOps.One, confidenceInterval);
        var lessThan = _numOps.LessThan(_numOps.One, confidenceLevel) ? _numOps.One : confidenceLevel;

        return _numOps.GreaterThan(_numOps.Zero, lessThan) ? _numOps.Zero : lessThan;
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.Good:
                recommendations.Add("The model shows good fit across all datasets based on bootstrap analysis.");
                recommendations.Add("Consider deploying the model or fine-tuning for even better performance.");
                break;
            case FitType.Overfit:
                recommendations.Add("Bootstrap analysis indicates potential overfitting. Consider:");
                recommendations.Add("- Increasing the amount of training data");
                recommendations.Add("- Applying regularization techniques");
                recommendations.Add("- Simplifying the model architecture");
                break;
            case FitType.Underfit:
                recommendations.Add("Bootstrap analysis suggests underfitting. Consider:");
                recommendations.Add("- Increasing model complexity");
                recommendations.Add("- Adding more relevant features");
                recommendations.Add("- Reducing regularization if applied");
                break;
            case FitType.HighVariance:
                recommendations.Add("Bootstrap analysis shows high variance. Consider:");
                recommendations.Add("- Increasing the amount of training data");
                recommendations.Add("- Applying feature selection techniques");
                recommendations.Add("- Using ensemble methods");
                break;
            case FitType.Unstable:
                recommendations.Add("Bootstrap analysis indicates unstable performance across datasets. Consider:");
                recommendations.Add("- Investigating data quality and consistency");
                recommendations.Add("- Applying cross-validation techniques");
                recommendations.Add("- Using more robust feature selection methods");
                break;
        }

        recommendations.Add($"Bootstrap analysis performed with {_options.NumberOfBootstraps} resamples and {_options.ConfidenceInterval * 100}% confidence interval.");

        return recommendations;
    }

    private List<BootstrapResult<T>> PerformBootstrap(
        PredictionStats<T> trainingPredictionStats,
        PredictionStats<T> validationPredictionStats,
        PredictionStats<T> testPredictionStats)
    {
        var results = new List<BootstrapResult<T>>();

        for (int i = 0; i < _options.NumberOfBootstraps; i++)
        {
            var bootstrapTrainingR2 = ResampleR2(trainingPredictionStats.R2);
            var bootstrapValidationR2 = ResampleR2(validationPredictionStats.R2);
            var bootstrapTestR2 = ResampleR2(testPredictionStats.R2);

            results.Add(new BootstrapResult<T>
            {
                TrainingR2 = bootstrapTrainingR2,
                ValidationR2 = bootstrapValidationR2,
                TestR2 = bootstrapTestR2
            });
        }

        return results;
    }

    private T ResampleR2(T originalR2)
    {
        // Simulate resampling by adding some noise to the original R2
        var noise = _random.NextDouble() * 0.1 - 0.05; // Random noise between -0.05 and 0.05
        var resampledR2 = Math.Max(0, Math.Min(1, Convert.ToDouble(originalR2) + noise));

        return _numOps.FromDouble(resampledR2);
    }
}