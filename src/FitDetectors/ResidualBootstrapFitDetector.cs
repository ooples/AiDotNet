using AiDotNet.Models.Results;

namespace AiDotNet.FitDetectors;

public class ResidualBootstrapFitDetector<T> : FitDetectorBase<T>
{
    private readonly ResidualBootstrapFitDetectorOptions _options;
    private readonly Random _random;

    public ResidualBootstrapFitDetector(ResidualBootstrapFitDetectorOptions? options = null)
    {
        _options = options ?? new ResidualBootstrapFitDetectorOptions();
        _random = new Random(_options.Seed ?? Environment.TickCount);
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
        var bootstrapMSEs = PerformResidualBootstrap(evaluationData);
        var originalMSE = evaluationData.TestSet.ErrorStats.MSE;

        var meanBootstrapMSE = StatisticsHelper<T>.CalculateMean(bootstrapMSEs);
        var stdDevBootstrapMSE = StatisticsHelper<T>.CalculateStandardDeviation(bootstrapMSEs);

        var zScore = _numOps.Divide(_numOps.Subtract(originalMSE, meanBootstrapMSE), stdDevBootstrapMSE);

        if (_numOps.GreaterThan(zScore, _numOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (_numOps.LessThan(zScore, _numOps.FromDouble(_options.UnderfitThreshold)))
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
        var bootstrapMSEs = PerformResidualBootstrap(evaluationData);
        var originalMSE = evaluationData.TestSet.ErrorStats.MSE;

        var meanBootstrapMSE = StatisticsHelper<T>.CalculateMean(bootstrapMSEs);
        var stdDevBootstrapMSE = StatisticsHelper<T>.CalculateStandardDeviation(bootstrapMSEs);

        var zScore = _numOps.Abs(_numOps.Divide(_numOps.Subtract(originalMSE, meanBootstrapMSE), stdDevBootstrapMSE));
        
        return _numOps.Subtract(_numOps.One, _numOps.Divide(zScore, _numOps.FromDouble(3.0))); // Normalize to [0, 1]
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.Overfit:
                recommendations.Add("Consider increasing regularization to reduce model complexity.");
                recommendations.Add("Try reducing the number of features or using feature selection techniques.");
                recommendations.Add("Increase the size of the training dataset if possible.");
                break;
            case FitType.Underfit:
                recommendations.Add("Consider increasing model complexity by adding more features or interactions.");
                recommendations.Add("Reduce regularization if it's being used.");
                recommendations.Add("Ensure that all relevant features are included in the model.");
                break;
            case FitType.GoodFit:
                recommendations.Add("The model appears to have a good fit based on residual bootstrap analysis.");
                recommendations.Add("Continue monitoring performance on new, unseen data.");
                recommendations.Add("Consider ensemble methods or advanced techniques to potentially improve performance further.");
                break;
        }

        return recommendations;
    }

    private Vector<T> PerformResidualBootstrap(ModelEvaluationData<T> evaluationData)
    {
        var actual = evaluationData.ModelStats.Actual;
        var predicted = evaluationData.ModelStats.Predicted;
        var sampleSize = actual.Length;

        if (sampleSize < _options.MinSampleSize)
        {
            throw new InvalidOperationException($"Sample size ({sampleSize}) is too small for residual bootstrap. Minimum required: {_options.MinSampleSize}");
        }

        var residuals = new Vector<T>(sampleSize);
        for (int i = 0; i < sampleSize; i++)
        {
            residuals[i] = _numOps.Subtract(actual[i], predicted[i]);
        }

        var bootstrapMSEs = new Vector<T>(_options.NumBootstrapSamples);

        for (int i = 0; i < _options.NumBootstrapSamples; i++)
        {
            var bootstrapSample = new Vector<T>(sampleSize);
            var bootstrapPredicted = new Vector<T>(sampleSize);

            for (int j = 0; j < sampleSize; j++)
            {
                int randomIndex = _random.Next(sampleSize);
                bootstrapSample[j] = _numOps.Add(predicted[j], residuals[randomIndex]);
                bootstrapPredicted[j] = predicted[j];
            }

            bootstrapMSEs[i] = StatisticsHelper<T>.CalculateMeanSquaredError(bootstrapSample, bootstrapPredicted);
        }

        return bootstrapMSEs;
    }
}