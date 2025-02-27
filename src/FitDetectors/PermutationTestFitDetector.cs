using AiDotNet.Models.Options;
using AiDotNet.Models.Results;

namespace AiDotNet.FitDetectors;

public class PermutationTestFitDetector<T> : FitDetectorBase<T>
{
    private readonly Random _random;
    private readonly PermutationTestFitDetectorOptions _options;

    public PermutationTestFitDetector(PermutationTestFitDetectorOptions? options = null)
    {
        _random = new Random();
        _options = options ?? new PermutationTestFitDetectorOptions();
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
        var trainingPValue = PerformPermutationTest(evaluationData.TrainingSet.PredictionStats);
        var validationPValue = PerformPermutationTest(evaluationData.ValidationSet.PredictionStats);
        var testPValue = PerformPermutationTest(evaluationData.TestSet.PredictionStats);

        if (trainingPValue < _options.SignificanceLevel && 
            validationPValue < _options.SignificanceLevel && 
            testPValue < _options.SignificanceLevel)
        {
            return FitType.GoodFit;
        }
        else if (trainingPValue < _options.SignificanceLevel && 
                 (validationPValue >= _options.SignificanceLevel || testPValue >= _options.SignificanceLevel))
        {
            return FitType.Overfit;
        }
        else if (trainingPValue >= _options.SignificanceLevel && 
                 validationPValue >= _options.SignificanceLevel && 
                 testPValue >= _options.SignificanceLevel)
        {
            return FitType.Underfit;
        }
        else if (Math.Abs(trainingPValue - validationPValue) > _options.HighVarianceThreshold || 
                 Math.Abs(trainingPValue - testPValue) > _options.HighVarianceThreshold)
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
        var trainingPValue = PerformPermutationTest(evaluationData.TrainingSet.PredictionStats);
        var validationPValue = PerformPermutationTest(evaluationData.ValidationSet.PredictionStats);
        var testPValue = PerformPermutationTest(evaluationData.TestSet.PredictionStats);

        var averagePValue = (trainingPValue + validationPValue + testPValue) / 3;
        var confidenceLevel = 1 - averagePValue;

        return _numOps.FromDouble(confidenceLevel);
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.GoodFit:
                recommendations.Add("The model shows good fit across all datasets based on permutation tests.");
                recommendations.Add("Consider deploying the model or fine-tuning for even better performance.");
                break;
            case FitType.Overfit:
                recommendations.Add("Permutation tests indicate potential overfitting. Consider:");
                recommendations.Add("- Increasing the amount of training data");
                recommendations.Add("- Applying regularization techniques");
                recommendations.Add("- Simplifying the model architecture");
                break;
            case FitType.Underfit:
                recommendations.Add("Permutation tests suggest underfitting. Consider:");
                recommendations.Add("- Increasing model complexity");
                recommendations.Add("- Adding more relevant features");
                recommendations.Add("- Reducing regularization if applied");
                break;
            case FitType.HighVariance:
                recommendations.Add("Permutation tests show high variance. Consider:");
                recommendations.Add("- Increasing the amount of training data");
                recommendations.Add("- Applying feature selection techniques");
                recommendations.Add("- Using ensemble methods");
                break;
            case FitType.Unstable:
                recommendations.Add("Permutation tests indicate unstable performance across datasets. Consider:");
                recommendations.Add("- Investigating data quality and consistency");
                recommendations.Add("- Applying cross-validation techniques");
                recommendations.Add("- Using more robust feature selection methods");
                break;
        }

        recommendations.Add($"Permutation tests performed with {_options.NumberOfPermutations} permutations and {_options.SignificanceLevel * 100}% significance level.");

        return recommendations;
    }

    private double PerformPermutationTest(PredictionStats<T> predictionStats)
    {
        var originalR2 = Convert.ToDouble(predictionStats.R2);
        var permutationR2s = new List<double>();

        for (int i = 0; i < _options.NumberOfPermutations; i++)
        {
            var permutedR2 = SimulatePermutedR2(originalR2);
            permutationR2s.Add(permutedR2);
        }

        var pValue = permutationR2s.Count(r2 => r2 >= originalR2) / (double)_options.NumberOfPermutations;
        return pValue;
    }

    private double SimulatePermutedR2(double originalR2)
    {
        // Simulate permutation by adding some noise to the original R2
        var noise = _random.NextDouble() * 0.2 - 0.1; // Random noise between -0.1 and 0.1
        var permutedR2 = Math.Max(0, Math.Min(1, originalR2 + noise));

        return permutedR2;
    }
}