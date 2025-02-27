using AiDotNet.Models.Options;
using AiDotNet.Models.Results;

namespace AiDotNet.FitDetectors;

public class JackknifeFitDetector<T> : FitDetectorBase<T>
{
    private readonly JackknifeFitDetectorOptions _options;

    public JackknifeFitDetector(JackknifeFitDetectorOptions? options = null)
    {
        _options = options ?? new JackknifeFitDetectorOptions();
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
        var jackknifeMSE = PerformJackknifeResampling(evaluationData);
        var originalMSE = evaluationData.TestSet.ErrorStats.MSE;

        var relativeDifference = _numOps.Divide(_numOps.Subtract(jackknifeMSE, originalMSE), originalMSE);

        if (_numOps.GreaterThan(relativeDifference, _numOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else if (_numOps.LessThan(relativeDifference, _numOps.Negate(_numOps.FromDouble(_options.UnderfitThreshold))))
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
        var jackknifeMSE = PerformJackknifeResampling(evaluationData);
        var originalMSE = evaluationData.TestSet.ErrorStats.MSE;

        var relativeDifference = _numOps.Abs(_numOps.Divide(_numOps.Subtract(jackknifeMSE, originalMSE), originalMSE));
        
        return _numOps.Subtract(_numOps.One, _numOps.LessThan(relativeDifference, _numOps.One) ? relativeDifference : _numOps.One);
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.Overfit:
                recommendations.Add("Consider increasing regularization or reducing model complexity.");
                recommendations.Add("Try collecting more training data if possible.");
                recommendations.Add("Implement feature selection to reduce the number of input variables.");
                break;
            case FitType.Underfit:
                recommendations.Add("Consider increasing model complexity or reducing regularization.");
                recommendations.Add("Explore additional relevant features that could improve model performance.");
                recommendations.Add("Increase the number of training iterations or epochs if applicable.");
                break;
            case FitType.GoodFit:
                recommendations.Add("The model appears to have a good fit based on jackknife resampling.");
                recommendations.Add("Continue monitoring performance on new, unseen data.");
                recommendations.Add("Consider ensemble methods to potentially improve performance further.");
                break;
        }

        return recommendations;
    }

    private T PerformJackknifeResampling(ModelEvaluationData<T> evaluationData)
    {
        var actual = evaluationData.ModelStats.Actual;
        var predicted = evaluationData.ModelStats.Predicted;
        var sampleSize = actual.Length;

        if (sampleSize < _options.MinSampleSize)
        {
            throw new InvalidOperationException($"Sample size ({sampleSize}) is too small for jackknife resampling. Minimum required: {_options.MinSampleSize}");
        }

        var jackknifeMSEs = new Vector<T>(Math.Min(sampleSize, _options.MaxIterations));

        for (int i = 0; i < Math.Min(sampleSize, _options.MaxIterations); i++)
        {
            var jackknifeSample = new Vector<T>(sampleSize - 1);
            var jackknifePredicted = new Vector<T>(sampleSize - 1);
            int index = 0;

            for (int j = 0; j < sampleSize; j++)
            {
                if (j != i)
                {
                    jackknifeSample[index] = actual[j];
                    jackknifePredicted[index] = predicted[j];
                    index++;
                }
            }

            var mse = StatisticsHelper<T>.CalculateMeanSquaredError(jackknifeSample, jackknifePredicted);
            jackknifeMSEs[i] = mse;
        }

        return StatisticsHelper<T>.CalculateMean(jackknifeMSEs);
    }
}