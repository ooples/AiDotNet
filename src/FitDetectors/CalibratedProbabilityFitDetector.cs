using AiDotNet.Models.Options;
using AiDotNet.Models.Results;

namespace AiDotNet.FitDetectors;

public class CalibratedProbabilityFitDetector<T> : FitDetectorBase<T>
{
    private readonly CalibratedProbabilityFitDetectorOptions _options;

    public CalibratedProbabilityFitDetector(CalibratedProbabilityFitDetectorOptions? options = null)
    {
        _options = options ?? new CalibratedProbabilityFitDetectorOptions();
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
        var (expectedCalibration, observedCalibration) = CalculateCalibration(evaluationData);

        var calibrationError = CalculateCalibrationError(expectedCalibration, observedCalibration);

        if (_numOps.LessThan(calibrationError, _numOps.FromDouble(_options.GoodFitThreshold)))
        {
            return FitType.GoodFit;
        }
        else if (_numOps.GreaterThan(calibrationError, _numOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else
        {
            return FitType.Underfit;
        }
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var (expectedCalibration, observedCalibration) = CalculateCalibration(evaluationData);

        var calibrationError = CalculateCalibrationError(expectedCalibration, observedCalibration);

        // Normalize confidence level to [0, 1]
        return _numOps.Subtract(_numOps.One, _numOps.Divide(calibrationError, _numOps.FromDouble(_options.MaxCalibrationError)));
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();

        switch (fitType)
        {
            case FitType.Overfit:
                recommendations.Add("The model appears to be overconfident. Consider the following:");
                recommendations.Add("1. Apply regularization techniques to reduce model complexity.");
                recommendations.Add("2. Use probability calibration methods like Platt scaling or isotonic regression.");
                recommendations.Add("3. Increase the diversity of your training data.");
                break;
            case FitType.Underfit:
                recommendations.Add("The model appears to be underconfident. Consider the following:");
                recommendations.Add("1. Increase model complexity by adding more features or using a more sophisticated algorithm.");
                recommendations.Add("2. Ensure you have enough training data to capture the complexity of the problem.");
                recommendations.Add("3. Review your feature engineering process to ensure important information isn't being lost.");
                break;
            case FitType.GoodFit:
                recommendations.Add("The model appears to be well-calibrated. Consider the following:");
                recommendations.Add("1. Continue monitoring the model's performance on new data.");
                recommendations.Add("2. Periodically retrain the model to maintain its calibration.");
                recommendations.Add("3. Consider ensemble methods to potentially improve performance further.");
                break;
        }

        return recommendations;
    }

    private (Vector<T>, Vector<T>) CalculateCalibration(ModelEvaluationData<T> evaluationData)
    {
        var predicted = evaluationData.ModelStats.Predicted;
        var actual = evaluationData.ModelStats.Actual;

        var numBins = _options.NumCalibrationBins;
        var binSize = _numOps.Divide(_numOps.One, _numOps.FromDouble(numBins));

        var expectedCalibration = new Vector<T>(numBins);
        var observedCalibration = new Vector<T>(numBins);

        for (int i = 0; i < numBins; i++)
        {
            var lowerBound = _numOps.Multiply(_numOps.FromDouble(i), binSize);
            var upperBound = _numOps.Multiply(_numOps.FromDouble(i + 1), binSize);

            var binIndices = predicted.Select((p, idx) => new { Prob = p, Index = idx })
                                      .Where(x => _numOps.GreaterThanOrEquals(x.Prob, lowerBound) && _numOps.LessThan(x.Prob, upperBound))
                                      .Select(x => _numOps.FromDouble(x.Index))
                                      .ToList();

            if (binIndices.Count > 0)
            {
                expectedCalibration[i] = _numOps.Divide(_numOps.Add(lowerBound, upperBound), _numOps.FromDouble(2));
    
                var sum = binIndices.Aggregate(_numOps.Zero, (acc, idx) => 
                    _numOps.Add(acc, actual[_numOps.ToInt32(idx)])
                );
    
                observedCalibration[i] = _numOps.Divide(
                    sum,
                    _numOps.FromDouble(binIndices.Count)
                );
            }
            else
            {
                expectedCalibration[i] = _numOps.Zero;
                observedCalibration[i] = _numOps.Zero;
            }
        }

        return (expectedCalibration, observedCalibration);
    }

    private T CalculateCalibrationError(Vector<T> expected, Vector<T> observed)
    {
        var squaredErrors = new Vector<T>(expected.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            var diff = _numOps.Subtract(expected[i], observed[i]);
            squaredErrors[i] = _numOps.Multiply(diff, diff);
        }

        return _numOps.Sqrt(_numOps.Divide(squaredErrors.Sum(), _numOps.FromDouble(expected.Length)));
    }
}