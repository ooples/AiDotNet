using AiDotNet.Models.Options;
using AiDotNet.Models.Results;

namespace AiDotNet.FitDetectors;

public class LearningCurveFitDetector<T> : FitDetectorBase<T>
{
    private readonly LearningCurveFitDetectorOptions _options;

    public LearningCurveFitDetector(LearningCurveFitDetectorOptions? options = null)
    {
        _options = options ?? new();
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
        var minDataPoints = _options.MinDataPoints;
        if (evaluationData.TrainingPredictionStats.LearningCurve.Count < minDataPoints || evaluationData.ValidationPredictionStats.LearningCurve.Count < minDataPoints)
        {
            return FitType.Unstable;
        }

        var trainingSlope = CalculateSlope(evaluationData.TrainingPredictionStats.LearningCurve);
        var validationSlope = CalculateSlope(evaluationData.ValidationPredictionStats.LearningCurve);
        var convergenceThreshold = _numOps.FromDouble(_options.ConvergenceThreshold);

        if (_numOps.LessThan(_numOps.Abs(trainingSlope), convergenceThreshold) &&
            _numOps.LessThan(_numOps.Abs(validationSlope), convergenceThreshold))
        {
            return FitType.GoodFit;
        }

        if (_numOps.LessThan(trainingSlope, _numOps.Zero) && _numOps.GreaterThan(validationSlope, _numOps.Zero))
        {
            return FitType.Overfit;
        }

        if (_numOps.GreaterThan(trainingSlope, _numOps.Zero) && _numOps.GreaterThan(validationSlope, _numOps.Zero))
        {
            return FitType.Underfit;
        }

        return FitType.Unstable;
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var trainingVariance = CalculateVariance(evaluationData.TrainingPredictionStats.LearningCurve);
        var validationVariance = CalculateVariance(evaluationData.ValidationPredictionStats.LearningCurve);

        var totalVariance = _numOps.Add(trainingVariance, validationVariance);
        return _numOps.Subtract(_numOps.One, _numOps.Divide(totalVariance, _numOps.FromDouble(2)));
    }

    private T CalculateSlope(List<T> curve)
    {
        if (curve.Count < 2)
            return _numOps.Zero;

        var x = Enumerable.Range(0, curve.Count).Select(i => _numOps.FromDouble(i)).ToList();
        var n = _numOps.FromDouble(curve.Count);

        var sumX = x.Aggregate(_numOps.Zero, (acc, val) => _numOps.Add(acc, val));
        var sumY = curve.Aggregate(_numOps.Zero, (acc, val) => _numOps.Add(acc, val));
        var sumXY = x.Zip(curve, (xi, yi) => _numOps.Multiply(xi, yi)).Aggregate(_numOps.Zero, (acc, val) => _numOps.Add(acc, val));
        var sumX2 = x.Select(xi => _numOps.Multiply(xi, xi)).Aggregate(_numOps.Zero, (acc, val) => _numOps.Add(acc, val));

        var numerator = _numOps.Subtract(_numOps.Multiply(n, sumXY), _numOps.Multiply(sumX, sumY));
        var denominator = _numOps.Subtract(_numOps.Multiply(n, sumX2), _numOps.Multiply(sumX, sumX));

        return _numOps.Divide(numerator, denominator);
    }

    private T CalculateVariance(List<T> curve)
    {
        var mean = curve.Aggregate(_numOps.Zero, (acc, val) => _numOps.Add(acc, val));
        mean = _numOps.Divide(mean, _numOps.FromDouble(curve.Count));

        var variance = curve.Select(x => _numOps.Multiply(_numOps.Subtract(x, mean), _numOps.Subtract(x, mean)))
                            .Aggregate(_numOps.Zero, (acc, val) => _numOps.Add(acc, val));

        return _numOps.Divide(variance, _numOps.FromDouble(curve.Count - 1));
    }
}