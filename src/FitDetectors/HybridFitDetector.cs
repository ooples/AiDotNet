namespace AiDotNet.FitDetectors;

public class HybridFitDetector<T> : FitDetectorBase<T>
{
    private readonly ResidualAnalysisFitDetector<T> _residualAnalyzer;
    private readonly LearningCurveFitDetector<T> _learningCurveDetector;
    private readonly HybridFitDetectorOptions _options;

    public HybridFitDetector(
        ResidualAnalysisFitDetector<T> residualAnalyzer,
        LearningCurveFitDetector<T> learningCurveDetector,
        HybridFitDetectorOptions? options = null)
    {
        _residualAnalyzer = residualAnalyzer;
        _learningCurveDetector = learningCurveDetector;
        _options = options ?? new HybridFitDetectorOptions();
    }

    public override FitDetectorResult<T> DetectFit(
        ErrorStats<T> trainingErrorStats,
        ErrorStats<T> validationErrorStats,
        ErrorStats<T> testErrorStats,
        BasicStats<T> trainingBasicStats,
        BasicStats<T> validationBasicStats,
        BasicStats<T> testBasicStats,
        BasicStats<T> trainingTargetStats,
        BasicStats<T> validationTargetStats,
        BasicStats<T> testTargetStats,
        PredictionStats<T> trainingPredictionStats,
        PredictionStats<T> validationPredictionStats,
        PredictionStats<T> testPredictionStats)
    {
        var residualResult = _residualAnalyzer.DetectFit(
            trainingErrorStats, validationErrorStats, testErrorStats,
            trainingBasicStats, validationBasicStats, testBasicStats,
            trainingTargetStats, validationTargetStats, testTargetStats,
            trainingPredictionStats, validationPredictionStats, testPredictionStats);

        var learningCurveResult = _learningCurveDetector.DetectFit(
            trainingErrorStats, validationErrorStats, testErrorStats,
            trainingBasicStats, validationBasicStats, testBasicStats,
            trainingTargetStats, validationTargetStats, testTargetStats,
            trainingPredictionStats, validationPredictionStats, testPredictionStats);

        var hybridFitType = CombineFitTypes(residualResult.FitType, learningCurveResult.FitType);
        var hybridConfidence = CombineConfidenceLevels(residualResult.ConfidenceLevel ?? _numOps.Zero, learningCurveResult.ConfidenceLevel ?? _numOps.Zero);

        var recommendations = new List<string>();
        recommendations.AddRange(residualResult.Recommendations);
        recommendations.AddRange(learningCurveResult.Recommendations);

        return new FitDetectorResult<T>
        {
            FitType = hybridFitType,
            ConfidenceLevel = hybridConfidence,
            Recommendations = recommendations
        };
    }

    protected override FitType DetermineFitType(
        ErrorStats<T> trainingErrorStats,
        ErrorStats<T> validationErrorStats,
        ErrorStats<T> testErrorStats,
        BasicStats<T> trainingBasicStats,
        BasicStats<T> validationBasicStats,
        BasicStats<T> testBasicStats,
        BasicStats<T> trainingTargetStats,
        BasicStats<T> validationTargetStats,
        BasicStats<T> testTargetStats,
        PredictionStats<T> trainingPredictionStats,
        PredictionStats<T> validationPredictionStats,
        PredictionStats<T> testPredictionStats)
    {
        var residualFitType = _residualAnalyzer.DetectFit(
            trainingErrorStats, validationErrorStats, testErrorStats,
            trainingBasicStats, validationBasicStats, testBasicStats,
            trainingTargetStats, validationTargetStats, testTargetStats,
            trainingPredictionStats, validationPredictionStats, testPredictionStats).FitType;

        var learningCurveFitType = _learningCurveDetector.DetectFit(
            trainingErrorStats, validationErrorStats, testErrorStats,
            trainingBasicStats, validationBasicStats, testBasicStats,
            trainingTargetStats, validationTargetStats, testTargetStats,
            trainingPredictionStats, validationPredictionStats, testPredictionStats).FitType;

        return CombineFitTypes(residualFitType, learningCurveFitType);
    }

    protected override T CalculateConfidenceLevel(
        ErrorStats<T> trainingErrorStats,
        ErrorStats<T> validationErrorStats,
        ErrorStats<T> testErrorStats,
        BasicStats<T> trainingBasicStats,
        BasicStats<T> validationBasicStats,
        BasicStats<T> testBasicStats,
        BasicStats<T> trainingTargetStats,
        BasicStats<T> validationTargetStats,
        BasicStats<T> testTargetStats,
        PredictionStats<T> trainingPredictionStats,
        PredictionStats<T> validationPredictionStats,
        PredictionStats<T> testPredictionStats)
    {
        var residualConfidence = _residualAnalyzer.DetectFit(
            trainingErrorStats, validationErrorStats, testErrorStats,
            trainingBasicStats, validationBasicStats, testBasicStats,
            trainingTargetStats, validationTargetStats, testTargetStats,
            trainingPredictionStats, validationPredictionStats, testPredictionStats).ConfidenceLevel;

        var learningCurveConfidence = _learningCurveDetector.DetectFit(
            trainingErrorStats, validationErrorStats, testErrorStats,
            trainingBasicStats, validationBasicStats, testBasicStats,
            trainingTargetStats, validationTargetStats, testTargetStats,
            trainingPredictionStats, validationPredictionStats, testPredictionStats).ConfidenceLevel;

        return CombineConfidenceLevels(residualConfidence ?? _numOps.Zero, learningCurveConfidence ?? _numOps.Zero);
    }

    private FitType CombineFitTypes(FitType residualFitType, FitType learningCurveFitType)
    {
        var fitTypeScores = new Dictionary<FitType, int>
        {
            { FitType.Good, 0 },
            { FitType.Overfit, 0 },
            { FitType.Underfit, 0 },
            { FitType.HighVariance, 0 },
            { FitType.HighBias, 0 },
            { FitType.Unstable, 0 }
        };

        // Assign scores based on the severity and combination of fit types
        fitTypeScores[residualFitType] += 2;
        fitTypeScores[learningCurveFitType] += 2;

        // Additional rules for combining fit types
        if (residualFitType == FitType.Overfit && learningCurveFitType == FitType.HighVariance)
        {
            fitTypeScores[FitType.Overfit] += 1;
            fitTypeScores[FitType.HighVariance] += 1;
        }
        else if (residualFitType == FitType.Underfit && learningCurveFitType == FitType.HighBias)
        {
            fitTypeScores[FitType.Underfit] += 1;
            fitTypeScores[FitType.HighBias] += 1;
        }

        // If either detector indicates instability, increase the score for Unstable
        if (residualFitType == FitType.Unstable || learningCurveFitType == FitType.Unstable)
        {
            fitTypeScores[FitType.Unstable] += 3;
        }

        // Return the FitType with the highest score
        return fitTypeScores.OrderByDescending(kvp => kvp.Value).First().Key;
    }

    private T CombineConfidenceLevels(T residualConfidence, T learningCurveConfidence)
    {
        // Calculate the difference between confidence levels
        var confidenceDifference = _numOps.Abs(_numOps.Subtract(residualConfidence, learningCurveConfidence));

        // If the difference is small, use a simple average
        if (_numOps.LessThan(confidenceDifference, _numOps.FromDouble(0.2)))
        {
            return _numOps.Divide(_numOps.Add(residualConfidence, learningCurveConfidence), _numOps.FromDouble(2.0));
        }

        // If the difference is large, use a weighted average favoring the higher confidence
        var weight = _numOps.FromDouble(0.7); // 70% weight to the higher confidence
        if (_numOps.GreaterThan(residualConfidence, learningCurveConfidence))
        {
            return _numOps.Add(
                _numOps.Multiply(residualConfidence, weight),
                _numOps.Multiply(learningCurveConfidence, _numOps.Subtract(_numOps.FromDouble(1.0), weight))
            );
        }
        else
        {
            return _numOps.Add(
                _numOps.Multiply(learningCurveConfidence, weight),
                _numOps.Multiply(residualConfidence, _numOps.Subtract(_numOps.FromDouble(1.0), weight))
            );
        }
    }
}