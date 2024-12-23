using AiDotNet.Models.Options;
using AiDotNet.Models.Results;

namespace AiDotNet.FitDetectors;

public class AdaptiveFitDetector<T> : FitDetectorBase<T>
{
    private readonly ResidualAnalysisFitDetector<T> _residualAnalyzer;
    private readonly LearningCurveFitDetector<T> _learningCurveDetector;
    private readonly HybridFitDetector<T> _hybridDetector;
    private readonly AdaptiveFitDetectorOptions _options;

    public AdaptiveFitDetector(AdaptiveFitDetectorOptions? options = null)
    {
        _options = options ?? new AdaptiveFitDetectorOptions();
        _residualAnalyzer = new ResidualAnalysisFitDetector<T>(_options.ResidualAnalysisOptions);
        _learningCurveDetector = new LearningCurveFitDetector<T>(_options.LearningCurveOptions);
        _hybridDetector = new HybridFitDetector<T>(_residualAnalyzer, _learningCurveDetector, _options.HybridOptions);
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
        var dataComplexity = AssessDataComplexity(evaluationData.TrainingPredictedBasicStats, evaluationData.ValidationPredictedBasicStats, evaluationData.TestPredictedBasicStats);
        var modelPerformance = AssessModelPerformance(evaluationData.TrainingPredictionStats, evaluationData.ValidationPredictionStats, evaluationData.TestPredictionStats);

        FitDetectorResult<T> result;

        if (dataComplexity == DataComplexity.Simple && modelPerformance == ModelPerformance.Good)
        {
            result = _residualAnalyzer.DetectFit(evaluationData);
        }
        else if (dataComplexity == DataComplexity.Moderate || modelPerformance == ModelPerformance.Moderate)
        {
            result = _learningCurveDetector.DetectFit(evaluationData);
        }
        else
        {
            result = _hybridDetector.DetectFit(evaluationData);
        }

        return result.FitType;
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var dataComplexity = AssessDataComplexity(evaluationData.TrainingPredictedBasicStats, evaluationData.ValidationPredictedBasicStats, evaluationData.TestPredictedBasicStats);
        var modelPerformance = AssessModelPerformance(evaluationData.TrainingPredictionStats, evaluationData.ValidationPredictionStats, evaluationData.TestPredictionStats);

        FitDetectorResult<T> result;

        if (dataComplexity == DataComplexity.Simple && modelPerformance == ModelPerformance.Good)
        {
            result = _residualAnalyzer.DetectFit(evaluationData);
        }
        else if (dataComplexity == DataComplexity.Moderate || modelPerformance == ModelPerformance.Moderate)
        {
            result = _learningCurveDetector.DetectFit(evaluationData);
        }
        else
        {
            result = _hybridDetector.DetectFit(evaluationData);
        }

        return result.ConfidenceLevel ?? _numOps.Zero;
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new List<string>();
        var dataComplexity = AssessDataComplexity(evaluationData.TrainingPredictedBasicStats, evaluationData.ValidationPredictedBasicStats, evaluationData.TestPredictedBasicStats);
        var modelPerformance = AssessModelPerformance(evaluationData.TrainingPredictionStats, evaluationData.ValidationPredictionStats, evaluationData.TestPredictionStats);

        recommendations.Add(GetAdaptiveRecommendation(dataComplexity, modelPerformance));

        return recommendations;
    }

    private DataComplexity AssessDataComplexity(BasicStats<T> trainingStats, BasicStats<T> validationStats, BasicStats<T> testStats)
    {
        var overallVariance = _numOps.Add(_numOps.Add(trainingStats.Variance, validationStats.Variance), testStats.Variance);
        var threshold = _numOps.FromDouble(_options.ComplexityThreshold);

        if (_numOps.LessThan(overallVariance, threshold))
            return DataComplexity.Simple;
        else if (_numOps.LessThan(overallVariance, _numOps.Multiply(threshold, _numOps.FromDouble(2))))
            return DataComplexity.Moderate;
        else
            return DataComplexity.Complex;
    }

    private ModelPerformance AssessModelPerformance(PredictionStats<T> trainingStats, PredictionStats<T> validationStats, PredictionStats<T> testStats)
    {
        var averageR2 = _numOps.Divide(
            _numOps.Add(_numOps.Add(trainingStats.R2, validationStats.R2), testStats.R2),
            _numOps.FromDouble(3)
        );

        var threshold = _numOps.FromDouble(_options.PerformanceThreshold);

        if (_numOps.GreaterThan(averageR2, threshold))
            return ModelPerformance.Good;
        else if (_numOps.GreaterThan(averageR2, _numOps.Multiply(threshold, _numOps.FromDouble(0.5))))
            return ModelPerformance.Moderate;
        else
            return ModelPerformance.Poor;
    }

    private string GetAdaptiveRecommendation(DataComplexity dataComplexity, ModelPerformance modelPerformance)
    {
        return $"Based on data complexity ({dataComplexity}) and model performance ({modelPerformance}), " +
               $"the adaptive fit detector used the {GetUsedDetectorName(dataComplexity, modelPerformance)} for analysis. " +
               $"Consider {GetAdditionalRecommendation(dataComplexity, modelPerformance)}";
    }

    private string GetUsedDetectorName(DataComplexity dataComplexity, ModelPerformance modelPerformance)
    {
        if (dataComplexity == DataComplexity.Simple && modelPerformance == ModelPerformance.Good)
            return "Residual Analysis Detector";
        else if (dataComplexity == DataComplexity.Moderate || modelPerformance == ModelPerformance.Moderate)
            return "Learning Curve Detector";
        else
            return "Hybrid Detector";
    }

    private string GetAdditionalRecommendation(DataComplexity dataComplexity, ModelPerformance modelPerformance)
    {
        if (dataComplexity == DataComplexity.Complex && modelPerformance == ModelPerformance.Poor)
            return "using more advanced modeling techniques or feature engineering to handle the complex data and improve performance.";
        else if (dataComplexity == DataComplexity.Complex)
            return "exploring additional feature engineering techniques to better capture the complexity of the data.";
        else if (modelPerformance == ModelPerformance.Poor)
            return "trying different model architectures or hyperparameter tuning to improve performance.";
        else
            return "fine-tuning the model and monitoring its performance on new data.";
    }
}