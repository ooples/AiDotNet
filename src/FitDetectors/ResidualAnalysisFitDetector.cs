using AiDotNet.Models.Results;

namespace AiDotNet.FitDetectors;

public class ResidualAnalysisFitDetector<T> : FitDetectorBase<T>
{
    private readonly ResidualAnalysisFitDetectorOptions _options;

    public ResidualAnalysisFitDetector(ResidualAnalysisFitDetectorOptions? options = null)
    {
        _options = options ?? new ResidualAnalysisFitDetectorOptions();
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
        // Check for autocorrelation using Durbin-Watson statistic
        if (_numOps.LessThan(evaluationData.TestSet.ErrorStats.DurbinWatsonStatistic, _numOps.FromDouble(1.5)) || 
            _numOps.GreaterThan(evaluationData.TestSet.ErrorStats.DurbinWatsonStatistic, _numOps.FromDouble(2.5)))
        {
            return FitType.Unstable;
        }

        // Check MAPE for overall fit
        if (_numOps.GreaterThan(evaluationData.TestSet.ErrorStats.MAPE, _numOps.FromDouble(_options.MapeThreshold)))
        {
            return FitType.Underfit;
        }

        // Analyze residuals across datasets
        var meanThreshold = _numOps.FromDouble(_options.MeanThreshold);
        var stdThreshold = _numOps.FromDouble(_options.StdThreshold);

        var trainingResidualMean = evaluationData.TrainingSet.ErrorStats.MeanBiasError;
        var validationResidualMean = evaluationData.ValidationSet.ErrorStats.MeanBiasError;
        var testResidualMean = evaluationData.TestSet.ErrorStats.MeanBiasError;

        var trainingResidualStd = evaluationData.TrainingSet.ErrorStats.PopulationStandardError;
        var validationResidualStd = evaluationData.ValidationSet.ErrorStats.PopulationStandardError;
        var testResidualStd = evaluationData.TestSet.ErrorStats.PopulationStandardError;

        if (_numOps.LessThan(_numOps.Abs(trainingResidualMean), meanThreshold) &&
            _numOps.LessThan(_numOps.Abs(validationResidualMean), meanThreshold) &&
            _numOps.LessThan(_numOps.Abs(testResidualMean), meanThreshold))
        {
            if (_numOps.LessThan(trainingResidualStd, stdThreshold) &&
                _numOps.LessThan(validationResidualStd, stdThreshold) &&
                _numOps.LessThan(testResidualStd, stdThreshold))
            {
                return FitType.GoodFit;
            }
            else
            {
                return FitType.HighVariance;
            }
        }
        else if (_numOps.GreaterThan(_numOps.Abs(trainingResidualMean), meanThreshold) &&
                 _numOps.GreaterThan(_numOps.Abs(validationResidualMean), meanThreshold) &&
                 _numOps.GreaterThan(_numOps.Abs(testResidualMean), meanThreshold))
        {
            return FitType.HighBias;
        }
        else if (_numOps.LessThan(_numOps.Abs(trainingResidualMean), meanThreshold) &&
                 _numOps.GreaterThan(_numOps.Abs(validationResidualMean), meanThreshold))
        {
            return FitType.Overfit;
        }

        // Check for significant differences in R-squared values
        var r2Threshold = _numOps.FromDouble(_options.R2Threshold);
        if (_numOps.GreaterThan(_numOps.Abs(_numOps.Subtract(evaluationData.TrainingSet.PredictionStats.R2, evaluationData.ValidationSet.PredictionStats.R2)), r2Threshold) ||
            _numOps.GreaterThan(_numOps.Abs(_numOps.Subtract(evaluationData.TrainingSet.PredictionStats.R2, evaluationData.TestSet.PredictionStats.R2)), r2Threshold))
        {
            return FitType.Unstable;
        }

        return FitType.GoodFit;
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var trainingConfidence = _numOps.Subtract(_numOps.One, _numOps.Divide(evaluationData.TrainingSet.ErrorStats.PopulationStandardError, evaluationData.TrainingSet.ErrorStats.MeanBiasError));
        var validationConfidence = _numOps.Subtract(_numOps.One, _numOps.Divide(evaluationData.ValidationSet.ErrorStats.PopulationStandardError, evaluationData.ValidationSet.ErrorStats.MeanBiasError));
        var testConfidence = _numOps.Subtract(_numOps.One, _numOps.Divide(evaluationData.TestSet.ErrorStats.PopulationStandardError, evaluationData.TestSet.ErrorStats.MeanBiasError));

        var averageConfidence = _numOps.Divide(_numOps.Add(_numOps.Add(trainingConfidence, validationConfidence), testConfidence), _numOps.FromDouble(3));

        // Adjust confidence based on R-squared values
        var r2Adjustment = _numOps.Divide(_numOps.Add(_numOps.Add(evaluationData.TrainingSet.PredictionStats.R2, evaluationData.ValidationSet.PredictionStats.R2), evaluationData.TestSet.PredictionStats.R2), _numOps.FromDouble(3));
        
        return _numOps.Multiply(averageConfidence, r2Adjustment);
    }
}