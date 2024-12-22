using AiDotNet.Models.Options;

namespace AiDotNet.FitDetectors;

public class EnsembleFitDetector<T> : FitDetectorBase<T>
{
    private readonly List<IFitDetector<T>> _detectors;
    private readonly EnsembleFitDetectorOptions _options;

    public EnsembleFitDetector(List<IFitDetector<T>> detectors, EnsembleFitDetectorOptions? options = null)
    {
        _detectors = detectors ?? throw new ArgumentNullException(nameof(detectors));
        if (_detectors.Count == 0)
            throw new ArgumentException("At least one detector must be provided.", nameof(detectors));
        _options = options ?? new EnsembleFitDetectorOptions();
    }

    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T> evaluationData)
    {
        if (evaluationData == null)
            throw new ArgumentNullException(nameof(evaluationData));

        var results = _detectors.Select(d => d.DetectFit(evaluationData)).ToList();

        var fitType = DetermineFitType(evaluationData);
        var confidenceLevel = CalculateConfidenceLevel(evaluationData);
        var recommendations = GenerateRecommendations(fitType, evaluationData);

        return new FitDetectorResult<T>
        {
            FitType = fitType,
            ConfidenceLevel = confidenceLevel,
            Recommendations = recommendations,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "IndividualResults", results },
                { "DetectorWeights", _options.DetectorWeights }
            }
        };
    }

    protected override FitType DetermineFitType(ModelEvaluationData<T> evaluationData)
    {
        var weightedFitTypes = _detectors.Select((d, i) =>
        {
            var result = d.DetectFit(evaluationData);
            var weight = i < _options.DetectorWeights.Count ? _options.DetectorWeights[i] : 1.0;
            return (result.FitType, Weight: weight);
        }).ToList();

        var totalWeight = weightedFitTypes.Sum(wft => wft.Weight);
        var weightedSum = weightedFitTypes.Sum(wft => (int)wft.FitType * wft.Weight);

        var averageFitType = weightedSum / totalWeight;

        if (averageFitType <= 1.5)
            return FitType.VeryPoorFit;
        else if (averageFitType <= 2.5)
            return FitType.PoorFit;
        else if (averageFitType <= 3.5)
            return FitType.Moderate;
        else
            return FitType.GoodFit;
    }

    protected override T CalculateConfidenceLevel(ModelEvaluationData<T> evaluationData)
    {
        var weightedConfidences = _detectors.Select((d, i) =>
        {
            var result = d.DetectFit(evaluationData);
            var weight = i < _options.DetectorWeights.Count ? _options.DetectorWeights[i] : 1.0;
            return _numOps.Multiply(result.ConfidenceLevel ?? _numOps.Zero, _numOps.FromDouble(weight));
        }).ToList();

        var totalWeight = _numOps.FromDouble(_options.DetectorWeights.Sum());
        var sumConfidence = weightedConfidences.Aggregate(_numOps.Zero, _numOps.Add);

        return _numOps.Divide(sumConfidence, totalWeight);
    }

    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T> evaluationData)
    {
        var recommendations = new HashSet<string>();

        foreach (var detector in _detectors)
        {
            var result = detector.DetectFit(evaluationData);
            recommendations.UnionWith(result.Recommendations);
        }

        var generalRecommendation = fitType switch
        {
            FitType.GoodFit => "The ensemble of detectors suggests a good fit. Consider fine-tuning for potential improvements.",
            FitType.Moderate => "The ensemble indicates moderate performance. Review individual detector results for specific areas of improvement.",
            FitType.PoorFit => "The ensemble suggests poor fit. Carefully analyze each detector's recommendations and consider significant model changes.",
            FitType.VeryPoorFit => "The ensemble indicates very poor fit. Reassess your approach, including data quality, feature selection, and model choice.",
            _ => throw new ArgumentOutOfRangeException(nameof(fitType))
        };

        recommendations.Add(generalRecommendation);

        if (recommendations.Count > _options.MaxRecommendations)
        {
            return [.. recommendations.Take(_options.MaxRecommendations)];
        }

        return [.. recommendations];
    }
}