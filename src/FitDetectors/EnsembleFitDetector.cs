namespace AiDotNet.FitDetectors;

/// <summary>
/// A fit detector that combines the results of multiple individual fit detectors to provide a more robust assessment.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> An ensemble approach combines the opinions of multiple "experts" (in this case, 
/// different fit detectors) to make a more reliable decision. This is similar to getting second and third 
/// opinions from different doctors before making an important medical decision.
/// </para>
/// <para>
/// This detector aggregates the results from multiple fit detectors, potentially giving different weights 
/// to each detector's opinion, to determine the overall fit type, confidence level, and recommendations.
/// </para>
/// </remarks>
public class EnsembleFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// The list of individual fit detectors that make up the ensemble.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These are the different "expert opinions" that will be combined to form 
    /// the ensemble's assessment. Each detector may use different techniques to evaluate model fit.
    /// </remarks>
    private readonly List<IFitDetector<T, TInput, TOutput>> _detectors;

    /// <summary>
    /// Configuration options for the ensemble fit detector.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These settings control how the ensemble combines the results from individual 
    /// detectors, including the weights assigned to each detector and the maximum number of recommendations 
    /// to return.
    /// </remarks>
    private readonly EnsembleFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the EnsembleFitDetector class.
    /// </summary>
    /// <param name="detectors">The list of individual fit detectors to include in the ensemble.</param>
    /// <param name="options">Optional configuration options. If not provided, default options are used.</param>
    /// <exception cref="ArgumentNullException">Thrown when detectors is null.</exception>
    /// <exception cref="ArgumentException">Thrown when detectors is an empty list.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new ensemble fit detector with a specified list 
    /// of individual detectors and optional configuration settings.
    /// </para>
    /// <para>
    /// You must provide at least one detector to create an ensemble. The ensemble will combine the results 
    /// from all provided detectors to make its assessment.
    /// </para>
    /// <para>
    /// The options parameter allows you to customize how the ensemble works, including:
    /// <list type="bullet">
    /// <item><description>Detector weights: How much influence each detector has on the final result</description></item>
    /// <item><description>Maximum recommendations: The maximum number of recommendations to include in the result</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public EnsembleFitDetector(List<IFitDetector<T, TInput, TOutput>> detectors, EnsembleFitDetectorOptions? options = null)
    {
        Guard.NotNull(detectors);
        _detectors = detectors;
        if (_detectors.Count == 0)
            throw new ArgumentException("At least one detector must be provided.", nameof(detectors));
        _options = options ?? new EnsembleFitDetectorOptions();
    }

    /// <summary>
    /// Detects the fit type of a model by combining the results of multiple individual fit detectors.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A result object containing the ensemble's assessment of fit type, confidence level, recommendations, and additional information.</returns>
    /// <exception cref="ArgumentNullException">Thrown when evaluationData is null.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method runs each individual detector on the evaluation data, then 
    /// combines their results to form an ensemble assessment.
    /// </para>
    /// <para>
    /// The result includes:
    /// <list type="bullet">
    /// <item><description>FitType: The ensemble's assessment of the model's fit type</description></item>
    /// <item><description>ConfidenceLevel: The ensemble's confidence in its assessment</description></item>
    /// <item><description>Recommendations: A combined list of recommendations from all detectors</description></item>
    /// <item><description>AdditionalInfo: Contains the individual results from each detector and the weights used</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T, TInput, TOutput> evaluationData)
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

    /// <summary>
    /// Determines the fit type by combining the assessments of all individual detectors.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>The ensemble's assessment of the model's fit type.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates a weighted average of the fit types reported by 
    /// all individual detectors, then maps this average to a final fit type.
    /// </para>
    /// <para>
    /// The process works as follows:
    /// <list type="number">
    /// <item><description>Run each detector and get its fit type assessment</description></item>
    /// <item><description>Apply the corresponding weight to each detector's assessment</description></item>
    /// <item><description>Calculate the weighted average of all assessments</description></item>
    /// <item><description>Map this average to a final fit type category</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The mapping from average to fit type uses these thresholds (lower enum values indicate better fit):
    /// <list type="bullet">
    /// <item><description>&lt;= 0.5: Good Fit (GoodFit=0)</description></item>
    /// <item><description>&lt;= 2.5: Moderate Fit</description></item>
    /// <item><description>&lt;= 5.5: Poor Fit</description></item>
    /// <item><description>&gt; 5.5: Very Poor Fit</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
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

        // FitType enum: GoodFit=0, Overfit=1, Underfit=2, etc.
        // Lower values indicate better fit, so thresholds should reflect this
        if (averageFitType <= 0.5)
            return FitType.GoodFit;
        else if (averageFitType <= 2.5)
            return FitType.Moderate;
        else if (averageFitType <= 5.5)
            return FitType.PoorFit;
        else
            return FitType.VeryPoorFit;
    }

    /// <summary>
    /// Calculates the confidence level by combining the confidence levels of all individual detectors.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>The ensemble's confidence level in its assessment.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates a weighted average of the confidence levels reported 
    /// by all individual detectors.
    /// </para>
    /// <para>
    /// The process works as follows:
    /// <list type="number">
    /// <item><description>Run each detector and get its confidence level</description></item>
    /// <item><description>Apply the corresponding weight to each detector's confidence level</description></item>
    /// <item><description>Calculate the weighted average of all confidence levels</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The resulting confidence level is a value between 0 and 1, with higher values indicating greater 
    /// confidence in the ensemble's assessment.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        double totalWeight = 0.0;
        var weightedConfidences = _detectors.Select((d, i) =>
        {
            var result = d.DetectFit(evaluationData);
            var weight = i < _options.DetectorWeights.Count ? _options.DetectorWeights[i] : 1.0;
            totalWeight += weight;
            return NumOps.Multiply(result.ConfidenceLevel ?? NumOps.Zero, NumOps.FromDouble(weight));
        }).ToList();

        // Guard against division by zero if no detectors or all weights are zero
        if (totalWeight <= 0.0)
        {
            return NumOps.FromDouble(0.5); // Return neutral confidence
        }

        var sumConfidence = weightedConfidences.Aggregate(NumOps.Zero, NumOps.Add);
        var avgConfidence = NumOps.Divide(sumConfidence, NumOps.FromDouble(totalWeight));

        // Clamp confidence to [0, 1] range
        var zero = NumOps.Zero;
        var one = NumOps.One;
        if (NumOps.LessThan(avgConfidence, zero))
        {
            return zero;
        }
        if (NumOps.GreaterThan(avgConfidence, one))
        {
            return one;
        }

        return avgConfidence;
    }

    /// <summary>
    /// Generates recommendations by combining the recommendations of all individual detectors.
    /// </summary>
    /// <param name="fitType">The ensemble's assessment of the model's fit type.</param>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A list of recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method collects recommendations from all individual detectors, 
    /// removes duplicates, adds a general recommendation based on the ensemble's fit type assessment, 
    /// and limits the total number of recommendations if necessary.
    /// </para>
    /// <para>
    /// The process works as follows:
    /// <list type="number">
    /// <item><description>Run each detector and collect its recommendations</description></item>
    /// <item><description>Combine all recommendations into a single set (removing duplicates)</description></item>
    /// <item><description>Add a general recommendation based on the ensemble's fit type assessment</description></item>
    /// <item><description>Limit the number of recommendations to the maximum specified in the options</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The general recommendation provides an overall assessment and suggestion based on the ensemble's 
    /// fit type determination.
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
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
