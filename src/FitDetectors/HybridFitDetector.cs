namespace AiDotNet.FitDetectors;

/// <summary>
/// A detector that combines multiple fit detection approaches to provide more robust model evaluation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class combines two different ways of checking how well your model is performing:
/// 1. Residual analysis - which looks at the differences between predicted and actual values
/// 2. Learning curve analysis - which examines how your model performs as it sees more training data
/// 
/// By using both approaches together, this detector can give you more reliable insights about whether
/// your model is a good fit, overfitting, underfitting, or has other issues. Think of it like getting
/// a second opinion from another doctor - having multiple perspectives leads to better diagnosis.
/// </para>
/// </remarks>
public class HybridFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// The residual analysis component of the hybrid detector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This component analyzes the differences between your model's predictions and the actual values.
    /// It helps identify patterns in these differences (called residuals) that can reveal whether your model
    /// is missing important trends in the data (underfitting) or being too influenced by random noise (overfitting).
    /// </para>
    /// <para>
    /// Residual analysis is particularly good at detecting issues like:
    /// <list type="bullet">
    /// <item><description>Heteroscedasticity (when prediction errors vary widely across different input values)</description></item>
    /// <item><description>Systematic bias (when predictions are consistently too high or too low)</description></item>
    /// <item><description>Non-linearity (when the model misses curved patterns in the data)</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    private readonly ResidualAnalysisFitDetector<T, TInput, TOutput> _residualAnalyzer;

    /// <summary>
    /// The learning curve component of the hybrid detector.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This component examines how your model's performance changes as it sees more training data.
    /// Learning curves plot training and validation errors against the amount of training data used, revealing
    /// important patterns about how your model learns.
    /// </para>
    /// <para>
    /// Learning curve analysis is especially effective at detecting:
    /// <list type="bullet">
    /// <item><description>High variance (when training error is much lower than validation error, suggesting overfitting)</description></item>
    /// <item><description>High bias (when both training and validation errors are high, suggesting underfitting)</description></item>
    /// <item><description>Insufficient data (when errors are still decreasing as more data is added)</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    private readonly LearningCurveFitDetector<T, TInput, TOutput> _learningCurveDetector;

    /// <summary>
    /// Configuration options that control how the hybrid detector combines and weighs results from its components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These settings determine how the hybrid detector combines the results from the
    /// residual analysis and learning curve components. They control aspects like:
    /// <list type="bullet">
    /// <item><description>How much weight to give each component's assessment</description></item>
    /// <item><description>Thresholds for determining when to favor one component's opinion over the other</description></item>
    /// <item><description>How to resolve conflicts when the components disagree about the model's fit</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// The default options are designed to work well for most common scenarios, but advanced users can
    /// customize these settings to better suit specific modeling challenges.
    /// </para>
    /// </remarks>
    private readonly HybridFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the HybridFitDetector class.
    /// </summary>
    /// <param name="residualAnalyzer">The residual analysis detector component.</param>
    /// <param name="learningCurveDetector">The learning curve detector component.</param>
    /// <param name="options">Optional configuration settings for the hybrid detector.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the hybrid detector by combining two specialized detectors:
    /// - The residual analyzer looks at prediction errors (the difference between what your model predicts and the actual values)
    /// - The learning curve detector examines how your model's performance changes as it sees more training examples
    /// 
    /// You can also provide optional settings to customize how the hybrid detector works.
    /// </para>
    /// </remarks>
    public HybridFitDetector(
        ResidualAnalysisFitDetector<T, TInput, TOutput> residualAnalyzer,
        LearningCurveFitDetector<T, TInput, TOutput> learningCurveDetector,
        HybridFitDetectorOptions? options = null)
    {
        _residualAnalyzer = residualAnalyzer;
        _learningCurveDetector = learningCurveDetector;
        _options = options ?? new HybridFitDetectorOptions();
    }

    /// <summary>
    /// Detects the fit type of a model by combining results from residual analysis and learning curve analysis.
    /// </summary>
    /// <param name="evaluationData">The data used to evaluate the model's performance.</param>
    /// <returns>A result containing the determined fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method is the main function that analyzes your model and tells you how well it's performing.
    /// It works by:
    /// 1. Running both the residual analysis and learning curve detectors
    /// 2. Combining their results to determine the overall fit type (like "good fit" or "overfitting")
    /// 3. Calculating how confident it is in this assessment
    /// 4. Gathering recommendations from both detectors to help you improve your model
    /// 
    /// The result gives you a complete picture of your model's performance and specific steps you can take
    /// to make it better.
    /// </para>
    /// </remarks>
    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var residualResult = _residualAnalyzer.DetectFit(evaluationData);

        var learningCurveResult = _learningCurveDetector.DetectFit(evaluationData);

        var hybridFitType = CombineFitTypes(residualResult.FitType, learningCurveResult.FitType);
        var hybridConfidence = CombineConfidenceLevels(residualResult.ConfidenceLevel ?? NumOps.Zero, learningCurveResult.ConfidenceLevel ?? NumOps.Zero);

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

    /// <summary>
    /// Determines the fit type of a model by combining results from multiple detection methods.
    /// </summary>
    /// <param name="evaluationData">The data used to evaluate the model's performance.</param>
    /// <returns>The determined fit type of the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method focuses specifically on determining what category your model's performance
    /// falls into (like "good fit", "overfitting", etc.). It gets opinions from both the residual analyzer
    /// and learning curve detector, then combines them to make a final decision about your model's fit type.
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var residualFitType = _residualAnalyzer.DetectFit(evaluationData).FitType;

        var learningCurveFitType = _learningCurveDetector.DetectFit(evaluationData).FitType;

        return CombineFitTypes(residualFitType, learningCurveFitType);
    }

    /// <summary>
    /// Calculates the confidence level in the fit type determination by combining confidence levels from multiple detection methods.
    /// </summary>
    /// <param name="evaluationData">The data used to evaluate the model's performance.</param>
    /// <returns>A numeric value representing the confidence level in the fit determination.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how confident the detector is in its assessment of your model.
    /// It combines the confidence levels from both the residual analyzer and learning curve detector
    /// to give you an overall confidence score. A higher number means the detector is more certain
    /// about its conclusions regarding your model's performance.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var residualConfidence = _residualAnalyzer.DetectFit(evaluationData).ConfidenceLevel;

        var learningCurveConfidence = _learningCurveDetector.DetectFit(evaluationData).ConfidenceLevel;

        return CombineConfidenceLevels(residualConfidence ?? NumOps.Zero, learningCurveConfidence ?? NumOps.Zero);
    }

    /// <summary>
    /// Combines fit types from different detection methods into a single, consensus fit type.
    /// </summary>
    /// <param name="residualFitType">The fit type determined by residual analysis.</param>
    /// <param name="learningCurveFitType">The fit type determined by learning curve analysis.</param>
    /// <returns>The combined fit type that best represents the model's performance.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method acts like a referee when the two detectors have different opinions
    /// about your model's performance. It uses a scoring system to decide which fit type is most likely correct.
    /// 
    /// For example:
    /// - If both detectors say your model is overfitting, this method will also conclude overfitting
    /// - If one detector says overfitting and the other says high variance (which are related issues),
    ///   it might still conclude overfitting but with some consideration of high variance
    /// - If either detector thinks your model is unstable, this method will give extra weight to that concern
    /// 
    /// The goal is to give you the most accurate assessment of your model's performance by considering
    /// multiple perspectives.
    /// </para>
    /// </remarks>
    private static FitType CombineFitTypes(FitType residualFitType, FitType learningCurveFitType)
    {
        var fitTypeScores = new Dictionary<FitType, int>
        {
            { FitType.GoodFit, 0 },
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

    /// <summary>
    /// Combines confidence levels from different detection methods into a single, weighted confidence score.
    /// </summary>
    /// <param name="residualConfidence">The confidence level from residual analysis.</param>
    /// <param name="learningCurveConfidence">The confidence level from learning curve analysis.</param>
    /// <returns>A combined confidence level that represents the overall certainty in the fit determination.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method combines the confidence scores from our two different detectors into one
    /// final confidence score. Think of confidence as "how sure we are" about our assessment of your model.
    /// 
    /// The method works like this:
    /// 1. First, it checks how different the two confidence scores are from each other
    /// 2. If they're pretty close (less than 0.2 or 20% different), it just takes the average of both scores
    /// 3. If they're quite different, it gives more weight (70%) to the detector that's more confident
    ///    and less weight (30%) to the less confident detector
    /// 
    /// This approach is like asking two experts for their opinion - if they're equally confident,
    /// you might average their advice. But if one expert is much more confident than the other,
    /// you'd probably lean more toward the confident expert's opinion.
    /// </para>
    /// </remarks>
    private T CombineConfidenceLevels(T residualConfidence, T learningCurveConfidence)
    {
        // Calculate the difference between confidence levels
        var confidenceDifference = NumOps.Abs(NumOps.Subtract(residualConfidence, learningCurveConfidence));

        // If the difference is small, use a simple average
        if (NumOps.LessThan(confidenceDifference, NumOps.FromDouble(0.2)))
        {
            return NumOps.Divide(NumOps.Add(residualConfidence, learningCurveConfidence), NumOps.FromDouble(2.0));
        }

        // If the difference is large, use a weighted average favoring the higher confidence
        var weight = NumOps.FromDouble(0.7); // 70% weight to the higher confidence
        if (NumOps.GreaterThan(residualConfidence, learningCurveConfidence))
        {
            return NumOps.Add(
                NumOps.Multiply(residualConfidence, weight),
                NumOps.Multiply(learningCurveConfidence, NumOps.Subtract(NumOps.FromDouble(1.0), weight))
            );
        }
        else
        {
            return NumOps.Add(
                NumOps.Multiply(learningCurveConfidence, weight),
                NumOps.Multiply(residualConfidence, NumOps.Subtract(NumOps.FromDouble(1.0), weight))
            );
        }
    }
}
