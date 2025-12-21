namespace AiDotNet.FitDetectors;

/// <summary>
/// A detector that evaluates model fit by analyzing learning curves from training and validation data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class helps determine if your machine learning model is a good fit for your data
/// by looking at "learning curves." Learning curves show how your model's performance improves as it
/// sees more training examples.
/// 
/// By comparing the trends in training and validation performance, this detector can identify common problems:
/// - Overfitting: When your model performs great on training data but poorly on new data
/// - Underfitting: When your model is too simple to capture important patterns in your data
/// - Good Fit: When your model has learned the underlying patterns without memorizing the training data
/// - Unstable: When your model's performance is inconsistent or unpredictable
/// </para>
/// </remarks>
public class LearningCurveFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the Learning Curve fit detector.
    /// </summary>
    private readonly LearningCurveFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the Learning Curve fit detector.
    /// </summary>
    /// <param name="options">Optional configuration settings for the detector. If not provided, default settings will be used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the constructor that creates a new detector object. You can customize how it works by providing options,
    /// or just use the default settings if you're not sure what to change.
    /// </para>
    /// </remarks>
    public LearningCurveFitDetector(LearningCurveFitDetectorOptions? options = null)
    {
        _options = options ?? new();
    }

    /// <summary>
    /// Analyzes model performance data to detect the type of fit and provide recommendations.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics from training and validation sets.</param>
    /// <returns>A result object containing the detected fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method you'll use to check if your model is working well.
    /// It examines how your model performed during training and tells you:
    /// 1. What type of fit your model has (good, overfit, underfit, etc.)
    /// 2. How confident the detector is about this assessment
    /// 3. Specific recommendations to improve your model
    /// </para>
    /// </remarks>
    public override FitDetectorResult<T> DetectFit(ModelEvaluationData<T, TInput, TOutput> evaluationData)
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

    /// <summary>
    /// Determines the type of fit based on learning curve analysis.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>The detected fit type (Overfit, Underfit, GoodFit, or Unstable).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method looks at the trends in your model's performance during training to determine if there are any problems.
    /// 
    /// It analyzes the "slopes" of your learning curves, which show how quickly your model is improving:
    /// - If both training and validation curves have flattened out (near-zero slopes), your model has a good fit
    /// - If training performance is decreasing (negative slope) while validation is increasing (positive slope), your model is overfitting
    /// - If both training and validation performance are still improving (positive slopes), your model might be underfitting
    /// - If there's not enough data or the patterns are inconsistent, your model is considered unstable
    /// 
    /// Terminology explained:
    /// - Learning curve: A graph showing how model performance changes as it sees more training examples
    /// - Slope: How steep the learning curve is at a particular point (positive slope means improving, negative means worsening)
    /// - Convergence: When the learning curve flattens out, indicating the model has stopped improving significantly
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var minDataPoints = _options.MinDataPoints;
        if (evaluationData.TrainingSet.PredictionStats.LearningCurve.Count < minDataPoints || evaluationData.ValidationSet.PredictionStats.LearningCurve.Count < minDataPoints)
        {
            return FitType.Unstable;
        }

        var trainingSlope = CalculateSlope(evaluationData.TrainingSet.PredictionStats.LearningCurve);
        var validationSlope = CalculateSlope(evaluationData.ValidationSet.PredictionStats.LearningCurve);
        var convergenceThreshold = NumOps.FromDouble(_options.ConvergenceThreshold);

        if (NumOps.LessThan(NumOps.Abs(trainingSlope), convergenceThreshold) &&
            NumOps.LessThan(NumOps.Abs(validationSlope), convergenceThreshold))
        {
            return FitType.GoodFit;
        }

        if (NumOps.LessThan(trainingSlope, NumOps.Zero) && NumOps.GreaterThan(validationSlope, NumOps.Zero))
        {
            return FitType.Overfit;
        }

        if (NumOps.GreaterThan(trainingSlope, NumOps.Zero) && NumOps.GreaterThan(validationSlope, NumOps.Zero))
        {
            return FitType.Underfit;
        }

        return FitType.Unstable;
    }

    /// <summary>
    /// Calculates the confidence level of the fit detection.
    /// </summary>
    /// <param name="evaluationData">Data containing model performance metrics.</param>
    /// <returns>A value between 0 and 1 representing the confidence level.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how confident we are in our assessment of your model.
    /// 
    /// The confidence is based on how consistent your learning curves are:
    /// - If your learning curves have low variance (don't jump up and down a lot), confidence will be high (closer to 1)
    /// - If your learning curves are very erratic with large swings in performance, confidence will be lower
    /// 
    /// A higher confidence value means you can trust the fit assessment more.
    /// 
    /// Terminology explained:
    /// - Variance: A measure of how spread out the values are in your learning curve
    /// - Low variance: Points on the curve are close to each other (smooth curve)
    /// - High variance: Points on the curve vary widely (jagged curve)
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var trainingVariance = CalculateVariance(evaluationData.TrainingSet.PredictionStats.LearningCurve);
        var validationVariance = CalculateVariance(evaluationData.ValidationSet.PredictionStats.LearningCurve);

        var totalVariance = NumOps.Add(trainingVariance, validationVariance);
        var result = NumOps.Subtract(NumOps.One, NumOps.Divide(totalVariance, NumOps.FromDouble(2)));

        // Clamp confidence to [0, 1]
        if (NumOps.LessThan(result, NumOps.Zero)) result = NumOps.Zero;
        if (NumOps.GreaterThan(result, NumOps.One)) result = NumOps.One;

        return result;
    }

    /// <summary>
    /// Calculates the slope of a learning curve using linear regression.
    /// </summary>
    /// <param name="curve">The learning curve data points.</param>
    /// <returns>The slope of the best-fit line through the learning curve.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates how quickly your model's performance is changing over time.
    /// 
    /// It uses a technique called "linear regression" to find the best straight line that fits your learning curve:
    /// - A positive slope means your model's performance is improving
    /// - A negative slope means your model's performance is getting worse
    /// - A slope close to zero means your model's performance has stabilized
    /// 
    /// The calculation uses the formula for the slope of a best-fit line:
    /// slope = (n * sum(xy) - sum(x) * sum(y)) / (n * sum(x²) - sum(x)²)
    /// where n is the number of points, x represents the position in the sequence, and y represents the performance value.
    /// </para>
    /// </remarks>
    private T CalculateSlope(List<T> curve)
    {
        if (curve.Count < 2)
            return NumOps.Zero;

        var x = Enumerable.Range(0, curve.Count).Select(i => NumOps.FromDouble(i)).ToList();
        var n = NumOps.FromDouble(curve.Count);

        var sumX = x.Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));
        var sumY = curve.Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));
        var sumXY = x.Zip(curve, (xi, yi) => NumOps.Multiply(xi, yi)).Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));
        var sumX2 = x.Select(xi => NumOps.Multiply(xi, xi)).Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));

        var numerator = NumOps.Subtract(NumOps.Multiply(n, sumXY), NumOps.Multiply(sumX, sumY));
        var denominator = NumOps.Subtract(NumOps.Multiply(n, sumX2), NumOps.Multiply(sumX, sumX));

        return NumOps.Divide(numerator, denominator);
    }

    /// <summary>
    /// Calculates the statistical variance of a learning curve.
    /// </summary>
    /// <param name="curve">The learning curve data points.</param>
    /// <returns>The variance of the learning curve values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method measures how spread out or scattered the points in your learning curve are.
    /// 
    /// Variance is calculated in these steps:
    /// 1. Find the average (mean) of all points in your learning curve
    /// 2. Calculate how far each point is from this average
    /// 3. Square these differences (to make all values positive)
    /// 4. Calculate the average of these squared differences
    /// 
    /// A high variance means your learning curve has large ups and downs (inconsistent performance).
    /// A low variance means your learning curve is smooth and stable (consistent performance).
    /// 
    /// This information helps determine how reliable our assessment of your model is. Models with
    /// erratic learning curves (high variance) are harder to evaluate with confidence.
    /// </para>
    /// </remarks>
    private T CalculateVariance(List<T> curve)
    {
        var mean = curve.Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));
        mean = NumOps.Divide(mean, NumOps.FromDouble(curve.Count));

        var variance = curve.Select(x => NumOps.Multiply(NumOps.Subtract(x, mean), NumOps.Subtract(x, mean)))
                            .Aggregate(NumOps.Zero, (acc, val) => NumOps.Add(acc, val));

        return NumOps.Divide(variance, NumOps.FromDouble(curve.Count - 1));
    }
}
