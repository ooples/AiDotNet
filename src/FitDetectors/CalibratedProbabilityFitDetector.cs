namespace AiDotNet.FitDetectors;

/// <summary>
/// A fit detector that analyzes the calibration of probability predictions to assess model fit.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Probability calibration refers to how well a model's predicted probabilities 
/// match the actual frequencies of events. A well-calibrated model should predict probabilities that 
/// match the true likelihood of events.
/// </para>
/// <para>
/// For example, if a model predicts a 70% probability for 100 different samples, approximately 70 of 
/// those samples should actually belong to the positive class. This detector analyzes how well your 
/// model's probability predictions match the actual outcomes.
/// </para>
/// </remarks>
public class CalibratedProbabilityFitDetector<T, TInput, TOutput> : FitDetectorBase<T, TInput, TOutput>
{
    /// <summary>
    /// Configuration options for the calibrated probability fit detector.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> These settings control how the detector analyzes probability calibration, 
    /// including the number of bins used for calibration analysis and thresholds for determining 
    /// different types of model fit.
    /// </remarks>
    private readonly CalibratedProbabilityFitDetectorOptions _options;

    /// <summary>
    /// Initializes a new instance of the CalibratedProbabilityFitDetector class.
    /// </summary>
    /// <param name="options">Optional configuration options. If not provided, default options are used.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new calibrated probability fit detector with 
    /// either custom options or default settings.
    /// </para>
    /// <para>
    /// The default settings typically include:
    /// <list type="bullet">
    /// <item><description>Number of calibration bins (often 10)</description></item>
    /// <item><description>Thresholds for determining good fit, overfitting, and underfitting</description></item>
    /// <item><description>Maximum calibration error for normalization</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    public CalibratedProbabilityFitDetector(CalibratedProbabilityFitDetectorOptions? options = null)
    {
        _options = options ?? new CalibratedProbabilityFitDetectorOptions();
    }

    /// <summary>
    /// Detects the fit type of a model based on probability calibration analysis.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A result object containing the detected fit type, confidence level, and recommendations.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes how well your model's probability predictions match 
    /// the actual outcomes to determine if it's underfitting, overfitting, or has a good fit.
    /// </para>
    /// <para>
    /// The result includes:
    /// <list type="bullet">
    /// <item><description>FitType: Whether the model is underfitting, overfitting, or has a good fit</description></item>
    /// <item><description>ConfidenceLevel: How confident the detector is in its assessment</description></item>
    /// <item><description>Recommendations: Suggestions for improving the model based on the detected fit type</description></item>
    /// </list>
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
    /// Determines the fit type based on probability calibration analysis.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>The detected fit type based on calibration analysis.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes the calibration error between expected and observed 
    /// probabilities to determine what type of fit your model has.
    /// </para>
    /// <para>
    /// The calibration error measures how much the model's predicted probabilities deviate from the 
    /// actual frequencies of events. Based on this error:
    /// <list type="bullet">
    /// <item><description>Low calibration error indicates a good fit</description></item>
    /// <item><description>High calibration error may indicate overfitting (model is overconfident)</description></item>
    /// <item><description>Moderate calibration error may indicate underfitting (model is underconfident)</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    protected override FitType DetermineFitType(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var (expectedCalibration, observedCalibration) = CalculateCalibration(evaluationData);

        var calibrationError = CalculateCalibrationError(expectedCalibration, observedCalibration);

        if (NumOps.LessThan(calibrationError, NumOps.FromDouble(_options.GoodFitThreshold)))
        {
            return FitType.GoodFit;
        }
        else if (NumOps.GreaterThan(calibrationError, NumOps.FromDouble(_options.OverfitThreshold)))
        {
            return FitType.Overfit;
        }
        else
        {
            return FitType.Underfit;
        }
    }

    /// <summary>
    /// Calculates the confidence level of the calibration-based fit detection.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A value indicating the confidence level of the detection.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method determines how confident the detector is in its assessment 
    /// of your model's fit. The confidence is based on the calibration error between expected and 
    /// observed probabilities.
    /// </para>
    /// <para>
    /// A lower calibration error indicates higher confidence in the fit assessment. The confidence 
    /// level is calculated as 1 minus the normalized calibration error, resulting in a value 
    /// between 0 and 1, where higher values indicate greater confidence.
    /// </para>
    /// </remarks>
    protected override T CalculateConfidenceLevel(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var (expectedCalibration, observedCalibration) = CalculateCalibration(evaluationData);

        var calibrationError = CalculateCalibrationError(expectedCalibration, observedCalibration);

        // Normalize confidence level to [0, 1]
        return NumOps.Subtract(NumOps.One, NumOps.Divide(calibrationError, NumOps.FromDouble(_options.MaxCalibrationError)));
    }

    /// <summary>
    /// Generates recommendations based on the detected fit type.
    /// </summary>
    /// <param name="fitType">The detected fit type.</param>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A list of recommendations for improving the model.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides practical suggestions for addressing the specific 
    /// type of fit issue detected in your model based on probability calibration analysis.
    /// </para>
    /// <para>
    /// Different types of calibration issues require different approaches:
    /// <list type="bullet">
    /// <item><description>Overfit (Overconfident): The model is too confident in its predictions and needs to be regularized</description></item>
    /// <item><description>Underfit (Underconfident): The model is not confident enough and may need more complexity</description></item>
    /// <item><description>Good Fit (Well-calibrated): The model's probability predictions match actual frequencies well</description></item>
    /// </list>
    /// </para>
    /// </remarks>
    protected override List<string> GenerateRecommendations(FitType fitType, ModelEvaluationData<T, TInput, TOutput> evaluationData)
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

    /// <summary>
    /// Calculates the expected and observed calibration values.
    /// </summary>
    /// <param name="evaluationData">Data containing model predictions and actual values.</param>
    /// <returns>A tuple containing vectors of expected and observed calibration values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method divides the predicted probabilities into bins and 
    /// calculates the expected and observed probabilities for each bin.
    /// </para>
    /// <para>
    /// For each bin:
    /// <list type="bullet">
    /// <item><description>Expected calibration: The average predicted probability in the bin</description></item>
    /// <item><description>Observed calibration: The actual fraction of positive samples in the bin</description></item>
    /// </list>
    /// </para>
    /// <para>
    /// In a well-calibrated model, these values should be close to each other across all bins.
    /// </para>
    /// </remarks>
    private (Vector<T>, Vector<T>) CalculateCalibration(ModelEvaluationData<T, TInput, TOutput> evaluationData)
    {
        var predicted = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Predicted);
        var actual = ConversionsHelper.ConvertToVector<T, TOutput>(evaluationData.ModelStats.Actual);

        var numBins = _options.NumCalibrationBins;
        var binSize = NumOps.Divide(NumOps.One, NumOps.FromDouble(numBins));

        var expectedCalibration = new Vector<T>(numBins);
        var observedCalibration = new Vector<T>(numBins);

        for (int i = 0; i < numBins; i++)
        {
            var lowerBound = NumOps.Multiply(NumOps.FromDouble(i), binSize);
            var upperBound = NumOps.Multiply(NumOps.FromDouble(i + 1), binSize);

            var binIndices = predicted.Select((p, idx) => new { Prob = p, Index = idx })
                                      .Where(x => NumOps.GreaterThanOrEquals(x.Prob, lowerBound) && NumOps.LessThan(x.Prob, upperBound))
                                      .Select(x => NumOps.FromDouble(x.Index))
                                      .ToList();

            if (binIndices.Count > 0)
            {
                expectedCalibration[i] = NumOps.Divide(NumOps.Add(lowerBound, upperBound), NumOps.FromDouble(2));

                var sum = binIndices.Aggregate(NumOps.Zero, (acc, idx) =>
                    NumOps.Add(acc, actual[NumOps.ToInt32(idx)])
                );

                observedCalibration[i] = NumOps.Divide(
                    sum,
                    NumOps.FromDouble(binIndices.Count)
                );
            }
            else
            {
                expectedCalibration[i] = NumOps.Zero;
                observedCalibration[i] = NumOps.Zero;
            }
        }

        return (expectedCalibration, observedCalibration);
    }

    /// <summary>
    /// Calculates the calibration error between expected and observed calibration values.
    /// </summary>
    /// <param name="expected">Vector of expected calibration values.</param>
    /// <param name="observed">Vector of observed calibration values.</param>
    /// <returns>The root mean squared error between expected and observed calibration values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This private method calculates how much the model's predicted probabilities 
    /// deviate from the actual frequencies of events. It uses the root mean squared error (RMSE) between 
    /// the expected and observed calibration values.
    /// </para>
    /// <para>
    /// A lower calibration error indicates a better calibrated model. Perfect calibration would result 
    /// in an error of 0, meaning the model's probability predictions perfectly match the actual frequencies.
    /// </para>
    /// </remarks>
    private T CalculateCalibrationError(Vector<T> expected, Vector<T> observed)
    {
        var squaredErrors = new Vector<T>(expected.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            var diff = NumOps.Subtract(expected[i], observed[i]);
            squaredErrors[i] = NumOps.Multiply(diff, diff);
        }

        return NumOps.Sqrt(NumOps.Divide(squaredErrors.Sum(), NumOps.FromDouble(expected.Length)));
    }
}
