using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.UncertaintyQuantification.ConformalPrediction;

/// <summary>
/// Implements Split Conformal Prediction for regression tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Conformal Prediction is a framework that provides prediction intervals
/// with guaranteed coverage, regardless of the underlying model.
///
/// Key concepts:
/// - Instead of a single prediction, you get a prediction interval: [lower_bound, upper_bound]
/// - The interval is guaranteed to contain the true value with a specified probability (e.g., 90%)
/// - This guarantee holds for ANY model (neural network, random forest, etc.)
///
/// Example:
/// If you set confidence level = 90%:
/// - The model predicts: "House price will be between $180K and $220K"
/// - You're guaranteed that at least 90% of such intervals contain the true price
///
/// How it works:
/// 1. Split data into training set, calibration set, and test set
/// 2. Train your model on the training set
/// 3. Use the calibration set to compute "non-conformity scores" (prediction errors)
/// 4. Use these scores to build prediction intervals for new data
///
/// This is particularly valuable when you NEED reliability guarantees, such as:
/// - Medical diagnosis (must know uncertainty bounds)
/// - Safety-critical systems (must guarantee coverage)
/// - Scientific applications (need statistically valid uncertainty)
/// </para>
/// </remarks>
public class SplitConformalPredictor<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> _model;
    private Vector<T>? _calibrationScores;
    private bool _isCalibrated;

    /// <summary>
    /// Initializes a new instance of the SplitConformalPredictor class.
    /// </summary>
    /// <param name="model">The underlying prediction model (must be already trained).</param>
    /// <remarks>
    /// <b>For Beginners:</b> The model should be trained on a separate training set before
    /// passing it here. This class wraps the model to add conformal prediction intervals.
    /// </remarks>
    public SplitConformalPredictor(IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _numOps = MathHelper.GetNumericOperations<T>();
        _isCalibrated = false;
    }

    /// <summary>
    /// Calibrates the conformal predictor using a calibration dataset.
    /// </summary>
    /// <param name="calibrationInputs">Input features from the calibration set.</param>
    /// <param name="calibrationTargets">True target values from the calibration set.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a critical step! You MUST calibrate before making predictions.
    ///
    /// The calibration set should be:
    /// - Separate from your training set (otherwise intervals will be too narrow)
    /// - Separate from your test set (otherwise you're "cheating")
    /// - Representative of the data you'll see in practice
    /// - Typically 20-30% of your total data
    ///
    /// During calibration, the method:
    /// 1. Runs the model on calibration data
    /// 2. Computes prediction errors (non-conformity scores)
    /// 3. Stores these scores for later use in building intervals
    /// </para>
    /// </remarks>
    public void Calibrate(Matrix<T> calibrationInputs, Vector<T> calibrationTargets)
    {
        if (calibrationInputs.Rows != calibrationTargets.Length)
            throw new ArgumentException("Number of inputs must match number of targets");

        var numCalibration = calibrationInputs.Rows;
        if (numCalibration == 0)
            throw new ArgumentException("Calibration set must contain at least one sample.", nameof(calibrationInputs));
        _calibrationScores = new Vector<T>(numCalibration);

        // Compute non-conformity scores (absolute residuals)
        for (int i = 0; i < numCalibration; i++)
        {
            var input = calibrationInputs.GetRow(i);
            var prediction = _model.Predict(new Tensor<T>([input.Length], input));
            if (prediction.Length != 1)
                throw new InvalidOperationException($"Model must return scalar output for regression. Got length={prediction.Length} shape=[{string.Join(", ", prediction.Shape)}] (sample index {i}).");
            var predValue = prediction[0]; // Assuming scalar output

            var residual = _numOps.Abs(_numOps.Subtract(calibrationTargets[i], predValue));
            _calibrationScores[i] = residual;
        }

        // Sort calibration scores for quantile computation
        var scoresArray = _calibrationScores.ToArray();
        Array.Sort(scoresArray);
        _calibrationScores = new Vector<T>(scoresArray);

        _isCalibrated = true;
    }

    /// <summary>
    /// Predicts with a conformal prediction interval.
    /// </summary>
    /// <param name="input">The input features for prediction.</param>
    /// <param name="confidenceLevel">The desired coverage level (e.g., 0.9 for 90% coverage).</param>
    /// <returns>A tuple containing (point_prediction, lower_bound, upper_bound).</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns both a point prediction and an interval around it.
    ///
    /// The confidence level determines how wide the interval is:
    /// - 0.90 (90%): Narrower interval, 90% guaranteed coverage
    /// - 0.95 (95%): Wider interval, 95% guaranteed coverage
    /// - 0.99 (99%): Even wider interval, 99% guaranteed coverage
    ///
    /// Example output: (prediction=200, lower=180, upper=220)
    /// Interpretation: "We predict 200, and we're 90% confident the true value is between 180 and 220"
    ///
    /// The guarantee: If you use 90% confidence, then 90% of your prediction intervals
    /// will contain the true value (assuming calibration set is representative).
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown if Calibrate hasn't been called first.</exception>
    public (T prediction, T lowerBound, T upperBound) PredictWithInterval(Tensor<T> input, double confidenceLevel = 0.9)
    {
        if (!_isCalibrated || _calibrationScores == null)
            throw new InvalidOperationException("Must calibrate predictor before making predictions");

        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentException("Confidence level must be between 0 and 1", nameof(confidenceLevel));

        // Get point prediction
        var prediction = _model.Predict(input);
        if (prediction.Length != 1)
            throw new InvalidOperationException($"Model must return scalar output for regression. Got length={prediction.Length} shape=[{string.Join(", ", prediction.Shape)}].");
        var predValue = prediction[0]; // Assuming scalar output

        // Compute quantile for interval width
        var quantile = ComputeQuantile(_calibrationScores, confidenceLevel);

        // Build prediction interval
        var lowerBound = _numOps.Subtract(predValue, quantile);
        var upperBound = _numOps.Add(predValue, quantile);

        return (predValue, lowerBound, upperBound);
    }

    /// <summary>
    /// Computes the quantile of calibration scores.
    /// </summary>
    /// <param name="scores">Sorted calibration scores.</param>
    /// <param name="confidenceLevel">The desired confidence level.</param>
    /// <returns>The quantile value.</returns>
    private T ComputeQuantile(Vector<T> scores, double confidenceLevel)
    {
        var n = scores.Length;

        // Adjusted quantile level to ensure coverage guarantee
        // Using ceil((n+1)*alpha)/n as per conformal prediction theory
        var adjustedLevel = Math.Ceiling((n + 1) * confidenceLevel) / n;
        if (adjustedLevel > 1.0) adjustedLevel = 1.0;

        var index = (int)Math.Ceiling(n * adjustedLevel) - 1;
        if (index < 0) index = 0;
        if (index >= n) index = n - 1;

        return scores[index];
    }

    /// <summary>
    /// Evaluates the empirical coverage of prediction intervals on a test set.
    /// </summary>
    /// <param name="testInputs">Test input features.</param>
    /// <param name="testTargets">True test target values.</param>
    /// <param name="confidenceLevel">The confidence level used for intervals.</param>
    /// <returns>The empirical coverage (proportion of intervals containing true value).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Use this to verify that your intervals achieve the desired coverage.
    /// For example, with 90% confidence level, the empirical coverage should be around 0.90.
    /// If it's much lower, your calibration set might not be representative.
    /// </remarks>
    public double EvaluateCoverage(Matrix<T> testInputs, Vector<T> testTargets, double confidenceLevel = 0.9)
    {
        if (!_isCalibrated)
            throw new InvalidOperationException("Must calibrate predictor before evaluation");

        int coveredCount = 0;
        int totalCount = testInputs.Rows;

        for (int i = 0; i < totalCount; i++)
        {
            var input = new Tensor<T>([testInputs.Columns], testInputs.GetRow(i));
            var (_, lower, upper) = PredictWithInterval(input, confidenceLevel);
            var trueValue = testTargets[i];

            // Check if true value is within the interval
            if (_numOps.GreaterThanOrEquals(trueValue, lower) && _numOps.LessThanOrEquals(trueValue, upper))
            {
                coveredCount++;
            }
        }

        return (double)coveredCount / totalCount;
    }

    /// <summary>
    /// Computes the average interval width on a test set.
    /// </summary>
    /// <param name="testInputs">Test input features.</param>
    /// <param name="confidenceLevel">The confidence level used for intervals.</param>
    /// <returns>The average width of prediction intervals.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Narrower intervals are better (more precise), but coverage must
    /// still be maintained. This metric helps you understand the precision of your uncertainty estimates.
    /// </remarks>
    public double ComputeAverageIntervalWidth(Matrix<T> testInputs, double confidenceLevel = 0.9)
    {
        if (!_isCalibrated)
            throw new InvalidOperationException("Must calibrate predictor before computing interval width");

        double totalWidth = 0.0;
        int count = testInputs.Rows;

        for (int i = 0; i < count; i++)
        {
            var input = new Tensor<T>([testInputs.Columns], testInputs.GetRow(i));
            var (_, lower, upper) = PredictWithInterval(input, confidenceLevel);

            var width = _numOps.Subtract(upper, lower);
            totalWidth += Convert.ToDouble(width);
        }

        return totalWidth / count;
    }
}
