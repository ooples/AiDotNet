using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Validation;

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
    [AiDotNet.Attributes.FittedParameter]
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
        Guard.NotNull(model);
        _model = model;
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
        var engine = AiDotNetEngine.Current;

        // Non-conformity scores: absolute residuals |y_i - f(x_i)|
        var predictions = PredictRows(calibrationInputs, includeSampleIndex: true);
        var residuals = engine.TensorAbs(engine.TensorSubtract(calibrationTargets.AsTensor(), predictions));

        // Sort calibration scores for quantile computation
        var (sorted, _) = engine.TensorSort(residuals, 0, false);
        _calibrationScores = sorted.ToVector();

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
            throw new InvalidOperationException($"Model must return scalar output for regression. Got length={prediction.Length} shape=[{string.Join(", ", prediction._shape)}].");
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
    /// <returns>The quantile value; positive infinity when the calibration set is too small for the level.</returns>
    /// <remarks>
    /// Split conformal prediction (Lei et al. 2018; Angelopoulos and Bates 2021, Sec. 1) uses the k-th smallest
    /// score with k = ceil((n + 1) * confidenceLevel). When k &gt; n no finite threshold guarantees the coverage,
    /// so the quantile is +infinity and the interval is unbounded.
    /// </remarks>
    internal static T ComputeQuantile(Vector<T> scores, double confidenceLevel)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var n = scores.Length;
        var k = (int)Math.Ceiling((n + 1) * confidenceLevel);
        if (k > n)
            return numOps.FromDouble(double.PositiveInfinity);

        return scores[Math.Max(k, 1) - 1];
    }

    /// <summary>
    /// Runs the model on every row and returns the scalar predictions as an [n] tensor.
    /// </summary>
    private Tensor<T> PredictRows(Matrix<T> inputs, bool includeSampleIndex)
    {
        var outputs = new Tensor<T>[inputs.Rows];
        for (int i = 0; i < inputs.Rows; i++)
        {
            var row = inputs.GetRow(i);
            var prediction = _model.Predict(new Tensor<T>([row.Length], row));
            if (prediction.Length != 1)
            {
                string where = includeSampleIndex ? $" (sample index {i})" : string.Empty;
                throw new InvalidOperationException($"Model must return scalar output for regression. Got length={prediction.Length} shape=[{string.Join(", ", prediction._shape)}]{where}.");
            }

            outputs[i] = AiDotNetEngine.Current.Reshape(prediction, new[] { 1 });
        }

        return outputs.Length == 0 ? new Tensor<T>([0]) : AiDotNetEngine.Current.TensorConcatenate(outputs, 0);
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
    public T EvaluateCoverage(Matrix<T> testInputs, Vector<T> testTargets, double confidenceLevel = 0.9)
    {
        if (!_isCalibrated || _calibrationScores == null)
            throw new InvalidOperationException("Must calibrate predictor before evaluation");

        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentException("Confidence level must be between 0 and 1", nameof(confidenceLevel));

        if (testTargets.Length != testInputs.Rows)
            throw new ArgumentException(
                $"testTargets has {testTargets.Length} entries but testInputs has {testInputs.Rows} rows.", nameof(testTargets));

        var engine = AiDotNetEngine.Current;
        int totalCount = testInputs.Rows;
        var quantile = ComputeQuantile(_calibrationScores, confidenceLevel);
        var predictions = PredictRows(testInputs, includeSampleIndex: false);

        // Covered when |y - f(x)| <= q, i.e. y lies in [f(x) - q, f(x) + q]; counted as int.
        var residuals = engine.TensorAbs(engine.TensorSubtract(testTargets.AsTensor(), predictions));
        var covered = engine.TensorAdd(engine.TensorLessThan(residuals, quantile), engine.TensorEquals(residuals, quantile)).Cast<int>();
        int coveredCount = engine.TensorSum(covered);

        return _numOps.Divide(_numOps.FromDouble(coveredCount), _numOps.FromDouble(totalCount));
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
    public T ComputeAverageIntervalWidth(Matrix<T> testInputs, double confidenceLevel = 0.9)
    {
        if (!_isCalibrated || _calibrationScores == null)
            throw new InvalidOperationException("Must calibrate predictor before computing interval width");

        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentException("Confidence level must be between 0 and 1", nameof(confidenceLevel));

        var engine = AiDotNetEngine.Current;
        var quantile = ComputeQuantile(_calibrationScores, confidenceLevel);
        var predictions = PredictRows(testInputs, includeSampleIndex: false);

        // Each interval is [f(x) - q, f(x) + q].
        var widths = engine.TensorSubtract(engine.TensorAddScalar(predictions, quantile), engine.TensorSubtractScalar(predictions, quantile));
        return engine.TensorMean(widths);
    }
}
