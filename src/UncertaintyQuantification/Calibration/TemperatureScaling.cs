namespace AiDotNet.UncertaintyQuantification.Calibration;



/// <summary>
/// Implements temperature scaling for probability calibration.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Temperature scaling is a simple but effective way to calibrate neural network probabilities.
///
/// Neural networks often output probabilities that don't match reality:
/// - They might be overconfident: predicting 99% when accuracy is only 70%
/// - Or underconfident: predicting 60% when accuracy is actually 95%
///
/// Temperature scaling fixes this by dividing the network's logits (pre-softmax values) by a learned
/// temperature parameter before applying softmax:
/// - Temperature > 1: Makes predictions less confident (softer)
/// - Temperature &lt; 1: Makes predictions more confident (sharper)
/// - Temperature = 1: No change
///
/// This is calibrated on a held-out validation set to find the temperature that makes the predicted
/// probabilities match actual frequencies.
///
/// Example:
/// If a model predicts 80% confidence for 100 cases, and 80 of them are actually correct,
/// the model is well-calibrated. If only 60 are correct, the model needs calibration.
/// </para>
/// </remarks>
public class TemperatureScaling<T>
{
    private readonly INumericOperations<T> _numOps;

    private T _temperature;


    /// <summary>
    /// Gets or sets the temperature parameter.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the scaling factor applied to logits before softmax.
    /// It's learned from validation data to improve calibration.
    /// </remarks>
    public T Temperature
    {
        get => _temperature;
        set
        {
            if (_numOps.LessThanOrEquals(value, _numOps.Zero))
                throw new ArgumentException("Temperature must be positive");
            _temperature = value;
        }
    }

    /// <summary>
    /// Initializes a new instance of the TemperatureScaling class.
    /// </summary>
    /// <param name="initialTemperature">The initial temperature value (default: 1.0).</param>
    public TemperatureScaling(double initialTemperature = 1.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _temperature = _numOps.Zero;
        if (initialTemperature <= 0)
            throw new ArgumentOutOfRangeException(nameof(initialTemperature), "Initial temperature must be positive.");

        Temperature = _numOps.FromDouble(initialTemperature);
    }

    /// <summary>
    /// Applies temperature scaling to logits.
    /// </summary>
    /// <param name="logits">The input logits from the neural network.</param>
    /// <returns>The temperature-scaled logits.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This divides each logit by the temperature before you apply softmax.
    /// Use this during inference after calibrating the temperature on validation data.
    /// </remarks>
    public Tensor<T> ScaleLogits(Tensor<T> logits)
    {
        return AiDotNetEngine.Current.TensorDivideScalar(logits, _temperature);
    }

    /// <summary>
    /// Calibrates the temperature parameter using validation data.
    /// </summary>
    /// <param name="logits">The logits from the model for validation samples.</param>
    /// <param name="labels">The true labels for validation samples.</param>
    /// <param name="learningRate">Learning rate for optimization (default: 0.01).</param>
    /// <param name="maxIterations">Maximum number of optimization iterations (default: 100).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This finds the best temperature value using a validation set.
    ///
    /// The process:
    /// 1. Try different temperature values
    /// 2. For each temperature, compute how well calibrated the predictions are
    /// 3. Choose the temperature that gives the best calibration (lowest negative log-likelihood)
    ///
    /// You should call this method once on a validation set (not your training or test set)
    /// before using the model for predictions.
    /// </para>
    /// </remarks>
    public void Calibrate(Matrix<T> logits, Vector<int> labels, double learningRate = 0.01, int maxIterations = 100)
    {
        if (logits.Rows != labels.Length)
            throw new ArgumentException("Number of logit samples must match number of labels");

        var engine = AiDotNetEngine.Current;
        var lr = _numOps.FromDouble(learningRate);
        var floor = _numOps.FromDouble(0.01);
        var sampleCount = _numOps.FromDouble(logits.Rows);

        var z = Tensor<T>.FromMatrix(logits);
        var labelIndex = Tensor<int>.FromVector(labels, new[] { labels.Length, 1 });

        // sum_i z_{i, y_i} does not depend on T, so it is gathered once.
        var trueLogitSum = engine.ReduceSum(engine.TensorTakeAlongDim(z, labelIndex, 1), new[] { 0, 1 }, keepDims: true);

        // The temperature lives in a [1, 1] tensor during the fit so every update is an engine op.
        var temperature = Tensor<T>.CreateDefault(new[] { 1, 1 }, _temperature);

        for (int iter = 0; iter < maxIterations; iter++)
        {
            var gradient = engine.TensorDivideScalar(ComputeGradient(z, trueLogitSum, temperature), sampleCount);

            // Update temperature using gradient descent and keep it positive.
            temperature = engine.TensorClampMin(
                engine.TensorSubtract(temperature, engine.TensorMultiplyScalar(gradient, lr)), floor);
        }

        _temperature = temperature[0, 0];
    }

    /// <summary>
    /// Computes the summed gradient of the negative log-likelihood with respect to temperature.
    /// </summary>
    /// <remarks>
    /// NLL_i = -z_y/T + log(sum_k exp(z_k/T)), so dNLL_i/dT = (1/T^2) (z_y - sum_k p_k z_k).
    /// Returns the sum over samples as a [1, 1] tensor.
    /// </remarks>
    private static Tensor<T> ComputeGradient(Tensor<T> logits, Tensor<T> trueLogitSum, Tensor<T> temperature)
    {
        var engine = AiDotNetEngine.Current;
        var broadcastTemperature = engine.TensorBroadcastTo(temperature, logits._shape);
        var probs = engine.TensorSoftmax(engine.TensorDivide(logits, broadcastTemperature), 1);
        var expectedLogitSum = engine.ReduceSum(engine.TensorMultiply(probs, logits), new[] { 0, 1 }, keepDims: true);
        var diff = engine.TensorSubtract(trueLogitSum, expectedLogitSum);
        return engine.TensorDivide(engine.TensorDivide(diff, temperature), temperature);
    }
}
