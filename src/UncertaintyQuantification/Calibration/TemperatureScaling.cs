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
/// - Temperature < 1: Makes predictions more confident (sharper)
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
    private T _temperature = default!;

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
        var scaled = new Tensor<T>(logits.Shape);
        for (int i = 0; i < logits.Length; i++)
        {
            scaled[i] = _numOps.Divide(logits[i], _temperature);
        }
        return scaled;
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

        var lr = _numOps.FromDouble(learningRate);

        for (int iter = 0; iter < maxIterations; iter++)
        {
            var gradient = ComputeGradient(logits, labels);

            // Update temperature using gradient descent
            _temperature = _numOps.Subtract(_temperature, _numOps.Multiply(lr, gradient));

            // Ensure temperature stays positive
            if (_numOps.LessThan(_temperature, _numOps.FromDouble(0.01)))
            {
                _temperature = _numOps.FromDouble(0.01);
            }
        }
    }

    /// <summary>
    /// Computes the gradient of negative log-likelihood with respect to temperature.
    /// </summary>
    private T ComputeGradient(Matrix<T> logits, Vector<int> labels)
    {
        var gradient = _numOps.Zero;
        var numSamples = logits.Rows;

        for (int i = 0; i < numSamples; i++)
        {
            var logit = logits.GetRow(i);
            var label = labels[i];

            // Compute softmax probabilities with current temperature
            var scaledLogits = new Vector<T>(logit.Length);
            for (int j = 0; j < logit.Length; j++)
            {
                scaledLogits[j] = _numOps.Divide(logit[j], _temperature);
            }

            var probs = Softmax(scaledLogits);

            // Compute gradient: ∂NLL/∂T = (1/T²) * (z_y - Σ_k(p_k * z_k))
            // Derivation: NLL = -log(p_y) = -z_y/T + log(Σ exp(z_k/T))
            // ∂NLL/∂T = z_y/T² - (1/T²)Σ p_k z_k = (1/T²)(z_y - Σ p_k z_k)
            var trueClassLogit = logit[label];
            var weightedSum = _numOps.Zero;

            for (int k = 0; k < logit.Length; k++)
            {
                weightedSum = _numOps.Add(weightedSum, _numOps.Multiply(probs[k], logit[k]));
            }

            var diff = _numOps.Subtract(trueClassLogit, weightedSum);
            var tempSquared = _numOps.Multiply(_temperature, _temperature);
            var sampleGrad = _numOps.Divide(diff, tempSquared);

            gradient = _numOps.Add(gradient, sampleGrad);
        }

        // Average gradient
        gradient = _numOps.Divide(gradient, _numOps.FromDouble(numSamples));
        return gradient;
    }

    /// <summary>
    /// Computes softmax probabilities from logits.
    /// </summary>
    private Vector<T> Softmax(Vector<T> logits)
    {
        // Find max for numerical stability
        var max = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (_numOps.GreaterThan(logits[i], max))
                max = logits[i];
        }

        // Compute exp(logit - max)
        var exps = new Vector<T>(logits.Length);
        var sum = _numOps.Zero;
        for (int i = 0; i < logits.Length; i++)
        {
            exps[i] = _numOps.Exp(_numOps.Subtract(logits[i], max));
            sum = _numOps.Add(sum, exps[i]);
        }

        // Normalize
        var probs = new Vector<T>(logits.Length);
        for (int i = 0; i < logits.Length; i++)
        {
            probs[i] = _numOps.Divide(exps[i], sum);
        }

        return probs;
    }
}
