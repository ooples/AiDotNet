namespace AiDotNet.UncertaintyQuantification.Layers;

/// <summary>
/// Implements Monte Carlo Dropout layer for uncertainty estimation in neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Monte Carlo Dropout is a simple yet powerful technique for estimating uncertainty.
///
/// Unlike regular dropout which is only active during training, MC Dropout keeps dropout active
/// during prediction as well. By running multiple predictions with different random dropout masks,
/// we get a distribution of predictions. The spread of this distribution tells us how uncertain
/// the model is.
///
/// Think of it like asking multiple slightly different versions of the same expert for their opinion.
/// If they all agree, you can be confident. If they disagree widely, there's high uncertainty.
///
/// This is particularly useful for:
/// - Detecting out-of-distribution samples
/// - Active learning (selecting which data to label next)
/// - Safety-critical applications (knowing when to defer to a human expert)
/// </para>
/// </remarks>
public class MCDropoutLayer<T> : LayerBase<T>
{
    private readonly T _dropoutRate;
    private readonly T _scale;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _dropoutMask;
    private bool _mcMode; // Monte Carlo mode - always apply dropout

    /// <summary>
    /// Gets or sets whether Monte Carlo mode is enabled (applies dropout during inference).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> When MC mode is on, dropout is applied even during prediction,
    /// allowing you to estimate uncertainty by running multiple forward passes.
    /// </remarks>
    public bool MonteCarloMode
    {
        get => _mcMode;
        set => _mcMode = value;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports training mode.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the MCDropoutLayer class.
    /// </summary>
    /// <param name="dropoutRate">The probability of dropping out a neuron (between 0 and 1).</param>
    /// <param name="mcMode">Whether to enable Monte Carlo mode by default.</param>
    /// <exception cref="ArgumentException">Thrown when dropout rate is not between 0 and 1.</exception>
    public MCDropoutLayer(double dropoutRate = 0.5, bool mcMode = false)
        : base(Array.Empty<int>(), [])
    {
        if (dropoutRate < 0 || dropoutRate >= 1)
            throw new ArgumentException("Dropout rate must be between 0 and 1", nameof(dropoutRate));

        _dropoutRate = NumOps.FromDouble(dropoutRate);
        _scale = NumOps.FromDouble(1.0 / (1.0 - dropoutRate));
        _mcMode = mcMode;
    }

    /// <summary>
    /// Performs the forward pass of the MC dropout layer.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The output tensor with dropout applied if in training or MC mode.</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Apply dropout if in training mode OR Monte Carlo mode
        if (!IsTrainingMode && !_mcMode)
            return input;

        _dropoutMask = new Tensor<T>(input.Shape);
        var output = new Tensor<T>(input.Shape);

        for (int i = 0; i < input.Length; i++)
        {
            if (Random.NextDouble() > Convert.ToDouble(_dropoutRate))
            {
                _dropoutMask[i] = _scale;
                output[i] = NumOps.Multiply(input[i], _scale);
            }
            else
            {
                _dropoutMask[i] = NumOps.Zero;
                output[i] = NumOps.Zero;
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass of the MC dropout layer.
    /// </summary>
    /// <param name="outputGradient">The gradient from the next layer.</param>
    /// <returns>The gradient to pass to the previous layer.</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _dropoutMask == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        if (!IsTrainingMode && !_mcMode)
            return outputGradient;

        var inputGradient = new Tensor<T>(_lastInput.Shape);

        for (int i = 0; i < outputGradient.Length; i++)
        {
            inputGradient[i] = NumOps.Multiply(outputGradient[i], _dropoutMask[i]);
        }

        return inputGradient;
    }

    /// <summary>
    /// Updates the parameters (no-op for dropout layers).
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        // No parameters to update
    }

    /// <summary>
    /// Gets the trainable parameters (empty for dropout layers).
    /// </summary>
    public override Vector<T> GetParameters()
    {
        return new Vector<T>(0);
    }

    /// <summary>
    /// Sets the trainable parameters (no-op for dropout layers).
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != 0)
        {
            throw new ArgumentException($"Expected 0 parameters, but got {parameters.Length}");
        }
    }

    /// <summary>
    /// Resets the internal state of the layer.
    /// </summary>
    public override void ResetState()
    {
        _lastInput = null;
        _dropoutMask = null;
    }
}
