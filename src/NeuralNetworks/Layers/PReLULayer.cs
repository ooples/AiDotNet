using AiDotNet.Attributes;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements a Parametric ReLU (PReLU) layer with learnable negative-slope coefficients.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// PReLU is a learnable variant of Leaky ReLU where the negative-slope coefficient <c>α</c> is
/// a trainable parameter rather than a fixed hyperparameter. The function is:
/// </para>
/// <para>
/// <c>f(x) = max(0, x) + α * min(0, x) = ReLU(x) - α * ReLU(-x)</c>
/// </para>
/// <para>
/// Two parameterization modes are supported, matching PyTorch's <c>nn.PReLU</c>:
/// </para>
/// <list type="bullet">
/// <item><description><c>numParameters == 1</c>: a single shared α across the whole tensor.</description></item>
/// <item><description><c>numParameters == channels</c>: one α per channel, broadcast along the
/// specified channel axis (default axis 1 for NCHW conv inputs).</description></item>
/// </list>
/// <para>
/// Introduced in He et al., "Delving Deep into Rectifiers" (2015), which also introduced the Kaiming
/// initialization. The paper's recommended initial α is 0.25, which is this layer's default.
/// </para>
/// <para>
/// <b>For Beginners:</b> PReLU is like Leaky ReLU except the "leakiness" is learned from data instead
/// of being set by you. For positive inputs the output is unchanged; for negative inputs the output
/// is the input scaled by α. The network adjusts α during training to control how much signal to let
/// through on the negative side. Per-channel α lets different channels learn different leaks, which
/// often works better than a single shared α for convolutional networks.
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Activation)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, TestInputShape = "1, 4", TestConstructorArgs = "4, 1, 0.25")]
public class PReLULayer<T> : LayerBase<T>
{
    private readonly int _numParameters;
    private readonly int _channelAxis;
    private int[] _alphaBroadcastShape;

    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _alpha;

    private Tensor<T>? _lastInput;

    /// <summary>
    /// Gets a value indicating that this layer has trainable parameters (α).
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the total number of trainable parameters (the length of α).
    /// </summary>
    public override int ParameterCount => _alpha.Length;

    /// <summary>
    /// Gets the current α tensor. Shape is <c>[numParameters]</c>.
    /// </summary>
    public Tensor<T> GetAlphaTensor() => _alpha;

    /// <summary>
    /// Initializes a new <see cref="PReLULayer{T}"/>.
    /// </summary>
    /// <param name="inputShape">The shape of the input tensor this layer will receive.</param>
    /// <param name="numParameters">
    /// Number of learnable α coefficients. Pass <c>1</c> for a single shared α, or the size of the
    /// channel dimension for per-channel α. Matches <c>torch.nn.PReLU(num_parameters)</c>.
    /// </param>
    /// <param name="channelAxis">
    /// Axis that per-channel α broadcasts over. Defaults to <c>1</c> (NCHW convention). Ignored
    /// when <paramref name="numParameters"/> is 1.
    /// </param>
    /// <param name="initialAlpha">Initial value for every α. Defaults to 0.25 per He et al. 2015.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="numParameters"/> is less than 1, or when it is greater than 1 but
    /// does not match <c>inputShape[channelAxis]</c>.
    /// </exception>
    public PReLULayer(int numParameters = 1, int channelAxis = 1, double initialAlpha = 0.25)
        : base(new[] { -1 }, new[] { -1 })
    {
        if (numParameters < 1)
            throw new ArgumentException("numParameters must be at least 1.", nameof(numParameters));
        if (numParameters > 1 && channelAxis < 0)
            throw new ArgumentException(
                $"channelAxis {channelAxis} must be non-negative when numParameters > 1.",
                nameof(channelAxis));

        _numParameters = numParameters;
        _channelAxis = channelAxis;

        _alpha = Tensor<T>.CreateDefault(new[] { numParameters }, NumOps.FromDouble(initialAlpha));

        // Broadcast shape resolved on first forward. Shared α uses [1].
        _alphaBroadcastShape = new[] { 1 };

        RegisterTrainableParameter(_alpha, PersistentTensorRole.Weights);
    }

    /// <summary>
    /// Resolves broadcast shape and validates channel-count compatibility on first forward.
    /// </summary>
    protected override void OnFirstForward(Tensor<T> input)
    {
        var shape = input.Shape.ToArray();
        if (_numParameters > 1)
        {
            if (_channelAxis >= shape.Length)
                throw new ArgumentException(
                    $"channelAxis {_channelAxis} is out of range for input of rank {shape.Length}.",
                    nameof(input));
            if (shape[_channelAxis] != _numParameters)
                throw new ArgumentException(
                    $"numParameters ({_numParameters}) must match input.Shape[{_channelAxis}] ({shape[_channelAxis]}).",
                    nameof(input));

            _alphaBroadcastShape = new int[shape.Length];
            for (int i = 0; i < shape.Length; i++) _alphaBroadcastShape[i] = 1;
            _alphaBroadcastShape[_channelAxis] = _numParameters;
        }

        ResolveShapes(shape, shape);
    }

    /// <summary>
    /// Performs the forward pass: <c>ReLU(x) - α · ReLU(-x)</c>, all ops on the gradient tape.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);
        if (IsTrainingMode)
            _lastInput = input;

        var positivePart = Engine.ReLU(input);
        var negated = Engine.TensorNegate(input);
        var negativePart = Engine.ReLU(negated);

        Tensor<T> scaledNegative;
        if (_numParameters == 1)
        {
            // Shared scalar α stored as a [1] tensor — TensorBroadcastMultiply handles the broadcast.
            scaledNegative = Engine.TensorBroadcastMultiply(negativePart, _alpha);
        }
        else
        {
            // Per-channel α: reshape [C] → [1, …, C, …, 1] so it broadcasts across batch + spatial dims.
            var alphaBroadcast = Engine.Reshape(_alpha, _alphaBroadcastShape);
            scaledNegative = Engine.TensorBroadcastMultiply(negativePart, alphaBroadcast);
        }

        return Engine.TensorSubtract(positivePart, scaledNegative);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters() => _alpha.ToVector();

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != _alpha.Length)
            throw new ArgumentException(
                $"Expected {_alpha.Length} parameters, but got {parameters.Length}");

        var span = _alpha.Data.Span;
        for (int i = 0; i < _alpha.Length; i++)
            span[i] = parameters[i];

        Engine.InvalidatePersistentTensor(_alpha);
    }

    /// <summary>
    /// Legacy scalar-learning-rate parameter update. Tape-based training uses
    /// <see cref="SetParameters"/> after computing gradients via <c>GradientTape&lt;T&gt;</c>,
    /// so this override is a no-op.
    /// </summary>
    public override void UpdateParameters(T learningRate)
    {
        // Tape-based training flows through SetParameters after the optimizer applies the update.
        // The scalar-LR path is not used for this layer.
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
    }

    /// <inheritdoc/>
    public override LayerBase<T> Clone()
    {
        var copy = new PReLULayer<T>(_numParameters, _channelAxis,
            Convert.ToDouble(_alpha.Data.Span[0]));
        // Copy current α values so Clone preserves trained state, not just init state.
        var dst = copy._alpha.Data.Span;
        var src = _alpha.Data.Span;
        for (int i = 0; i < _alpha.Length; i++) dst[i] = src[i];
        return copy;
    }
}
