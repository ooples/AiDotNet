using AiDotNet.Autodiff;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Represents a Group Normalization layer that normalizes inputs across groups of channels.
/// </summary>
/// <remarks>
/// <para>
/// Group Normalization divides channels into groups and normalizes the features within each group.
/// This makes it invariant to batch size, making it suitable for small batch sizes or applications
/// where batch statistics are not reliable (like VAEs and generative models).
/// </para>
/// <para><b>For Beginners:</b> This layer helps stabilize training for convolutional networks.
///
/// Think of Group Normalization like organizing students into study groups:
/// - Each group (of channels) studies together and normalizes their behavior
/// - It works the same regardless of class size (batch size)
/// - This is especially useful for generative models like VAEs where batch sizes may be small
///
/// Key advantages:
/// - Works well with small batch sizes (even batch size of 1)
/// - More stable than Batch Normalization for generative models
/// - Used extensively in modern architectures like Stable Diffusion VAE
///
/// Typical usage:
/// - numGroups=32 for 256+ channels
/// - numGroups=16 for 128 channels
/// - numGroups=8 for 64 channels
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class GroupNormalizationLayer<T> : LayerBase<T>
{
    private readonly T _epsilon;
    private readonly int _numGroups;
    private readonly int _numChannels;
    private Tensor<T> _gamma;
    private Tensor<T> _beta;
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastMean;
    private Tensor<T>? _lastVariance;
    private Tensor<T>? _gammaGradient;
    private Tensor<T>? _betaGradient;

    public override bool SupportsTraining => true;

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["NumGroups"] = _numGroups.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumChannels"] = _numChannels.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Epsilon"] = Convert.ToDouble(_epsilon, System.Globalization.CultureInfo.InvariantCulture)
            .ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    public int NumGroups => _numGroups;
    public int NumChannels => _numChannels;
    public Vector<T> GetGamma() => _gamma.ToVector();
    public Tensor<T> GetGammaTensor() => _gamma;
    public Vector<T> GetBeta() => _beta.ToVector();
    public Tensor<T> GetBetaTensor() => _beta;
    public T GetEpsilon() => _epsilon;

    public GroupNormalizationLayer(int numGroups, int numChannels, double epsilon = NumericalStabilityHelper.LargeEpsilon)
        : base([numChannels], [numChannels])
    {
        if (numGroups <= 0)
            throw new ArgumentOutOfRangeException(nameof(numGroups), "Number of groups must be positive.");
        if (numChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(numChannels), "Number of channels must be positive.");
        if (numChannels % numGroups != 0)
            throw new ArgumentException($"Number of channels ({numChannels}) must be divisible by number of groups ({numGroups}).");

        _numGroups = numGroups;
        _numChannels = numChannels;
        _epsilon = NumericalStabilityHelper.GetEpsilon<T>(epsilon);
        _gamma = Tensor<T>.CreateDefault([numChannels], NumOps.One);
        _beta = Tensor<T>.CreateDefault([numChannels], NumOps.Zero);
    }

    /// <summary>
    /// Tracks whether we added a batch dimension to a 3D input.
    /// </summary>
    private bool _addedBatchDimension;

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        var shape = input.Shape;

        // Determine channel count based on input rank:
        // 4D [N, C, H, W]: channels = shape[1]
        // 3D [C, H, W]: channels = shape[0]
        // 2D [N, C]: channels = shape[1]
        int channels;
        Tensor<T> input4D;

        if (shape.Length == 4)
        {
            // Standard NCHW format
            channels = shape[1];
            input4D = input;
            _addedBatchDimension = false;
        }
        else if (shape.Length == 3)
        {
            // CHW format without batch - add batch dimension
            channels = shape[0];
            input4D = input.Reshape(1, shape[0], shape[1], shape[2]);
            _addedBatchDimension = true;
        }
        else if (shape.Length == 2)
        {
            // [N, C] format
            channels = shape[1];
            input4D = input;
            _addedBatchDimension = false;
        }
        else
        {
            throw new ArgumentException($"GroupNormalization expects 2D, 3D, or 4D input, got {shape.Length}D.");
        }

        if (channels != _numChannels)
            throw new ArgumentException($"Input has {channels} channels but layer expects {_numChannels} channels.");

        // Use Engine for GPU/CPU accelerated Group Normalization
        var output = Engine.GroupNorm(
            input4D,
            _numGroups,
            _gamma,
            _beta,
            NumOps.ToDouble(_epsilon),
            out var mean,
            out var variance);

        _lastMean = mean;
        _lastVariance = variance;

        // Remove batch dimension if we added it
        return _addedBatchDimension
            ? output.Reshape(output.Shape[1], output.Shape[2], output.Shape[3])
            : output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastMean == null || _lastVariance == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Handle 3D gradients by adding batch dimension if needed
        Tensor<T> grad4D = (_addedBatchDimension && outputGradient.Shape.Length == 3)
            ? outputGradient.Reshape(1, outputGradient.Shape[0], outputGradient.Shape[1], outputGradient.Shape[2])
            : outputGradient;

        // Get input with batch dimension for backward pass
        Tensor<T> input4D = (_addedBatchDimension && _lastInput.Shape.Length == 3)
            ? _lastInput.Reshape(1, _lastInput.Shape[0], _lastInput.Shape[1], _lastInput.Shape[2])
            : _lastInput;

        // Use Engine for GPU/CPU accelerated backward pass
        var inputGradient = Engine.GroupNormBackward(
            grad4D,
            input4D,
            _numGroups,
            _gamma,
            _lastMean,
            _lastVariance,
            NumOps.ToDouble(_epsilon),
            out var gradGamma,
            out var gradBeta);

        _gammaGradient = gradGamma;
        _betaGradient = gradBeta;

        // Remove batch dimension if we added it
        return _addedBatchDimension
            ? inputGradient.Reshape(inputGradient.Shape[1], inputGradient.Shape[2], inputGradient.Shape[3])
            : inputGradient;
    }

    public override void UpdateParameters(T learningRate)
    {
        if (_gammaGradient == null || _betaGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        _gamma = Engine.TensorSubtract(_gamma, Engine.TensorMultiplyScalar(_gammaGradient, learningRate));
        _beta = Engine.TensorSubtract(_beta, Engine.TensorMultiplyScalar(_betaGradient, learningRate));
    }

    public override Vector<T> GetParameters()
    {
        return Vector<T>.Concatenate(_gamma.ToVector(), _beta.ToVector());
    }

    public override void SetParameters(Vector<T> parameters)
    {
        int totalParams = _gamma.Length + _beta.Length;

        if (parameters.Length != totalParams)
            throw new ArgumentException($"Expected {totalParams} parameters, but got {parameters.Length}");

        var gammaVec = parameters.Slice(0, _gamma.Length);
        var betaVec = parameters.Slice(_gamma.Length, _beta.Length);

        _gamma = Tensor<T>.FromVector(gammaVec, _gamma.Shape);
        _beta = Tensor<T>.FromVector(betaVec, _beta.Shape);
    }

    public override void ResetState()
    {
        _lastInput = null;
        _lastMean = null;
        _lastVariance = null;
        _gammaGradient = null;
        _betaGradient = null;
        _addedBatchDimension = false;
    }

    /// <summary>
    /// Gets a value indicating whether this layer supports JIT compilation.
    /// </summary>
    public override bool SupportsJitCompilation => true;

    /// <summary>
    /// Exports the computation graph for JIT compilation.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the GroupNormalization operation.</returns>
    /// <remarks>
    /// <para>
    /// This method builds a computation graph representing the GroupNormalization layer.
    /// The graph divides channels into groups and normalizes within each group,
    /// then applies learned scale (gamma) and shift (beta) parameters per channel.
    /// </para>
    /// </remarks>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes is null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (InputShape is null || InputShape.Length == 0)
            throw new InvalidOperationException("Layer input shape not configured.");

        // Create symbolic input node with batch dimension
        var symbolicInput = new Tensor<T>(new int[] { 1 }.Concat(InputShape).ToArray());
        var inputNode = TensorOperations<T>.Variable(symbolicInput, "input");
        inputNodes.Add(inputNode);

        // Create gamma and beta parameter nodes
        var gammaNode = TensorOperations<T>.Constant(_gamma, "gamma");
        var betaNode = TensorOperations<T>.Constant(_beta, "beta");

        // Apply GroupNorm operation
        var outputNode = TensorOperations<T>.GroupNorm(
            inputNode,
            _numGroups,
            gammaNode,
            betaNode,
            NumOps.ToDouble(_epsilon));

        return outputNode;
    }
}
