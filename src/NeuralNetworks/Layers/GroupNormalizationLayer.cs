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

    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        var shape = input.Shape;
        int channels = shape.Length > 1 ? shape[1] : 1;

        if (channels != _numChannels)
            throw new ArgumentException($"Input has {channels} channels but layer expects {_numChannels} channels.");

        // Use Engine for GPU/CPU accelerated Group Normalization
        var output = Engine.GroupNorm(
            input,
            _numGroups,
            _gamma,
            _beta,
            NumOps.ToDouble(_epsilon),
            out var mean,
            out var variance);

        _lastMean = mean;
        _lastVariance = variance;

        return output;
    }

    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastMean == null || _lastVariance == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Use Engine for GPU/CPU accelerated backward pass
        var inputGradient = Engine.GroupNormBackward(
            outputGradient,
            _lastInput,
            _numGroups,
            _gamma,
            _lastMean,
            _lastVariance,
            NumOps.ToDouble(_epsilon),
            out var gradGamma,
            out var gradBeta);

        _gammaGradient = gradGamma;
        _betaGradient = gradBeta;

        return inputGradient;
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
    }

    public override bool SupportsJitCompilation => false;

    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        throw new NotSupportedException(
            "GroupNormalization JIT compilation is not yet implemented. " +
            "Use the layer in interpreted mode by setting SupportsJitCompilation = false.");
    }
}
