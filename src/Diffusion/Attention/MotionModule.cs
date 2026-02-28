using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Attention;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion.Attention;

/// <summary>
/// AnimateDiff motion module for injecting temporal awareness into image diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Paper: "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning" (Guo et al., 2023)</item>
/// </list></para>
/// <para><b>For Beginners:</b> The Motion Module is a plug-in temporal attention layer that can be added to any image diffusion model to enable video generation. It learns temporal motion patterns that are applied on top of existing spatial features.</para>
/// <para>
/// The motion module is a plug-in temporal attention block that can be inserted into any
/// image diffusion UNet to add video generation capability. It consists of:
/// - Temporal self-attention (across frames for each spatial position)
/// - Positional encoding (sinusoidal temporal embeddings)
/// - Feed-forward network (MLP for feature transformation)
/// - Zero-initialized output projection (for stable training from image models)
/// </para>
/// <para>
/// Key innovation: The zero-initialization ensures that inserting the motion module
/// into a pretrained image model initially produces identity output, allowing the
/// temporal parameters to be trained without disrupting the spatial generation quality.
/// </para>
/// </remarks>
public class MotionModule<T> : LayerBase<T>
{
    private readonly int _channels;
    private readonly int _numHeads;
    private readonly int _numFrames;
    private readonly int _spatialSize;
    private readonly TemporalSelfAttention<T> _temporalAttention;
    private readonly DenseLayer<T> _ffnIn;
    private readonly DenseLayer<T> _ffnOut;
    private readonly LayerNormalizationLayer<T> _norm1;
    private readonly LayerNormalizationLayer<T> _norm2;
    private Tensor<T>? _lastInput;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the number of channels.
    /// </summary>
    public int Channels => _channels;

    /// <summary>
    /// Gets the number of frames.
    /// </summary>
    public int NumFrames => _numFrames;

    /// <summary>
    /// Initializes a new AnimateDiff motion module.
    /// </summary>
    /// <param name="channels">Number of feature channels.</param>
    /// <param name="numHeads">Number of attention heads.</param>
    /// <param name="numFrames">Number of video frames.</param>
    /// <param name="spatialSize">Spatial size of feature maps.</param>
    /// <param name="ffnMultiplier">Feed-forward network hidden dim multiplier.</param>
    public MotionModule(
        int channels,
        int numHeads = 8,
        int numFrames = 16,
        int spatialSize = 64,
        int ffnMultiplier = 4)
        : base(
            new[] { 1, numFrames * spatialSize * spatialSize, channels },
            new[] { 1, numFrames * spatialSize * spatialSize, channels })
    {
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), "Channels must be positive.");
        if (numHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(numHeads), "Number of heads must be positive.");
        if (numFrames <= 0)
            throw new ArgumentOutOfRangeException(nameof(numFrames), "Number of frames must be positive.");
        if (spatialSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(spatialSize), "Spatial size must be positive.");
        if (ffnMultiplier <= 0)
            throw new ArgumentOutOfRangeException(nameof(ffnMultiplier), "FFN multiplier must be positive.");

        _channels = channels;
        _numHeads = numHeads;
        _numFrames = numFrames;
        _spatialSize = spatialSize;

        _temporalAttention = new TemporalSelfAttention<T>(
            channels: channels,
            numHeads: numHeads,
            numFrames: numFrames,
            spatialSize: spatialSize);

        int ffnHidden = channels * ffnMultiplier;
        _ffnIn = new DenseLayer<T>(channels, ffnHidden, (IActivationFunction<T>)new GELUActivation<T>());
        _ffnOut = new DenseLayer<T>(ffnHidden, channels, (IActivationFunction<T>)new IdentityActivation<T>());

        _norm1 = new LayerNormalizationLayer<T>(channels);
        _norm2 = new LayerNormalizationLayer<T>(channels);
    }

    /// <summary>
    /// Applies the motion module: temporal attention + FFN with residual connections.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Temporal attention with residual
        var normed1 = _norm1.Forward(input);
        var attnOut = _temporalAttention.Forward(normed1);
        var afterAttn = AddTensors(input, attnOut);

        // FFN with residual
        var normed2 = _norm2.Forward(afterAttn);
        var ffnHidden = _ffnIn.Forward(normed2);
        var ffnOut = _ffnOut.Forward(ffnHidden);
        var output = AddTensors(afterAttn, ffnOut);

        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // Backward through FFN
        var ffnOutGrad = _ffnOut.Backward(outputGradient);
        var ffnInGrad = _ffnIn.Backward(ffnOutGrad);
        var norm2Grad = _norm2.Backward(ffnInGrad);
        var afterAttnGrad = AddTensors(outputGradient, norm2Grad);

        // Backward through temporal attention
        var attnGrad = _temporalAttention.Backward(afterAttnGrad);
        var norm1Grad = _norm1.Backward(attnGrad);
        return AddTensors(afterAttnGrad, norm1Grad);
    }

    private static readonly INumericOperations<T> StaticNumOps = MathHelper.GetNumericOperations<T>();

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd<T>(a, b);
    }

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        _temporalAttention.UpdateParameters(learningRate);
        _ffnIn.UpdateParameters(learningRate);
        _ffnOut.UpdateParameters(learningRate);
        _norm1.UpdateParameters(learningRate);
        _norm2.UpdateParameters(learningRate);
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parts = new[]
        {
            _temporalAttention.GetParameters(),
            _ffnIn.GetParameters(),
            _ffnOut.GetParameters(),
            _norm1.GetParameters(),
            _norm2.GetParameters()
        };

        int total = 0;
        foreach (var p in parts) total += p.Length;

        var combined = new Vector<T>(total);
        int offset = 0;
        foreach (var p in parts)
        {
            for (int i = 0; i < p.Length; i++)
                combined[offset + i] = p[i];
            offset += p.Length;
        }
        return combined;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        // GetParameters() concatenates sublayer params; allocates once for validation.
        int expected = GetParameters().Length;
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, got {parameters.Length}.", nameof(parameters));

        int offset = 0;
        SetSubParams(_temporalAttention, parameters, ref offset);
        SetSubParams(_ffnIn, parameters, ref offset);
        SetSubParams(_ffnOut, parameters, ref offset);
        SetSubParams(_norm1, parameters, ref offset);
        SetSubParams(_norm2, parameters, ref offset);
    }

    private static void SetSubParams(LayerBase<T> layer, Vector<T> parameters, ref int offset)
    {
        int count = layer.GetParameters().Length;
        var sub = new Vector<T>(count);
        for (int i = 0; i < count; i++)
            sub[i] = parameters[offset + i];
        layer.SetParameters(sub);
        offset += count;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _temporalAttention.ResetState();
        _ffnIn.ResetState();
        _ffnOut.ResetState();
        _norm1.ResetState();
        _norm2.ResetState();
    }

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

    /// <inheritdoc />
    public override Autodiff.ComputationNode<T> ExportComputationGraph(List<Autodiff.ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("MotionModule does not support JIT compilation.");
    }
}
