using AiDotNet.Helpers;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A one-dimensional convolutional layer that slides a learnable kernel along the
/// length (time) axis of a <c>[batch, channels, length]</c> signal.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// 1-D convolution is the standard building block for raw-waveform and sequence
/// models (e.g. Silero VAD, WaveNet-style encoders, Conv-TasNet). It is mathematically
/// a 2-D convolution whose kernel and padding/stride are degenerate on one spatial axis,
/// so this layer reshapes the rank-3 signal <c>[B, C, L]</c> to <c>[B, C, 1, L]</c> and
/// dispatches the engine's per-axis Conv2D (stride <c>[1, S]</c>, padding <c>[0, P]</c>,
/// kernel <c>[outC, inC, 1, K]</c>). This reuses the engine's tape-aware, accelerated
/// convolution path, so the kernel and bias appear as graph leaves and are updated by
/// the optimizer under <c>TrainWithTape</c> exactly like <see cref="ConvolutionalLayer{T}"/>.
/// </para>
/// <para><b>For Beginners:</b> A regular convolutional layer scans a 2-D image with a
/// little square window. A 1-D convolutional layer scans a 1-D signal (like an audio
/// waveform) with a window that slides left-to-right in time. It's the right tool when
/// your data has one sequential axis instead of two spatial ones.</para>
/// </remarks>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "1, 1, 16", TestConstructorArgs = "1, 2, 3")]
public class Conv1DLayer<T> : LayerBase<T>
{
    /// <summary>Number of input channels the layer expects.</summary>
    public int InputChannels { get; }

    /// <summary>Number of output channels (filters) the layer produces.</summary>
    public int OutputChannels { get; }

    /// <summary>Length of the 1-D convolution kernel.</summary>
    public int KernelSize { get; }

    /// <summary>Step size of the kernel along the length axis.</summary>
    public int Stride { get; }

    /// <summary>Zero-padding applied to both ends of the length axis.</summary>
    public int Padding { get; }

    // Kernel stored in NCHW-compatible 4-D layout [outChannels, inChannels, 1, kernelSize]
    // so it can be fed directly to the engine's Conv2D without a per-forward reshape.
    private Tensor<T> _kernels;
    private Tensor<T> _biases;

    private static readonly int[] ConvDilation2D = { 1, 1 };

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new <see cref="Conv1DLayer{T}"/>.
    /// </summary>
    /// <param name="inputChannels">Number of input channels.</param>
    /// <param name="outputChannels">Number of output channels (filters).</param>
    /// <param name="kernelSize">Length of the convolution kernel.</param>
    /// <param name="stride">Step size along the length axis (default 1).</param>
    /// <param name="padding">Zero-padding on each end of the length axis (default 0).</param>
    /// <param name="activationFunction">Element-wise activation (default identity).</param>
    public Conv1DLayer(
        int inputChannels,
        int outputChannels,
        int kernelSize,
        int stride = 1,
        int padding = 0,
        IActivationFunction<T>? activationFunction = null)
        : base(new[] { inputChannels, -1 }, new[] { outputChannels, -1 },
               activationFunction ?? new IdentityActivation<T>())
    {
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels), "inputChannels must be positive.");
        if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels), "outputChannels must be positive.");
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize), "kernelSize must be positive.");
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride), "stride must be positive.");
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding), "padding cannot be negative.");

        InputChannels = inputChannels;
        OutputChannels = outputChannels;
        KernelSize = kernelSize;
        Stride = stride;
        Padding = padding;

        _kernels = new Tensor<T>([outputChannels, inputChannels, 1, kernelSize]);
        _biases = new Tensor<T>([outputChannels]);

        InitializeParameters();

        RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
    }

    private void InitializeParameters()
    {
        // He-uniform (Kaiming) initialization: fan_in = inChannels * kernelSize.
        int fanIn = InputChannels * KernelSize;
        double limit = Math.Sqrt(6.0 / fanIn);
        var rng = RandomHelper.CreateSecureRandom();

        var kSpan = _kernels.Data.Span;
        for (int i = 0; i < kSpan.Length; i++)
        {
            double v = (rng.NextDouble() * 2.0 - 1.0) * limit;
            kSpan[i] = NumOps.FromDouble(v);
        }
        _biases.Fill(NumOps.Zero);
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Accept [C, L] (no batch) or [B, C, L].
        Tensor<T> input3D = input.Rank == 2
            ? Engine.Reshape(input, [1, input.Shape[0], input.Shape[1]])
            : input;

        if (input3D.Rank != 3)
            throw new ArgumentException(
                $"Conv1DLayer expects a rank-2 [C, L] or rank-3 [B, C, L] input, got rank {input.Rank}.",
                nameof(input));

        int batch = input3D.Shape[0];
        int inC = input3D.Shape[1];
        int length = input3D.Shape[2];

        if (inC != InputChannels)
            throw new ArgumentException(
                $"Conv1DLayer expected {InputChannels} input channels, got {inC}.", nameof(input));

        // [B, C, L] -> [B, C, 1, L] so a per-axis Conv2D performs a 1-D convolution
        // along the length (width) axis only. stride [1, S] / padding [0, P] leave the
        // singleton height axis (kernel height 1) untouched.
        var input4D = Engine.Reshape(input3D, [batch, inC, 1, length]);
        int[] stride2D = { 1, Stride };
        int[] padding2D = { 0, Padding };

        var conv = Engine.Conv2D(input4D, _kernels, stride2D, padding2D, ConvDilation2D);

        // Tape-aware broadcast bias add: [1, outC, 1, 1] over [B, outC, 1, outL].
        var biasReshaped = Engine.Reshape(_biases, [1, OutputChannels, 1, 1]);
        var biased = Engine.TensorBroadcastAdd(conv, biasReshaped);

        int outLength = biased.Shape[3];
        var output3D = Engine.Reshape(biased, [batch, OutputChannels, outLength]);

        return ApplyActivation(output3D);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Training uses the gradient tape: <see cref="Forward"/> records the engine
    /// convolution and the registered <c>_kernels</c>/<c>_biases</c> are updated in place
    /// by the optimizer in <c>TrainWithTape</c>. This SGD-from-stored-gradients entry
    /// point is therefore a no-op, matching <see cref="ConvolutionalLayer{T}"/>.
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
        // No-op: parameters are updated by the optimizer through the gradient tape.
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        int total = _kernels.Length + _biases.Length;
        var result = new Vector<T>(total);
        int idx = 0;
        var kSpan = _kernels.Data.Span;
        for (int i = 0; i < kSpan.Length; i++) result[idx++] = kSpan[i];
        var bSpan = _biases.Data.Span;
        for (int i = 0; i < bSpan.Length; i++) result[idx++] = bSpan[i];
        return result;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        int expected = _kernels.Length + _biases.Length;
        if (parameters.Length != expected)
            throw new ArgumentException(
                $"Conv1DLayer expected {expected} parameters, got {parameters.Length}.", nameof(parameters));

        // Copy into the existing tensors in place to preserve the trainable-parameter
        // registration (object identity) the gradient tape and optimizer bind to.
        int idx = 0;
        var kSpan = _kernels.Data.Span;
        for (int i = 0; i < kSpan.Length; i++) kSpan[i] = parameters[idx++];
        var bSpan = _biases.Data.Span;
        for (int i = 0; i < bSpan.Length; i++) bSpan[i] = parameters[idx++];
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        // Stateless between forward passes (no recurrent or cached activation state).
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["InputChannels"] = InputChannels.ToString();
        metadata["OutputChannels"] = OutputChannels.ToString();
        metadata["KernelSize"] = KernelSize.ToString();
        metadata["Stride"] = Stride.ToString();
        metadata["Padding"] = Padding.ToString();
        return metadata;
    }

    /// <inheritdoc/>
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);
        writer.Write(InputChannels);
        writer.Write(OutputChannels);
        writer.Write(KernelSize);
        writer.Write(Stride);
        writer.Write(Padding);

        var kSpan = _kernels.Data.Span;
        for (int i = 0; i < kSpan.Length; i++) writer.Write(Convert.ToDouble(kSpan[i]));
        var bSpan = _biases.Data.Span;
        for (int i = 0; i < bSpan.Length; i++) writer.Write(Convert.ToDouble(bSpan[i]));
    }

    /// <inheritdoc/>
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);
        // Configuration is fixed at construction; validate the persisted values match
        // so a mismatched stream surfaces clearly rather than corrupting weights.
        int inC = reader.ReadInt32();
        int outC = reader.ReadInt32();
        int k = reader.ReadInt32();
        int s = reader.ReadInt32();
        int p = reader.ReadInt32();
        if (inC != InputChannels || outC != OutputChannels || k != KernelSize || s != Stride || p != Padding)
            throw new InvalidOperationException(
                "Conv1DLayer.Deserialize: serialized configuration does not match this layer instance.");

        var kSpan = _kernels.Data.Span;
        for (int i = 0; i < kSpan.Length; i++) kSpan[i] = NumOps.FromDouble(reader.ReadDouble());
        var bSpan = _biases.Data.Span;
        for (int i = 0; i < bSpan.Length; i++) bSpan[i] = NumOps.FromDouble(reader.ReadDouble());
    }
}
