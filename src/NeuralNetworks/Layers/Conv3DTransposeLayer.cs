using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// 3D transposed convolution ("up-convolution") for volumetric data — the learnable
/// upsampling primitive the 3D U-Net decoder is built from (Cicek et al. 2016).
/// Operates on rank-5 input <c>[B, C_in, D, H, W]</c> and produces
/// <c>[B, C_out, D*stride, H*stride, W*stride]</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> An ordinary convolution shrinks a volume; this one grows it. The
/// decoder half of a U-Net needs to get back to full resolution, and doing that with a LEARNED
/// operation rather than a fixed rule lets the network decide how to fill in the detail. Cicek
/// et al. also halve the channel count here, which is why the U-Net decoder is cheap enough to
/// run at all.
/// </para>
/// <para>
/// <b>How it is implemented.</b> The engine exposes <c>Conv3D</c> but no <c>ConvTranspose3D</c>,
/// and <c>IEngine</c> lives in the separate AiDotNet.Tensors package. For the case this layer
/// supports -- <c>stride == kernelSize</c>, which is the 2x2x2/stride-2 up-convolution the paper
/// specifies -- a transposed convolution is EXACTLY equivalent to a 1x1x1 convolution producing
/// <c>C_out * stride^3</c> channels followed by a depth-to-space rearrangement. That is the
/// sub-pixel convolution identity of Shi et al. (2016): with stride equal to kernel size the
/// output blocks are disjoint, so every output voxel is one input voxel times one kernel weight,
/// and grouping those weights by their position within the block turns the whole operation into a
/// channel projection plus a reshuffle. It is not an approximation.
/// </para>
/// <para>
/// The rearrangement is done as three rank-6 reshape/permute stages, one per spatial axis, rather
/// than a single rank-8 permute, because a rank-8 tensor is the more likely thing for a backend
/// to reject. Every operation used is an existing engine op, so the tape autodiff backward comes
/// for free and there is no hand-written gradient to get wrong.
/// </para>
/// <para>
/// Strides other than <c>kernelSize</c> are REJECTED rather than approximated. The sub-pixel
/// identity does not hold when the output blocks overlap, and silently computing something that
/// is not a transposed convolution would be worse than refusing.
/// </para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerTask(LayerTask.SpatialProcessing)]
[LayerProperty(NormalizesInput = true, IsTrainable = true, ChangesShape = true, ExpectedInputRank = 5, Cost = ComputeCost.Medium, TestInputShape = "1, 4, 4, 4, 4", TestConstructorArgs = "2, 2, 2, (AiDotNet.Interfaces.IActivationFunction<double>?)null")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Depth, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Depth, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class Conv3DTransposeLayer<T> : LayerBase<T>, IShapeContract
{
    private int _inputChannels;
    private readonly int _outputChannels;
    private readonly int _kernelSize;
    private readonly int _stride;

    private Tensor<T> _kernels;
    private Tensor<T> _biases;

    /// <inheritdoc/>
    public override bool HasUninitializedParameters => !IsShapeResolved;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Creates an up-convolution that grows each spatial axis by <paramref name="stride"/>.
    /// </summary>
    /// <param name="outputChannels">Number of output feature maps.</param>
    /// <param name="kernelSize">Kernel extent per axis. Must equal <paramref name="stride"/>.</param>
    /// <param name="stride">Spatial expansion factor. The 3D U-Net uses 2.</param>
    /// <param name="activation">Optional scalar activation.</param>
    /// <param name="initializationStrategy">Optional weight initialization (defaults to He).</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when a dimension is not positive, or when the stride differs from the kernel size.
    /// </exception>
    public Conv3DTransposeLayer(
        [LayerState] int outputChannels,
        [LayerState] int kernelSize = 2,
        [LayerState] int stride = 2,
        IActivationFunction<T>? activation = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(new[] { -1, -1, -1, -1 }, new[] { outputChannels, -1, -1, -1 },
               activation ?? new AiDotNet.ActivationFunctions.IdentityActivation<T>())
    {
        if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels));
        if (kernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));
        if (stride != kernelSize)
        {
            throw new ArgumentOutOfRangeException(
                nameof(stride),
                $"Conv3DTransposeLayer supports only stride == kernelSize (got stride {stride}, "
                    + $"kernelSize {kernelSize}). The sub-pixel identity it is built on holds only "
                    + "when the output blocks are disjoint; any other stride would compute "
                    + "something that is not a transposed convolution.");
        }

        InitializationStrategy = initializationStrategy ?? Initialization.InitializationStrategies<T>.He;

        _inputChannels = -1;
        _outputChannels = outputChannels;
        _kernelSize = kernelSize;
        _stride = stride;

        _kernels = new Tensor<T>([0, 0, 0, 0, 0]);
        _biases = new Tensor<T>([0]);
    }

    /// <summary>
    /// Declares the output shape relation: channels are fixed, and every spatial axis is scaled
    /// by the stride.
    /// </summary>
    /// <param name="inputRank">The rank of the input being reasoned about.</param>
    /// <returns>The per-axis contracts, or null when the rank is not supported.</returns>
    /// <remarks>
    /// <para>
    /// Unlike the 1D transposed convolution, there is no affine offset to express here. This
    /// layer only accepts stride == kernelSize with no padding, for which the output length is
    /// exactly input * stride, so each spatial axis is a clean Scaled relation.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 4 && inputRank != 5) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
            new OutputAxisContract(TensorAxis.Channels, AxisRelation.Fixed(_outputChannels)),
            new OutputAxisContract(TensorAxis.Depth, AxisRelation.Scaled(TensorAxis.Depth, _stride)),
            new OutputAxisContract(TensorAxis.Height, AxisRelation.Scaled(TensorAxis.Height, _stride)),
            new OutputAxisContract(TensorAxis.Width, AxisRelation.Scaled(TensorAxis.Width, _stride)),
        };
    }

    /// <inheritdoc/>
    protected override void OnFirstForward(Tensor<T> input)
    {
        int rank = input.Shape.Length;
        if (rank != 4 && rank != 5)
        {
            throw new ArgumentException(
                $"Conv3DTransposeLayer requires rank-4 [C, D, H, W] or rank-5 [B, C, D, H, W] "
                    + $"input; got rank {rank}.",
                nameof(input));
        }

        // Rank-4 is the unbatched form Conv3DLayer also accepts; the channel axis shifts.
        int cIn = rank == 5 ? input.Shape[1] : input.Shape[0];
        int projected = _outputChannels * _stride * _stride * _stride;

        _inputChannels = cIn;

        // Idempotent: a clone or deserialize may already have installed trained weights.
        if (!WeightsAlreadyAllocated(_kernels, projected, cIn, 1, 1, 1))
        {
            _kernels = AllocateLazyWeight([projected, cIn, 1, 1, 1]);
            _biases = AllocateLazyWeight([projected]);
            InitializeLayerWeights(_kernels, cIn, projected);
            InitializeLayerBiases(_biases);
            RegisterTrainableParameter(_kernels, PersistentTensorRole.Weights);
            RegisterTrainableParameter(_biases, PersistentTensorRole.Biases);
        }

        int sd = rank == 5 ? input.Shape[2] : input.Shape[1];
        int sh = rank == 5 ? input.Shape[3] : input.Shape[2];
        int sw = rank == 5 ? input.Shape[4] : input.Shape[3];

        ResolveShapes(
            new[] { cIn, sd, sh, sw },
            new[] { _outputChannels, LayerShape.Dynamic, LayerShape.Dynamic, LayerShape.Dynamic });
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        EnsureInitializedFromInput(input);

        // Conv3D and the reshapes below all assume a batch axis. Add one for a rank-4 input and
        // strip it again on the way out, which is what Conv3DLayer does for the same shape.
        bool addedBatch = input.Shape.Length == 4;
        if (addedBatch)
        {
            input = Engine.Reshape(input,
                new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3] });
        }

        int b = input.Shape[0];
        int d = input.Shape[2];
        int h = input.Shape[3];
        int w = input.Shape[4];
        int s = _stride;

        // 1x1x1 projection to C_out * s^3 channels: one channel per (output channel, position
        // within the s x s x s block).
        var projected = Engine.Conv3D(input, _kernels, [1, 1, 1], [0, 0, 0], [1, 1, 1]);

        var biasReshaped = Engine.Reshape(_biases, new[] { 1, _outputChannels * s * s * s, 1, 1, 1 });
        var withBias = Engine.TensorAdd(projected, biasReshaped);
        var activated = ApplyActivation(withBias);

        // Depth-to-space, one axis at a time. Each stage peels the next factor of s off the
        // channel axis and interleaves it into a spatial axis.
        var x = activated;

        // Depth.
        x = Engine.Reshape(x, new[] { b, _outputChannels * s * s, s, d, h, w });
        x = Engine.TensorPermute(x, new[] { 0, 1, 3, 2, 4, 5 });
        x = Engine.Reshape(x, new[] { b, _outputChannels * s * s, d * s, h, w });

        // Height.
        x = Engine.Reshape(x, new[] { b, _outputChannels * s, s, d * s, h, w });
        x = Engine.TensorPermute(x, new[] { 0, 1, 3, 4, 2, 5 });
        x = Engine.Reshape(x, new[] { b, _outputChannels * s, d * s, h * s, w });

        // Width.
        x = Engine.Reshape(x, new[] { b, _outputChannels, s, d * s, h * s, w });
        x = Engine.TensorPermute(x, new[] { 0, 1, 3, 4, 5, 2 });
        x = Engine.Reshape(x, new[] { b, _outputChannels, d * s, h * s, w * s });

        if (addedBatch)
        {
            x = Engine.Reshape(x, new[] { _outputChannels, d * s, h * s, w * s });
        }

        return x;
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
    }

    /// <summary>
    /// Serialization metadata — the hyper-parameters are not recoverable from the input and
    /// output shapes alone, so they round-trip here for Clone and Deserialize.
    /// </summary>
    /// <returns>The metadata dictionary.</returns>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["OutputChannels"] = _outputChannels.ToString();
        metadata["KernelSize"] = _kernelSize.ToString();
        metadata["Stride"] = _stride.ToString();
        if (_inputChannels > 0)
            metadata["InputChannels"] = _inputChannels.ToString();
        if (ScalarActivation is not null)
        {
            metadata["ScalarActivationType"] = ScalarActivation.GetType().AssemblyQualifiedName
                ?? ScalarActivation.GetType().FullName ?? string.Empty;
        }

        return metadata;
    }
}
