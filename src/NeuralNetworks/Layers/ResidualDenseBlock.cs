using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Engines;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Residual Dense Block (RDB) as used in ESRGAN and Real-ESRGAN generators.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This implements the Residual Dense Block from the ESRGAN paper (Wang et al., 2018).
/// It differs from DenseNet's Dense Block by using LeakyReLU, no batch normalization,
/// and a local residual connection with scaling.
/// </para>
/// <para>
/// The architecture consists of 5 convolutional layers with dense connections:
/// <code>
/// x0 = input (numFeatures channels)
/// x1 = LeakyReLU(Conv(x0))                              [growthChannels]
/// x2 = LeakyReLU(Conv(concat(x0, x1)))                  [growthChannels]
/// x3 = LeakyReLU(Conv(concat(x0, x1, x2)))              [growthChannels]
/// x4 = LeakyReLU(Conv(concat(x0, x1, x2, x3)))          [growthChannels]
/// x5 = Conv(concat(x0, x1, x2, x3, x4))                 [numFeatures, no activation]
/// output = x5 * residualScale + x0                      [local residual learning]
/// </code>
/// </para>
/// <para>
/// <b>For Beginners:</b> This block is the building block of ESRGAN's generator.
/// It combines ideas from DenseNet (dense connections) and ResNet (residual learning):
///
/// 1. **Dense connections**: Each conv layer can see ALL previous features
/// 2. **Local residual**: The block's output is added to its input (helps training)
/// 3. **Residual scaling**: The residual is scaled by 0.2 (prevents instability)
/// 4. **LeakyReLU**: Uses LeakyReLU(0.2) instead of ReLU (better gradients)
/// 5. **No batch norm**: ESRGAN generator doesn't use batch normalization
///
/// The default parameters (64 features, 32 growth, 0.2 scale) are from the paper.
/// </para>
/// <para>
/// <b>Reference:</b> Wang et al., "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks",
/// ECCV 2018 Workshops. https://arxiv.org/abs/1809.00219
/// </para>
/// </remarks>
[LayerCategory(LayerCategory.Residual)]
[LayerCategory(LayerCategory.Convolution)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ExpectedInputRank = 3, Cost = ComputeCost.High, TestInputShape = "4, 8, 8", TestConstructorArgs = "4, 4")]
// SHAPE-PRESERVING, and it is the residual connection that makes it so rather than any of the five
// convolutions: ForwardTraced ends at "AddResidual(x5, x0, _residualScale)", which can only add x5 to
// the untouched input if the two agree exactly. OnFirstForward says the same thing directly -
// "ResolveShapes(new[] { _numFeatures, inH, inW }, new[] { _numFeatures, inH, inW })" - input and output
// are the SAME array contents. The interior channel growth (_growthChannels per conv, concatenated) is
// entirely internal and never reaches the boundary.
//
// Roles and ranks come from this layer's own guard, which rejects everything else:
// "requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input". BatchOptional covers both from one declaration,
// including the rank 3 its [LayerProperty(ExpectedInputRank = 3)] is tested at.
//
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    BatchOptional = true, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class ResidualDenseBlock<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Every axis is <c>Same</c>, so this looks like the case the generator handles - but the generator
    /// keys its arms on the DECLARED layout length and does not expand <c>BatchOptional</c>, so it would
    /// emit rank 4 only and return null for the rank 3 this layer's
    /// <c>[LayerProperty(ExpectedInputRank = 3, TestInputShape = "4, 8, 8")]</c> exercises. Writing both
    /// ranks by hand is what makes the unbatched form resolvable rather than silently declined.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        // Ranks 3 and 4 only - the two forms OnFirstForward accepts; it throws for anything else.
        if (inputRank is not (3 or 4)) return null;

        var channels = new OutputAxisContract(TensorAxis.Channels, AxisRelation.Same(TensorAxis.Channels));
        var height = new OutputAxisContract(TensorAxis.Height, AxisRelation.Same(TensorAxis.Height));
        var width = new OutputAxisContract(TensorAxis.Width, AxisRelation.Same(TensorAxis.Width));

        return inputRank == 3
            ? new[] { channels, height, width }
            : new[]
            {
                new OutputAxisContract(TensorAxis.Batch, AxisRelation.Same(TensorAxis.Batch)),
                channels, height, width,
            };
    }

    #region Fields

    /// <summary>
    /// The 5 convolutional layers in the dense block.
    /// </summary>
    private readonly ConvolutionalLayer<T>[] _convLayers;

    /// <summary>
    /// LeakyReLU activation with negative slope 0.2 (from ESRGAN paper).
    /// </summary>
    private readonly LeakyReLUActivation<T> _activation;

    /// <summary>
    /// Residual scaling factor. Default: 0.2 from the paper.
    /// </summary>
    private readonly double _residualScale;

    /// <summary>
    /// Number of output channels for intermediate conv layers (growth rate).
    /// </summary>
    private readonly int _growthChannels;

    /// <summary>
    /// Number of input/output channels (feature channels).
    /// </summary>
    private readonly int _numFeatures;

    /// <summary>
    /// Input height — non-readonly: lazy ctor leaves -1 until OnFirstForward.
    /// </summary>
    private int _inputHeight;

    /// <summary>
    /// Input width — non-readonly: lazy ctor leaves -1 until OnFirstForward.
    /// </summary>
    private int _inputWidth;

    /// <summary>
    /// Cached input for backpropagation.
    /// </summary>
    [Scratch]
    private Tensor<T>? _lastInput;

    /// <summary>
    /// Cached intermediate conv outputs (before activation) for backpropagation.
    /// </summary>
    private Tensor<T>[]? _convOutputs;

    /// <summary>
    /// Cached intermediate activation outputs for backpropagation.
    /// </summary>
    private Tensor<T>[]? _activationOutputs;

    /// <summary>
    /// Cached concatenated inputs to each conv layer for backpropagation.
    /// </summary>
    private Tensor<T>[]? _concatInputs;

    #endregion

    #region Properties

    /// <summary>
    /// Gets the residual scaling factor.
    /// </summary>
    public double ResidualScale => _residualScale;

    /// <summary>
    /// Gets the growth channels (intermediate conv output channels).
    /// </summary>
    public int GrowthChannels => _growthChannels;

    /// <summary>
    /// Gets the number of feature channels.
    /// </summary>
    public int NumFeatures => _numFeatures;

    public override bool SupportsTraining => true;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new Residual Dense Block.
    /// </summary>
    /// <param name="numFeatures">Number of input/output feature channels. Default: 64 (from paper).</param>
    /// <param name="growthChannels">Number of channels for intermediate convolutions. Default: 32 (from paper).</param>
    /// <param name="inputHeight">Height of input feature maps.</param>
    /// <param name="inputWidth">Width of input feature maps.</param>
    /// <param name="residualScale">Residual scaling factor. Default: 0.2 (from paper).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a Residual Dense Block for ESRGAN:
    /// <code>
    /// var rdb = new ResidualDenseBlock&lt;float&gt;(
    ///     numFeatures: 64,       // Main feature channels (paper default)
    ///     growthChannels: 32,    // Intermediate channels (paper default)
    ///     inputHeight: 128,
    ///     inputWidth: 128,
    ///     residualScale: 0.2     // Residual scaling (paper default)
    /// );
    /// </code>
    /// </para>
    /// </remarks>
    public ResidualDenseBlock(
        int numFeatures = 64,
        int growthChannels = 32,
        double residualScale = 0.2)
        : base(
            [numFeatures, -1, -1],
            [numFeatures, -1, -1])
    {
        if (numFeatures <= 0)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be positive.");
        if (growthChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(growthChannels), "Growth channels must be positive.");
        if (residualScale < 0 || residualScale > 1)
            throw new ArgumentOutOfRangeException(nameof(residualScale), "Residual scale must be between 0 and 1.");

        _numFeatures = numFeatures;
        _growthChannels = growthChannels;
        _inputHeight = -1; // resolved in OnFirstForward
        _inputWidth = -1;  // resolved in OnFirstForward
        _residualScale = residualScale;
        _activation = new LeakyReLUActivation<T>(0.2); // ESRGAN uses 0.2 negative slope

        // Create 5 convolutional layers with increasing input channels due to concatenation
        // All use 3x3 kernels with padding=1 to preserve spatial dimensions
        _convLayers = new ConvolutionalLayer<T>[5];

        // Conv1: numFeatures -> growthChannels
        _convLayers[0] = new ConvolutionalLayer<T>(
            outputDepth: growthChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // Conv2: numFeatures + growthChannels -> growthChannels
        _convLayers[1] = new ConvolutionalLayer<T>(
            outputDepth: growthChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // Conv3: numFeatures + 2*growthChannels -> growthChannels
        _convLayers[2] = new ConvolutionalLayer<T>(
            outputDepth: growthChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // Conv4: numFeatures + 3*growthChannels -> growthChannels
        _convLayers[3] = new ConvolutionalLayer<T>(
            outputDepth: growthChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        // Conv5: numFeatures + 4*growthChannels -> numFeatures (back to original channels)
        _convLayers[4] = new ConvolutionalLayer<T>(
            outputDepth: numFeatures,
            kernelSize: 3,
            stride: 1,
            padding: 1,
            activationFunction: null);

        foreach (var conv in _convLayers)
            RegisterSubLayer(conv);
    }

    /// <inheritdoc/>
    /// <remarks>
    /// Resolves input height/width from <c>input.Shape</c> and propagates
    /// to each conv sub-layer via <see cref="LayerBase{T}.ResolveFromShape"/>
    /// so weights are allocated up front and downstream Forward calls see
    /// fully resolved shapes.
    /// </remarks>
    protected override void OnFirstForward(Tensor<T> input)
    {
        var s = input._shape;
        int inChannels, inH, inW;
        if (s.Length == 3) { inChannels = s[0]; inH = s[1]; inW = s[2]; }
        else if (s.Length == 4) { inChannels = s[1]; inH = s[2]; inW = s[3]; }
        else
            throw new ArgumentException(
                $"ResidualDenseBlock requires rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank {s.Length}.",
                nameof(input));
        if (inChannels != _numFeatures)
            throw new ArgumentException(
                $"ResidualDenseBlock expected {_numFeatures} input channels, got {inChannels}.",
                nameof(input));

        _inputHeight = inH;
        _inputWidth = inW;

        // Each conv's input is the running concat of x0 + previous growth-channel
        // outputs, so the channel counts grow by _growthChannels per layer.
        int currentChannels = _numFeatures;
        for (int i = 0; i < 4; i++)
        {
            _convLayers[i].ResolveFromShape(new[] { currentChannels, inH, inW });
            currentChannels += _growthChannels;
        }
        _convLayers[4].ResolveFromShape(new[] { currentChannels, inH, inW });

        // Propagate parent's training mode to each conv (RegisterSubLayer
        // doesn't backfill it for sub-layers allocated before SetTrainingMode).
        foreach (var conv in _convLayers)
            conv.SetTrainingMode(IsTrainingMode);

        ResolveShapes(
            new[] { _numFeatures, inH, inW },
            new[] { _numFeatures, inH, inW });

        // Replay any Deserialize-buffered parameters now that conv shapes are resolved.
        if (_pendingParameters is not null)
        {
            var pending = _pendingParameters;
            _pendingParameters = null;
            int offset = 0;
            foreach (var conv in _convLayers)
            {
                int count = conv.GetParameters().Length;
                conv.SetParameters(pending.SubVector(offset, count));
                offset += count;
            }
        }
    }

    #endregion

    #region Forward Pass

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (!IsShapeResolved) OnFirstForward(input);

        _lastInput = ShouldCacheForBackward ? input : null; // #1668: skip in inference (arena safety)
        _convOutputs = new Tensor<T>[5];
        _activationOutputs = new Tensor<T>[4]; // Only first 4 have activation
        _concatInputs = new Tensor<T>[5];

        // x0 = input
        var x0 = input;

        // Conv1: x1 = LeakyReLU(Conv(x0))
        _concatInputs[0] = x0;
        _convOutputs[0] = _convLayers[0].Forward(x0);
        _activationOutputs[0] = ApplyLeakyReLU(_convOutputs[0]);
        var x1 = _activationOutputs[0];

        // Conv2: x2 = LeakyReLU(Conv(concat(x0, x1)))
        var concat1 = ConcatenateChannels(x0, x1);
        _concatInputs[1] = concat1;
        _convOutputs[1] = _convLayers[1].Forward(concat1);
        _activationOutputs[1] = ApplyLeakyReLU(_convOutputs[1]);
        var x2 = _activationOutputs[1];

        // Conv3: x3 = LeakyReLU(Conv(concat(x0, x1, x2)))
        var concat2 = ConcatenateChannels(concat1, x2);
        _concatInputs[2] = concat2;
        _convOutputs[2] = _convLayers[2].Forward(concat2);
        _activationOutputs[2] = ApplyLeakyReLU(_convOutputs[2]);
        var x3 = _activationOutputs[2];

        // Conv4: x4 = LeakyReLU(Conv(concat(x0, x1, x2, x3)))
        var concat3 = ConcatenateChannels(concat2, x3);
        _concatInputs[3] = concat3;
        _convOutputs[3] = _convLayers[3].Forward(concat3);
        _activationOutputs[3] = ApplyLeakyReLU(_convOutputs[3]);
        var x4 = _activationOutputs[3];

        // Conv5: x5 = Conv(concat(x0, x1, x2, x3, x4)) - NO activation
        var concat4 = ConcatenateChannels(concat3, x4);
        _concatInputs[4] = concat4;
        _convOutputs[4] = _convLayers[4].Forward(concat4);
        var x5 = _convOutputs[4];

        // Local residual learning: output = x5 * residualScale + x0
        return AddResidual(x5, x0, _residualScale);
    }

    #endregion

    #region Backward Pass

    #endregion

    #region Helper Methods

    /// <summary>
    /// Concatenates two tensors along the channel dimension.
    /// </summary>
    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        // Channel axis is 0 for 3D [C,H,W] and 1 for 4D [B,C,H,W]
        int channelAxis = a.Rank == 4 ? 1 : 0;
        return Engine.TensorConcatenate([a, b], axis: channelAxis);
    }

    /// <summary>
    /// Splits a tensor along the channel dimension.
    /// </summary>
    private (Tensor<T> first, Tensor<T> second) SplitGradient(Tensor<T> grad, int firstChannels, int secondChannels)
    {
        if (grad.Rank == 3)
        {
            int height = grad.Shape[1];
            int width = grad.Shape[2];
            int spatialSize = height * width;

            var first = new Tensor<T>([firstChannels, height, width]);
            var second = new Tensor<T>([secondChannels, height, width]);

            for (int c = 0; c < firstChannels; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    first.Data.Span[c * spatialSize + hw] = grad.Data.Span[c * spatialSize + hw];
                }
            }

            for (int c = 0; c < secondChannels; c++)
            {
                for (int hw = 0; hw < spatialSize; hw++)
                {
                    second.Data.Span[c * spatialSize + hw] = grad.Data.Span[(firstChannels + c) * spatialSize + hw];
                }
            }

            return (first, second);
        }
        else // 4D
        {
            int batch = grad.Shape[0];
            int height = grad.Shape[2];
            int width = grad.Shape[3];
            int spatialSize = height * width;
            int totalChannels = firstChannels + secondChannels;

            var first = new Tensor<T>([batch, firstChannels, height, width]);
            var second = new Tensor<T>([batch, secondChannels, height, width]);

            for (int n = 0; n < batch; n++)
            {
                for (int c = 0; c < firstChannels; c++)
                {
                    for (int hw = 0; hw < spatialSize; hw++)
                    {
                        int srcIdx = n * totalChannels * spatialSize + c * spatialSize + hw;
                        int dstIdx = n * firstChannels * spatialSize + c * spatialSize + hw;
                        first.Data.Span[dstIdx] = grad.Data.Span[srcIdx];
                    }
                }

                for (int c = 0; c < secondChannels; c++)
                {
                    for (int hw = 0; hw < spatialSize; hw++)
                    {
                        int srcIdx = n * totalChannels * spatialSize + (firstChannels + c) * spatialSize + hw;
                        int dstIdx = n * secondChannels * spatialSize + c * spatialSize + hw;
                        second.Data.Span[dstIdx] = grad.Data.Span[srcIdx];
                    }
                }
            }

            return (first, second);
        }
    }

    /// <summary>
    /// Applies LeakyReLU activation to the input tensor.
    /// </summary>
    private Tensor<T> ApplyLeakyReLU(Tensor<T> input)
    {
        // ENGINE OP, NOT A RAW SPAN LOOP. Filling a rented tensor element by element produces a
        // tensor the autodiff tape has never seen: DifferentiableOps is internal to
        // AiDotNet.Tensors, so an Engine call is the ONLY way a layer's math reaches the tape.
        //
        // Measured directly before this change, running the layer under a tape and asking for a
        // gradient on every cached tensor: _convOutputs[0..3] (the pre-activations) came back
        // ABSENT while their _activationOutputs did have gradients, and _convOutputs[4] -- the one
        // conv whose output never passes through here -- was on the tape. The chain was cut at
        // exactly these four calls, so conv1..conv4's weights and the input's route through them
        // lost their gradient entirely. What survived was the input's path via the recorded
        // TensorConcatenate and AddResidual, which is why the input VJP was partially right
        // (~1.5-12% off) rather than absent, and why the error was independent of activation sign.
        //
        // Engine.LeakyReLU is the same primitive UNetDiscriminator and GraphAttentionLayer already
        // use, and it allocates through TensorAllocator.Rent internally exactly as this did, so
        // nothing is given up on the allocation path. Alpha still comes from _activation, so the
        // ESRGAN 0.2 negative slope is unchanged.
        return Engine.LeakyReLU(input, _activation.Alpha);
    }

    /// <summary>
    /// Backward pass through LeakyReLU activation.
    /// </summary>
    private Tensor<T> BackwardActivation(Tensor<T> activationOutput, Tensor<T> gradient)
    {
        var output = TensorAllocator.Rent<T>(gradient._shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            output.Data.Span[i] = NumOps.Multiply(
                gradient.Data.Span[i],
                _activation.Derivative(activationOutput.Data.Span[i]));
        }
        return output;
    }

    /// <summary>
    /// Adds residual with scaling: output = a * scale + b.
    /// </summary>
    private Tensor<T> AddResidual(Tensor<T> a, Tensor<T> b, double scale)
    {
        var scaleT = NumOps.FromDouble(scale);
        var scaled = Engine.TensorMultiplyScalar(a, scaleT);
        return Engine.TensorAdd(scaled, b);
    }

    /// <summary>
    /// Scales a tensor by a factor.
    /// </summary>
    private Tensor<T> ScaleGradient(Tensor<T> gradient, double scale)
    {
        var scaleT = NumOps.FromDouble(scale);
        return Engine.TensorMultiplyScalar(gradient, scaleT);
    }

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    #endregion

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        foreach (var conv in _convLayers)
        {
            conv.UpdateParameters(learningRate);
        }
    }

    public override Vector<T> GetParameterGradients()
    {
        var gradVectors = _convLayers
            .Select(c => c.GetParameterGradients())
            .ToArray();
        return Vector<T>.Concatenate(gradVectors);
    }

    public override void ClearGradients()
    {
        foreach (var conv in _convLayers)
            conv.ClearGradients();
    }

    [Scratch]
    private Vector<T>? _pendingParameters;

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _convOutputs = null;
        _activationOutputs = null;
        _concatInputs = null;

        foreach (var conv in _convLayers)
        {
            conv.ResetState();
        }
    }


    /// <summary>
    /// Builds a Conv2D computation node from a ConvolutionalLayer.
    /// </summary>
    private ComputationNode<T> BuildConvNode(ConvolutionalLayer<T> conv, ComputationNode<T> input, string namePrefix)
    {
        var biases = conv.GetBiases();
        return TensorOperations<T>.Conv2D(
            input,
            TensorOperations<T>.Constant(conv.GetFilters(), $"{namePrefix}kernel"),
            biases is not null ? TensorOperations<T>.Constant(biases, $"{namePrefix}bias") : null,
            stride: new int[] { conv.Stride, conv.Stride },
            padding: new int[] { conv.Padding, conv.Padding });
    }

    /// <summary>
    /// Scales a computation node by a scalar value using element-wise multiplication.
    /// </summary>
    private static ComputationNode<T> ScaleNode(ComputationNode<T> node, double scale, string name)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var scaleValue = numOps.FromDouble(scale);

        // Create a constant tensor filled with the scale value matching the input shape
        var scaleTensor = new Tensor<T>(node.Value._shape);
        for (int i = 0; i < scaleTensor.Length; i++)
        {
            scaleTensor.Data.Span[i] = scaleValue;
        }

        var scaleNode = TensorOperations<T>.Constant(scaleTensor, name);
        return TensorOperations<T>.ElementwiseMultiply(node, scaleNode);
    }

    #endregion

}
