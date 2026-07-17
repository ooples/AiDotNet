using System.IO;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Motion;

/// <summary>
/// Recurrent All-pairs Field Transforms (RAFT) for optical flow estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> RAFT is a state-of-the-art optical flow estimation model that predicts
/// the motion between two consecutive video frames. Optical flow represents how pixels move
/// from one frame to the next, useful for:
/// - Motion analysis and tracking
/// - Video stabilization
/// - Action recognition
/// - Video compression
/// - Self-driving car perception
///
/// RAFT iteratively refines its flow estimate using a recurrent update mechanism,
/// making it very accurate while remaining efficient.
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Feature extraction using CNN encoder
/// - 4D correlation volumes for all-pairs matching
/// - GRU-based iterative update operator
/// - Multi-scale feature pyramids
/// </para>
/// <para>
/// <b>Reference:</b> Teed and Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow"
/// ECCV 2020.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create a RAFT model for recurrent optical flow estimation
/// var raft = new RAFT&lt;double&gt;();
///
/// // Or configure with custom architecture
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 256, inputWidth: 256, inputDepth: 3, outputSize: 2);
/// var model = new RAFT&lt;double&gt;(architecture, numFeatures: 128, numIterations: 12);
/// </code>
/// </example>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("RAFT: Recurrent All-Pairs Field Transforms for Optical Flow",
    "https://arxiv.org/abs/2003.12039",
    Year = 2020,
    Authors = "Zachary Teed, Jia Deng")]
public class RAFT<T> : OpticalFlowBase<T>
{
    private readonly RAFTOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly int _height;
    private readonly int _width;
    private readonly int _channels;
    private readonly int _numFeatures;
    private readonly int _correlationLevels;
    private readonly int _correlationRadius;


    // Feature encoder
    private readonly List<ConvolutionalLayer<T>> _featureEncoder;

    // Context encoder
    private readonly List<ConvolutionalLayer<T>> _contextEncoder;

    // Correlation lookup
    private ConvolutionalLayer<T>? _correlationConv;

    // GRU update block
    private ConvolutionalLayer<T>? _gruConvZ;
    private ConvolutionalLayer<T>? _gruConvR;
    private ConvolutionalLayer<T>? _gruConvH;

    // Flow update heads
    private ConvolutionalLayer<T>? _flowHead;
    private ConvolutionalLayer<T>? _deltaFlowHead;

    // Upsampling
    private ConvolutionalLayer<T>? _upsampleConv;

    private const int DefaultNumFeatures = 256;
    private const int DefaultCorrelationLevels = 4;
    private const int DefaultCorrelationRadius = 4;
    private const int DefaultNumIterations = 12;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether training is supported.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the input height for frames.
    /// </summary>
    internal int InputHeight => _height;

    /// <summary>
    /// Gets the input width for frames.
    /// </summary>
    internal int InputWidth => _width;

    /// <summary>
    /// Gets the number of input channels.
    /// </summary>
    internal int InputChannels => _channels;

    /// <summary>
    /// Gets the number of refinement iterations.
    /// </summary>


    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public RAFT()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputHeight: 256, inputWidth: 256, inputDepth: 3,
            outputSize: 2))
    {
    }

    /// <summary>
    /// Initializes a new instance of the RAFT class.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">The number of features in intermediate layers.</param>
    /// <param name="correlationLevels">The number of levels in the correlation pyramid.</param>
    /// <param name="correlationRadius">The search radius for correlation lookup.</param>
    /// <param name="numIterations">The number of GRU refinement iterations.</param>
    public RAFT(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = DefaultNumFeatures,
        int correlationLevels = DefaultCorrelationLevels,
        int correlationRadius = DefaultCorrelationRadius,
        int numIterations = DefaultNumIterations,
        RAFTOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new RAFTOptions();
        Options = _options;
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 640;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFeatures = numFeatures;
        _correlationLevels = correlationLevels;
        _correlationRadius = correlationRadius;
        NumIterations = numIterations;

        _featureEncoder = [];
        _contextEncoder = [];

        // DEFAULT training optimizer (overridable — pass your own `optimizer`, or configure one via the
        // builder). RAFT's correlation-pyramid + GRU-refinement gradients overshoot at the framework
        // default 1e-3 Adam and drive the weights to NaN within a few steps (post-train outputs went NaN);
        // 1e-4 is the standard small optical-flow fine-tune LR and keeps training finite. Gradient clipping
        // is already on (OpticalFlowBase maxGradNorm = 1.0) but is a near-no-op under Adam, so the LR is the
        // effective lever.
        SetBaseTrainOptimizer(optimizer ?? new AiDotNet.Optimizers.AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = 1e-4 }));

        InitializeNativeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Estimates optical flow between two frames.
    /// </summary>
    /// <param name="frame1">The first frame tensor [C, H, W] or [B, C, H, W].</param>
    /// <param name="frame2">The second frame tensor [C, H, W] or [B, C, H, W].</param>
    /// <returns>The optical flow tensor [2, H, W] or [B, 2, H, W].</returns>
    public override Tensor<T> EstimateFlow(Tensor<T> frame1, Tensor<T> frame2)
    {
        bool hasBatch = frame1.Rank == 4;
        if (!hasBatch)
        {
            frame1 = AddBatchDimension(frame1);
            frame2 = AddBatchDimension(frame2);
        }

        var concatenated = ConcatenateChannels(frame1, frame2);
        var result = Predict(concatenated);

        if (!hasBatch)
        {
            result = RemoveBatchDimension(result);
        }

        return result;
    }

    /// <summary>
    /// Estimates optical flow with intermediate flow predictions.
    /// </summary>
    /// <param name="frame1">The first frame tensor.</param>
    /// <param name="frame2">The second frame tensor.</param>
    /// <returns>List of flow predictions at each iteration.</returns>
    public List<Tensor<T>> EstimateFlowIterative(Tensor<T> frame1, Tensor<T> frame2)
    {
        bool hasBatch = frame1.Rank == 4;
        if (!hasBatch)
        {
            frame1 = AddBatchDimension(frame1);
            frame2 = AddBatchDimension(frame2);
        }

        var flowIterations = ForwardIterative(frame1, frame2);

        if (!hasBatch)
        {
            for (int i = 0; i < flowIterations.Count; i++)
            {
                flowIterations[i] = RemoveBatchDimension(flowIterations[i]);
            }
        }

        return flowIterations;
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var frame1 = SliceChannels(input, 0, _channels);
        var frame2 = SliceChannels(input, _channels, _channels * 2);
        var flowIterations = ForwardIterative(frame1, frame2);
        return flowIterations[^1];
    }

    /// <inheritdoc/>
    /// <remarks>
    /// RAFT's computation graph is the iterative correlation/GRU update in
    /// <see cref="ForwardIterative"/>, NOT a sequential pass over the flat
    /// <c>Layers</c> list. The base <see cref="NeuralNetworkBase{T}.ForwardForTraining"/>
    /// runs the layers sequentially, which feeds the 2-frame channel-concat
    /// into the single-frame feature encoder ("Expected input depth 3, got 6").
    /// Route the training forward through the same graph Predict uses so the
    /// tape records the real operations.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        var frame1 = SliceChannels(input, 0, _channels);
        var frame2 = SliceChannels(input, _channels, _channels * 2);
        var flowIterations = ForwardIterative(frame1, frame2);
        return flowIterations[^1];
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    #endregion

    #region Private Methods

    /// <summary>
    /// Throws if the layer fields have not been initialized via <see cref="InitializeNativeLayers"/>.
    /// </summary>
    private void ThrowIfNotInitialized()
    {
        if (_correlationConv is null || _gruConvZ is null || _gruConvR is null ||
            _gruConvH is null || _flowHead is null || _deltaFlowHead is null ||
            _upsampleConv is null)
        {
            throw new InvalidOperationException(
                $"{nameof(RAFT<T>)} has not been initialized. Ensure the constructor completed successfully.");
        }
    }

    private void InitializeNativeLayers()
    {
        // Check for user-provided custom layers
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            var layers = LayerHelper<T>.CreateRAFTLayers(
                channels: _channels, height: _height, width: _width,
                numFeatures: _numFeatures, correlationLevels: _correlationLevels,
                correlationRadius: _correlationRadius).ToList();
            Layers.AddRange(layers);
        }

        // Distribute layers to sub-lists for forward pass
        int idx = 0;
        // Feature encoder (5 layers)
        for (int i = 0; i < 5; i++)
            _featureEncoder.Add((ConvolutionalLayer<T>)Layers[idx++]);
        // Context encoder (5 layers)
        for (int i = 0; i < 5; i++)
            _contextEncoder.Add((ConvolutionalLayer<T>)Layers[idx++]);
        // Correlation conv
        _correlationConv = (ConvolutionalLayer<T>)Layers[idx++];
        // GRU update block
        _gruConvZ = (ConvolutionalLayer<T>)Layers[idx++];
        _gruConvR = (ConvolutionalLayer<T>)Layers[idx++];
        _gruConvH = (ConvolutionalLayer<T>)Layers[idx++];
        // Flow heads
        _flowHead = (ConvolutionalLayer<T>)Layers[idx++];
        _deltaFlowHead = (ConvolutionalLayer<T>)Layers[idx++];
        // Upsample conv
        _upsampleConv = (ConvolutionalLayer<T>)Layers[idx++];
    }

    private List<Tensor<T>> ForwardIterative(Tensor<T> frame1, Tensor<T> frame2)
    {
        ThrowIfNotInitialized();

        var correlationConv = _correlationConv ?? throw new InvalidOperationException("Correlation conv not initialized.");
        var flowHead = _flowHead ?? throw new InvalidOperationException("Flow head not initialized.");
        var deltaFlowHead = _deltaFlowHead ?? throw new InvalidOperationException("Delta flow head not initialized.");

        int batchSize = frame1.Shape[0];

        var fmap1 = ExtractFeatures(frame1);
        var fmap2 = ExtractFeatures(frame2);
        var context = ExtractContext(frame1);

        // Derive the flow-field resolution from the ACTUAL feature-map spatial
        // dims rather than architecture._height/8. The feature encoder
        // downsamples whatever input it is given by 8×, so when the real input
        // size differs from the architecture's configured size (e.g. a 64×64
        // test frame against the parameterless ctor's 256×256 default), a
        // _height/8 flow grid (32×32) no longer matches the encoder output
        // (8×8) and the GRU-input ConcatenateChannels fails with
        // "Mismatch at axis 2". Sourcing the grid from fmap1 keeps flow,
        // correlation, and context spatially aligned at any input size.
        int featHeight = fmap1.Shape[2];
        int featWidth = fmap1.Shape[3];

        var flow = new Tensor<T>([batchSize, 2, featHeight, featWidth]);
        var hiddenState = context;

        var flowPredictions = new List<Tensor<T>>();

        for (int iter = 0; iter < NumIterations; iter++)
        {
            var correlation = ComputeCorrelation(fmap1, fmap2, flow);
            var corrFeatures = correlationConv.Forward(correlation);

            var gruInput = ConcatenateChannels(
                ConcatenateChannels(context, corrFeatures),
                flow);

            hiddenState = GRUUpdate(hiddenState, gruInput);

            var flowFeatures = flowHead.Forward(hiddenState);
            var deltaFlow = deltaFlowHead.Forward(flowFeatures);

            flow = AddTensors(flow, deltaFlow);

            var fullResFlow = UpsampleFlow(flow, hiddenState, 8);
            flowPredictions.Add(fullResFlow);
        }

        return flowPredictions;
    }

    private Tensor<T> ExtractFeatures(Tensor<T> frame)
    {
        var features = frame;
        foreach (var encoder in _featureEncoder)
        {
            features = encoder.Forward(features);
        }
        return features;
    }

    private Tensor<T> ExtractContext(Tensor<T> frame)
    {
        var context = frame;
        foreach (var encoder in _contextEncoder)
        {
            context = encoder.Forward(context);
        }
        return context;
    }

    private Tensor<T> ComputeCorrelation(Tensor<T> fmap1, Tensor<T> fmap2, Tensor<T> flow)
    {
        int batchSize = fmap1.Shape[0];
        int channels = fmap1.Shape[1];
        int height = fmap1.Shape[2];
        int width = fmap1.Shape[3];

        // Flow-guided local correlation (Teed & Deng 2020, sec 3.2), tape-aware. The paper looks the flow
        // up in a precomputed all-pairs cost volume; the mathematically-equivalent, cheap, differentiable
        // formulation is: warp fmap2 by the CURRENT flow ONCE (a single GridSample), then take, for each
        // (dh,dw) in the level-scaled window, the per-pixel dot product of fmap1 with the integer-shifted
        // warped fmap2. Shifting is a pad + slice (both differentiable), so a forward costs a couple of
        // GridSamples rather than one per window cell. The scalar loop this replaces severed the tape
        // (starving both feature encoders of gradient); an earlier one-GridSample-per-cell version was
        // correct but ~24 s/step. Gradients reach fmap1, fmap2 and the flow, exactly as the lookup does.
        var warped = WarpByFlow(fmap2, flow); // [B,C,H,W] fmap2 resampled at the current flow

        int radius = _correlationRadius;
        int maxScale = 1 << (_correlationLevels - 1);
        int pad = radius * maxScale;
        var padded = Engine.Pad(warped, pad, pad, pad, pad, NumOps.Zero); // [B,C,H+2p,W+2p]

        var corrChannels = new List<Tensor<T>>();
        for (int level = 0; level < _correlationLevels; level++)
        {
            int scale = 1 << level;
            for (int dh = -radius; dh <= radius; dh++)
            {
                for (int dw = -radius; dw <= radius; dw++)
                {
                    // Window cell (dh,dw) at this pyramid level = the warped fmap2 shifted by (dh,dw)*scale
                    // pixels, read out of the padded tensor as a plain (differentiable) slice.
                    var shifted = Engine.TensorSlice(
                        padded,
                        new[] { 0, 0, pad + dh * scale, pad + dw * scale },
                        new[] { batchSize, channels, height, width });
                    var prod = Engine.TensorMultiply(fmap1, shifted);                     // [B,C,H,W]
                    corrChannels.Add(Engine.ReduceSum(prod, new[] { 1 }, keepDims: true)); // [B,1,H,W]
                }
            }
        }

        return Engine.TensorConcatenate(corrChannels.ToArray(), axis: 1); // [B, corrDim, H, W]
    }

    private Tensor<T> WarpByFlow(Tensor<T> image, Tensor<T> flow)
    {
        // Differentiable backward warp: resample `image` [B,C,H,W] at each pixel displaced by `flow`
        // [B,2,H,W] (x=dx, y=dy) via a normalized identity grid + flow offset + GridSample — the same
        // construction RIFE/VFI use. Gradients flow to both `image` and `flow`.
        int batchSize = image.Shape[0];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var identityTheta = new Tensor<T>([batchSize, 2, 3]);
        for (int b = 0; b < batchSize; b++)
        {
            identityTheta[b, 0, 0] = NumOps.One; // x scale
            identityTheta[b, 1, 1] = NumOps.One; // y scale
        }
        var baseGrid = Engine.AffineGrid(identityTheta, height, width); // [B,H,W,2] identity sampling grid

        // A dx-pixel shift is 2*dx/(W-1) in the normalized [-1,1] grid coordinates.
        double sx = width > 1 ? 2.0 / (width - 1) : 0.0;
        double sy = height > 1 ? 2.0 / (height - 1) : 0.0;
        var flowNHWC = Engine.TensorPermute(flow, new[] { 0, 2, 3, 1 }); // [B,H,W,2] (x=dx, y=dy)
        var normScale = new Tensor<T>([batchSize, height, width, 2]);
        var nsSpan = normScale.Data.Span;
        for (int i = 0; i + 1 < nsSpan.Length; i += 2) { nsSpan[i] = NumOps.FromDouble(sx); nsSpan[i + 1] = NumOps.FromDouble(sy); }
        var grid = Engine.TensorAdd(baseGrid, Engine.TensorMultiply(flowNHWC, normScale));

        var imageNHWC = Engine.TensorPermute(image, new[] { 0, 2, 3, 1 }); // [B,H,W,C]
        var warpedNHWC = Engine.GridSample(imageNHWC, grid);               // [B,H,W,C]
        return Engine.TensorPermute(warpedNHWC, new[] { 0, 3, 1, 2 });     // [B,C,H,W]
    }

    private Tensor<T> GRUUpdate(Tensor<T> hiddenState, Tensor<T> gruInput)
    {
        ThrowIfNotInitialized();

        var gruConvZ = _gruConvZ ?? throw new InvalidOperationException("GRU conv Z not initialized.");
        var gruConvR = _gruConvR ?? throw new InvalidOperationException("GRU conv R not initialized.");
        var gruConvH = _gruConvH ?? throw new InvalidOperationException("GRU conv H not initialized.");

        var z = Engine.Sigmoid(gruConvZ.Forward(gruInput));
        var r = Engine.Sigmoid(gruConvR.Forward(gruInput));
        var hNew = Engine.Tanh(gruConvH.Forward(gruInput));

        var ones = Tensor<T>.CreateDefault(z._shape, NumOps.One);
        var oneMinusZ = Engine.TensorSubtract(ones, z);
        var term1 = Engine.TensorMultiply(oneMinusZ, hiddenState);
        var term2 = Engine.TensorMultiply(z, hNew);

        return Engine.TensorAdd(term1, term2);
    }

    private Tensor<T> UpsampleFlow(Tensor<T> flow, Tensor<T> features, int factor)
    {
        // Paper-faithful convex upsampling (Teed & Deng 2020, sec 3.3): predict
        // per-output-pixel 3×3 mask weights from the GRU hidden state, soft-max
        // them so the mask is a convex combination, then synthesize each
        // full-resolution flow pixel as a learnable weighted sum of the 3×3
        // low-resolution flow neighborhood (scaled by `factor` so flow
        // magnitudes match the new pixel grid).
        //
        // mask[b, i·F + j, k, h, w] = softmax_k of upsample_conv(features),
        // up_flow[b, c, h·F + i, w·F + j] = Σ_k mask[…] · factor · flow[b, c, h+dh_k, w+dw_k]
        //
        // The reshape-multiply-reduce-pixel-shuffle sequence is entirely on the
        // tape, so _upsampleConv participates in the backward sweep and trains
        // jointly with the rest of the recurrent flow refiner.
        var upsampleConv = _upsampleConv ?? throw new InvalidOperationException("Upsample conv not initialized.");

        int B = flow.Shape[0];
        int H = flow.Shape[2];
        int W = flow.Shape[3];
        int F2 = factor * factor;

        // 1. Mask: upsampleConv(features) ∈ [B, 9·F², H, W]. Reshape so that
        //    the 9-neighbor axis is contiguous, then soft-max along it to get
        //    a convex combination over the 3×3 source neighborhood per sub-pixel.
        var mask = upsampleConv.Forward(features);
        var maskGrouped = Engine.Reshape(mask, new[] { B, F2, 9, H, W });
        var maskNormalized = Engine.Softmax(maskGrouped, axis: 2);

        // 2. Unfolded flow: ×factor magnitude rescale, then pad+crop nine 3×3
        //    spatial offsets and concat along a new neighbor axis. Result:
        //    [B, 2, 9, H, W] containing each low-res pixel's 3×3 neighborhood.
        var flowScaled = Engine.TensorMultiplyScalar(flow, NumOps.FromDouble(factor));
        var flowPadded = Engine.Pad(flowScaled, 1, 1, 1, 1, NumOps.Zero);

        var patchTensors = new Tensor<T>[9];
        for (int dy = 0; dy < 3; dy++)
        {
            for (int dx = 0; dx < 3; dx++)
            {
                var patch = Engine.Crop(flowPadded, dy, dx, H, W);
                patchTensors[dy * 3 + dx] = Engine.Reshape(patch, new[] { B, 2, 1, H, W });
            }
        }
        var flowUnfolded = Engine.TensorConcatenate(patchTensors, axis: 2);

        // 3. For each flow component c ∈ {0, 1}, slice [B, 1, 9, H, W], broadcast-
        //    multiply against the [B, F², 9, H, W] mask, and sum across the 9-
        //    neighbor axis. Keeps the two flow channels in separate accumulators
        //    so the final stack lands in PixelShuffle's canonical [c, sub-pixel]
        //    channel order — avoids an N-D transpose we don't have on the tape.
        var sliceStart = new int[] { 0, 0, 0, 0, 0 };
        var sliceLen = new int[] { B, 1, 9, H, W };
        var subPixelByChannel = new Tensor<T>[2];
        for (int c = 0; c < 2; c++)
        {
            sliceStart[1] = c;
            var flowC = Engine.TensorSlice(flowUnfolded, sliceStart, sliceLen);
            var product = Engine.TensorBroadcastMultiply(maskNormalized, flowC);
            subPixelByChannel[c] = Engine.ReduceSum(product, new[] { 2 }, keepDims: false);
        }

        // 4. Stack the per-channel sub-pixel maps in PixelShuffle's expected
        //    layout [B, C·F², H, W] with C outer, F² inner. PixelShuffle then
        //    reshape-permute-reshapes to [B, C, F·H, F·W] (a tape-tracked
        //    depth-to-space — the only N-D permute we have available here).
        var stacked = Engine.TensorConcatenate(subPixelByChannel, axis: 1);
        return Engine.PixelShuffle(stacked, factor);
    }

    private Tensor<T> ConcatenateChannels(Tensor<T> t1, Tensor<T> t2)
    {
        return Engine.TensorConcatenate([t1, t2], axis: 1);
    }

    private Tensor<T> SliceChannels(Tensor<T> input, int startChannel, int endChannel)
    {
        // Tape-aware channel slice (was a scalar copy loop that severed the tape). Input is [B, C, H, W].
        int batchSize = input.Shape[0];
        int numChannels = endChannel - startChannel;
        int height = input.Shape[2];
        int width = input.Shape[3];
        return Engine.TensorSlice(input,
            new[] { 0, startChannel, 0, 0 },
            new[] { batchSize, numChannels, height, width });
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorAdd(a, b);
    }

    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        // Convert [C, H, W] to [1, C, H, W]
        int c = tensor.Shape[0];
        int h = tensor.Shape[1];
        int w = tensor.Shape[2];

        var result = new Tensor<T>([1, c, h, w]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        // Convert [1, C, H, W] to [C, H, W]
        int c = tensor.Shape[1];
        int h = tensor.Shape[2];
        int w = tensor.Shape[3];

        var result = new Tensor<T>([c, h, w]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    #endregion

    #region Abstract Implementation

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        ClearLayers();

        foreach (var layer in _featureEncoder) Layers.Add(layer);
        foreach (var layer in _contextEncoder) Layers.Add(layer);
        if (_correlationConv is not null) Layers.Add(_correlationConv);
        if (_gruConvZ is not null) Layers.Add(_gruConvZ);
        if (_gruConvR is not null) Layers.Add(_gruConvR);
        if (_gruConvH is not null) Layers.Add(_gruConvH);
        if (_flowHead is not null) Layers.Add(_flowHead);
        if (_deltaFlowHead is not null) Layers.Add(_deltaFlowHead);
        if (_upsampleConv is not null) Layers.Add(_upsampleConv);
    }

    private bool _shapesProbed;

    /// <inheritdoc/>
    /// <remarks>
    /// RAFT's forward is non-linear — a per-frame feature/context encoder, a correlation volume over
    /// the 6-channel feature pair, and GRU update convs — so the base linear Layers-walk mis-sizes the
    /// correlation / GRU convs (it resolved one to 3 while the real forward feeds 6, throwing
    /// "Expected input depth 3, but got 6"). Resolve every lazy conv through a real forward on a small
    /// dummy frame-pair instead, so each conv sees exactly the input its production forward feeds it.
    /// </remarks>
    protected override void ResolveLazyLayerShapes()
    {
        if (_shapesProbed || _featureEncoder.Count == 0) return;
        _shapesProbed = true;
        int c = _channels > 0 ? _channels : 3;
        // Keep the probe cheap: ONE GRU iteration resolves every conv (the update convs are reused
        // each iteration), and a 32×32 pair downsamples to 4×4 features so the all-pairs correlation
        // is trivial — the full 12-iteration / 64×64 forward here made construction blow the test timeout.
        int savedIters = NumIterations;
        NumIterations = 1;
        try { _ = PredictCore(new Tensor<T>([1, c * 2, 32, 32])); }
        finally { NumIterations = savedIters; }
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        foreach (var layer in Layers)
        {
            var layerParams = layer.GetParameters();
            int paramCount = layerParams.Length;
            if (paramCount > 0 && offset + paramCount <= parameters.Length)
            {
                var slice = new Vector<T>(paramCount);
                for (int i = 0; i < paramCount; i++)
                {
                    slice[i] = parameters[offset + i];
                }
                layer.SetParameters(slice);
                offset += paramCount;
            }
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "RAFT" },
            { "Description", "Recurrent All-Pairs Field Transforms for Optical Flow" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "InputChannels", _channels },
            { "NumFeatures", _numFeatures },
            { "CorrelationLevels", _correlationLevels },
            { "CorrelationRadius", _correlationRadius },
            { "NumIterations", NumIterations },
            { "NumLayers", Layers.Count }
        };

        return new ModelMetadata<T>
        {
            AdditionalInfo = additionalInfo,
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height);
        writer.Write(_width);
        writer.Write(_channels);
        writer.Write(_numFeatures);
        writer.Write(_correlationLevels);
        writer.Write(_correlationRadius);
        writer.Write(NumIterations);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new RAFT<T>(Architecture, _numFeatures, _correlationLevels, _correlationRadius, NumIterations);
    }

    #endregion

    #region Base Class Abstract Methods

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames)
    {
        return NormalizeFrames(rawFrames);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        return modelOutput;
    }

    #endregion

}
