using System.IO;
using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.Options;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// BasicVSR++ (Basic Video Super-Resolution++) for temporal video super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// BasicVSR++ (Chan et al. 2022) improves upon BasicVSR with:
/// - Second-order grid propagation for better temporal modeling
/// - Flow-guided deformable alignment for accurate feature alignment
/// - Bidirectional propagation for utilizing both past and future frames
/// </para>
/// <para>
/// <b>Implementation fidelity note:</b> This native implementation realizes flow-guided
/// deformable alignment and SECOND-ORDER bidirectional grid propagation faithfully and
/// trains them end-to-end through the autodiff tape (see <see cref="Train"/>). Each
/// propagation step aggregates BOTH the immediately adjacent frame (i±1) and the second
/// neighbour (i±2), with the i→i±2 flow formed by COMPOSING the two adjacent flows
/// (Chan et al. 2022, Sec. 3.1); the deformable-alignment layer resolves its widened
/// (current + i±1 + i±2) input-channel count lazily on the first forward. One
/// simplification remains: the SPyNet flow estimator acts as a fixed sampling guide — its
/// warp keeps the sampled features on the tape (gradients reach the reconstruction
/// network) but SPyNet's own weights are not fine-tuned here, matching the common
/// "pre-trained flow" setup rather than the paper's fully joint training.
/// </para>
/// <para>
/// <b>For Beginners:</b> BasicVSR++ is a video super-resolution model that upscales
/// low-resolution videos to higher resolution while maintaining temporal consistency.
/// Unlike single-image methods (like RealESRGAN), it uses information from multiple
/// frames to produce sharper and more consistent results.
///
/// Key concepts:
/// 1. <b>Bidirectional Propagation:</b> Uses both past and future frames to enhance
///    the current frame, ensuring temporal coherence.
/// 2. <b>Optical Flow:</b> Estimates how pixels move between frames to align features.
/// 3. <b>Deformable Alignment:</b> Uses learned offsets to precisely align features
///    even with complex motions.
///
/// Example usage:
/// <code>
/// // Create architecture for video input
/// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.ThreeDimensional,
///     inputHeight: 64, inputWidth: 64, inputDepth: 3);
///
/// // Create model with 4x upscaling
/// var model = new BasicVSRPlusPlus&lt;double&gt;(arch, scaleFactor: 4);
///
/// // Super-resolve video frames (shape: [numFrames, 3, H, W])
/// var highResFrames = model.EnhanceVideo(lowResFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> Chan et al., "BasicVSR++: Improving Video Super-Resolution with
/// Enhanced Propagation and Alignment", CVPR 2022. https://arxiv.org/abs/2104.13371
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment",
    "https://arxiv.org/abs/2104.13371",
    Year = 2022,
    Authors = "Kelvin C.K. Chan, Shangchen Zhou, Xiangyu Xu, Chen Change Loy")]
public class BasicVSRPlusPlus<T> : VideoSuperResolutionBase<T>
{
    private readonly BasicVSRPlusPlusOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Execution Mode

    /// <summary>
    /// Indicates whether this model uses native layers (true) or ONNX model (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// The ONNX inference session for the model.
    /// </summary>
    private readonly InferenceSession? _onnxSession;

    /// <summary>
    /// Path to the ONNX model file.
    /// </summary>
    private readonly string? _onnxModelPath;

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// The SPyNet flow estimator for optical flow between frames.
    /// </summary>
    private SpyNetLayer<T>? _flowEstimator;

    /// <summary>
    /// Feature extraction layer at the input.
    /// </summary>
    private ConvolutionalLayer<T>? _featExtract;

    /// <summary>
    /// Residual blocks for feature extraction.
    /// </summary>
    private readonly List<ResidualDenseBlock<T>> _residualBlocks;

    /// <summary>
    /// Deformable alignment modules for backward propagation.
    /// </summary>
    private readonly List<DeformableConvolutionalLayer<T>> _backwardAlignments;

    /// <summary>
    /// Deformable alignment modules for forward propagation.
    /// </summary>
    private readonly List<DeformableConvolutionalLayer<T>> _forwardAlignments;

    /// <summary>
    /// Propagation convolutions for backward pass.
    /// </summary>
    private readonly List<ConvolutionalLayer<T>> _backwardConvs;

    /// <summary>
    /// Propagation convolutions for forward pass.
    /// </summary>
    private readonly List<ConvolutionalLayer<T>> _forwardConvs;

    /// <summary>
    /// Upsampling layers using PixelShuffle.
    /// </summary>
    private readonly List<PixelShuffleLayer<T>> _upsampleLayers;

    /// <summary>
    /// Final convolution for output reconstruction.
    /// </summary>
    private ConvolutionalLayer<T>? _outputConv;

    /// <summary>
    /// The upscaling factor (2 or 4).
    /// </summary>
    private readonly int _scaleFactor;

    /// <summary>
    /// Number of feature channels in the network.
    /// </summary>
    private readonly int _numFeatures;

    /// <summary>
    /// Number of residual blocks.
    /// </summary>
    private readonly int _numResidualBlocks;

    /// <summary>
    /// Number of propagation iterations per direction.
    /// </summary>
    private readonly int _numPropagations;

    /// <summary>
    /// Learning rate for training.
    /// </summary>
    private readonly double _learningRate;

    /// <summary>
    /// Test-only hook: when true, the first-order (i±1) warped neighbour is zeroed in every propagation
    /// step, leaving ONLY the second-order (i±2) contribution. This isolates the direct two-step
    /// propagation path so a test can prove that i→i±2 information transfer works, independent of the
    /// ordinary i±1 recurrence that would otherwise carry the same information transitively (2→1→0).
    /// Defaults to false, so production behaviour is unchanged.
    /// </summary>
    internal bool DisableFirstOrderPropagationForTesting { get; set; }

    /// <summary>
    /// Test-only hook: if set, invoked for each frame index during BACKWARD propagation with that frame's
    /// deformable-alignment input tensor [current | first-order warp | second-order warp] (channel-axis
    /// concatenation). Lets a test record and inspect the second-order channel slice
    /// (<c>[2·numFeatures, 3·numFeatures)</c>) to verify it responds to a two-step-away perturbation.
    /// Null in production, so it adds no overhead there.
    /// </summary>
    internal Action<int, Tensor<T>>? BackwardAlignInputRecorder { get; set; }

    #endregion

    #region Properties

    private ConvolutionalLayer<T> FeatExtract => _featExtract ?? throw new InvalidOperationException(
        $"{GetType().Name}: Native BasicVSR++ layers are not initialized. This member is only available in native mode after successful construction.");

    private ConvolutionalLayer<T> OutputConv => _outputConv ?? throw new InvalidOperationException(
        $"{GetType().Name}: Native BasicVSR++ layers are not initialized. This member is only available in native mode after successful construction.");

    private SpyNetLayer<T> FlowEstimator => _flowEstimator ?? throw new InvalidOperationException(
        $"{GetType().Name}: Flow estimator is only available in native mode after successful native-layer initialization.");

    /// <summary>
    /// Gets whether this model uses native mode (true) or ONNX mode (false).
    /// </summary>
    internal bool UseNativeMode => _useNativeMode;

    /// <summary>
    /// Gets whether training is supported (only in native mode).
    /// </summary>
    public override bool SupportsTraining => _useNativeMode;

    /// <summary>
    /// Gets the upscaling factor for this model.
    /// </summary>
    internal int UpscaleFactor => _scaleFactor;

    /// <summary>
    /// Gets the number of feature channels.
    /// </summary>
    internal int NumFeatures => _numFeatures;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public BasicVSRPlusPlus()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.ThreeDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputHeight: 256, inputWidth: 256, inputDepth: 3,
            outputSize: 2))
    {
    }

    /// <summary>
    /// Creates a BasicVSR++ model for native training and inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="scaleFactor">Upscaling factor (2 or 4). Default: 4.</param>
    /// <param name="numFeatures">Number of feature channels. Default: 64.</param>
    /// <param name="numResidualBlocks">Number of residual blocks. Default: 15.</param>
    /// <param name="numPropagations">Number of propagation iterations. Default: 2.</param>
    /// <param name="learningRate">Learning rate for training. Default: 0.0001.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Create a BasicVSR++ model with sensible defaults:
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 64, inputWidth: 64, inputDepth: 3);
    /// var model = new BasicVSRPlusPlus&lt;double&gt;(arch);
    /// </code>
    ///
    /// For faster training (lower quality):
    /// <code>
    /// var model = new BasicVSRPlusPlus&lt;double&gt;(arch,
    ///     scaleFactor: 2,
    ///     numFeatures: 32,
    ///     numResidualBlocks: 8);
    /// </code>
    /// </para>
    /// </remarks>
    public BasicVSRPlusPlus(
        NeuralNetworkArchitecture<T> architecture,
        int scaleFactor = 4,
        int numFeatures = 64,
        int numResidualBlocks = 15,
        int numPropagations = 2,
        double learningRate = 0.0001,
        BasicVSRPlusPlusOptions? options = null)
        : base(architecture, new CharbonnierLoss<T>())
    {
        _options = options ?? new BasicVSRPlusPlusOptions();
        Options = _options;

        if (scaleFactor != 2 && scaleFactor != 4)
            throw new ArgumentOutOfRangeException(nameof(scaleFactor), "Scale factor must be 2 or 4.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be at least 1.");
        if (numResidualBlocks < 1)
            throw new ArgumentOutOfRangeException(nameof(numResidualBlocks), "Number of residual blocks must be at least 1.");

        _useNativeMode = true;
        _scaleFactor = scaleFactor;
        ScaleFactor = scaleFactor;
        _numFeatures = numFeatures;
        _numResidualBlocks = numResidualBlocks;
        _numPropagations = numPropagations;
        _learningRate = learningRate;

        _residualBlocks = [];
        _backwardAlignments = [];
        _forwardAlignments = [];
        _backwardConvs = [];
        _forwardConvs = [];
        _upsampleLayers = [];

        InitializeNativeLayers(architecture);
    }

    /// <summary>
    /// Creates a BasicVSR++ model using a pretrained ONNX model for inference.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="onnxModelPath">Path to the pretrained ONNX model.</param>
    /// <param name="scaleFactor">Upscaling factor of the pretrained model. Default: 4.</param>
    /// <param name="numFeatures">Number of feature channels in the model. Default: 64 (standard BasicVSR++ architecture).</param>
    /// <param name="numResidualBlocks">Number of residual blocks. Default: 15 (standard BasicVSR++ architecture).</param>
    /// <param name="numPropagations">Number of propagation iterations. Default: 2 (standard BasicVSR++ architecture).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this when you have a pretrained BasicVSR++ model:
    /// <code>
    /// var arch = new NeuralNetworkArchitecture&lt;double&gt;(
    ///     inputType: InputType.ThreeDimensional,
    ///     inputHeight: 64, inputWidth: 64, inputDepth: 3);
    /// var model = new BasicVSRPlusPlus&lt;double&gt;(arch, "basicvsrpp_x4.onnx");
    /// var hrFrames = model.EnhanceVideo(lrFrames);
    /// </code>
    /// </para>
    /// <para>
    /// <b>Note:</b> The architecture parameters (numFeatures, numResidualBlocks, numPropagations)
    /// default to the standard BasicVSR++ values. If your ONNX model uses a different architecture,
    /// provide the correct values to ensure accurate metadata reporting.
    /// </para>
    /// </remarks>
    public BasicVSRPlusPlus(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int scaleFactor = 4,
        int numFeatures = 64,
        int numResidualBlocks = 15,
        int numPropagations = 2,
        BasicVSRPlusPlusOptions? options = null)
        : base(architecture, new CharbonnierLoss<T>())
    {
        _options = options ?? new BasicVSRPlusPlusOptions();
        Options = _options;

        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"BasicVSR++ ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _scaleFactor = scaleFactor;
        ScaleFactor = scaleFactor;
        _numFeatures = numFeatures;
        _numResidualBlocks = numResidualBlocks;
        _numPropagations = numPropagations;
        _learningRate = 0.0001;

        _residualBlocks = [];
        _backwardAlignments = [];
        _forwardAlignments = [];
        _backwardConvs = [];
        _forwardConvs = [];
        _upsampleLayers = [];

        try
        {
            _onnxSession = new InferenceSession(onnxModelPath);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to load ONNX model: {ex.Message}", ex);
        }

        InitializeLayers();
    }

    private void InitializeNativeLayers(NeuralNetworkArchitecture<T> arch)
    {
        int height = arch.InputHeight > 0 ? arch.InputHeight : 64;
        int width = arch.InputWidth > 0 ? arch.InputWidth : 64;
        int channels = arch.InputDepth > 0 ? arch.InputDepth : 3;

        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            var layers = LayerHelper<T>.CreateBasicVSRPlusPlusLayers(
                channels, height, width,
                _numFeatures, _scaleFactor,
                _numResidualBlocks, _numPropagations,
                numLevels: 5, deformGroups: 8, growthChannels: 32).ToList();
            Layers.AddRange(layers);
        }

        // Distribute layers to sub-lists for forward pass
        int idx = 0;

        // SPyNet flow estimator
        _flowEstimator = (SpyNetLayer<T>)Layers[idx++];

        // Feature extraction
        _featExtract = (ConvolutionalLayer<T>)Layers[idx++];

        // Residual blocks
        for (int i = 0; i < _numResidualBlocks; i++)
            _residualBlocks.Add((ResidualDenseBlock<T>)Layers[idx++]);

        // Deformable alignment + propagation convolutions per propagation iteration
        for (int i = 0; i < _numPropagations; i++)
        {
            _backwardAlignments.Add((DeformableConvolutionalLayer<T>)Layers[idx++]);
            _forwardAlignments.Add((DeformableConvolutionalLayer<T>)Layers[idx++]);
            _backwardConvs.Add((ConvolutionalLayer<T>)Layers[idx++]);
            _forwardConvs.Add((ConvolutionalLayer<T>)Layers[idx++]);
        }

        // Upsampling layers
        int numUpsample = _scaleFactor == 4 ? 2 : 1;
        for (int i = 0; i < numUpsample; i++)
            _upsampleLayers.Add((PixelShuffleLayer<T>)Layers[idx++]);

        // Output convolution
        _outputConv = (ConvolutionalLayer<T>)Layers[idx++];
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Enhances a sequence of video frames using temporal super-resolution.
    /// </summary>
    /// <param name="frames">Input frames tensor with shape [numFrames, channels, height, width].</param>
    /// <returns>Enhanced frames tensor with shape [numFrames, channels, height*scale, width*scale].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Pass your low-resolution video frames as a 4D tensor:
    /// <code>
    /// // Load video frames into tensor [numFrames, 3, H, W]
    /// var lrFrames = LoadVideoFrames("input.mp4");
    /// var hrFrames = model.EnhanceVideo(lrFrames);
    /// SaveVideoFrames(hrFrames, "output_4x.mp4");
    /// </code>
    /// </para>
    /// </remarks>
    public Tensor<T> EnhanceVideo(Tensor<T> frames)
    {
        if (frames == null)
            throw new ArgumentNullException(nameof(frames));

        if (_useNativeMode)
        {
            return EnhanceVideoNative(frames);
        }
        else
        {
            return EnhanceVideoOnnx(frames);
        }
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        return EnhanceVideo(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");

        var loss = LossFunction as LossFunctionBase<T>
            ?? throw new InvalidOperationException(
                "BasicVSR++ tape training requires a LossFunctionBase<T> (Charbonnier by default).");

        // BasicVSR++ is a RECURRENT architecture: the same alignment/propagation
        // layers are reapplied across frames and propagation iterations, and the
        // reconstruction is assembled by stacking per-frame outputs. The generic
        // sequential layer-chain forward (NeuralNetworkBase.ForwardForTraining) does
        // not model this — it would feed SPyNet a single frame and throw
        // ("expects two stacked frames"). Instead we drive the model's real forward
        // (EnhanceVideo) through the autodiff tape so gradients reach every
        // trainable layer, exactly as PyTorch's loss.backward() would over the
        // recurrent graph. This is the paper's end-to-end training of the
        // reconstruction network under the Charbonnier objective (Chan et al. 2022).
        SetTrainingMode(true);
        try
        {
            var trainableLayers = Layers.OfType<ITrainableLayer<T>>().ToList();
            T learningRate = NumOps.FromDouble(_learningRate);

            LastLoss = TapeTrainingStep<T>.Step(
                trainableLayers,
                input,
                expectedOutput,
                learningRate,
                forward: EnhanceVideo,
                computeLoss: (prediction, target) => loss.ComputeTapeLoss(prediction, target));
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc/>
    /// <remarks>
    /// BasicVSR++ is recurrent, so the generic sequential <see cref="NeuralNetworkBase{T}.ForwardForTraining"/>
    /// (which would feed SPyNet a single frame and throw) does not model it. Route the tape forward
    /// through the real recurrent <see cref="EnhanceVideo"/> so that <see cref="NeuralNetworkBase{T}.ComputeGradients"/>
    /// — and the finite-difference gradient check built on it (#1872) — trace the true computation graph.
    /// </remarks>
    public override Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        return _useNativeMode ? EnhanceVideo(input) : base.ForwardForTraining(input);
    }

    #endregion

    #region Native Implementation

    private Tensor<T> EnhanceVideoNative(Tensor<T> frames)
    {
        int numFrames = frames.Shape[0];

        // --- 1. Shallow feature extraction per frame (tape-differentiable) ---
        // FeatExtract lifts each RGB frame [C, H, W] into the numFeatures feature
        // space that every downstream stage operates in.
        var frameFeatures = new List<Tensor<T>>(numFrames);
        for (int i = 0; i < numFrames; i++)
        {
            var frame = ExtractFrameTape(frames, i);
            frameFeatures.Add(FeatExtract.Forward(frame));
        }

        // --- 2. Optical flow between adjacent frames (motion guidance) ---
        // SPyNet estimates the flow between each adjacent pair; the flow is used
        // only to warp features toward their neighbours. As in Chan et al. 2022,
        // where the flow network is pre-trained and the reconstruction network is
        // what the training objective optimizes, the flow acts as a fixed sampling
        // guide — but the warped FEATURES stay on the autodiff tape (via
        // Engine.GridSample), so their gradients reach the layers that produced them.
        var flows = ComputeFlows(frames, numFrames);

        // --- 3. Flow-guided bidirectional grid propagation ---
        var propagated = BidirectionalPropagation(frameFeatures, flows, numFrames);

        // --- 4. Per-frame reconstruction: residual blocks -> upsample -> head ---
        var outputFrames = new List<Tensor<T>>(numFrames);
        for (int i = 0; i < numFrames; i++)
        {
            var feat = propagated[i];

            foreach (var block in _residualBlocks)
                feat = block.Forward(feat);

            foreach (var upsample in _upsampleLayers)
                feat = upsample.Forward(feat);

            outputFrames.Add(OutputConv.Forward(feat));
        }

        // --- 5. Stack the per-frame outputs into [numFrames, C, H*scale, W*scale] ---
        // TensorStack keeps every frame's reconstruction on the tape, so a loss over
        // the assembled clip backpropagates into every layer above. (The previous
        // implementation copied each frame into a fresh tensor with StoreFrameBatch,
        // severing the graph and leaving Train() with no gradient path — which is why
        // it needed the hand-rolled backward pass this rewrite removes.)
        return Engine.TensorStack(outputFrames.ToArray(), axis: 0);
    }

    /// <summary>
    /// Extracts frame <paramref name="frameIndex"/> from a [frames, C, H, W] clip as a
    /// tape-differentiable [C, H, W] slice.
    /// </summary>
    private Tensor<T> ExtractFrameTape(Tensor<T> frames, int frameIndex)
        => Engine.TensorSliceAxis(frames, axis: 0, index: frameIndex);

    /// <summary>
    /// Concatenates two feature maps along the channel axis on the autodiff tape.
    /// Handles both [C, H, W] (rank 3) and [batch, C, H, W] (rank 4) layouts.
    /// </summary>
    private Tensor<T> ConcatenateFeaturesTape(Tensor<T> feat1, Tensor<T> feat2)
    {
        int channelAxis = feat1.Rank == 4 ? 1 : 0;
        return Engine.TensorConcatenate(new[] { feat1, feat2 }, channelAxis);
    }

    /// <summary>
    /// Three-input channel-axis concatenation on the tape — used by second-order propagation to fuse the
    /// current feature with its first-order (i±1) and second-order (i±2) warped neighbours.
    /// </summary>
    private Tensor<T> ConcatenateFeaturesTape(Tensor<T> feat1, Tensor<T> feat2, Tensor<T> feat3)
    {
        int channelAxis = feat1.Rank == 4 ? 1 : 0;
        return Engine.TensorConcatenate(new[] { feat1, feat2, feat3 }, channelAxis);
    }

    /// <summary>
    /// Composes two optical-flow fields into the two-hop flow used by second-order propagation (Chan et
    /// al. 2022): the flow from frame i to frame i±2 is <paramref name="baseFlow"/> (i→i±1) followed by
    /// <paramref name="nextFlow"/> (i±1→i±2). At pixel p the composed displacement is
    /// baseFlow(p) + nextFlow(p + baseFlow(p)) — sample the second hop at the location the first hop lands
    /// on, then add the two hops. Flows are FIXED sampling guides (their values are not differentiated,
    /// exactly like the single-hop flows), so composition runs on the materialized data — matching how
    /// <see cref="BuildFlowGrid"/> already reads <c>flow.Data.Span</c> directly.
    /// </summary>
    private Tensor<T> ComposeFlow(Tensor<T> baseFlow, Tensor<T> nextFlow)
    {
        // Warp the second-hop flow into the first frame's grid (bilinear grid-sample), then add the
        // first-hop flow. WarpFeatureTape treats the [2, H, W] flow field exactly like a 2-channel
        // feature map, so p+baseFlow(p) sampling is reused verbatim.
        var warpedNext = WarpFeatureTape(nextFlow, baseFlow);
        var composed = new Tensor<T>(baseFlow._shape);
        var b = baseFlow.Data.Span;
        var w = warpedNext.Data.Span;
        var c = composed.Data.Span;
        int len = c.Length;
        for (int i = 0; i < len; i++)
            c[i] = NumOps.Add(b[i], w[i]);
        return composed;
    }

    /// <summary>
    /// A zero feature map matching <paramref name="template"/>'s shape — the second-order slot for an edge
    /// frame that has no i±2 neighbour, so the alignment input keeps a constant channel count.
    /// </summary>
    private static Tensor<T> ZerosLike(Tensor<T> template) => new Tensor<T>(template._shape);

    private List<(Tensor<T> forward, Tensor<T> backward)> ComputeFlows(Tensor<T> frames, int numFrames)
    {
        var flows = new List<(Tensor<T> forward, Tensor<T> backward)>();

        for (int i = 0; i < numFrames - 1; i++)
        {
            var frame1 = ExtractFrameTape(frames, i);
            var frame2 = ExtractFrameTape(frames, i + 1);

            // Forward flow: frame i -> frame i+1; backward flow: frame i+1 -> frame i.
            var forwardFlow = FlowEstimator.EstimateFlow(frame1, frame2);
            var backwardFlow = FlowEstimator.EstimateFlow(frame2, frame1);

            flows.Add((forwardFlow, backwardFlow));
        }

        return flows;
    }

    /// <summary>
    /// Flow-guided bidirectional propagation (Chan et al. 2022). For each propagation
    /// iteration, features are propagated backward (last -> first) and then forward
    /// (first -> last). Each step warps the neighbouring frame's feature along the
    /// optical flow, refines the alignment with a deformable-convolution alignment
    /// layer, and fuses it with the current feature through a propagation conv. Every
    /// operation runs through tape-aware Engine ops (grid sample, concat, conv) so the
    /// whole recurrent graph is differentiable end-to-end.
    /// </summary>
    private List<Tensor<T>> BidirectionalPropagation(
        List<Tensor<T>> features,
        List<(Tensor<T> forward, Tensor<T> backward)> flows,
        int numFrames)
    {
        var propagatedFeatures = new List<Tensor<T>>(features);

        for (int iter = 0; iter < _numPropagations; iter++)
        {
            // Backward propagation (last -> first): SECOND-ORDER grid propagation (Chan et al. 2022,
            // Sec. 3.1). Each frame aggregates BOTH its immediate successor (i+1) and its second successor
            // (i+2). GridSample maps a target pixel p to source p + flow[p], so aligning frame i+1 onto
            // frame i uses the forward flow i -> i+1 (flows[i].forward); the second-order warp of i+2 uses
            // the paper's COMPOSITION of the two adjacent forward flows (i -> i+1 then i+1 -> i+2). At the
            // tail edge (no i+2) the second-order slot is zero-filled so the alignment layer always sees
            // the same channel count (current + first-order + second-order = 3 * numFeatures, resolved
            // lazily on the deformable-alignment layer's first forward).
            var backwardFeats = new List<Tensor<T>>(propagatedFeatures);
            for (int i = numFrames - 2; i >= 0; i--)
            {
                var warp1 = WarpFeatureTape(backwardFeats[i + 1], flows[i].forward);
                var warp2 = (i + 2 < numFrames)
                    ? WarpFeatureTape(backwardFeats[i + 2], ComposeFlow(flows[i].forward, flows[i + 1].forward))
                    : ZerosLike(warp1);
                // Test-only isolation: zero the first-order slot so the only route by which a two-step-away
                // frame reaches this one is the second-order (i+2) warp above.
                if (DisableFirstOrderPropagationForTesting) warp1 = ZerosLike(warp1);
                var alignInput = ConcatenateFeaturesTape(propagatedFeatures[i], warp1, warp2);
                BackwardAlignInputRecorder?.Invoke(i, alignInput);
                var aligned = _backwardAlignments[iter].Forward(alignInput);
                var fuseInput = ConcatenateFeaturesTape(propagatedFeatures[i], aligned);
                backwardFeats[i] = _backwardConvs[iter].Forward(fuseInput);
            }

            // Forward propagation (first -> last): mirror the second-order pass over predecessors i-1 and
            // i-2. Aligning frame i-1 onto frame i uses the backward flow i -> i-1 (flows[i-1].backward,
            // since flows[i-1] = (i-1 -> i, i -> i-1)); the i -> i-2 flow composes flows[i-1].backward with
            // flows[i-2].backward. The head edge (no i-2) zero-fills the second-order slot.
            var forwardFeats = new List<Tensor<T>>(backwardFeats);
            for (int i = 1; i < numFrames; i++)
            {
                var warp1 = WarpFeatureTape(forwardFeats[i - 1], flows[i - 1].backward);
                var warp2 = (i - 2 >= 0)
                    ? WarpFeatureTape(forwardFeats[i - 2], ComposeFlow(flows[i - 1].backward, flows[i - 2].backward))
                    : ZerosLike(warp1);
                if (DisableFirstOrderPropagationForTesting) warp1 = ZerosLike(warp1);
                var alignInput = ConcatenateFeaturesTape(backwardFeats[i], warp1, warp2);
                var aligned = _forwardAlignments[iter].Forward(alignInput);
                var fuseInput = ConcatenateFeaturesTape(backwardFeats[i], aligned);
                forwardFeats[i] = _forwardConvs[iter].Forward(fuseInput);
            }

            propagatedFeatures = forwardFeats;
        }

        return propagatedFeatures;
    }

    /// <summary>
    /// Warps <paramref name="feature"/> along an optical-flow field using bilinear grid
    /// sampling on the autodiff tape. The flow is a fixed sampling guide (its own values
    /// are not differentiated), but the sampled feature stays on the tape so gradients
    /// flow back to whatever produced it.
    /// </summary>
    private Tensor<T> WarpFeatureTape(Tensor<T> feature, Tensor<T> flow)
    {
        bool hasBatch = feature.Rank == 4;
        int height = hasBatch ? feature.Shape[2] : feature.Shape[1];
        int width = hasBatch ? feature.Shape[3] : feature.Shape[2];

        // Sampling grid in GridSample's [batch, H, W, 2] normalized-coordinate layout.
        var grid = BuildFlowGrid(flow, height, width);

        // Engine.GridSample is NCHW (PyTorch F.grid_sample convention, Tensors #777): input
        // [batch, C, H, W], grid [batch, outH, outW, 2] -> output [batch, C, outH, outW]. Features
        // are already channels-first, so pass them directly — no permute.
        var feature4D = hasBatch ? feature : Engine.TensorExpandDims(feature, 0);
        var warped = Engine.GridSample(feature4D, grid);

        return hasBatch ? warped : Engine.TensorSqueeze(warped, axis: 0);
    }

    /// <summary>
    /// Builds a GridSample sampling grid [1, H, W, 2] from a [2, H, W] (or [1, 2, H, W])
    /// optical-flow field. Grid coordinates are normalized to [-1, 1] via
    /// grid = (pixel + flow) * 2 / (size - 1) - 1, matching SPyNet's warp convention.
    /// </summary>
    private Tensor<T> BuildFlowGrid(Tensor<T> flow, int height, int width)
    {
        var grid = new Tensor<T>(new[] { 1, height, width, 2 });

        T widthNorm = NumOps.FromDouble(width > 1 ? 2.0 / (width - 1) : 0.0);
        T heightNorm = NumOps.FromDouble(height > 1 ? 2.0 / (height - 1) : 0.0);
        T one = NumOps.One;
        int plane = height * width;

        // Flow channel 0 = dx, channel 1 = dy (channels-first [2, H, W]); a leading
        // batch dim of 1 leaves these per-pixel offsets unchanged.
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                int pixel = h * width + w;
                T dx = flow.Data.Span[pixel];
                T dy = flow.Data.Span[plane + pixel];

                T srcX = NumOps.Add(NumOps.FromDouble(w), dx);
                T srcY = NumOps.Add(NumOps.FromDouble(h), dy);

                T gridX = NumOps.Subtract(NumOps.Multiply(srcX, widthNorm), one);
                T gridY = NumOps.Subtract(NumOps.Multiply(srcY, heightNorm), one);

                int gridIdx = pixel * 2;
                grid.Data.Span[gridIdx] = gridX;
                grid.Data.Span[gridIdx + 1] = gridY;
            }
        }

        return grid;
    }

    #endregion

    #region ONNX Implementation

    private Tensor<T> EnhanceVideoOnnx(Tensor<T> frames)
    {
        if (_onnxSession == null)
            throw new InvalidOperationException("ONNX session is not initialized.");

        // Convert to ONNX format
        var inputData = new float[frames.Length];
        for (int i = 0; i < frames.Length; i++)
        {
            inputData[i] = Convert.ToSingle(frames.Data.Span[i]);
        }

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, frames._shape);
        var inputMeta = _onnxSession.InputMetadata;
        string inputName = inputMeta.Keys.First();

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, onnxInput)
        };

        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        var outputShape = outputTensor.Dimensions.ToArray();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++)
        {
            outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        }

        return new Tensor<T>(outputShape, new Vector<T>(outputData));
    }

    #endregion

    #region Named Activations

    /// <summary>
    /// Captures intermediate activations from BasicVSR++'s real recurrent forward pass
    /// for inspection/debugging. The generic base implementation chains every layer in
    /// <c>Layers</c> sequentially, which is invalid here: the first layer is SPyNet,
    /// which needs two stacked frames and rejects a single RGB frame. This override runs
    /// the true forward — per-frame feature extraction, flow-guided bidirectional
    /// propagation, and reconstruction — and records the meaningful stage outputs.
    /// </summary>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        if (!_useNativeMode)
            return base.GetNamedLayerActivations(input);

        var activations = new Dictionary<string, Tensor<T>>();
        int numFrames = input.Shape[0];

        // Shallow feature extraction per frame.
        var frameFeatures = new List<Tensor<T>>(numFrames);
        for (int i = 0; i < numFrames; i++)
        {
            var feat = FeatExtract.Forward(ExtractFrameTape(input, i));
            frameFeatures.Add(feat);
            activations[$"FeatExtract_Frame{i}"] = feat.Clone();
        }

        // Flow-guided bidirectional propagation.
        var flows = ComputeFlows(input, numFrames);
        if (flows.Count > 0)
            activations["Flow_Forward_0"] = flows[0].forward.Clone();

        var propagated = BidirectionalPropagation(frameFeatures, flows, numFrames);
        for (int i = 0; i < numFrames; i++)
            activations[$"Propagated_Frame{i}"] = propagated[i].Clone();

        // Per-frame reconstruction output.
        for (int i = 0; i < numFrames; i++)
        {
            var feat = propagated[i];
            foreach (var block in _residualBlocks)
                feat = block.Forward(feat);
            foreach (var upsample in _upsampleLayers)
                feat = upsample.Forward(feat);
            activations[$"Output_Frame{i}"] = OutputConv.Forward(feat).Clone();
        }

        return activations;
    }

    #endregion

    #region Layer Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        ClearLayers();

        // Mirror InitializeNativeLayers' construction order exactly so the flat
        // Layers list is consistent with the sub-list distribution (and with the
        // GetParameters/UpdateParameters ordering below): flow estimator, feature
        // extraction, residual blocks, then per-iteration [backward align, forward
        // align, backward conv, forward conv], upsampling, and the output head.
        // Residual blocks were previously omitted here, and the alignment/conv
        // layers were grouped instead of interleaved — leaving Layers out of sync
        // with the model's actual structure whenever this ran.
        if (_flowEstimator is not null) Layers.Add(_flowEstimator);
        if (_featExtract is not null) Layers.Add(_featExtract);
        foreach (var block in _residualBlocks) Layers.Add(block);
        for (int i = 0; i < _numPropagations; i++)
        {
            if (i < _backwardAlignments.Count) Layers.Add(_backwardAlignments[i]);
            if (i < _forwardAlignments.Count) Layers.Add(_forwardAlignments[i]);
            if (i < _backwardConvs.Count) Layers.Add(_backwardConvs[i]);
            if (i < _forwardConvs.Count) Layers.Add(_forwardConvs[i]);
        }
        foreach (var layer in _upsampleLayers) Layers.Add(layer);
        if (_outputConv is not null) Layers.Add(_outputConv);
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    /// <remarks>
    /// Distributes the flat parameter vector across the trainable layers in the SAME order the base
    /// <see cref="NeuralNetworkBase{T}.GetParameters"/> collects them — the canonical <c>Layers</c>
    /// order (flow estimator -> feature extraction -> residual blocks -> per-iteration [backward align,
    /// forward align, backward conv, forward conv] -> upsampling -> output conv). The previous override
    /// used a different feature -> residuals -> flow ordering, so a GetParameters/UpdateParameters
    /// round-trip wrote each slice into the wrong layer.
    /// </remarks>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Parameter updates are not supported in ONNX mode.");

        // Validate the flat vector length UP FRONT against the total trainable parameter count (the exact
        // set this loop consumes). A mismatched vector must RAISE rather than silently applying a partial
        // (corrupt) update via an early break — mirrors the sibling VideoCLIP / SelfOrganizingMap
        // UpdateParameters overrides (#1789 review).
        long expected = 0;
        foreach (var layer in Layers)
            if (layer.SupportsTraining) expected += layer.ParameterCount;
        if (parameters.Length != expected)
            throw new ArgumentException(
                $"Expected {expected} parameters (sum over trainable layers), got {parameters.Length}.",
                nameof(parameters));

        int offset = 0;
        foreach (var layer in Layers)
        {
            if (!layer.SupportsTraining || layer.ParameterCount == 0)
                continue;
            int count = checked((int)layer.ParameterCount);
            var slice = new Vector<T>(count);
            for (int i = 0; i < count; i++)
                slice[i] = parameters[offset + i];
            layer.UpdateParameters(slice);
            offset += count;
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "BasicVSRPlusPlus" },
            { "ScaleFactor", _scaleFactor },
            { "NumFeatures", _numFeatures },
            { "NumResidualBlocks", _numResidualBlocks },
            { "NumPropagations", _numPropagations },
            { "UseNativeMode", _useNativeMode }
        };

        if (!_useNativeMode && _onnxModelPath != null)
        {
            additionalInfo["OnnxModelPath"] = _onnxModelPath;
        }

        return new ModelMetadata<T>
        {
            AdditionalInfo = additionalInfo,
            ModelData = _useNativeMode ? this.Serialize() : Array.Empty<byte>()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Serialization is not supported in ONNX mode.");

        writer.Write(_scaleFactor);
        writer.Write(_numFeatures);
        writer.Write(_numResidualBlocks);
        writer.Write(_numPropagations);
        writer.Write(_learningRate);

        // Serialize layer parameters
        SerializeLayerParameters(writer, FeatExtract.GetParameters());
        SerializeLayerParameters(writer, OutputConv.GetParameters());

        foreach (var block in _residualBlocks)
        {
            SerializeLayerParameters(writer, block.GetParameters());
        }
    }

    private void SerializeLayerParameters(BinaryWriter writer, Vector<T> parameters)
    {
        writer.Write(parameters.Length);
        for (int i = 0; i < parameters.Length; i++)
        {
            writer.Write(NumOps.ToDouble(parameters[i]));
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Deserialization is not supported in ONNX mode.");

        // Read configuration (already set in constructor)
        _ = reader.ReadInt32(); // scaleFactor
        _ = reader.ReadInt32(); // numFeatures
        _ = reader.ReadInt32(); // numResidualBlocks
        _ = reader.ReadInt32(); // numPropagations
        _ = reader.ReadDouble(); // learningRate

        // Load layer parameters
        FeatExtract.SetParameters(DeserializeLayerParameters(reader));
        OutputConv.SetParameters(DeserializeLayerParameters(reader));

        foreach (var block in _residualBlocks)
        {
            block.SetParameters(DeserializeLayerParameters(reader));
        }
    }

    private Vector<T> DeserializeLayerParameters(BinaryReader reader)
    {
        int count = reader.ReadInt32();
        var parameters = new T[count];
        for (int i = 0; i < count; i++)
        {
            parameters[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        return new Vector<T>(parameters);
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new BasicVSRPlusPlus<T>(
            Architecture,
            _scaleFactor,
            _numFeatures,
            _numResidualBlocks,
            _numPropagations,
            _learningRate);
    }

    #endregion

    #region Base Class Abstract Methods

    /// <inheritdoc/>
    public override Tensor<T> Upscale(Tensor<T> lowResFrames)
    {
        return Forward(lowResFrames);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames)
    {
        return NormalizeFrames(rawFrames);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        return DenormalizeFrames(modelOutput);
    }

    #endregion

}
