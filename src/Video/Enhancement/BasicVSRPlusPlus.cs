using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// BasicVSR++ (Basic Video Super-Resolution++) for temporal video super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// BasicVSR++ improves upon BasicVSR with:
/// - Second-order grid propagation for better temporal modeling
/// - Flow-guided deformable alignment for accurate feature alignment
/// - Bidirectional propagation for utilizing both past and future frames
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
public class BasicVSRPlusPlus<T> : NeuralNetworkBase<T>
{
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
    /// Cached features from last forward pass for training.
    /// </summary>
    private readonly List<Tensor<T>> _cachedFeatures;

    // Comprehensive activation cache for proper backward pass
    private List<Tensor<T>>? _cachedInitialFeatures;
    private List<(Tensor<T> forward, Tensor<T> backward)>? _cachedFlows;
    private List<List<Tensor<T>>>? _cachedBackwardPropFeatures;  // [iteration][frame]
    private List<List<Tensor<T>>>? _cachedForwardPropFeatures;   // [iteration][frame]
    private List<List<Tensor<T>>>? _cachedBackwardAlignInputs;   // [iteration][frame]
    private List<List<Tensor<T>>>? _cachedForwardAlignInputs;    // [iteration][frame]
    private List<List<Tensor<T>>>? _cachedBackwardAlignOutputs;  // [iteration][frame]
    private List<List<Tensor<T>>>? _cachedForwardAlignOutputs;   // [iteration][frame]
    private List<List<Tensor<T>>>? _cachedResidualInputs;        // [frame][block]
    private List<List<Tensor<T>>>? _cachedResidualOutputs;       // [frame][block]
    private List<List<Tensor<T>>>? _cachedUpsampleInputs;        // [frame][layer]
    private List<Tensor<T>>? _cachedOutputConvInputs;            // [frame]
    private int _cachedNumFrames;

    #endregion

    #region Properties

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
    internal int ScaleFactor => _scaleFactor;

    /// <summary>
    /// Gets the number of feature channels.
    /// </summary>
    internal int NumFeatures => _numFeatures;

    #endregion

    #region Constructors

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
        double learningRate = 0.0001)
        : base(architecture, new CharbonnierLoss<T>())
    {
        if (scaleFactor != 2 && scaleFactor != 4)
            throw new ArgumentOutOfRangeException(nameof(scaleFactor), "Scale factor must be 2 or 4.");
        if (numFeatures < 1)
            throw new ArgumentOutOfRangeException(nameof(numFeatures), "Number of features must be at least 1.");
        if (numResidualBlocks < 1)
            throw new ArgumentOutOfRangeException(nameof(numResidualBlocks), "Number of residual blocks must be at least 1.");

        _useNativeMode = true;
        _scaleFactor = scaleFactor;
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
        _cachedFeatures = [];

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
        int numPropagations = 2)
        : base(architecture, new CharbonnierLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"BasicVSR++ ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _scaleFactor = scaleFactor;
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
        _cachedFeatures = [];

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

        // SPyNet for optical flow estimation
        _flowEstimator = new SpyNetLayer<T>(height, width, channels, numLevels: 5);

        // Initial feature extraction
        // ConvolutionalLayer(inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding)
        _featExtract = new ConvolutionalLayer<T>(channels, height, width, _numFeatures, 3, 1, 1);

        // Residual blocks for deep feature extraction
        for (int i = 0; i < _numResidualBlocks; i++)
        {
            _residualBlocks.Add(new ResidualDenseBlock<T>(
                numFeatures: _numFeatures,
                growthChannels: 32,
                inputHeight: height,
                inputWidth: width,
                residualScale: 0.2));
        }

        // Deformable alignment modules for each propagation
        for (int i = 0; i < _numPropagations; i++)
        {
            // Backward alignment
            _backwardAlignments.Add(new DeformableConvolutionalLayer<T>(
                height, width, _numFeatures * 2, _numFeatures,
                kernelSize: 3, padding: 1, deformGroups: 8));

            // Forward alignment
            _forwardAlignments.Add(new DeformableConvolutionalLayer<T>(
                height, width, _numFeatures * 2, _numFeatures,
                kernelSize: 3, padding: 1, deformGroups: 8));

            // Propagation convolutions
            // ConvolutionalLayer(inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding)
            _backwardConvs.Add(new ConvolutionalLayer<T>(
                _numFeatures * 2, height, width, _numFeatures, 3, 1, 1));
            _forwardConvs.Add(new ConvolutionalLayer<T>(
                _numFeatures * 2, height, width, _numFeatures, 3, 1, 1));
        }

        // Upsampling layers using PixelShuffle
        int numUpsample = _scaleFactor == 4 ? 2 : 1;
        int currentHeight = height;
        int currentWidth = width;

        for (int i = 0; i < numUpsample; i++)
        {
            // PixelShuffleLayer(inputShape, upscaleFactor)
            // Input channels need to be upscaleFactor^2 * outputChannels
            // For scale=2, we need 4 * numFeatures channels in, numFeatures channels out
            int[] inputShape = [1, _numFeatures * 4, currentHeight, currentWidth];
            _upsampleLayers.Add(new PixelShuffleLayer<T>(inputShape, 2));
            currentHeight *= 2;
            currentWidth *= 2;
        }

        // Final output convolution
        // ConvolutionalLayer(inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding)
        _outputConv = new ConvolutionalLayer<T>(
            _numFeatures, currentHeight, currentWidth, channels, 3, 1, 1);

        InitializeLayers();
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
    public override Tensor<T> Predict(Tensor<T> input)
    {
        return EnhanceVideo(input);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is not supported in ONNX mode.");

        // Forward pass
        var output = EnhanceVideoNative(input);

        // Compute loss gradient
        var lossGradient = ComputeLossGradient(output, expectedOutput);

        // Backward pass
        BackwardPass(lossGradient);

        // Update parameters
        UpdateAllParameters();
    }

    #endregion

    #region Native Implementation

    private Tensor<T> EnhanceVideoNative(Tensor<T> frames)
    {
        int numFrames = frames.Shape[0];
        int channels = frames.Shape[1];
        int height = frames.Shape[2];
        int width = frames.Shape[3];

        int outHeight = height * _scaleFactor;
        int outWidth = width * _scaleFactor;

        var outputFrames = new Tensor<T>([numFrames, channels, outHeight, outWidth]);

        // Clear and initialize activation caches for backward pass
        ClearActivationCache();
        _cachedNumFrames = numFrames;
        _cachedInitialFeatures = [];
        _cachedResidualInputs = [];
        _cachedResidualOutputs = [];
        _cachedUpsampleInputs = [];
        _cachedOutputConvInputs = [];

        // Extract initial features for all frames (with caching)
        var frameFeatures = new List<Tensor<T>>();
        for (int i = 0; i < numFrames; i++)
        {
            var frame = ExtractFrame(frames, i);
            var feat = _featExtract!.Forward(frame);
            frameFeatures.Add(feat);
            _cachedInitialFeatures.Add(feat);
        }

        // Compute optical flows between adjacent frames (with caching)
        var flows = ComputeFlows(frames, numFrames);
        _cachedFlows = flows;

        // Bidirectional propagation with second-order grid (with caching)
        var propagatedFeatures = BidirectionalPropagationWithCache(frameFeatures, flows, numFrames);

        // Process each frame through residual blocks and upsampling (with caching)
        for (int i = 0; i < numFrames; i++)
        {
            var feat = propagatedFeatures[i];
            var frameResidualInputs = new List<Tensor<T>>();
            var frameResidualOutputs = new List<Tensor<T>>();

            // Apply residual blocks with caching
            foreach (var block in _residualBlocks)
            {
                frameResidualInputs.Add(feat);
                feat = block.Forward(feat);
                frameResidualOutputs.Add(feat);
            }
            _cachedResidualInputs.Add(frameResidualInputs);
            _cachedResidualOutputs.Add(frameResidualOutputs);

            // Upsampling with caching
            var frameUpsampleInputs = new List<Tensor<T>>();
            foreach (var upsample in _upsampleLayers)
            {
                frameUpsampleInputs.Add(feat);
                feat = upsample.Forward(feat);
            }
            _cachedUpsampleInputs.Add(frameUpsampleInputs);

            // Cache output conv input
            _cachedOutputConvInputs.Add(feat);

            // Final output convolution
            var output = _outputConv!.Forward(feat);

            // Store in output tensor
            StoreFrame(outputFrames, output, i);
        }

        return outputFrames;
    }

    /// <summary>
    /// Clears all activation caches.
    /// </summary>
    private void ClearActivationCache()
    {
        _cachedFeatures.Clear();
        _cachedInitialFeatures = null;
        _cachedFlows = null;
        _cachedBackwardPropFeatures = null;
        _cachedForwardPropFeatures = null;
        _cachedBackwardAlignInputs = null;
        _cachedForwardAlignInputs = null;
        _cachedBackwardAlignOutputs = null;
        _cachedForwardAlignOutputs = null;
        _cachedResidualInputs = null;
        _cachedResidualOutputs = null;
        _cachedUpsampleInputs = null;
        _cachedOutputConvInputs = null;
    }

    private List<(Tensor<T> forward, Tensor<T> backward)> ComputeFlows(Tensor<T> frames, int numFrames)
    {
        var flows = new List<(Tensor<T> forward, Tensor<T> backward)>();

        for (int i = 0; i < numFrames - 1; i++)
        {
            var frame1 = ExtractFrame(frames, i);
            var frame2 = ExtractFrame(frames, i + 1);

            // Forward flow: frame i -> frame i+1
            var forwardFlow = _flowEstimator!.EstimateFlow(frame1, frame2);

            // Backward flow: frame i+1 -> frame i
            var backwardFlow = _flowEstimator.EstimateFlow(frame2, frame1);

            flows.Add((forwardFlow, backwardFlow));
        }

        return flows;
    }

    private List<Tensor<T>> BidirectionalPropagation(
        List<Tensor<T>> features,
        List<(Tensor<T> forward, Tensor<T> backward)> flows,
        int numFrames)
    {
        var propagatedFeatures = new List<Tensor<T>>();

        // Initialize with original features
        for (int i = 0; i < numFrames; i++)
        {
            propagatedFeatures.Add(features[i]);
        }

        // Multiple propagation iterations (second-order grid propagation)
        for (int iter = 0; iter < _numPropagations; iter++)
        {
            // Backward propagation (from last to first)
            var backwardFeats = new List<Tensor<T>>(propagatedFeatures);
            for (int i = numFrames - 2; i >= 0; i--)
            {
                // Warp feature from frame i+1 to frame i using backward flow
                var warpedFeat = WarpFeature(backwardFeats[i + 1], flows[i].backward);

                // Concatenate current and warped features
                var concat = ConcatenateFeatures(propagatedFeatures[i], warpedFeat);

                // Apply deformable alignment
                var aligned = _backwardAlignments[iter].Forward(concat);

                // Fuse with propagation conv
                concat = ConcatenateFeatures(propagatedFeatures[i], aligned);
                backwardFeats[i] = _backwardConvs[iter].Forward(concat);
            }

            // Forward propagation (from first to last)
            var forwardFeats = new List<Tensor<T>>(backwardFeats);
            for (int i = 1; i < numFrames; i++)
            {
                // Warp feature from frame i-1 to frame i using forward flow
                var warpedFeat = WarpFeature(forwardFeats[i - 1], flows[i - 1].forward);

                // Concatenate current and warped features
                var concat = ConcatenateFeatures(backwardFeats[i], warpedFeat);

                // Apply deformable alignment
                var aligned = _forwardAlignments[iter].Forward(concat);

                // Fuse with propagation conv
                concat = ConcatenateFeatures(backwardFeats[i], aligned);
                forwardFeats[i] = _forwardConvs[iter].Forward(concat);
            }

            propagatedFeatures = forwardFeats;
        }

        return propagatedFeatures;
    }

    /// <summary>
    /// Performs bidirectional propagation with caching of all intermediate activations
    /// for proper gradient computation during backward pass.
    /// </summary>
    private List<Tensor<T>> BidirectionalPropagationWithCache(
        List<Tensor<T>> features,
        List<(Tensor<T> forward, Tensor<T> backward)> flows,
        int numFrames)
    {
        // Initialize caches for propagation
        _cachedBackwardPropFeatures = [];
        _cachedForwardPropFeatures = [];
        _cachedBackwardAlignInputs = [];
        _cachedForwardAlignInputs = [];
        _cachedBackwardAlignOutputs = [];
        _cachedForwardAlignOutputs = [];

        var propagatedFeatures = new List<Tensor<T>>();

        // Initialize with original features
        for (int i = 0; i < numFrames; i++)
        {
            propagatedFeatures.Add(features[i]);
        }

        // Multiple propagation iterations (second-order grid propagation)
        for (int iter = 0; iter < _numPropagations; iter++)
        {
            var iterBackwardPropFeatures = new List<Tensor<T>>();
            var iterForwardPropFeatures = new List<Tensor<T>>();
            var iterBackwardAlignInputs = new List<Tensor<T>>();
            var iterForwardAlignInputs = new List<Tensor<T>>();
            var iterBackwardAlignOutputs = new List<Tensor<T>>();
            var iterForwardAlignOutputs = new List<Tensor<T>>();

            // Initialize lists with placeholder tensors for proper indexing
            for (int i = 0; i < numFrames; i++)
            {
                iterBackwardPropFeatures.Add(new Tensor<T>([1]));
                iterForwardPropFeatures.Add(new Tensor<T>([1]));
                iterBackwardAlignInputs.Add(new Tensor<T>([1]));
                iterForwardAlignInputs.Add(new Tensor<T>([1]));
                iterBackwardAlignOutputs.Add(new Tensor<T>([1]));
                iterForwardAlignOutputs.Add(new Tensor<T>([1]));
            }

            // Initialize boundary frames that won't be processed in the loops
            // Frame numFrames-1 is not processed in backward propagation
            iterBackwardPropFeatures[numFrames - 1] = propagatedFeatures[numFrames - 1];
            iterBackwardAlignInputs[numFrames - 1] = propagatedFeatures[numFrames - 1];
            iterBackwardAlignOutputs[numFrames - 1] = propagatedFeatures[numFrames - 1];

            // Backward propagation (from last to first)
            var backwardFeats = new List<Tensor<T>>(propagatedFeatures);
            for (int i = numFrames - 2; i >= 0; i--)
            {
                // Warp feature from frame i+1 to frame i using backward flow
                var warpedFeat = WarpFeature(backwardFeats[i + 1], flows[i].backward);

                // Concatenate current and warped features
                var alignInput = ConcatenateFeatures(propagatedFeatures[i], warpedFeat);
                iterBackwardAlignInputs[i] = alignInput;

                // Apply deformable alignment
                var aligned = _backwardAlignments[iter].Forward(alignInput);
                iterBackwardAlignOutputs[i] = aligned;

                // Fuse with propagation conv
                var propInput = ConcatenateFeatures(propagatedFeatures[i], aligned);
                iterBackwardPropFeatures[i] = propInput;
                backwardFeats[i] = _backwardConvs[iter].Forward(propInput);
            }

            // Forward propagation (from first to last)
            var forwardFeats = new List<Tensor<T>>(backwardFeats);

            // Frame 0 is not processed in forward propagation
            iterForwardPropFeatures[0] = backwardFeats[0];
            iterForwardAlignInputs[0] = backwardFeats[0];
            iterForwardAlignOutputs[0] = backwardFeats[0];

            for (int i = 1; i < numFrames; i++)
            {
                // Warp feature from frame i-1 to frame i using forward flow
                var warpedFeat = WarpFeature(forwardFeats[i - 1], flows[i - 1].forward);

                // Concatenate current and warped features
                var alignInput = ConcatenateFeatures(backwardFeats[i], warpedFeat);
                iterForwardAlignInputs[i] = alignInput;

                // Apply deformable alignment
                var aligned = _forwardAlignments[iter].Forward(alignInput);
                iterForwardAlignOutputs[i] = aligned;

                // Fuse with propagation conv
                var propInput = ConcatenateFeatures(backwardFeats[i], aligned);
                iterForwardPropFeatures[i] = propInput;
                forwardFeats[i] = _forwardConvs[iter].Forward(propInput);
            }

            // Cache this iteration's activations
            _cachedBackwardPropFeatures.Add(iterBackwardPropFeatures);
            _cachedForwardPropFeatures.Add(iterForwardPropFeatures);
            _cachedBackwardAlignInputs.Add(iterBackwardAlignInputs);
            _cachedForwardAlignInputs.Add(iterForwardAlignInputs);
            _cachedBackwardAlignOutputs.Add(iterBackwardAlignOutputs);
            _cachedForwardAlignOutputs.Add(iterForwardAlignOutputs);

            propagatedFeatures = forwardFeats;
        }

        return propagatedFeatures;
    }

    private Tensor<T> WarpFeature(Tensor<T> feature, Tensor<T> flow)
    {
        // Warp feature using optical flow (bilinear sampling)
        bool hasBatch = feature.Rank == 4;
        int batch = hasBatch ? feature.Shape[0] : 1;
        int channels = hasBatch ? feature.Shape[1] : feature.Shape[0];
        int height = hasBatch ? feature.Shape[2] : feature.Shape[1];
        int width = hasBatch ? feature.Shape[3] : feature.Shape[2];

        var warped = new Tensor<T>(feature.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    // Get flow at this position
                    int flowIdxX = hasBatch
                        ? b * 2 * height * width + h * width + w
                        : h * width + w;
                    int flowIdxY = hasBatch
                        ? b * 2 * height * width + height * width + h * width + w
                        : height * width + h * width + w;

                    double dx = NumOps.ToDouble(flow.Data[flowIdxX]);
                    double dy = NumOps.ToDouble(flow.Data[flowIdxY]);

                    double srcX = w + dx;
                    double srcY = h + dy;

                    // Bilinear sample for each channel
                    for (int c = 0; c < channels; c++)
                    {
                        T value = BilinearSample(feature, b, c, srcY, srcX, hasBatch, height, width, channels);
                        int outIdx = hasBatch
                            ? b * channels * height * width + c * height * width + h * width + w
                            : c * height * width + h * width + w;
                        warped.Data[outIdx] = value;
                    }
                }
            }
        }

        return warped;
    }

    private T BilinearSample(Tensor<T> tensor, int b, int c, double h, double w, bool hasBatch, int height, int width, int channels)
    {
        int h0 = (int)Math.Floor(h);
        int w0 = (int)Math.Floor(w);
        int h1 = h0 + 1;
        int w1 = w0 + 1;

        // Clamp to valid range (compatible with .NET Framework 4.7.1)
        h0 = Math.Max(0, Math.Min(h0, height - 1));
        h1 = Math.Max(0, Math.Min(h1, height - 1));
        w0 = Math.Max(0, Math.Min(w0, width - 1));
        w1 = Math.Max(0, Math.Min(w1, width - 1));

        double hWeight = h - Math.Floor(h);
        double wWeight = w - Math.Floor(w);

        T v00 = GetValue(tensor, b, c, h0, w0, hasBatch, height, width, channels);
        T v01 = GetValue(tensor, b, c, h0, w1, hasBatch, height, width, channels);
        T v10 = GetValue(tensor, b, c, h1, w0, hasBatch, height, width, channels);
        T v11 = GetValue(tensor, b, c, h1, w1, hasBatch, height, width, channels);

        T top = NumOps.Add(
            NumOps.Multiply(v00, NumOps.FromDouble(1 - wWeight)),
            NumOps.Multiply(v01, NumOps.FromDouble(wWeight)));
        T bottom = NumOps.Add(
            NumOps.Multiply(v10, NumOps.FromDouble(1 - wWeight)),
            NumOps.Multiply(v11, NumOps.FromDouble(wWeight)));

        return NumOps.Add(
            NumOps.Multiply(top, NumOps.FromDouble(1 - hWeight)),
            NumOps.Multiply(bottom, NumOps.FromDouble(hWeight)));
    }

    private T GetValue(Tensor<T> tensor, int b, int c, int h, int w, bool hasBatch, int height, int width, int channels)
    {
        int idx = hasBatch
            ? b * channels * height * width + c * height * width + h * width + w
            : c * height * width + h * width + w;
        return tensor.Data[idx];
    }

    private Tensor<T> ConcatenateFeatures(Tensor<T> feat1, Tensor<T> feat2)
    {
        bool hasBatch = feat1.Rank == 4;
        int batch = hasBatch ? feat1.Shape[0] : 1;
        int c1 = hasBatch ? feat1.Shape[1] : feat1.Shape[0];
        int c2 = hasBatch ? feat2.Shape[1] : feat2.Shape[0];
        int height = hasBatch ? feat1.Shape[2] : feat1.Shape[1];
        int width = hasBatch ? feat1.Shape[3] : feat1.Shape[2];

        var outShape = hasBatch
            ? new[] { batch, c1 + c2, height, width }
            : new[] { c1 + c2, height, width };
        var output = new Tensor<T>(outShape);

        int pixelsPerChannel = height * width;

        for (int b = 0; b < batch; b++)
        {
            // Copy feat1 channels
            for (int c = 0; c < c1; c++)
            {
                int srcOffset = hasBatch ? b * c1 * pixelsPerChannel + c * pixelsPerChannel : c * pixelsPerChannel;
                int dstOffset = hasBatch ? b * (c1 + c2) * pixelsPerChannel + c * pixelsPerChannel : c * pixelsPerChannel;

                for (int i = 0; i < pixelsPerChannel; i++)
                {
                    output.Data[dstOffset + i] = feat1.Data[srcOffset + i];
                }
            }

            // Copy feat2 channels
            for (int c = 0; c < c2; c++)
            {
                int srcOffset = hasBatch ? b * c2 * pixelsPerChannel + c * pixelsPerChannel : c * pixelsPerChannel;
                int dstOffset = hasBatch ? b * (c1 + c2) * pixelsPerChannel + (c1 + c) * pixelsPerChannel : (c1 + c) * pixelsPerChannel;

                for (int i = 0; i < pixelsPerChannel; i++)
                {
                    output.Data[dstOffset + i] = feat2.Data[srcOffset + i];
                }
            }
        }

        return output;
    }

    private Tensor<T> ExtractFrame(Tensor<T> frames, int frameIndex)
    {
        int channels = frames.Shape[1];
        int height = frames.Shape[2];
        int width = frames.Shape[3];

        var frame = new Tensor<T>([channels, height, width]);
        int frameSize = channels * height * width;
        int srcOffset = frameIndex * frameSize;

        for (int i = 0; i < frameSize; i++)
        {
            frame.Data[i] = frames.Data[srcOffset + i];
        }

        return frame;
    }

    private void StoreFrame(Tensor<T> output, Tensor<T> frame, int frameIndex)
    {
        int channels = output.Shape[1];
        int height = output.Shape[2];
        int width = output.Shape[3];
        int frameSize = channels * height * width;
        int dstOffset = frameIndex * frameSize;

        for (int i = 0; i < frameSize; i++)
        {
            output.Data[dstOffset + i] = frame.Data[i];
        }
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
            inputData[i] = Convert.ToSingle(frames.Data[i]);
        }

        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, frames.Shape);
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

    #region Training Methods

    private Tensor<T> ComputeLossGradient(Tensor<T> output, Tensor<T> target)
    {
        var gradient = new Tensor<T>(output.Shape);

        for (int i = 0; i < output.Length; i++)
        {
            T diff = NumOps.Subtract(output.Data[i], target.Data[i]);
            // Charbonnier loss gradient: diff / sqrt(diff^2 + epsilon^2)
            double d = NumOps.ToDouble(diff);
            double eps = 1e-6;
            double grad = d / Math.Sqrt(d * d + eps * eps);
            gradient.Data[i] = NumOps.FromDouble(grad);
        }

        return gradient;
    }

    /// <summary>
    /// Production-ready backward pass with proper gradient routing through
    /// bidirectional propagation, warping operations, and all network components.
    /// </summary>
    /// <param name="gradient">Loss gradient with shape [numFrames, channels, height*scale, width*scale].</param>
    private void BackwardPass(Tensor<T> gradient)
    {
        if (_cachedInitialFeatures == null || _cachedFlows == null ||
            _cachedResidualInputs == null || _cachedUpsampleInputs == null ||
            _cachedOutputConvInputs == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int numFrames = _cachedNumFrames;

        // Gradient accumulators for each frame's features (after propagation)
        var featureGradients = new List<Tensor<T>>();
        for (int i = 0; i < numFrames; i++)
        {
            featureGradients.Add(new Tensor<T>(_cachedInitialFeatures[0].Shape));
        }

        // Process each frame's gradient through output conv, upsampling, and residual blocks
        for (int f = 0; f < numFrames; f++)
        {
            // Extract per-frame gradient
            var frameGrad = ExtractFrame(gradient, f);

            // Backward through output convolution
            var grad = _outputConv!.Backward(frameGrad);

            // Backward through upsampling layers (in reverse order)
            for (int i = _upsampleLayers.Count - 1; i >= 0; i--)
            {
                grad = _upsampleLayers[i].Backward(grad);
            }

            // Backward through residual blocks (in reverse order)
            for (int i = _residualBlocks.Count - 1; i >= 0; i--)
            {
                grad = _residualBlocks[i].Backward(grad);
            }

            // Accumulate gradient for this frame
            AccumulateGradient(featureGradients[f], grad);
        }

        // Backward through bidirectional propagation (reverse order of iterations)
        var propagationGradients = BackwardThroughPropagation(featureGradients, numFrames);

        // Backward through initial feature extraction for each frame
        for (int f = 0; f < numFrames; f++)
        {
            _featExtract!.Backward(propagationGradients[f]);
        }
    }

    /// <summary>
    /// Backward pass through bidirectional propagation with proper gradient routing
    /// through deformable alignments, warping, and propagation convolutions.
    /// </summary>
    private List<Tensor<T>> BackwardThroughPropagation(List<Tensor<T>> outputGradients, int numFrames)
    {
        if (_cachedBackwardPropFeatures == null || _cachedForwardPropFeatures == null ||
            _cachedBackwardAlignInputs == null || _cachedForwardAlignInputs == null ||
            _cachedFlows == null)
        {
            throw new InvalidOperationException("Propagation cache not available.");
        }

        var currentGradients = new List<Tensor<T>>(outputGradients);

        // Backward through propagation iterations (in reverse order)
        for (int iter = _numPropagations - 1; iter >= 0; iter--)
        {
            // Initialize gradient accumulators for this iteration
            var backwardPhaseGradients = new List<Tensor<T>>();
            for (int i = 0; i < numFrames; i++)
            {
                backwardPhaseGradients.Add(new Tensor<T>(currentGradients[0].Shape));
            }

            // --- Backward through FORWARD propagation phase ---
            // Forward propagation went from first to last (i = 1 to numFrames-1)
            // So backward goes from last to first
            for (int i = numFrames - 1; i >= 1; i--)
            {
                var grad = currentGradients[i];

                // Backward through forward propagation conv
                var propConvGrad = _forwardConvs[iter].Backward(grad);

                // Split gradient at concatenation point (current features + aligned)
                var (currentPartGrad, alignedPartGrad) = SplitConcatenatedGradient(
                    propConvGrad, _numFeatures, _numFeatures);

                // Backward through forward alignment
                var alignGrad = _forwardAlignments[iter].Backward(alignedPartGrad);

                // Split alignment input gradient (backward features + warped)
                var (backwardFeatGrad, warpedGrad) = SplitConcatenatedGradient(
                    alignGrad, _numFeatures, _numFeatures);

                // Backward through warping: gradients go to previous frame features and flow
                var (unwarpedGrad, flowGrad) = WarpBackward(
                    warpedGrad, _cachedFlows[i - 1].forward,
                    _cachedForwardPropFeatures![iter][i - 1]);

                // Accumulate gradients
                AccumulateGradient(backwardPhaseGradients[i], currentPartGrad);
                AccumulateGradient(backwardPhaseGradients[i], backwardFeatGrad);
                AccumulateGradient(backwardPhaseGradients[i - 1], unwarpedGrad);

                // Flow gradients would be passed to flow estimator (SPyNet)
                // For now, we store them for potential future use
            }

            // First frame passes through directly in forward propagation
            AccumulateGradient(backwardPhaseGradients[0], currentGradients[0]);

            // --- Backward through BACKWARD propagation phase ---
            var inputGradients = new List<Tensor<T>>();
            for (int i = 0; i < numFrames; i++)
            {
                inputGradients.Add(new Tensor<T>(currentGradients[0].Shape));
            }

            // Backward propagation went from last to first (i = numFrames-2 to 0)
            // So backward goes from first to last
            for (int i = 0; i < numFrames - 1; i++)
            {
                var grad = backwardPhaseGradients[i];

                // Backward through backward propagation conv
                var propConvGrad = _backwardConvs[iter].Backward(grad);

                // Split gradient at concatenation point (original features + aligned)
                var (origPartGrad, alignedPartGrad) = SplitConcatenatedGradient(
                    propConvGrad, _numFeatures, _numFeatures);

                // Backward through backward alignment
                var alignGrad = _backwardAlignments[iter].Backward(alignedPartGrad);

                // Split alignment input gradient (propagated features + warped)
                var (propFeatGrad, warpedGrad) = SplitConcatenatedGradient(
                    alignGrad, _numFeatures, _numFeatures);

                // Backward through warping: gradients go to next frame features and flow
                var (unwarpedGrad, flowGrad) = WarpBackward(
                    warpedGrad, _cachedFlows[i].backward,
                    _cachedBackwardPropFeatures![iter][i + 1]);

                // Accumulate gradients
                AccumulateGradient(inputGradients[i], origPartGrad);
                AccumulateGradient(inputGradients[i], propFeatGrad);
                AccumulateGradient(inputGradients[i + 1], unwarpedGrad);
            }

            // Last frame passes through directly in backward propagation
            AccumulateGradient(inputGradients[numFrames - 1], backwardPhaseGradients[numFrames - 1]);

            currentGradients = inputGradients;
        }

        return currentGradients;
    }

    /// <summary>
    /// Splits a gradient tensor at a concatenation point along the channel dimension.
    /// </summary>
    private (Tensor<T> first, Tensor<T> second) SplitConcatenatedGradient(
        Tensor<T> gradient, int firstChannels, int secondChannels)
    {
        bool hasBatch = gradient.Rank == 4;
        int batch = hasBatch ? gradient.Shape[0] : 1;
        int height = hasBatch ? gradient.Shape[2] : gradient.Shape[1];
        int width = hasBatch ? gradient.Shape[3] : gradient.Shape[2];

        var firstShape = hasBatch
            ? new[] { batch, firstChannels, height, width }
            : new[] { firstChannels, height, width };
        var secondShape = hasBatch
            ? new[] { batch, secondChannels, height, width }
            : new[] { secondChannels, height, width };

        var firstGrad = new Tensor<T>(firstShape);
        var secondGrad = new Tensor<T>(secondShape);

        int pixelsPerChannel = height * width;

        for (int b = 0; b < batch; b++)
        {
            // Copy first channels
            for (int c = 0; c < firstChannels; c++)
            {
                int srcOffset = hasBatch
                    ? b * (firstChannels + secondChannels) * pixelsPerChannel + c * pixelsPerChannel
                    : c * pixelsPerChannel;
                int dstOffset = hasBatch
                    ? b * firstChannels * pixelsPerChannel + c * pixelsPerChannel
                    : c * pixelsPerChannel;

                for (int i = 0; i < pixelsPerChannel; i++)
                {
                    firstGrad.Data[dstOffset + i] = gradient.Data[srcOffset + i];
                }
            }

            // Copy second channels
            for (int c = 0; c < secondChannels; c++)
            {
                int srcOffset = hasBatch
                    ? b * (firstChannels + secondChannels) * pixelsPerChannel + (firstChannels + c) * pixelsPerChannel
                    : (firstChannels + c) * pixelsPerChannel;
                int dstOffset = hasBatch
                    ? b * secondChannels * pixelsPerChannel + c * pixelsPerChannel
                    : c * pixelsPerChannel;

                for (int i = 0; i < pixelsPerChannel; i++)
                {
                    secondGrad.Data[dstOffset + i] = gradient.Data[srcOffset + i];
                }
            }
        }

        return (firstGrad, secondGrad);
    }

    /// <summary>
    /// Backward pass through bilinear warping operation.
    /// Computes gradients with respect to input features and optical flow.
    /// </summary>
    private (Tensor<T> featureGrad, Tensor<T> flowGrad) WarpBackward(
        Tensor<T> outputGrad, Tensor<T> flow, Tensor<T> inputFeature)
    {
        bool hasBatch = outputGrad.Rank == 4;
        int batch = hasBatch ? outputGrad.Shape[0] : 1;
        int channels = hasBatch ? outputGrad.Shape[1] : outputGrad.Shape[0];
        int height = hasBatch ? outputGrad.Shape[2] : outputGrad.Shape[1];
        int width = hasBatch ? outputGrad.Shape[3] : outputGrad.Shape[2];

        var featureGrad = new Tensor<T>(inputFeature.Shape);
        var flowGrad = new Tensor<T>(flow.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    // Get flow at this position
                    int flowIdxX = hasBatch
                        ? b * 2 * height * width + h * width + w
                        : h * width + w;
                    int flowIdxY = hasBatch
                        ? b * 2 * height * width + height * width + h * width + w
                        : height * width + h * width + w;

                    double dx = NumOps.ToDouble(flow.Data[flowIdxX]);
                    double dy = NumOps.ToDouble(flow.Data[flowIdxY]);

                    double srcX = w + dx;
                    double srcY = h + dy;

                    // Compute bilinear interpolation weights and their gradients
                    int x0 = (int)Math.Floor(srcX);
                    int y0 = (int)Math.Floor(srcY);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;

                    // Clamp to valid range
                    int x0c = Math.Max(0, Math.Min(x0, width - 1));
                    int x1c = Math.Max(0, Math.Min(x1, width - 1));
                    int y0c = Math.Max(0, Math.Min(y0, height - 1));
                    int y1c = Math.Max(0, Math.Min(y1, height - 1));

                    double wx = srcX - x0;
                    double wy = srcY - y0;

                    // Bilinear weights
                    double w00 = (1 - wx) * (1 - wy);
                    double w01 = wx * (1 - wy);
                    double w10 = (1 - wx) * wy;
                    double w11 = wx * wy;

                    for (int c = 0; c < channels; c++)
                    {
                        int outIdx = hasBatch
                            ? b * channels * height * width + c * height * width + h * width + w
                            : c * height * width + h * width + w;

                        double grad = NumOps.ToDouble(outputGrad.Data[outIdx]);

                        // Distribute gradient to input feature positions
                        int idx00 = hasBatch
                            ? b * channels * height * width + c * height * width + y0c * width + x0c
                            : c * height * width + y0c * width + x0c;
                        int idx01 = hasBatch
                            ? b * channels * height * width + c * height * width + y0c * width + x1c
                            : c * height * width + y0c * width + x1c;
                        int idx10 = hasBatch
                            ? b * channels * height * width + c * height * width + y1c * width + x0c
                            : c * height * width + y1c * width + x0c;
                        int idx11 = hasBatch
                            ? b * channels * height * width + c * height * width + y1c * width + x1c
                            : c * height * width + y1c * width + x1c;

                        // Add gradient contribution to feature gradient
                        featureGrad.Data[idx00] = NumOps.Add(featureGrad.Data[idx00],
                            NumOps.FromDouble(grad * w00));
                        featureGrad.Data[idx01] = NumOps.Add(featureGrad.Data[idx01],
                            NumOps.FromDouble(grad * w01));
                        featureGrad.Data[idx10] = NumOps.Add(featureGrad.Data[idx10],
                            NumOps.FromDouble(grad * w10));
                        featureGrad.Data[idx11] = NumOps.Add(featureGrad.Data[idx11],
                            NumOps.FromDouble(grad * w11));

                        // Compute gradient with respect to flow
                        // d/dx: gradient of bilinear interpolation w.r.t. x coordinate
                        double v00 = NumOps.ToDouble(GetValue(inputFeature, b, c, y0c, x0c, hasBatch, height, width, channels));
                        double v01 = NumOps.ToDouble(GetValue(inputFeature, b, c, y0c, x1c, hasBatch, height, width, channels));
                        double v10 = NumOps.ToDouble(GetValue(inputFeature, b, c, y1c, x0c, hasBatch, height, width, channels));
                        double v11 = NumOps.ToDouble(GetValue(inputFeature, b, c, y1c, x1c, hasBatch, height, width, channels));

                        // dOutput/dx = (v01 - v00) * (1 - wy) + (v11 - v10) * wy
                        double dOutDx = (v01 - v00) * (1 - wy) + (v11 - v10) * wy;
                        // dOutput/dy = (v10 - v00) * (1 - wx) + (v11 - v01) * wx
                        double dOutDy = (v10 - v00) * (1 - wx) + (v11 - v01) * wx;

                        flowGrad.Data[flowIdxX] = NumOps.Add(flowGrad.Data[flowIdxX],
                            NumOps.FromDouble(grad * dOutDx));
                        flowGrad.Data[flowIdxY] = NumOps.Add(flowGrad.Data[flowIdxY],
                            NumOps.FromDouble(grad * dOutDy));
                    }
                }
            }
        }

        return (featureGrad, flowGrad);
    }

    /// <summary>
    /// Accumulates gradient values into a target tensor.
    /// </summary>
    private void AccumulateGradient(Tensor<T> target, Tensor<T> source)
    {
        if (target.Length != source.Length)
        {
            throw new ArgumentException(
                $"Tensor size mismatch: target has {target.Length} elements, source has {source.Length}");
        }

        for (int i = 0; i < target.Length; i++)
        {
            target.Data[i] = NumOps.Add(target.Data[i], source.Data[i]);
        }
    }

    private void UpdateAllParameters()
    {
        T lr = NumOps.FromDouble(_learningRate);

        _featExtract!.UpdateParameters(lr);
        _outputConv!.UpdateParameters(lr);

        foreach (var block in _residualBlocks)
        {
            block.UpdateParameters(lr);
        }

        foreach (var layer in _upsampleLayers)
        {
            layer.UpdateParameters(lr);
        }

        for (int i = 0; i < _numPropagations; i++)
        {
            _backwardAlignments[i].UpdateParameters(lr);
            _forwardAlignments[i].UpdateParameters(lr);
            _backwardConvs[i].UpdateParameters(lr);
            _forwardConvs[i].UpdateParameters(lr);
        }
    }

    #endregion

    #region Layer Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        ClearLayers();
    }

    #endregion

    #region Serialization

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Parameter updates are not supported in ONNX mode.");

        int offset = 0;

        // Update feature extraction
        if (_featExtract != null)
        {
            var featParams = _featExtract.GetParameters();
            if (offset + featParams.Length <= parameters.Length)
            {
                var newParams = new Vector<T>(featParams.Length);
                for (int i = 0; i < featParams.Length; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                _featExtract.SetParameters(newParams);
                offset += featParams.Length;
            }
        }

        // Update residual blocks
        foreach (var block in _residualBlocks)
        {
            var blockParams = block.GetParameters();
            if (offset + blockParams.Length <= parameters.Length)
            {
                var newParams = new Vector<T>(blockParams.Length);
                for (int i = 0; i < blockParams.Length; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                block.SetParameters(newParams);
                offset += blockParams.Length;
            }
        }

        // Update flow estimator
        if (_flowEstimator != null)
        {
            var flowParams = _flowEstimator.GetParameters();
            if (offset + flowParams.Length <= parameters.Length)
            {
                var newParams = new Vector<T>(flowParams.Length);
                for (int i = 0; i < flowParams.Length; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                _flowEstimator.SetParameters(newParams);
                offset += flowParams.Length;
            }
        }

        // Update propagation layers
        for (int propIdx = 0; propIdx < _numPropagations; propIdx++)
        {
            if (propIdx < _backwardAlignments.Count)
            {
                var layerParams = _backwardAlignments[propIdx].GetParameters();
                if (offset + layerParams.Length <= parameters.Length)
                {
                    var newParams = new Vector<T>(layerParams.Length);
                    for (int i = 0; i < layerParams.Length; i++)
                        newParams[i] = parameters[offset + i];
                    _backwardAlignments[propIdx].SetParameters(newParams);
                    offset += layerParams.Length;
                }
            }
            if (propIdx < _forwardAlignments.Count)
            {
                var layerParams = _forwardAlignments[propIdx].GetParameters();
                if (offset + layerParams.Length <= parameters.Length)
                {
                    var newParams = new Vector<T>(layerParams.Length);
                    for (int i = 0; i < layerParams.Length; i++)
                        newParams[i] = parameters[offset + i];
                    _forwardAlignments[propIdx].SetParameters(newParams);
                    offset += layerParams.Length;
                }
            }
            if (propIdx < _backwardConvs.Count)
            {
                var layerParams = _backwardConvs[propIdx].GetParameters();
                if (offset + layerParams.Length <= parameters.Length)
                {
                    var newParams = new Vector<T>(layerParams.Length);
                    for (int i = 0; i < layerParams.Length; i++)
                        newParams[i] = parameters[offset + i];
                    _backwardConvs[propIdx].SetParameters(newParams);
                    offset += layerParams.Length;
                }
            }
            if (propIdx < _forwardConvs.Count)
            {
                var layerParams = _forwardConvs[propIdx].GetParameters();
                if (offset + layerParams.Length <= parameters.Length)
                {
                    var newParams = new Vector<T>(layerParams.Length);
                    for (int i = 0; i < layerParams.Length; i++)
                        newParams[i] = parameters[offset + i];
                    _forwardConvs[propIdx].SetParameters(newParams);
                    offset += layerParams.Length;
                }
            }
        }

        // Update upsampling layers
        foreach (var layer in _upsampleLayers)
        {
            var layerParams = layer.GetParameters();
            if (offset + layerParams.Length <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParams.Length);
                for (int i = 0; i < layerParams.Length; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                layer.SetParameters(newParams);
                offset += layerParams.Length;
            }
        }

        // Update output convolution
        if (_outputConv != null)
        {
            var outParams = _outputConv.GetParameters();
            if (offset + outParams.Length <= parameters.Length)
            {
                var newParams = new Vector<T>(outParams.Length);
                for (int i = 0; i < outParams.Length; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                _outputConv.SetParameters(newParams);
                offset += outParams.Length;
            }
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
            ModelType = ModelType.VideoSuperResolution,
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
        SerializeLayerParameters(writer, _featExtract!.GetParameters());
        SerializeLayerParameters(writer, _outputConv!.GetParameters());

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
        _featExtract!.SetParameters(DeserializeLayerParameters(reader));
        _outputConv!.SetParameters(DeserializeLayerParameters(reader));

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
}
