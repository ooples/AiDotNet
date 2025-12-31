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
    /// </remarks>
    public BasicVSRPlusPlus(
        NeuralNetworkArchitecture<T> architecture,
        string onnxModelPath,
        int scaleFactor = 4)
        : base(architecture, new CharbonnierLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"BasicVSR++ ONNX model not found: {onnxModelPath}");

        _useNativeMode = false;
        _onnxModelPath = onnxModelPath;
        _scaleFactor = scaleFactor;
        _numFeatures = 64;
        _numResidualBlocks = 15;
        _numPropagations = 2;
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
        _cachedFeatures.Clear();

        // Extract initial features for all frames
        var frameFeatures = new List<Tensor<T>>();
        for (int i = 0; i < numFrames; i++)
        {
            var frame = ExtractFrame(frames, i);
            var feat = _featExtract!.Forward(frame);
            frameFeatures.Add(feat);
        }

        // Compute optical flows between adjacent frames
        var flows = ComputeFlows(frames, numFrames);

        // Bidirectional propagation with second-order grid
        var propagatedFeatures = BidirectionalPropagation(frameFeatures, flows, numFrames);

        // Process each frame through residual blocks and upsampling
        for (int i = 0; i < numFrames; i++)
        {
            var feat = propagatedFeatures[i];

            // Apply residual blocks
            foreach (var block in _residualBlocks)
            {
                feat = block.Forward(feat);
            }

            // Upsampling
            foreach (var upsample in _upsampleLayers)
            {
                feat = upsample.Forward(feat);
            }

            // Final output convolution
            var output = _outputConv!.Forward(feat);

            // Store in output tensor
            StoreFrame(outputFrames, output, i);
        }

        return outputFrames;
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

    private void BackwardPass(Tensor<T> gradient)
    {
        // Simplified backward pass through the network
        var grad = _outputConv!.Backward(gradient);

        // Backward through upsampling
        for (int i = _upsampleLayers.Count - 1; i >= 0; i--)
        {
            grad = _upsampleLayers[i].Backward(grad);
        }

        // Backward through residual blocks
        for (int i = _residualBlocks.Count - 1; i >= 0; i--)
        {
            grad = _residualBlocks[i].Backward(grad);
        }

        // Backward through propagation modules
        for (int i = _numPropagations - 1; i >= 0; i--)
        {
            _forwardConvs[i].Backward(grad);
            _backwardConvs[i].Backward(grad);
            _forwardAlignments[i].Backward(grad);
            _backwardAlignments[i].Backward(grad);
        }

        _featExtract!.Backward(grad);
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
