using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// Real-time Intermediate Flow Estimation (RIFE) for video frame interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> RIFE is a state-of-the-art model for generating intermediate frames between
/// two existing video frames. This is useful for:
/// - Increasing video frame rate (e.g., 24fps to 60fps)
/// - Creating slow-motion effects
/// - Smoothing video playback
/// - Reducing temporal aliasing
///
/// RIFE uses a privileged distillation approach with intermediate flow estimation
/// to create realistic frames at arbitrary positions between input frames.
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Uses IFNet architecture for intermediate flow estimation
/// - Coarse-to-fine flow refinement across multiple scales
/// - Context-aware fusion with feature maps
/// - Supports arbitrary timestep interpolation (not just midpoint)
/// </para>
/// <para>
/// <b>Reference:</b> Huang et al., "RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation"
/// ECCV 2022.
/// </para>
/// </remarks>
public class RIFE<T> : FrameInterpolationBase<T>
{
    private readonly RIFEOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private int _height;
    private int _width;
    private int _channels;
    private int _numFeatures;
    private int _numFlowBlocks;

    // IFNet components - coarse to fine flow estimation
    private readonly List<ConvolutionalLayer<T>> _encoder;
    private readonly List<ConvolutionalLayer<T>> _flowDecoder;
    private readonly List<ConvolutionalLayer<T>> _contextEncoder;
    private ConvolutionalLayer<T>? _fusion;
    private ConvolutionalLayer<T>? _outputConv;

    // Flow refinement blocks
    private readonly List<ConvolutionalLayer<T>> _flowBlocks;

    private const int DefaultNumFeatures = 64;
    private const int DefaultNumFlowBlocks = 8;

    // Activation cache for backward pass
    private Tensor<T>? _cachedConcatenatedFrames;
    private Tensor<T>? _cachedFrame1;
    private Tensor<T>? _cachedFrame2;
    private Tensor<T>? _cachedFlow;
    private Tensor<T>? _cachedFlow_0_1;
    private Tensor<T>? _cachedFlow_1_0;
    private Tensor<T>? _cachedFlow_t_0;
    private Tensor<T>? _cachedFlow_t_1;
    private Tensor<T>? _cachedFrame1Warped;
    private Tensor<T>? _cachedFrame2Warped;
    private Tensor<T>? _cachedContext;
    private Tensor<T>? _cachedFusionInput;
    private Tensor<T>? _cachedFused;
    private double _cachedTimestep;
    private readonly List<Tensor<T>> _cachedEncoderOutputs;
    private readonly List<Tensor<T>> _cachedFlowDecoderOutputs;
    private readonly List<Tensor<T>> _cachedContextEncoderOutputs;
    private readonly List<Tensor<T>> _cachedFlowBlockInputs;
    private readonly List<Tensor<T>> _cachedFlowBlockOutputs;

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

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the RIFE class.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numFeatures">The number of features in intermediate layers.</param>
    /// <param name="numFlowBlocks">The number of flow refinement blocks.</param>
    public RIFE(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = DefaultNumFeatures,
        int numFlowBlocks = DefaultNumFlowBlocks,
        RIFEOptions? options = null)
        : base(architecture, new CharbonnierLoss<T>())
    {
        _options = options ?? new RIFEOptions();
        Options = _options;

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 640;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFeatures = numFeatures;
        _numFlowBlocks = numFlowBlocks;

        _encoder = [];
        _flowDecoder = [];
        _contextEncoder = [];
        _flowBlocks = [];

        // Initialize activation caches
        _cachedEncoderOutputs = [];
        _cachedFlowDecoderOutputs = [];
        _cachedContextEncoderOutputs = [];
        _cachedFlowBlockInputs = [];
        _cachedFlowBlockOutputs = [];

        InitializeNativeLayers();
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Interpolates frames between two input frames.
    /// </summary>
    /// <param name="frame1">The first frame tensor [C, H, W] or [B, C, H, W].</param>
    /// <param name="frame2">The second frame tensor [C, H, W] or [B, C, H, W].</param>
    /// <param name="timestep">The interpolation position (0.0 = frame1, 1.0 = frame2, 0.5 = midpoint).</param>
    /// <returns>The interpolated frame.</returns>
    public override Tensor<T> Interpolate(Tensor<T> frame1, Tensor<T> frame2, double timestep = 0.5)
    {
        timestep = Math.Max(0.0, Math.Min(1.0, timestep));

        bool hasBatch = frame1.Rank == 4;
        if (!hasBatch)
        {
            frame1 = AddBatchDimension(frame1);
            frame2 = AddBatchDimension(frame2);
        }

        var concatenated = ConcatenateChannels(frame1, frame2);
        var result = ProcessInterpolation(concatenated, timestep);

        if (!hasBatch)
        {
            result = RemoveBatchDimension(result);
        }

        return result;
    }

    /// <summary>
    /// Interpolates multiple frames between input frames for frame rate upsampling.
    /// </summary>
    /// <param name="frames">List of input frames.</param>
    /// <param name="factor">The upsampling factor (e.g., 2 doubles frame rate).</param>
    /// <returns>List of interpolated frames including original frames.</returns>
    public List<Tensor<T>> UpsampleFrameRate(List<Tensor<T>> frames, int factor = 2)
    {
        if (frames.Count < 2)
        {
            return [.. frames];
        }

        var result = new List<Tensor<T>>();

        for (int i = 0; i < frames.Count - 1; i++)
        {
            result.Add(frames[i]);

            for (int j = 1; j < factor; j++)
            {
                double timestep = (double)j / factor;
                var interpolated = Interpolate(frames[i], frames[i + 1], timestep);
                result.Add(interpolated);
            }
        }

        result.Add(frames[^1]);
        return result;
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Input should be concatenated frames [B, 2*C, H, W]
        return ProcessInterpolation(input, 0.5);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Forward pass
        var predicted = Predict(input);

        // Calculate loss gradient
        var lossGradient = predicted.Transform((v, idx) =>
            NumOps.Subtract(v, expectedOutput.Data.Span[idx]));

        // Backward pass
        BackwardPass(lossGradient);

        // Update parameters
        T lr = NumOps.FromDouble(0.0001);
        foreach (var layer in Layers)
        {
            layer.UpdateParameters(lr);
        }
    }

    #endregion

    #region Private Methods

    private void InitializeNativeLayers()
    {
        // Encoder: extract features from concatenated frames
        _encoder.Add(new ConvolutionalLayer<T>(
            _channels * 2, _height, _width, _numFeatures, 3, 1, 1));
        _encoder.Add(new ConvolutionalLayer<T>(
            _numFeatures, _height, _width, _numFeatures * 2, 3, 2, 1));
        _encoder.Add(new ConvolutionalLayer<T>(
            _numFeatures * 2, _height / 2, _width / 2, _numFeatures * 4, 3, 2, 1));

        // Context encoder
        _contextEncoder.Add(new ConvolutionalLayer<T>(
            _channels * 2, _height, _width, _numFeatures, 3, 1, 1));
        _contextEncoder.Add(new ConvolutionalLayer<T>(
            _numFeatures, _height, _width, _numFeatures, 3, 1, 1));

        // Flow decoder
        int decoderH = _height / 4;
        int decoderW = _width / 4;

        _flowDecoder.Add(new ConvolutionalLayer<T>(
            _numFeatures * 4, decoderH, decoderW, _numFeatures * 2, 3, 1, 1));
        _flowDecoder.Add(new ConvolutionalLayer<T>(
            _numFeatures * 2, decoderH * 2, decoderW * 2, _numFeatures, 3, 1, 1));
        _flowDecoder.Add(new ConvolutionalLayer<T>(
            _numFeatures, decoderH * 4, decoderW * 4, 4, 3, 1, 1));

        // Flow refinement blocks
        for (int i = 0; i < _numFlowBlocks; i++)
        {
            _flowBlocks.Add(new ConvolutionalLayer<T>(
                _numFeatures + 4, _height, _width, _numFeatures, 3, 1, 1));
        }

        // Fusion layer
        int fusionInputChannels = _channels * 2 + _numFeatures + 4;
        _fusion = new ConvolutionalLayer<T>(
            fusionInputChannels, _height, _width, _numFeatures, 3, 1, 1);

        // Output convolution
        _outputConv = new ConvolutionalLayer<T>(
            _numFeatures, _height, _width, _channels, 3, 1, 1);

        // Register all layers
        foreach (var layer in _encoder) Layers.Add(layer);
        foreach (var layer in _flowDecoder) Layers.Add(layer);
        foreach (var layer in _contextEncoder) Layers.Add(layer);
        foreach (var layer in _flowBlocks) Layers.Add(layer);
        Layers.Add(_fusion);
        Layers.Add(_outputConv);
    }

    private Tensor<T> ProcessInterpolation(Tensor<T> concatenatedFrames, double timestep)
    {
        // Clear and cache input for backward pass
        ClearActivationCache();
        _cachedConcatenatedFrames = concatenatedFrames;
        _cachedTimestep = timestep;

        _cachedFrame1 = SliceChannels(concatenatedFrames, 0, _channels);
        _cachedFrame2 = SliceChannels(concatenatedFrames, _channels, _channels * 2);

        // Encode features with caching
        var features = concatenatedFrames;
        _cachedEncoderOutputs.Add(features); // Cache input
        foreach (var encoder in _encoder)
        {
            features = encoder.Forward(features);
            _cachedEncoderOutputs.Add(features);
        }

        // Decode flow with caching
        var flowFeatures = features;
        _cachedFlowDecoderOutputs.Add(flowFeatures); // Cache input
        for (int i = 0; i < _flowDecoder.Count; i++)
        {
            flowFeatures = _flowDecoder[i].Forward(flowFeatures);
            _cachedFlowDecoderOutputs.Add(flowFeatures);
            if (i < _flowDecoder.Count - 1)
            {
                flowFeatures = BilinearUpsample(flowFeatures, 2);
                _cachedFlowDecoderOutputs.Add(flowFeatures); // Cache after upsample
            }
        }

        _cachedFlow = flowFeatures;
        _cachedFlow_0_1 = SliceChannels(_cachedFlow, 0, 2);
        _cachedFlow_1_0 = SliceChannels(_cachedFlow, 2, 4);

        var t = NumOps.FromDouble(timestep);
        var oneMinusT = NumOps.FromDouble(1.0 - timestep);

        // Scale flows correctly for interpolation at time t
        _cachedFlow_t_0 = ScaleFlow(_cachedFlow_0_1, t);
        _cachedFlow_t_1 = ScaleFlow(_cachedFlow_1_0, oneMinusT);

        _cachedFrame1Warped = WarpImage(_cachedFrame1, _cachedFlow_t_0);
        _cachedFrame2Warped = WarpImage(_cachedFrame2, _cachedFlow_t_1);

        // Context encoder with caching
        var context = concatenatedFrames;
        _cachedContextEncoderOutputs.Add(context); // Cache input
        foreach (var contextEnc in _contextEncoder)
        {
            context = contextEnc.Forward(context);
            _cachedContextEncoderOutputs.Add(context);
        }
        _cachedContext = context;

        _cachedFusionInput = ConcatenateChannels(
            ConcatenateChannels(_cachedFrame1Warped, _cachedFrame2Warped),
            ConcatenateChannels(_cachedContext, _cachedFlow));

        _cachedFused = _fusion!.Forward(_cachedFusionInput);

        // Flow blocks with caching
        var fused = _cachedFused;
        for (int i = 0; i < _flowBlocks.Count; i++)
        {
            var blockInput = ConcatenateChannels(fused, _cachedFlow);
            _cachedFlowBlockInputs.Add(blockInput);
            fused = _flowBlocks[i].Forward(blockInput);
            _cachedFlowBlockOutputs.Add(fused);
        }

        return _outputConv!.Forward(fused);
    }

    /// <summary>
    /// Performs the backward pass with proper gradient routing through all parallel branches.
    /// </summary>
    /// <remarks>
    /// The RIFE architecture has multiple parallel branches:
    /// 1. Encoder branch: processes concatenated frames to features
    /// 2. Flow decoder branch: decodes features to optical flow
    /// 3. Context encoder branch: extracts context features independently
    /// 4. Warping branch: warps frames using predicted flow
    /// 5. Flow blocks: refine features using flow information
    ///
    /// Gradient routing must properly split at branch points and accumulate at merge points.
    /// </remarks>
    private void BackwardPass(Tensor<T> gradient)
    {
        // 1. Backward through output convolution
        gradient = _outputConv!.Backward(gradient);

        // 2. Backward through flow blocks with gradient accumulation for flow
        var flowGradAccumulator = new Tensor<T>(_cachedFlow!.Shape);
        var fusedGradient = gradient;

        for (int i = _flowBlocks.Count - 1; i >= 0; i--)
        {
            // Flow block input was [fused, flow] concatenated
            var blockGradient = _flowBlocks[i].Backward(fusedGradient);

            // Split gradient for fused and flow components
            var (fusedGrad, flowGrad) = SplitConcatenatedGradient(
                blockGradient,
                _cachedFlowBlockOutputs[Math.Max(0, i - 1)].Shape[1],
                _cachedFlow.Shape[1]);

            // Accumulate flow gradients
            flowGradAccumulator = AddTensors(flowGradAccumulator, flowGrad);

            // Use fused gradient for next iteration
            fusedGradient = fusedGrad;
        }

        // 3. Backward through fusion layer
        var fusionGradient = _fusion!.Backward(fusedGradient);

        // Split fusion gradient: [frame1_warped, frame2_warped, context, flow]
        int warpedChannels = _channels;
        int contextChannels = _cachedContext!.Shape[1];
        int flowChannels = _cachedFlow.Shape[1];

        var (warpedGradients, contextFlowGrad) = SplitConcatenatedGradient(
            fusionGradient,
            warpedChannels * 2,
            contextChannels + flowChannels);

        var (frame1WarpedGrad, frame2WarpedGrad) = SplitConcatenatedGradient(
            warpedGradients, warpedChannels, warpedChannels);

        var (contextGrad, flowGradFromFusion) = SplitConcatenatedGradient(
            contextFlowGrad, contextChannels, flowChannels);

        // Accumulate flow gradient from fusion
        flowGradAccumulator = AddTensors(flowGradAccumulator, flowGradFromFusion);

        // 4. Backward through context encoder
        var contextEncoderGrad = contextGrad;
        for (int i = _contextEncoder.Count - 1; i >= 0; i--)
        {
            contextEncoderGrad = _contextEncoder[i].Backward(contextEncoderGrad);
        }

        // 5. Backward through warping operations
        // Compute gradients w.r.t. frames and flow from warping
        var (frame1Grad, flowGrad1) = WarpBackward(
            frame1WarpedGrad, _cachedFrame1!, _cachedFlow_t_0!);
        var (frame2Grad, flowGrad2) = WarpBackward(
            frame2WarpedGrad, _cachedFrame2!, _cachedFlow_t_1!);

        // Scale flow gradients by timestep (chain rule for flow scaling)
        var t = NumOps.FromDouble(_cachedTimestep);
        var oneMinusT = NumOps.FromDouble(1.0 - _cachedTimestep);
        flowGrad1 = ScaleFlow(flowGrad1, t);
        flowGrad2 = ScaleFlow(flowGrad2, oneMinusT);

        // Combine flow gradients for flow_0_1 and flow_1_0
        var flowGradCombined = CombineFlowGradients(flowGrad1, flowGrad2, flowGradAccumulator);

        // 6. Backward through flow decoder
        var flowDecoderGrad = flowGradCombined;
        for (int i = _flowDecoder.Count - 1; i >= 0; i--)
        {
            // Handle upsampling backward (downsample gradient)
            if (i < _flowDecoder.Count - 1)
            {
                flowDecoderGrad = BilinearDownsample(flowDecoderGrad, 2);
            }
            flowDecoderGrad = _flowDecoder[i].Backward(flowDecoderGrad);
        }

        // 7. Backward through encoder
        var encoderGrad = flowDecoderGrad;
        for (int i = _encoder.Count - 1; i >= 0; i--)
        {
            encoderGrad = _encoder[i].Backward(encoderGrad);
        }

        // 8. Accumulate gradients from encoder, context encoder, and frame warping
        // All these branches feed back to the concatenated frames input
        var inputGrad = AccumulateInputGradients(
            encoderGrad,
            contextEncoderGrad,
            frame1Grad,
            frame2Grad);
    }

    /// <summary>
    /// Computes gradients through the warping operation.
    /// </summary>
    private (Tensor<T> frameGrad, Tensor<T> flowGrad) WarpBackward(
        Tensor<T> outputGrad,
        Tensor<T> inputFrame,
        Tensor<T> flow)
    {
        int batchSize = outputGrad.Shape[0];
        int channels = outputGrad.Shape[1];
        int height = outputGrad.Shape[2];
        int width = outputGrad.Shape[3];

        var frameGrad = new Tensor<T>(inputFrame.Shape);
        var flowGrad = new Tensor<T>(flow.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double dx = Convert.ToDouble(flow[b, 0, h, w]);
                    double dy = Convert.ToDouble(flow[b, 1, h, w]);

                    double srcH = h + dy;
                    double srcW = w + dx;

                    // Compute bilinear interpolation coefficients
                    int h0 = (int)Math.Floor(srcH);
                    int w0 = (int)Math.Floor(srcW);
                    int h1 = h0 + 1;
                    int w1 = w0 + 1;

                    double hWeight = srcH - Math.Floor(srcH);
                    double wWeight = srcW - Math.Floor(srcW);

                    // Coefficients for bilinear interpolation
                    double c00 = (1 - hWeight) * (1 - wWeight);
                    double c01 = (1 - hWeight) * wWeight;
                    double c10 = hWeight * (1 - wWeight);
                    double c11 = hWeight * wWeight;

                    for (int c = 0; c < channels; c++)
                    {
                        T grad = outputGrad[b, c, h, w];

                        // Distribute gradient to source pixels (gradient w.r.t. input frame)
                        if (h0 >= 0 && h0 < height && w0 >= 0 && w0 < width)
                            frameGrad[b, c, h0, w0] = NumOps.Add(
                                frameGrad[b, c, h0, w0],
                                NumOps.Multiply(grad, NumOps.FromDouble(c00)));

                        if (h0 >= 0 && h0 < height && w1 >= 0 && w1 < width)
                            frameGrad[b, c, h0, w1] = NumOps.Add(
                                frameGrad[b, c, h0, w1],
                                NumOps.Multiply(grad, NumOps.FromDouble(c01)));

                        if (h1 >= 0 && h1 < height && w0 >= 0 && w0 < width)
                            frameGrad[b, c, h1, w0] = NumOps.Add(
                                frameGrad[b, c, h1, w0],
                                NumOps.Multiply(grad, NumOps.FromDouble(c10)));

                        if (h1 >= 0 && h1 < height && w1 >= 0 && w1 < width)
                            frameGrad[b, c, h1, w1] = NumOps.Add(
                                frameGrad[b, c, h1, w1],
                                NumOps.Multiply(grad, NumOps.FromDouble(c11)));

                        // Gradient w.r.t. flow (spatial derivatives of bilinear interpolation)
                        T v00 = GetPixelSafe(inputFrame, b, c, h0, w0, height, width);
                        T v01 = GetPixelSafe(inputFrame, b, c, h0, w1, height, width);
                        T v10 = GetPixelSafe(inputFrame, b, c, h1, w0, height, width);
                        T v11 = GetPixelSafe(inputFrame, b, c, h1, w1, height, width);

                        // dOutput/dx = derivative of bilinear interpolation w.r.t. x (width direction)
                        T dX = NumOps.Add(
                            NumOps.Multiply(
                                NumOps.Subtract(v01, v00),
                                NumOps.FromDouble(1 - hWeight)),
                            NumOps.Multiply(
                                NumOps.Subtract(v11, v10),
                                NumOps.FromDouble(hWeight)));

                        // dOutput/dy = derivative of bilinear interpolation w.r.t. y (height direction)
                        T dY = NumOps.Add(
                            NumOps.Multiply(
                                NumOps.Subtract(v10, v00),
                                NumOps.FromDouble(1 - wWeight)),
                            NumOps.Multiply(
                                NumOps.Subtract(v11, v01),
                                NumOps.FromDouble(wWeight)));

                        // Accumulate flow gradients
                        flowGrad[b, 0, h, w] = NumOps.Add(
                            flowGrad[b, 0, h, w],
                            NumOps.Multiply(grad, dX));
                        flowGrad[b, 1, h, w] = NumOps.Add(
                            flowGrad[b, 1, h, w],
                            NumOps.Multiply(grad, dY));
                    }
                }
            }
        }

        return (frameGrad, flowGrad);
    }

    /// <summary>
    /// Gets a pixel value safely with boundary handling.
    /// </summary>
    private T GetPixelSafe(Tensor<T> tensor, int b, int c, int h, int w, int height, int width)
    {
        h = Math.Max(0, Math.Min(h, height - 1));
        w = Math.Max(0, Math.Min(w, width - 1));
        return tensor[b, c, h, w];
    }

    /// <summary>
    /// Splits a concatenated gradient tensor along the channel dimension.
    /// </summary>
    private (Tensor<T> first, Tensor<T> second) SplitConcatenatedGradient(
        Tensor<T> gradient, int firstChannels, int secondChannels)
    {
        int batchSize = gradient.Shape[0];
        int height = gradient.Shape[2];
        int width = gradient.Shape[3];

        var first = new Tensor<T>([batchSize, firstChannels, height, width]);
        var second = new Tensor<T>([batchSize, secondChannels, height, width]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < firstChannels; c++)
                        first[b, c, h, w] = gradient[b, c, h, w];
                    for (int c = 0; c < secondChannels; c++)
                        second[b, c, h, w] = gradient[b, firstChannels + c, h, w];
                }
            }
        }

        return (first, second);
    }

    /// <summary>
    /// Combines flow gradients from different sources.
    /// </summary>
    private Tensor<T> CombineFlowGradients(
        Tensor<T> flowGrad1,
        Tensor<T> flowGrad2,
        Tensor<T> flowGradAccumulator)
    {
        // flow_0_1 channels 0-1, flow_1_0 channels 2-3
        int batchSize = flowGradAccumulator.Shape[0];
        int height = flowGradAccumulator.Shape[2];
        int width = flowGradAccumulator.Shape[3];

        var combined = new Tensor<T>(flowGradAccumulator.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    // Flow 0->1 gradients (channels 0-1)
                    combined[b, 0, h, w] = NumOps.Add(
                        flowGradAccumulator[b, 0, h, w],
                        flowGrad1[b, 0, h, w]);
                    combined[b, 1, h, w] = NumOps.Add(
                        flowGradAccumulator[b, 1, h, w],
                        flowGrad1[b, 1, h, w]);

                    // Flow 1->0 gradients (channels 2-3)
                    combined[b, 2, h, w] = NumOps.Add(
                        flowGradAccumulator[b, 2, h, w],
                        flowGrad2[b, 0, h, w]);
                    combined[b, 3, h, w] = NumOps.Add(
                        flowGradAccumulator[b, 3, h, w],
                        flowGrad2[b, 1, h, w]);
                }
            }
        }

        return combined;
    }

    /// <summary>
    /// Accumulates gradients from multiple branches back to the input.
    /// </summary>
    private Tensor<T> AccumulateInputGradients(
        Tensor<T> encoderGrad,
        Tensor<T> contextGrad,
        Tensor<T> frame1Grad,
        Tensor<T> frame2Grad)
    {
        // The input is concatenated frames [frame1, frame2]
        // Encoder and context encoder both process full concatenated input
        // Frame gradients only affect their respective parts

        int batchSize = encoderGrad.Shape[0];
        int totalChannels = encoderGrad.Shape[1];
        int height = encoderGrad.Shape[2];
        int width = encoderGrad.Shape[3];

        var result = new Tensor<T>(encoderGrad.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < totalChannels; c++)
                    {
                        // Accumulate encoder and context encoder gradients
                        T grad = NumOps.Add(encoderGrad[b, c, h, w], contextGrad[b, c, h, w]);

                        // Add frame-specific gradients
                        if (c < _channels)
                        {
                            grad = NumOps.Add(grad, frame1Grad[b, c, h, w]);
                        }
                        else if (c < _channels * 2)
                        {
                            grad = NumOps.Add(grad, frame2Grad[b, c - _channels, h, w]);
                        }

                        result[b, c, h, w] = grad;
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            result.Data.Span[i] = NumOps.Add(a.Data.Span[i], b.Data.Span[i]);
        }
        return result;
    }

    /// <summary>
    /// Downsamples a tensor using bilinear interpolation (backward of upsample).
    /// </summary>
    private Tensor<T> BilinearDownsample(Tensor<T> input, int factor)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outHeight = inHeight / factor;
        int outWidth = inWidth / factor;

        var output = new Tensor<T>([batchSize, channels, outHeight, outWidth]);

        // Average pooling as inverse of bilinear upsampling
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < outHeight; h++)
                {
                    for (int w = 0; w < outWidth; w++)
                    {
                        T sum = NumOps.Zero;
                        int count = 0;

                        for (int dy = 0; dy < factor; dy++)
                        {
                            for (int dx = 0; dx < factor; dx++)
                            {
                                int srcH = h * factor + dy;
                                int srcW = w * factor + dx;
                                if (srcH < inHeight && srcW < inWidth)
                                {
                                    sum = NumOps.Add(sum, input[b, c, srcH, srcW]);
                                    count++;
                                }
                            }
                        }

                        output[b, c, h, w] = count > 0
                            ? NumOps.Divide(sum, NumOps.FromDouble(count))
                            : NumOps.Zero;
                    }
                }
            }
        }

        return output;
    }

    /// <summary>
    /// Clears the activation cache.
    /// </summary>
    private void ClearActivationCache()
    {
        _cachedConcatenatedFrames = null;
        _cachedFrame1 = null;
        _cachedFrame2 = null;
        _cachedFlow = null;
        _cachedFlow_0_1 = null;
        _cachedFlow_1_0 = null;
        _cachedFlow_t_0 = null;
        _cachedFlow_t_1 = null;
        _cachedFrame1Warped = null;
        _cachedFrame2Warped = null;
        _cachedContext = null;
        _cachedFusionInput = null;
        _cachedFused = null;
        _cachedEncoderOutputs.Clear();
        _cachedFlowDecoderOutputs.Clear();
        _cachedContextEncoderOutputs.Clear();
        _cachedFlowBlockInputs.Clear();
        _cachedFlowBlockOutputs.Clear();
    }

    private Tensor<T> ConcatenateChannels(Tensor<T> t1, Tensor<T> t2)
    {
        int batchSize = t1.Shape[0];
        int c1 = t1.Shape[1];
        int c2 = t2.Shape[1];
        int height = t1.Shape[2];
        int width = t1.Shape[3];

        var result = new Tensor<T>([batchSize, c1 + c2, height, width]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    for (int c = 0; c < c1; c++)
                        result[b, c, h, w] = t1[b, c, h, w];
                    for (int c = 0; c < c2; c++)
                        result[b, c1 + c, h, w] = t2[b, c, h, w];
                }
            }
        }

        return result;
    }

    private Tensor<T> SliceChannels(Tensor<T> input, int startChannel, int endChannel)
    {
        int batchSize = input.Shape[0];
        int numChannels = endChannel - startChannel;
        int height = input.Shape[2];
        int width = input.Shape[3];

        var result = new Tensor<T>([batchSize, numChannels, height, width]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < numChannels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        result[b, c, h, w] = input[b, startChannel + c, h, w];
                    }
                }
            }
        }

        return result;
    }

    private Tensor<T> ScaleFlow(Tensor<T> flow, T scale)
    {
        return flow.Transform((v, _) => NumOps.Multiply(v, scale));
    }

    private Tensor<T> WarpImage(Tensor<T> image, Tensor<T> flow)
    {
        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var result = new Tensor<T>(image.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double dx = Convert.ToDouble(flow[b, 0, h, w]);
                    double dy = Convert.ToDouble(flow[b, 1, h, w]);

                    double srcH = h + dy;
                    double srcW = w + dx;

                    for (int c = 0; c < channels; c++)
                    {
                        result[b, c, h, w] = BilinearSample(image, b, c, srcH, srcW, height, width);
                    }
                }
            }
        }

        return result;
    }

    private T BilinearSample(Tensor<T> tensor, int b, int c, double h, double w, int height, int width)
    {
        int h0 = (int)Math.Floor(h);
        int w0 = (int)Math.Floor(w);
        int h1 = h0 + 1;
        int w1 = w0 + 1;

        h0 = Math.Max(0, Math.Min(h0, height - 1));
        h1 = Math.Max(0, Math.Min(h1, height - 1));
        w0 = Math.Max(0, Math.Min(w0, width - 1));
        w1 = Math.Max(0, Math.Min(w1, width - 1));

        double hWeight = h - Math.Floor(h);
        double wWeight = w - Math.Floor(w);

        T v00 = tensor[b, c, h0, w0];
        T v01 = tensor[b, c, h0, w1];
        T v10 = tensor[b, c, h1, w0];
        T v11 = tensor[b, c, h1, w1];

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

    private Tensor<T> BilinearUpsample(Tensor<T> input, int factor)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int inHeight = input.Shape[2];
        int inWidth = input.Shape[3];

        int outHeight = inHeight * factor;
        int outWidth = inWidth * factor;

        var output = new Tensor<T>([batchSize, channels, outHeight, outWidth]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < outHeight; h++)
                {
                    for (int w = 0; w < outWidth; w++)
                    {
                        double srcH = (h + 0.5) / factor - 0.5;
                        double srcW = (w + 0.5) / factor - 0.5;
                        output[b, c, h, w] = BilinearSample(input, b, c, srcH, srcW, inHeight, inWidth);
                    }
                }
            }
        }

        return output;
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
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;

        // Update encoder layers
        foreach (var layer in _encoder)
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

        // Update flow decoder layers
        foreach (var layer in _flowDecoder)
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

        // Update context encoder layers
        foreach (var layer in _contextEncoder)
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

        // Update flow blocks
        foreach (var layer in _flowBlocks)
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

        // Update fusion and output layers
        if (_fusion != null)
        {
            var layerParams = _fusion.GetParameters();
            if (offset + layerParams.Length <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParams.Length);
                for (int i = 0; i < layerParams.Length; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                _fusion.SetParameters(newParams);
                offset += layerParams.Length;
            }
        }

        if (_outputConv != null)
        {
            var layerParams = _outputConv.GetParameters();
            if (offset + layerParams.Length <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParams.Length);
                for (int i = 0; i < layerParams.Length; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                _outputConv.SetParameters(newParams);
            }
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var additionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "RIFE" },
            { "Description", "Real-time Intermediate Flow Estimation for Video Frame Interpolation" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "InputChannels", _channels },
            { "NumFeatures", _numFeatures },
            { "NumFlowBlocks", _numFlowBlocks },
            { "NumLayers", Layers.Count }
        };

        return new ModelMetadata<T>
        {
            ModelType = ModelType.FrameInterpolation,
            AdditionalInfo = additionalInfo,
            ModelData = this.Serialize()
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height);
        writer.Write(_width);
        writer.Write(_channels);
        writer.Write(_numFeatures);
        writer.Write(_numFlowBlocks);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _height = reader.ReadInt32();
        _width = reader.ReadInt32();
        _channels = reader.ReadInt32();
        _numFeatures = reader.ReadInt32();
        _numFlowBlocks = reader.ReadInt32();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new RIFE<T>(Architecture, _numFeatures, _numFlowBlocks);
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
        return DenormalizeFrames(modelOutput);
    }

    #endregion

}
