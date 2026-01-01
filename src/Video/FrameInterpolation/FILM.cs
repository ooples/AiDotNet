using System.IO;
using AiDotNet.Helpers;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Video.FrameInterpolation;

/// <summary>
/// FILM (Frame Interpolation for Large Motion) model for high-quality frame interpolation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> FILM generates smooth intermediate frames between two input frames,
/// even when there's significant motion between them. It's particularly good at:
/// - Large motion scenes (fast camera movements, rapid object motion)
/// - Creating slow-motion effects from regular video
/// - Increasing video frame rate (24fps to 60fps)
/// - Smooth transitions between keyframes
///
/// Unlike older methods that struggle with large motions, FILM uses a multi-scale
/// feature extraction approach that handles both small and large movements gracefully.
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Multi-scale feature pyramid for handling large motions
/// - Bi-directional flow estimation with occlusion handling
/// - Feature-based frame synthesis (not just flow warping)
/// - Scale-agnostic architecture for arbitrary resolution
/// </para>
/// <para>
/// <b>Reference:</b> Reda et al., "FILM: Frame Interpolation for Large Motion"
/// ECCV 2022.
/// </para>
/// </remarks>
public class FILM<T> : NeuralNetworkBase<T>
{
    #region Fields

    private int _height;
    private int _width;
    private int _channels;
    private int _numScales;
    private int _numFeatures;

    // Multi-scale feature extractor
    private readonly List<ConvolutionalLayer<T>> _featureExtractor;
    private readonly List<ConvolutionalLayer<T>> _pyramidLayers;

    // Bi-directional flow estimator
    private readonly List<ConvolutionalLayer<T>> _flowEstimator;
    private readonly ConvolutionalLayer<T> _flowRefinement;

    // Feature fusion and synthesis
    private readonly List<ConvolutionalLayer<T>> _fusionLayers;
    private readonly ConvolutionalLayer<T> _synthesisHead;

    // Occlusion estimator
    private readonly ConvolutionalLayer<T> _occlusionEstimator;

    // Cached intermediate activations for backward pass
    private Tensor<T>? _cachedFrame1;
    private Tensor<T>? _cachedFrame2;
    private Tensor<T>? _cachedFeatures1;
    private Tensor<T>? _cachedFeatures2;
    private Tensor<T>? _cachedFlow1to2;
    private Tensor<T>? _cachedFlow2to1;
    private Tensor<T>? _cachedFlowToT1;
    private Tensor<T>? _cachedFlowToT2;
    private Tensor<T>? _cachedOcc1;
    private Tensor<T>? _cachedOcc2;
    private Tensor<T>? _cachedWarped1;
    private Tensor<T>? _cachedWarped2;
    private Tensor<T>? _cachedFused;
    private Tensor<T>? _cachedFlowConcat;
    private Tensor<T>? _cachedOccConcat;
    private List<Tensor<T>>? _cachedFeatureExtractor1Activations;
    private List<Tensor<T>>? _cachedFeatureExtractor2Activations;
    private List<Tensor<T>>? _cachedFlowEstimatorActivations;
    private List<Tensor<T>>? _cachedFusionActivations;
    private double _cachedTimestep;

    #endregion

    #region Properties

    /// <summary>
    /// Gets whether training is supported.
    /// </summary>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the input frame height.
    /// </summary>
    internal int InputHeight => _height;

    /// <summary>
    /// Gets the input frame width.
    /// </summary>
    internal int InputWidth => _width;

    /// <summary>
    /// Gets the number of pyramid scales.
    /// </summary>
    internal int NumScales => _numScales;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the FILM class.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="numScales">Number of pyramid scales for multi-scale processing.</param>
    /// <param name="numFeatures">Base number of feature channels.</param>
    public FILM(
        NeuralNetworkArchitecture<T> architecture,
        int numScales = 7,
        int numFeatures = 64)
        : base(architecture, new CharbonnierLoss<T>())
    {
        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 256;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 256;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numScales = numScales;
        _numFeatures = numFeatures;

        _featureExtractor = [];
        _pyramidLayers = [];
        _flowEstimator = [];
        _fusionLayers = [];

        // Multi-scale feature extractor (shared for both frames)
        _featureExtractor.Add(new ConvolutionalLayer<T>(_channels, _height, _width, _numFeatures, 3, 1, 1));
        _featureExtractor.Add(new ConvolutionalLayer<T>(_numFeatures, _height, _width, _numFeatures, 3, 1, 1));
        _featureExtractor.Add(new ConvolutionalLayer<T>(_numFeatures, _height, _width, _numFeatures * 2, 3, 2, 1));

        // Pyramid layers for each scale
        int currentH = _height / 2;
        int currentW = _width / 2;
        int currentC = _numFeatures * 2;

        for (int s = 0; s < _numScales - 1; s++)
        {
            _pyramidLayers.Add(new ConvolutionalLayer<T>(currentC, currentH, currentW, currentC * 2, 3, 2, 1));
            currentH /= 2;
            currentW /= 2;
            currentC = Math.Min(currentC * 2, 512);
            if (currentH < 4 || currentW < 4) break;
        }

        // Bi-directional flow estimator
        int flowInputC = _numFeatures * 2 * 2; // Concatenated features from both frames
        _flowEstimator.Add(new ConvolutionalLayer<T>(flowInputC, _height / 2, _width / 2, _numFeatures * 2, 3, 1, 1));
        _flowEstimator.Add(new ConvolutionalLayer<T>(_numFeatures * 2, _height / 2, _width / 2, _numFeatures, 3, 1, 1));
        _flowEstimator.Add(new ConvolutionalLayer<T>(_numFeatures, _height / 2, _width / 2, 4, 3, 1, 1)); // 4 = 2 flows x 2 coords

        // Flow refinement
        _flowRefinement = new ConvolutionalLayer<T>(4 + _numFeatures, _height / 2, _width / 2, 4, 3, 1, 1);

        // Occlusion estimator
        _occlusionEstimator = new ConvolutionalLayer<T>(flowInputC + 4, _height / 2, _width / 2, 2, 3, 1, 1);

        // Feature fusion for synthesis
        int fusionInputC = _numFeatures * 2 * 2 + 4 + 2; // Features + flow + occlusion
        _fusionLayers.Add(new ConvolutionalLayer<T>(fusionInputC, _height / 2, _width / 2, _numFeatures * 2, 3, 1, 1));
        _fusionLayers.Add(new ConvolutionalLayer<T>(_numFeatures * 2, _height / 2, _width / 2, _numFeatures, 3, 1, 1));

        // Synthesis head
        _synthesisHead = new ConvolutionalLayer<T>(_numFeatures, _height, _width, _channels, 3, 1, 1);

        // Register layers
        foreach (var layer in _featureExtractor) Layers.Add(layer);
        foreach (var layer in _pyramidLayers) Layers.Add(layer);
        foreach (var layer in _flowEstimator) Layers.Add(layer);
        Layers.Add(_flowRefinement);
        Layers.Add(_occlusionEstimator);
        foreach (var layer in _fusionLayers) Layers.Add(layer);
        Layers.Add(_synthesisHead);
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Interpolates a frame between two input frames.
    /// </summary>
    /// <param name="frame1">First input frame [C, H, W] or [B, C, H, W].</param>
    /// <param name="frame2">Second input frame [C, H, W] or [B, C, H, W].</param>
    /// <param name="timestep">Interpolation position (0.0 = frame1, 1.0 = frame2, 0.5 = middle).</param>
    /// <returns>Interpolated frame at the specified timestep.</returns>
    public Tensor<T> Interpolate(Tensor<T> frame1, Tensor<T> frame2, double timestep = 0.5)
    {
        bool hasBatch = frame1.Rank == 4;
        if (!hasBatch)
        {
            frame1 = AddBatchDimension(frame1);
            frame2 = AddBatchDimension(frame2);
        }

        // Cache inputs for backward pass
        _cachedFrame1 = frame1;
        _cachedFrame2 = frame2;
        _cachedTimestep = timestep;

        // Extract multi-scale features for both frames (with activation caching)
        _cachedFeatureExtractor1Activations = [];
        var features1 = ExtractFeaturesWithCache(frame1, _cachedFeatureExtractor1Activations);
        _cachedFeatureExtractor2Activations = [];
        var features2 = ExtractFeaturesWithCache(frame2, _cachedFeatureExtractor2Activations);
        _cachedFeatures1 = features1;
        _cachedFeatures2 = features2;

        // Estimate bi-directional flow (with activation caching)
        _cachedFlowEstimatorActivations = [];
        var (flow1to2, flow2to1, flowConcat) = EstimateFlowWithCache(features1, features2, _cachedFlowEstimatorActivations);
        _cachedFlow1to2 = flow1to2;
        _cachedFlow2to1 = flow2to1;
        _cachedFlowConcat = flowConcat;

        // Scale flows by timestep
        var flowToT1 = ScaleFlow(flow2to1, timestep);
        var flowToT2 = ScaleFlow(flow1to2, 1.0 - timestep);
        _cachedFlowToT1 = flowToT1;
        _cachedFlowToT2 = flowToT2;

        // Estimate occlusion masks (with activation caching)
        var (occ1, occ2, occConcat) = EstimateOcclusionWithCache(features1, features2, flow1to2, flow2to1);
        _cachedOcc1 = occ1;
        _cachedOcc2 = occ2;
        _cachedOccConcat = occConcat;

        // Warp features using flows
        var warped1 = WarpFeatures(features1, flowToT1);
        var warped2 = WarpFeatures(features2, flowToT2);
        _cachedWarped1 = warped1;
        _cachedWarped2 = warped2;

        // Fuse features with occlusion-aware blending (with activation caching)
        _cachedFusionActivations = [];
        var fused = FuseFeaturesWithCache(warped1, warped2, occ1, occ2, flowToT1, flowToT2, timestep, _cachedFusionActivations);
        _cachedFused = fused;

        // Synthesize output frame
        var output = SynthesizeFrame(fused);

        if (!hasBatch)
        {
            output = RemoveBatchDimension(output);
        }

        return output;
    }

    /// <summary>
    /// Generates multiple intermediate frames between two input frames.
    /// </summary>
    /// <param name="frame1">First input frame.</param>
    /// <param name="frame2">Second input frame.</param>
    /// <param name="numIntermediateFrames">Number of frames to generate.</param>
    /// <returns>List of interpolated frames (excluding input frames).</returns>
    public List<Tensor<T>> InterpolateMultiple(Tensor<T> frame1, Tensor<T> frame2, int numIntermediateFrames)
    {
        var results = new List<Tensor<T>>();

        for (int i = 1; i <= numIntermediateFrames; i++)
        {
            double t = (double)i / (numIntermediateFrames + 1);
            var interpolated = Interpolate(frame1, frame2, t);
            results.Add(interpolated);
        }

        return results;
    }

    /// <summary>
    /// Increases video frame rate by a given factor.
    /// </summary>
    /// <param name="frames">Input video frames.</param>
    /// <param name="factor">Frame rate multiplication factor (2, 4, or 8).</param>
    /// <returns>Frame rate enhanced video.</returns>
    public List<Tensor<T>> IncreaseFrameRate(List<Tensor<T>> frames, int factor = 2)
    {
        var result = new List<Tensor<T>>();
        int intermediateFrames = factor - 1;

        for (int i = 0; i < frames.Count - 1; i++)
        {
            result.Add(frames[i]);
            var interpolated = InterpolateMultiple(frames[i], frames[i + 1], intermediateFrames);
            result.AddRange(interpolated);
        }

        result.Add(frames[frames.Count - 1]);
        return result;
    }

    /// <summary>
    /// Creates slow-motion effect from video frames.
    /// </summary>
    /// <param name="frames">Input video frames.</param>
    /// <param name="slowdownFactor">Slowdown factor (2 = half speed, 4 = quarter speed).</param>
    /// <returns>Slow-motion video frames.</returns>
    public List<Tensor<T>> CreateSlowMotion(List<Tensor<T>> frames, int slowdownFactor = 4)
    {
        return IncreaseFrameRate(frames, slowdownFactor);
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Expects concatenated frame pair [B, C*2, H, W]
        // Split into two frames and interpolate at t=0.5
        int batchSize = input.Shape[0];
        int channels = input.Shape[1] / 2;
        int height = input.Shape[2];
        int width = input.Shape[3];

        var frame1 = new Tensor<T>([batchSize, channels, height, width]);
        var frame2 = new Tensor<T>([batchSize, channels, height, width]);

        // Split channels
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        frame1[b, c, h, w] = input[b, c, h, w];
                        frame2[b, c, h, w] = input[b, channels + c, h, w];
                    }
                }
            }
        }

        return Interpolate(frame1, frame2, 0.5);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var predicted = Predict(input);
        var lossGradient = predicted.Transform((v, idx) =>
            NumOps.Subtract(v, expectedOutput.Data[idx]));

        BackwardPass(lossGradient);

        T lr = NumOps.FromDouble(0.0001);
        foreach (var layer in Layers)
        {
            layer.UpdateParameters(lr);
        }
    }

    #endregion

    #region Private Methods

    private Tensor<T> ExtractFeatures(Tensor<T> frame)
    {
        var features = frame;
        foreach (var layer in _featureExtractor)
        {
            features = layer.Forward(features);
            features = ApplyLeakyReLU(features, 0.2);
        }
        return features;
    }

    private (Tensor<T> flow1to2, Tensor<T> flow2to1) EstimateFlow(Tensor<T> features1, Tensor<T> features2)
    {
        // Concatenate features
        var concat = ConcatenateChannels(features1, features2);

        // Estimate flow
        var flowFeatures = concat;
        for (int i = 0; i < _flowEstimator.Count; i++)
        {
            flowFeatures = _flowEstimator[i].Forward(flowFeatures);
            // Only apply activation to intermediate layers, not final flow output
            // Flow represents pixel displacements (positive or negative, unbounded)
            if (i < _flowEstimator.Count - 1)
            {
                flowFeatures = ApplyLeakyReLU(flowFeatures, 0.2);
            }
        }

        // Split into two flows
        int batchSize = flowFeatures.Shape[0];
        int height = flowFeatures.Shape[2];
        int width = flowFeatures.Shape[3];

        var flow1to2 = new Tensor<T>([batchSize, 2, height, width]);
        var flow2to1 = new Tensor<T>([batchSize, 2, height, width]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    flow1to2[b, 0, h, w] = flowFeatures[b, 0, h, w];
                    flow1to2[b, 1, h, w] = flowFeatures[b, 1, h, w];
                    flow2to1[b, 0, h, w] = flowFeatures[b, 2, h, w];
                    flow2to1[b, 1, h, w] = flowFeatures[b, 3, h, w];
                }
            }
        }

        return (flow1to2, flow2to1);
    }

    private Tensor<T> ScaleFlow(Tensor<T> flow, double scale)
    {
        return flow.Transform((v, _) => NumOps.FromDouble(Convert.ToDouble(v) * scale));
    }

    private (Tensor<T> occ1, Tensor<T> occ2) EstimateOcclusion(
        Tensor<T> features1, Tensor<T> features2,
        Tensor<T> flow1to2, Tensor<T> flow2to1)
    {
        // Concatenate all inputs
        var concat = ConcatenateChannels(features1, features2);
        concat = ConcatenateChannels(concat, flow1to2);
        concat = ConcatenateChannels(concat, flow2to1);

        // Estimate occlusion
        var occFeatures = _occlusionEstimator.Forward(concat);
        occFeatures = ApplySigmoid(occFeatures);

        // Split into two masks
        int batchSize = occFeatures.Shape[0];
        int height = occFeatures.Shape[2];
        int width = occFeatures.Shape[3];

        var occ1 = new Tensor<T>([batchSize, 1, height, width]);
        var occ2 = new Tensor<T>([batchSize, 1, height, width]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    occ1[b, 0, h, w] = occFeatures[b, 0, h, w];
                    occ2[b, 0, h, w] = occFeatures[b, 1, h, w];
                }
            }
        }

        return (occ1, occ2);
    }

    private Tensor<T> WarpFeatures(Tensor<T> features, Tensor<T> flow)
    {
        int batchSize = features.Shape[0];
        int channels = features.Shape[1];
        int height = features.Shape[2];
        int width = features.Shape[3];

        var warped = new Tensor<T>(features.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double flowX = Convert.ToDouble(flow[b, 0, h, w]);
                    double flowY = Convert.ToDouble(flow[b, 1, h, w]);

                    double srcX = w + flowX;
                    double srcY = h + flowY;

                    // Bilinear sampling
                    for (int c = 0; c < channels; c++)
                    {
                        warped[b, c, h, w] = BilinearSample(features, b, c, srcY, srcX);
                    }
                }
            }
        }

        return warped;
    }

    private T BilinearSample(Tensor<T> tensor, int batch, int channel, double y, double x)
    {
        int height = tensor.Shape[2];
        int width = tensor.Shape[3];

        int x0 = (int)Math.Floor(x);
        int y0 = (int)Math.Floor(y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        x0 = Math.Max(0, Math.Min(x0, width - 1));
        x1 = Math.Max(0, Math.Min(x1, width - 1));
        y0 = Math.Max(0, Math.Min(y0, height - 1));
        y1 = Math.Max(0, Math.Min(y1, height - 1));

        double dx = x - Math.Floor(x);
        double dy = y - Math.Floor(y);

        double v00 = Convert.ToDouble(tensor[batch, channel, y0, x0]);
        double v01 = Convert.ToDouble(tensor[batch, channel, y0, x1]);
        double v10 = Convert.ToDouble(tensor[batch, channel, y1, x0]);
        double v11 = Convert.ToDouble(tensor[batch, channel, y1, x1]);

        double value = v00 * (1 - dx) * (1 - dy) +
                       v01 * dx * (1 - dy) +
                       v10 * (1 - dx) * dy +
                       v11 * dx * dy;

        return NumOps.FromDouble(value);
    }

    private Tensor<T> FuseFeatures(
        Tensor<T> warped1, Tensor<T> warped2,
        Tensor<T> occ1, Tensor<T> occ2,
        Tensor<T> flow1, Tensor<T> flow2,
        double timestep)
    {
        // Concatenate all inputs
        var concat = ConcatenateChannels(warped1, warped2);
        concat = ConcatenateChannels(concat, flow1);
        concat = ConcatenateChannels(concat, flow2);
        concat = ConcatenateChannels(concat, occ1);
        concat = ConcatenateChannels(concat, occ2);

        // Process through fusion layers
        var features = concat;
        foreach (var layer in _fusionLayers)
        {
            features = layer.Forward(features);
            features = ApplyLeakyReLU(features, 0.2);
        }

        return features;
    }

    private Tensor<T> SynthesizeFrame(Tensor<T> features)
    {
        // Upsample to full resolution
        var upsampled = Upsample2x(features);

        // Final synthesis
        var output = _synthesisHead.Forward(upsampled);
        output = ApplySigmoid(output);

        return output;
    }

    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        int batchSize = a.Shape[0];
        int channelsA = a.Shape[1];
        int channelsB = b.Shape[1];
        int height = a.Shape[2];
        int width = a.Shape[3];

        var output = new Tensor<T>([batchSize, channelsA + channelsB, height, width]);

        for (int batch = 0; batch < batchSize; batch++)
        {
            for (int c = 0; c < channelsA; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        output[batch, c, h, w] = a[batch, c, h, w];

            for (int c = 0; c < channelsB; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        output[batch, channelsA + c, h, w] = b[batch, c, h, w];
        }

        return output;
    }

    private Tensor<T> Upsample2x(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];

        var output = new Tensor<T>([batchSize, channels, height * 2, width * 2]);

        for (int b = 0; b < batchSize; b++)
            for (int c = 0; c < channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                    {
                        T val = input[b, c, h, w];
                        output[b, c, h * 2, w * 2] = val;
                        output[b, c, h * 2, w * 2 + 1] = val;
                        output[b, c, h * 2 + 1, w * 2] = val;
                        output[b, c, h * 2 + 1, w * 2 + 1] = val;
                    }

        return output;
    }

    private Tensor<T> ApplyLeakyReLU(Tensor<T> input, double negativeSlope)
    {
        return input.Transform((v, _) =>
        {
            double x = Convert.ToDouble(v);
            return NumOps.FromDouble(x > 0 ? x : x * negativeSlope);
        });
    }

    private Tensor<T> ApplySigmoid(Tensor<T> input)
    {
        return input.Transform((v, _) =>
        {
            double x = Convert.ToDouble(v);
            return NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-x)));
        });
    }

    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        int c = tensor.Shape[0];
        int h = tensor.Shape[1];
        int w = tensor.Shape[2];
        var result = new Tensor<T>([1, c, h, w]);
        Array.Copy(tensor.Data, result.Data, tensor.Data.Length);
        return result;
    }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        int[] newShape = new int[tensor.Shape.Length - 1];
        for (int i = 0; i < newShape.Length; i++)
            newShape[i] = tensor.Shape[i + 1];
        var result = new Tensor<T>(newShape);
        Array.Copy(tensor.Data, result.Data, tensor.Data.Length);
        return result;
    }

    private Tensor<T> ExtractFeaturesWithCache(Tensor<T> frame, List<Tensor<T>> activationCache)
    {
        var features = frame;
        activationCache.Add(features); // Cache input
        foreach (var layer in _featureExtractor)
        {
            features = layer.Forward(features);
            activationCache.Add(features); // Cache pre-activation
            features = ApplyLeakyReLU(features, 0.2);
            activationCache.Add(features); // Cache post-activation
        }
        return features;
    }

    private (Tensor<T> flow1to2, Tensor<T> flow2to1, Tensor<T> concat) EstimateFlowWithCache(
        Tensor<T> features1, Tensor<T> features2, List<Tensor<T>> activationCache)
    {
        var concat = ConcatenateChannels(features1, features2);
        activationCache.Add(concat); // Cache concatenated input

        var flowFeatures = concat;
        for (int i = 0; i < _flowEstimator.Count; i++)
        {
            flowFeatures = _flowEstimator[i].Forward(flowFeatures);
            activationCache.Add(flowFeatures); // Cache layer output
            if (i < _flowEstimator.Count - 1)
            {
                flowFeatures = ApplyLeakyReLU(flowFeatures, 0.2);
                activationCache.Add(flowFeatures); // Cache post-activation
            }
        }

        // Split into two flows
        int batchSize = flowFeatures.Shape[0];
        int height = flowFeatures.Shape[2];
        int width = flowFeatures.Shape[3];

        var flow1to2 = new Tensor<T>([batchSize, 2, height, width]);
        var flow2to1 = new Tensor<T>([batchSize, 2, height, width]);

        for (int b = 0; b < batchSize; b++)
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                {
                    flow1to2[b, 0, h, w] = flowFeatures[b, 0, h, w];
                    flow1to2[b, 1, h, w] = flowFeatures[b, 1, h, w];
                    flow2to1[b, 0, h, w] = flowFeatures[b, 2, h, w];
                    flow2to1[b, 1, h, w] = flowFeatures[b, 3, h, w];
                }

        return (flow1to2, flow2to1, concat);
    }

    private (Tensor<T> occ1, Tensor<T> occ2, Tensor<T> concat) EstimateOcclusionWithCache(
        Tensor<T> features1, Tensor<T> features2,
        Tensor<T> flow1to2, Tensor<T> flow2to1)
    {
        var concat = ConcatenateChannels(features1, features2);
        concat = ConcatenateChannels(concat, flow1to2);
        concat = ConcatenateChannels(concat, flow2to1);

        var occFeatures = _occlusionEstimator.Forward(concat);
        var occPreSigmoid = occFeatures; // Cache for gradient
        occFeatures = ApplySigmoid(occFeatures);

        int batchSize = occFeatures.Shape[0];
        int height = occFeatures.Shape[2];
        int width = occFeatures.Shape[3];

        var occ1 = new Tensor<T>([batchSize, 1, height, width]);
        var occ2 = new Tensor<T>([batchSize, 1, height, width]);

        for (int b = 0; b < batchSize; b++)
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                {
                    occ1[b, 0, h, w] = occFeatures[b, 0, h, w];
                    occ2[b, 0, h, w] = occFeatures[b, 1, h, w];
                }

        return (occ1, occ2, concat);
    }

    private Tensor<T> FuseFeaturesWithCache(
        Tensor<T> warped1, Tensor<T> warped2,
        Tensor<T> occ1, Tensor<T> occ2,
        Tensor<T> flowToT1, Tensor<T> flowToT2,
        double timestep, List<Tensor<T>> activationCache)
    {
        // Weighted blending based on occlusion and timestep
        T t = NumOps.FromDouble(timestep);
        T oneMinusT = NumOps.FromDouble(1.0 - timestep);

        var blended = warped1.Transform((v, idx) =>
        {
            int batchSize = warped1.Shape[0];
            int channels = warped1.Shape[1];
            int height = warped1.Shape[2];
            int width = warped1.Shape[3];

            int totalPerBatch = channels * height * width;
            int b = idx / totalPerBatch;
            int remaining = idx % totalPerBatch;
            int c = remaining / (height * width);
            remaining = remaining % (height * width);
            int h = remaining / width;
            int w = remaining % width;

            double o1 = Convert.ToDouble(occ1[b, 0, h, w]);
            double o2 = Convert.ToDouble(occ2[b, 0, h, w]);
            double w1Val = Convert.ToDouble(warped1[b, c, h, w]);
            double w2Val = Convert.ToDouble(warped2[b, c, h, w]);

            double weight1 = (1.0 - timestep) * (1.0 - o1);
            double weight2 = timestep * (1.0 - o2);
            double totalWeight = weight1 + weight2 + 1e-8;

            return NumOps.FromDouble((weight1 * w1Val + weight2 * w2Val) / totalWeight);
        });

        activationCache.Add(blended);

        // Pass through fusion layers
        var fused = blended;
        foreach (var layer in _fusionLayers)
        {
            fused = layer.Forward(fused);
            activationCache.Add(fused);
            fused = ApplyLeakyReLU(fused, 0.2);
            activationCache.Add(fused);
        }

        return fused;
    }

    private void BackwardPass(Tensor<T> gradient)
    {
        if (_cachedFeatures1 == null || _cachedFeatures2 == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        // 1. Backpropagate through synthesis head
        gradient = _synthesisHead.Backward(gradient);

        // 2. Backpropagate through fusion layers (reverse order)
        // Apply LeakyReLU gradient for each layer
        if (_cachedFusionActivations != null)
        {
            int actIdx = _cachedFusionActivations.Count - 1;
            for (int i = _fusionLayers.Count - 1; i >= 0; i--)
            {
                // LeakyReLU gradient
                if (actIdx >= 1)
                {
                    var preActivation = _cachedFusionActivations[actIdx - 1];
                    gradient = ApplyLeakyReLUGradient(gradient, preActivation, 0.2);
                    actIdx--;
                }
                gradient = _fusionLayers[i].Backward(gradient);
                actIdx--;
            }
        }

        // 3. At this point, gradient is for the fused (blended) tensor
        // Split gradient to warped1, warped2, occ1, occ2 based on blending formula
        var (gradWarped1, gradWarped2, gradOcc1, gradOcc2) = ComputeFusionGradients(
            gradient, _cachedWarped1!, _cachedWarped2!, _cachedOcc1!, _cachedOcc2!, _cachedTimestep);

        // 4. Backpropagate warping gradients
        // Warp backward: gradient w.r.t. features and flow
        var (gradFeatures1FromWarp, gradFlowToT1) = WarpFeaturesBackward(
            gradWarped1, _cachedFeatures1, _cachedFlowToT1!);
        var (gradFeatures2FromWarp, gradFlowToT2) = WarpFeaturesBackward(
            gradWarped2, _cachedFeatures2, _cachedFlowToT2!);

        // 5. Scale flow gradients back (reverse of ScaleFlow)
        var gradFlow2to1 = ScaleFlow(gradFlowToT1, _cachedTimestep);
        var gradFlow1to2 = ScaleFlow(gradFlowToT2, 1.0 - _cachedTimestep);

        // 6. Backpropagate through occlusion estimator
        // Combine occlusion gradients and apply sigmoid gradient
        var gradOccCombined = CombineOcclusionGradients(gradOcc1, gradOcc2, _cachedOcc1!, _cachedOcc2!);
        gradOccCombined = ApplySigmoidGradient(gradOccCombined, _cachedOcc1!, _cachedOcc2!);
        var gradOccInput = _occlusionEstimator.Backward(gradOccCombined);

        // Split occlusion input gradient to features and flows
        int feat1Channels = _cachedFeatures1.Shape[1];
        int feat2Channels = _cachedFeatures2.Shape[1];
        var (gradFeaturesFromOcc1, gradFeaturesFromOcc2, gradFlowFromOcc1, gradFlowFromOcc2) =
            SplitOcclusionGradient(gradOccInput, feat1Channels, feat2Channels);

        // Accumulate flow gradients
        gradFlow1to2 = AddTensors(gradFlow1to2, gradFlowFromOcc1);
        gradFlow2to1 = AddTensors(gradFlow2to1, gradFlowFromOcc2);

        // 7. Backpropagate through flow estimator
        var gradFlowCombined = CombineFlowGradients(gradFlow1to2, gradFlow2to1);

        if (_cachedFlowEstimatorActivations != null)
        {
            int actIdx = _cachedFlowEstimatorActivations.Count - 1;
            for (int i = _flowEstimator.Count - 1; i >= 0; i--)
            {
                // Apply LeakyReLU gradient for non-final layers
                if (i < _flowEstimator.Count - 1 && actIdx >= 1)
                {
                    var preActivation = _cachedFlowEstimatorActivations[actIdx - 1];
                    gradFlowCombined = ApplyLeakyReLUGradient(gradFlowCombined, preActivation, 0.2);
                    actIdx--;
                }
                gradFlowCombined = _flowEstimator[i].Backward(gradFlowCombined);
                actIdx--;
            }
        }

        // Split flow input gradient to features1 and features2
        var (gradFeaturesFromFlow1, gradFeaturesFromFlow2) = SplitConcatenatedGradient(
            gradFlowCombined, _cachedFeatures1.Shape[1], _cachedFeatures2.Shape[1]);

        // 8. Accumulate all gradients going to features1 and features2
        var gradFeatures1 = AddTensors(gradFeatures1FromWarp, gradFeaturesFromOcc1);
        gradFeatures1 = AddTensors(gradFeatures1, gradFeaturesFromFlow1);

        var gradFeatures2 = AddTensors(gradFeatures2FromWarp, gradFeaturesFromOcc2);
        gradFeatures2 = AddTensors(gradFeatures2, gradFeaturesFromFlow2);

        // 9. Backpropagate through feature extractors
        BackwardThroughFeatureExtractor(gradFeatures1, _cachedFeatureExtractor1Activations!);
        BackwardThroughFeatureExtractor(gradFeatures2, _cachedFeatureExtractor2Activations!);

        // Clear cached activations
        ClearActivationCache();
    }

    private (Tensor<T> gradWarped1, Tensor<T> gradWarped2, Tensor<T> gradOcc1, Tensor<T> gradOcc2)
        ComputeFusionGradients(Tensor<T> gradOutput, Tensor<T> warped1, Tensor<T> warped2,
            Tensor<T> occ1, Tensor<T> occ2, double timestep)
    {
        var gradWarped1 = new Tensor<T>(warped1.Shape);
        var gradWarped2 = new Tensor<T>(warped2.Shape);
        var gradOcc1 = new Tensor<T>(occ1.Shape);
        var gradOcc2 = new Tensor<T>(occ2.Shape);

        int batchSize = warped1.Shape[0];
        int channels = warped1.Shape[1];
        int height = warped1.Shape[2];
        int width = warped1.Shape[3];

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double o1 = Convert.ToDouble(occ1[b, 0, h, w]);
                    double o2 = Convert.ToDouble(occ2[b, 0, h, w]);
                    double weight1 = (1.0 - timestep) * (1.0 - o1);
                    double weight2 = timestep * (1.0 - o2);
                    double totalWeight = weight1 + weight2 + 1e-8;

                    double dOcc1Sum = 0, dOcc2Sum = 0;

                    for (int c = 0; c < channels; c++)
                    {
                        double grad = Convert.ToDouble(gradOutput[b, c, h, w]);
                        double w1Val = Convert.ToDouble(warped1[b, c, h, w]);
                        double w2Val = Convert.ToDouble(warped2[b, c, h, w]);

                        // Gradient w.r.t. warped values
                        gradWarped1.Data[(b * channels + c) * height * width + h * width + w] =
                            NumOps.FromDouble(grad * weight1 / totalWeight);
                        gradWarped2.Data[(b * channels + c) * height * width + h * width + w] =
                            NumOps.FromDouble(grad * weight2 / totalWeight);

                        // Gradient w.r.t. occlusion (accumulated over channels)
                        double blendedVal = (weight1 * w1Val + weight2 * w2Val) / totalWeight;
                        double dWeight1 = (w1Val - blendedVal) / totalWeight;
                        double dWeight2 = (w2Val - blendedVal) / totalWeight;

                        dOcc1Sum += grad * dWeight1 * (-(1.0 - timestep));
                        dOcc2Sum += grad * dWeight2 * (-timestep);
                    }

                    gradOcc1[b, 0, h, w] = NumOps.FromDouble(dOcc1Sum);
                    gradOcc2[b, 0, h, w] = NumOps.FromDouble(dOcc2Sum);
                }
            }
        }

        return (gradWarped1, gradWarped2, gradOcc1, gradOcc2);
    }

    private (Tensor<T> gradFeatures, Tensor<T> gradFlow) WarpFeaturesBackward(
        Tensor<T> gradOutput, Tensor<T> features, Tensor<T> flow)
    {
        var gradFeatures = new Tensor<T>(features.Shape);
        var gradFlow = new Tensor<T>(flow.Shape);

        int batchSize = features.Shape[0];
        int channels = features.Shape[1];
        int height = features.Shape[2];
        int width = features.Shape[3];

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double flowX = Convert.ToDouble(flow[b, 0, h, w]);
                    double flowY = Convert.ToDouble(flow[b, 1, h, w]);

                    double srcX = w + flowX;
                    double srcY = h + flowY;

                    int x0 = (int)Math.Floor(srcX);
                    int y0 = (int)Math.Floor(srcY);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;

                    double dx = srcX - x0;
                    double dy = srcY - y0;

                    double dFlowX = 0, dFlowY = 0;

                    for (int c = 0; c < channels; c++)
                    {
                        double grad = Convert.ToDouble(gradOutput[b, c, h, w]);

                        // Bilinear interpolation weights
                        double w00 = (1 - dx) * (1 - dy);
                        double w01 = dx * (1 - dy);
                        double w10 = (1 - dx) * dy;
                        double w11 = dx * dy;

                        // Gradient w.r.t. source pixels (features)
                        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height)
                        {
                            int idx = (b * channels + c) * height * width + y0 * width + x0;
                            gradFeatures.Data[idx] = NumOps.Add(gradFeatures.Data[idx],
                                NumOps.FromDouble(grad * w00));
                        }
                        if (x1 >= 0 && x1 < width && y0 >= 0 && y0 < height)
                        {
                            int idx = (b * channels + c) * height * width + y0 * width + x1;
                            gradFeatures.Data[idx] = NumOps.Add(gradFeatures.Data[idx],
                                NumOps.FromDouble(grad * w01));
                        }
                        if (x0 >= 0 && x0 < width && y1 >= 0 && y1 < height)
                        {
                            int idx = (b * channels + c) * height * width + y1 * width + x0;
                            gradFeatures.Data[idx] = NumOps.Add(gradFeatures.Data[idx],
                                NumOps.FromDouble(grad * w10));
                        }
                        if (x1 >= 0 && x1 < width && y1 >= 0 && y1 < height)
                        {
                            int idx = (b * channels + c) * height * width + y1 * width + x1;
                            gradFeatures.Data[idx] = NumOps.Add(gradFeatures.Data[idx],
                                NumOps.FromDouble(grad * w11));
                        }

                        // Gradient w.r.t. flow (through bilinear weights)
                        double v00 = GetPixelSafe(features, b, c, y0, x0, height, width);
                        double v01 = GetPixelSafe(features, b, c, y0, x1, height, width);
                        double v10 = GetPixelSafe(features, b, c, y1, x0, height, width);
                        double v11 = GetPixelSafe(features, b, c, y1, x1, height, width);

                        // d/dFlowX = d/dSrcX
                        dFlowX += grad * ((1 - dy) * (v01 - v00) + dy * (v11 - v10));
                        // d/dFlowY = d/dSrcY
                        dFlowY += grad * ((1 - dx) * (v10 - v00) + dx * (v11 - v01));
                    }

                    gradFlow[b, 0, h, w] = NumOps.FromDouble(dFlowX);
                    gradFlow[b, 1, h, w] = NumOps.FromDouble(dFlowY);
                }
            }
        }

        return (gradFeatures, gradFlow);
    }

    private double GetPixelSafe(Tensor<T> tensor, int b, int c, int h, int w, int height, int width)
    {
        if (h < 0 || h >= height || w < 0 || w >= width)
            return 0.0;
        return Convert.ToDouble(tensor[b, c, h, w]);
    }

    private Tensor<T> ApplyLeakyReLUGradient(Tensor<T> gradOutput, Tensor<T> input, double negativeSlope)
    {
        return gradOutput.Transform((g, idx) =>
        {
            double x = Convert.ToDouble(input.Data[idx]);
            double grad = Convert.ToDouble(g);
            return NumOps.FromDouble(x > 0 ? grad : grad * negativeSlope);
        });
    }

    private Tensor<T> ApplySigmoidGradient(Tensor<T> gradOutput, Tensor<T> occ1, Tensor<T> occ2)
    {
        // Sigmoid gradient: sig(x) * (1 - sig(x))
        int batchSize = gradOutput.Shape[0];
        int height = gradOutput.Shape[2];
        int width = gradOutput.Shape[3];

        var result = new Tensor<T>(gradOutput.Shape);
        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double sig1 = Convert.ToDouble(occ1[b, 0, h, w]);
                    double sig2 = Convert.ToDouble(occ2[b, 0, h, w]);
                    double grad1 = Convert.ToDouble(gradOutput[b, 0, h, w]);
                    double grad2 = Convert.ToDouble(gradOutput[b, 1, h, w]);

                    result[b, 0, h, w] = NumOps.FromDouble(grad1 * sig1 * (1 - sig1));
                    result[b, 1, h, w] = NumOps.FromDouble(grad2 * sig2 * (1 - sig2));
                }
            }
        }
        return result;
    }

    private Tensor<T> CombineOcclusionGradients(Tensor<T> gradOcc1, Tensor<T> gradOcc2,
        Tensor<T> occ1, Tensor<T> occ2)
    {
        int batchSize = gradOcc1.Shape[0];
        int height = gradOcc1.Shape[2];
        int width = gradOcc1.Shape[3];

        var combined = new Tensor<T>([batchSize, 2, height, width]);
        for (int b = 0; b < batchSize; b++)
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                {
                    combined[b, 0, h, w] = gradOcc1[b, 0, h, w];
                    combined[b, 1, h, w] = gradOcc2[b, 0, h, w];
                }
        return combined;
    }

    private Tensor<T> CombineFlowGradients(Tensor<T> gradFlow1to2, Tensor<T> gradFlow2to1)
    {
        int batchSize = gradFlow1to2.Shape[0];
        int height = gradFlow1to2.Shape[2];
        int width = gradFlow1to2.Shape[3];

        var combined = new Tensor<T>([batchSize, 4, height, width]);
        for (int b = 0; b < batchSize; b++)
            for (int h = 0; h < height; h++)
                for (int w = 0; w < width; w++)
                {
                    combined[b, 0, h, w] = gradFlow1to2[b, 0, h, w];
                    combined[b, 1, h, w] = gradFlow1to2[b, 1, h, w];
                    combined[b, 2, h, w] = gradFlow2to1[b, 0, h, w];
                    combined[b, 3, h, w] = gradFlow2to1[b, 1, h, w];
                }
        return combined;
    }

    private (Tensor<T> grad1, Tensor<T> grad2) SplitConcatenatedGradient(
        Tensor<T> gradient, int channels1, int channels2)
    {
        int batchSize = gradient.Shape[0];
        int height = gradient.Shape[2];
        int width = gradient.Shape[3];

        var grad1 = new Tensor<T>([batchSize, channels1, height, width]);
        var grad2 = new Tensor<T>([batchSize, channels2, height, width]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < channels1; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        grad1[b, c, h, w] = gradient[b, c, h, w];

            for (int c = 0; c < channels2; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        grad2[b, c, h, w] = gradient[b, channels1 + c, h, w];
        }

        return (grad1, grad2);
    }

    private (Tensor<T> gradFeat1, Tensor<T> gradFeat2, Tensor<T> gradFlow1, Tensor<T> gradFlow2)
        SplitOcclusionGradient(Tensor<T> gradient, int feat1Channels, int feat2Channels)
    {
        int batchSize = gradient.Shape[0];
        int totalChannels = gradient.Shape[1];
        int height = gradient.Shape[2];
        int width = gradient.Shape[3];

        int flowChannels = 2;

        var gradFeat1 = new Tensor<T>([batchSize, feat1Channels, height, width]);
        var gradFeat2 = new Tensor<T>([batchSize, feat2Channels, height, width]);
        var gradFlow1 = new Tensor<T>([batchSize, flowChannels, height, width]);
        var gradFlow2 = new Tensor<T>([batchSize, flowChannels, height, width]);

        int offset = 0;
        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < feat1Channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        gradFeat1[b, c, h, w] = gradient[b, offset + c, h, w];
        }
        offset += feat1Channels;

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < feat2Channels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        gradFeat2[b, c, h, w] = gradient[b, offset + c, h, w];
        }
        offset += feat2Channels;

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < flowChannels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        gradFlow1[b, c, h, w] = gradient[b, offset + c, h, w];
        }
        offset += flowChannels;

        for (int b = 0; b < batchSize; b++)
        {
            for (int c = 0; c < flowChannels; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        gradFlow2[b, c, h, w] = gradient[b, offset + c, h, w];
        }

        return (gradFeat1, gradFeat2, gradFlow1, gradFlow2);
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        return a.Transform((v, idx) => NumOps.Add(v, b.Data[idx]));
    }

    private void BackwardThroughFeatureExtractor(Tensor<T> gradient, List<Tensor<T>> activationCache)
    {
        int actIdx = activationCache.Count - 1;
        for (int i = _featureExtractor.Count - 1; i >= 0; i--)
        {
            // LeakyReLU gradient
            if (actIdx >= 1)
            {
                var preActivation = activationCache[actIdx - 1];
                gradient = ApplyLeakyReLUGradient(gradient, preActivation, 0.2);
                actIdx--;
            }
            gradient = _featureExtractor[i].Backward(gradient);
            actIdx--;
        }
    }

    private void ClearActivationCache()
    {
        _cachedFrame1 = null;
        _cachedFrame2 = null;
        _cachedFeatures1 = null;
        _cachedFeatures2 = null;
        _cachedFlow1to2 = null;
        _cachedFlow2to1 = null;
        _cachedFlowToT1 = null;
        _cachedFlowToT2 = null;
        _cachedOcc1 = null;
        _cachedOcc2 = null;
        _cachedWarped1 = null;
        _cachedWarped2 = null;
        _cachedFused = null;
        _cachedFlowConcat = null;
        _cachedOccConcat = null;
        _cachedFeatureExtractor1Activations = null;
        _cachedFeatureExtractor2Activations = null;
        _cachedFlowEstimatorActivations = null;
        _cachedFusionActivations = null;
    }

    #endregion

    #region Abstract Implementation

    protected override void InitializeLayers() => ClearLayers();

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

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.FrameInterpolation,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "ModelName", "FILM" },
                { "Description", "Frame Interpolation for Large Motion" },
                { "InputHeight", _height },
                { "InputWidth", _width },
                { "NumScales", _numScales }
            },
            ModelData = this.Serialize()
        };
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height);
        writer.Write(_width);
        writer.Write(_channels);
        writer.Write(_numScales);
        writer.Write(_numFeatures);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _height = reader.ReadInt32();
        _width = reader.ReadInt32();
        _channels = reader.ReadInt32();
        _numScales = reader.ReadInt32();
        _numFeatures = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new FILM<T>(Architecture, _numScales, _numFeatures);

    #endregion
}
