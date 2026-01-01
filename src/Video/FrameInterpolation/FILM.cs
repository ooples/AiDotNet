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

        // Extract multi-scale features for both frames
        var features1 = ExtractFeatures(frame1);
        var features2 = ExtractFeatures(frame2);

        // Estimate bi-directional flow
        var (flow1to2, flow2to1) = EstimateFlow(features1, features2);

        // Scale flows by timestep
        var flowToT1 = ScaleFlow(flow2to1, timestep);
        var flowToT2 = ScaleFlow(flow1to2, 1.0 - timestep);

        // Estimate occlusion masks
        var (occ1, occ2) = EstimateOcclusion(features1, features2, flow1to2, flow2to1);

        // Warp features using flows
        var warped1 = WarpFeatures(features1, flowToT1);
        var warped2 = WarpFeatures(features2, flowToT2);

        // Fuse features with occlusion-aware blending
        var fused = FuseFeatures(warped1, warped2, occ1, occ2, flowToT1, flowToT2, timestep);

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
        foreach (var layer in _flowEstimator)
        {
            flowFeatures = layer.Forward(flowFeatures);
            flowFeatures = ApplyLeakyReLU(flowFeatures, 0.2);
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

    private void BackwardPass(Tensor<T> gradient)
    {
        gradient = _synthesisHead.Backward(gradient);
        for (int i = _fusionLayers.Count - 1; i >= 0; i--)
            gradient = _fusionLayers[i].Backward(gradient);
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
