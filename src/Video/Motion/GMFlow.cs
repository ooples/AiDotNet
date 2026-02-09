using System.IO;
using AiDotNet.Helpers;
using AiDotNet.Video.Options;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Video.Motion;

/// <summary>
/// GMFlow (Global Matching Flow) for accurate optical flow estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> GMFlow estimates how pixels move between video frames using
/// a global matching approach. Unlike local methods that only look at small neighborhoods,
/// GMFlow considers the entire image when matching pixels, making it better at:
/// - Large displacements (fast motion)
/// - Textureless regions
/// - Occlusions and disocclusions
/// - Repetitive patterns
///
/// The output is a "flow field" where each pixel has (dx, dy) values indicating
/// where that pixel moved to in the next frame.
/// </para>
/// <para>
/// <b>Technical Details:</b>
/// - Transformer-based global matching architecture
/// - Cross-attention for finding correspondences
/// - Hierarchical refinement for sub-pixel accuracy
/// - Self-attention for context aggregation
/// </para>
/// <para>
/// <b>Reference:</b> Xu et al., "GMFlow: Learning Optical Flow via Global Matching"
/// CVPR 2022.
/// </para>
/// </remarks>
public class GMFlow<T> : NeuralNetworkBase<T>
{
    private readonly GMFlowOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Fields

    private readonly int _height;
    private readonly int _width;
    private readonly int _channels;
    private readonly int _numFeatures;
    private readonly int _numTransformerLayers;
    private readonly int _numHeads;

    // Feature encoder
    private readonly List<ConvolutionalLayer<T>> _encoder;

    // Transformer layers for global matching
    private readonly List<ConvolutionalLayer<T>> _selfAttention;
    private readonly List<ConvolutionalLayer<T>> _crossAttention;

    // Flow decoder
    private readonly List<ConvolutionalLayer<T>> _flowDecoder;
    private readonly ConvolutionalLayer<T> _flowHead;

    // Refinement module
    private readonly List<ConvolutionalLayer<T>> _refinement;

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
    /// Gets the number of transformer layers.
    /// </summary>
    internal int NumTransformerLayers => _numTransformerLayers;

    #endregion

    #region Constructors

    public GMFlow(
        NeuralNetworkArchitecture<T> architecture,
        int numFeatures = 128,
        int numTransformerLayers = 6,
        int numHeads = 8,
        GMFlowOptions? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new GMFlowOptions();
        Options = _options;

        _height = architecture.InputHeight > 0 ? architecture.InputHeight : 480;
        _width = architecture.InputWidth > 0 ? architecture.InputWidth : 640;
        _channels = architecture.InputDepth > 0 ? architecture.InputDepth : 3;
        _numFeatures = numFeatures;
        _numTransformerLayers = numTransformerLayers;
        _numHeads = numHeads;

        _encoder = [];
        _selfAttention = [];
        _crossAttention = [];
        _flowDecoder = [];
        _refinement = [];

        int featH = _height / 8;
        int featW = _width / 8;

        // Feature encoder (ResNet-like)
        _encoder.Add(new ConvolutionalLayer<T>(_channels, _height, _width, 64, 7, 2, 3));
        _encoder.Add(new ConvolutionalLayer<T>(64, _height / 2, _width / 2, 64, 3, 1, 1));
        _encoder.Add(new ConvolutionalLayer<T>(64, _height / 2, _width / 2, 96, 3, 2, 1));
        _encoder.Add(new ConvolutionalLayer<T>(96, _height / 4, _width / 4, 96, 3, 1, 1));
        _encoder.Add(new ConvolutionalLayer<T>(96, _height / 4, _width / 4, _numFeatures, 3, 2, 1));
        _encoder.Add(new ConvolutionalLayer<T>(_numFeatures, featH, featW, _numFeatures, 3, 1, 1));

        // Transformer layers
        for (int i = 0; i < _numTransformerLayers; i++)
        {
            // Self-attention
            _selfAttention.Add(new ConvolutionalLayer<T>(_numFeatures, featH, featW, _numFeatures, 1, 1, 0));
            _selfAttention.Add(new ConvolutionalLayer<T>(_numFeatures, featH, featW, _numFeatures, 1, 1, 0));

            // Cross-attention
            _crossAttention.Add(new ConvolutionalLayer<T>(_numFeatures * 2, featH, featW, _numFeatures, 1, 1, 0));
            _crossAttention.Add(new ConvolutionalLayer<T>(_numFeatures, featH, featW, _numFeatures, 1, 1, 0));
        }

        // Flow decoder
        _flowDecoder.Add(new ConvolutionalLayer<T>(_numFeatures, featH, featW, 128, 3, 1, 1));
        _flowDecoder.Add(new ConvolutionalLayer<T>(128, featH, featW, 64, 3, 1, 1));
        _flowHead = new ConvolutionalLayer<T>(64, featH, featW, 2, 3, 1, 1);

        // Refinement
        _refinement.Add(new ConvolutionalLayer<T>(_channels * 2 + 2, _height, _width, 64, 3, 1, 1));
        _refinement.Add(new ConvolutionalLayer<T>(64, _height, _width, 32, 3, 1, 1));
        _refinement.Add(new ConvolutionalLayer<T>(32, _height, _width, 2, 3, 1, 1));

        // Register layers
        foreach (var layer in _encoder) Layers.Add(layer);
        foreach (var layer in _selfAttention) Layers.Add(layer);
        foreach (var layer in _crossAttention) Layers.Add(layer);
        foreach (var layer in _flowDecoder) Layers.Add(layer);
        Layers.Add(_flowHead);
        foreach (var layer in _refinement) Layers.Add(layer);
    }

    #endregion

    #region Public Methods

    /// <summary>
    /// Estimates optical flow between two frames.
    /// </summary>
    public Tensor<T> EstimateFlow(Tensor<T> frame1, Tensor<T> frame2)
    {
        bool hasBatch = frame1.Rank == 4;
        if (!hasBatch)
        {
            frame1 = AddBatchDimension(frame1);
            frame2 = AddBatchDimension(frame2);
        }

        // Extract features
        var feat1 = EncodeFeatures(frame1);
        var feat2 = EncodeFeatures(frame2);

        // Global matching with transformers
        var (matchedFeat1, matchedFeat2) = GlobalMatching(feat1, feat2);

        // Decode flow
        var coarseFlow = DecodeFlow(matchedFeat1, matchedFeat2);

        // Upsample and refine
        var refinedFlow = RefineFlow(frame1, frame2, coarseFlow);

        if (!hasBatch)
        {
            refinedFlow = RemoveBatchDimension(refinedFlow);
        }

        return refinedFlow;
    }

    /// <summary>
    /// Computes forward and backward flow for consistency checking.
    /// </summary>
    public (Tensor<T> Forward, Tensor<T> Backward) EstimateBidirectionalFlow(Tensor<T> frame1, Tensor<T> frame2)
    {
        var forward = EstimateFlow(frame1, frame2);
        var backward = EstimateFlow(frame2, frame1);
        return (forward, backward);
    }

    /// <summary>
    /// Estimates flow with occlusion mask.
    /// </summary>
    public (Tensor<T> Flow, Tensor<T> Occlusion) EstimateFlowWithOcclusion(Tensor<T> frame1, Tensor<T> frame2)
    {
        var (forward, backward) = EstimateBidirectionalFlow(frame1, frame2);
        var occlusion = ComputeOcclusionMask(forward, backward);
        return (forward, occlusion);
    }

    /// <summary>
    /// Warps an image using the estimated flow.
    /// </summary>
    public Tensor<T> WarpImage(Tensor<T> image, Tensor<T> flow)
    {
        bool hasBatch = image.Rank == 4;
        if (!hasBatch)
        {
            image = AddBatchDimension(image);
            flow = AddBatchDimension(flow);
        }

        var warped = BilinearWarp(image, flow);

        if (!hasBatch)
        {
            warped = RemoveBatchDimension(warped);
        }

        return warped;
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        // Expects concatenated frame pair [B, C*2, H, W]
        // Split into two frames and estimate flow
        int batchSize = input.Shape[0];
        int channels = input.Shape[1] / 2;
        int height = input.Shape[2];
        int width = input.Shape[3];

        var frame1 = new Tensor<T>([batchSize, channels, height, width]);
        var frame2 = new Tensor<T>([batchSize, channels, height, width]);

        // Split channels
        int frameSize = channels * height * width;
        for (int b = 0; b < batchSize; b++)
        {
            input.Data.Span.Slice(b * 2 * frameSize, frameSize).CopyTo(frame1.Data.Span.Slice(b * frameSize, frameSize));
            input.Data.Span.Slice(b * 2 * frameSize + frameSize, frameSize).CopyTo(frame2.Data.Span.Slice(b * frameSize, frameSize));
        }

        return EstimateFlow(frame1, frame2);
    }

    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        var prediction = Predict(input);

        // Compute loss gradient
        var lossGradient = prediction.Transform((v, idx) =>
            NumOps.Subtract(v, expectedOutput.Data.Span[idx]));

        // Backward pass through all layers
        var gradient = lossGradient;
        foreach (var layer in _refinement.AsEnumerable().Reverse())
        {
            gradient = layer.Backward(gradient);
        }
        foreach (var layer in _flowDecoder.AsEnumerable().Reverse())
        {
            gradient = layer.Backward(gradient);
        }
        foreach (var layer in _crossAttention.AsEnumerable().Reverse())
        {
            gradient = layer.Backward(gradient);
        }
        foreach (var layer in _selfAttention.AsEnumerable().Reverse())
        {
            gradient = layer.Backward(gradient);
        }
        foreach (var layer in _encoder.AsEnumerable().Reverse())
        {
            gradient = layer.Backward(gradient);
        }

        // Update parameters
        T lr = NumOps.FromDouble(0.0001);
        foreach (var layer in Layers)
            layer.UpdateParameters(lr);
    }

    #endregion

    #region Private Methods

    private Tensor<T> EncodeFeatures(Tensor<T> input)
    {
        var features = input;
        foreach (var layer in _encoder)
        {
            features = layer.Forward(features);
            features = ApplyReLU(features);
        }
        return features;
    }

    private (Tensor<T>, Tensor<T>) GlobalMatching(Tensor<T> feat1, Tensor<T> feat2)
    {
        var f1 = feat1;
        var f2 = feat2;

        for (int i = 0; i < _numTransformerLayers; i++)
        {
            // Self-attention on each feature
            int selfIdx = i * 2;
            var q1 = _selfAttention[selfIdx].Forward(f1);
            var k1 = _selfAttention[selfIdx + 1].Forward(f1);
            f1 = ApplyAttention(q1, k1, f1);

            var q2 = _selfAttention[selfIdx].Forward(f2);
            var k2 = _selfAttention[selfIdx + 1].Forward(f2);
            f2 = ApplyAttention(q2, k2, f2);

            // Cross-attention between features
            int crossIdx = i * 2;
            var concat1 = ConcatenateChannels(f1, f2);
            var cross1 = _crossAttention[crossIdx].Forward(concat1);
            cross1 = _crossAttention[crossIdx + 1].Forward(cross1);
            f1 = AddTensors(f1, cross1);

            var concat2 = ConcatenateChannels(f2, f1);
            var cross2 = _crossAttention[crossIdx].Forward(concat2);
            cross2 = _crossAttention[crossIdx + 1].Forward(cross2);
            f2 = AddTensors(f2, cross2);
        }

        return (f1, f2);
    }

    /// <summary>
    /// Applies scaled dot-product attention following the Transformer mechanism.
    /// </summary>
    /// <param name="query">Query tensor [batch, channels, height, width].</param>
    /// <param name="key">Key tensor [batch, channels, height, width].</param>
    /// <param name="value">Value tensor [batch, channels, height, width].</param>
    /// <returns>Attention output tensor [batch, channels, height, width].</returns>
    /// <remarks>
    /// Implements: Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    /// Uses a local window attention pattern for efficiency on high-resolution feature maps.
    /// </remarks>
    private Tensor<T> ApplyAttention(Tensor<T> query, Tensor<T> key, Tensor<T> value)
    {
        int batchSize = query.Shape[0];
        int channels = query.Shape[1];
        int height = query.Shape[2];
        int width = query.Shape[3];

        var output = new Tensor<T>(value.Shape);
        double scale = 1.0 / Math.Sqrt(channels);

        // Use local window attention for efficiency (window size based on feature resolution)
        int windowSize = Math.Min(Math.Min(height, width), 8);
        int halfWindow = windowSize / 2;

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    // Define local attention window
                    int hStart = Math.Max(0, h - halfWindow);
                    int hEnd = Math.Min(height, h + halfWindow + 1);
                    int wStart = Math.Max(0, w - halfWindow);
                    int wEnd = Math.Min(width, w + halfWindow + 1);

                    int windowH = hEnd - hStart;
                    int windowW = wEnd - wStart;
                    int numPositions = windowH * windowW;

                    // Compute attention scores for all positions in window
                    var scores = new double[numPositions];
                    double maxScore = double.MinValue;

                    int pos = 0;
                    for (int h2 = hStart; h2 < hEnd; h2++)
                    {
                        for (int w2 = wStart; w2 < wEnd; w2++)
                        {
                            double score = 0;
                            for (int c = 0; c < channels; c++)
                            {
                                double qVal = Convert.ToDouble(query[b, c, h, w]);
                                double kVal = Convert.ToDouble(key[b, c, h2, w2]);
                                score += qVal * kVal;
                            }
                            score *= scale;
                            scores[pos] = score;
                            if (score > maxScore) maxScore = score;
                            pos++;
                        }
                    }

                    // Apply softmax to attention scores
                    double sumExp = 0;
                    var expScores = new double[numPositions];
                    for (int i = 0; i < numPositions; i++)
                    {
                        // Subtract max for numerical stability
                        expScores[i] = Math.Exp(scores[i] - maxScore);
                        sumExp += expScores[i];
                    }

                    // Normalize to get attention weights
                    for (int i = 0; i < numPositions; i++)
                    {
                        expScores[i] /= Math.Max(sumExp, 1e-12);
                    }

                    // Apply attention weights to value vectors
                    for (int c = 0; c < channels; c++)
                    {
                        double weightedSum = 0;
                        pos = 0;
                        for (int h2 = hStart; h2 < hEnd; h2++)
                        {
                            for (int w2 = wStart; w2 < wEnd; w2++)
                            {
                                double vVal = Convert.ToDouble(value[b, c, h2, w2]);
                                weightedSum += expScores[pos] * vVal;
                                pos++;
                            }
                        }
                        output[b, c, h, w] = NumOps.FromDouble(weightedSum);
                    }
                }
            }
        }

        return output;
    }

    private Tensor<T> DecodeFlow(Tensor<T> feat1, Tensor<T> feat2)
    {
        var diff = feat1.Transform((v, idx) => NumOps.Subtract(v, feat2.Data.Span[idx]));

        foreach (var layer in _flowDecoder)
        {
            diff = layer.Forward(diff);
            diff = ApplyReLU(diff);
        }

        return _flowHead.Forward(diff);
    }

    private Tensor<T> RefineFlow(Tensor<T> frame1, Tensor<T> frame2, Tensor<T> coarseFlow)
    {
        // Upsample coarse flow
        var upFlow = UpsampleFlow(coarseFlow, _height, _width);

        // Concatenate inputs
        var concat = ConcatenateChannels(frame1, frame2);
        concat = ConcatenateChannels(concat, upFlow);

        // Refine
        var residual = concat;
        foreach (var layer in _refinement)
        {
            residual = layer.Forward(residual);
            residual = ApplyReLU(residual);
        }

        return AddTensors(upFlow, residual);
    }

    private Tensor<T> UpsampleFlow(Tensor<T> flow, int targetH, int targetW)
    {
        int batchSize = flow.Shape[0];
        int srcH = flow.Shape[2];
        int srcW = flow.Shape[3];

        var upsampled = new Tensor<T>([batchSize, 2, targetH, targetW]);
        double scaleH = (double)srcH / targetH;
        double scaleW = (double)srcW / targetW;

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < targetH; h++)
            {
                for (int w = 0; w < targetW; w++)
                {
                    int srcY = Math.Min((int)(h * scaleH), srcH - 1);
                    int srcX = Math.Min((int)(w * scaleW), srcW - 1);

                    upsampled[b, 0, h, w] = NumOps.FromDouble(Convert.ToDouble(flow[b, 0, srcY, srcX]) / scaleW);
                    upsampled[b, 1, h, w] = NumOps.FromDouble(Convert.ToDouble(flow[b, 1, srcY, srcX]) / scaleH);
                }
            }
        }

        return upsampled;
    }

    private Tensor<T> ComputeOcclusionMask(Tensor<T> forward, Tensor<T> backward)
    {
        int batchSize = forward.Shape[0];
        int height = forward.Shape[2];
        int width = forward.Shape[3];

        var occlusion = new Tensor<T>([batchSize, 1, height, width]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double fx = Convert.ToDouble(forward[b, 0, h, w]);
                    double fy = Convert.ToDouble(forward[b, 1, h, w]);

                    int tx = Math.Max(0, Math.Min((int)(w + fx), width - 1));
                    int ty = Math.Max(0, Math.Min((int)(h + fy), height - 1));

                    double bx = Convert.ToDouble(backward[b, 0, ty, tx]);
                    double by = Convert.ToDouble(backward[b, 1, ty, tx]);

                    double consistency = Math.Sqrt((fx + bx) * (fx + bx) + (fy + by) * (fy + by));
                    double occ = consistency > 1.0 ? 1.0 : 0.0;

                    occlusion[b, 0, h, w] = NumOps.FromDouble(occ);
                }
            }
        }

        return occlusion;
    }

    private Tensor<T> BilinearWarp(Tensor<T> image, Tensor<T> flow)
    {
        int batchSize = image.Shape[0];
        int channels = image.Shape[1];
        int height = image.Shape[2];
        int width = image.Shape[3];

        var warped = new Tensor<T>(image.Shape);

        for (int b = 0; b < batchSize; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    double srcX = w + Convert.ToDouble(flow[b, 0, h, w]);
                    double srcY = h + Convert.ToDouble(flow[b, 1, h, w]);

                    int x0 = (int)Math.Floor(srcX);
                    int y0 = (int)Math.Floor(srcY);
                    int x1 = x0 + 1;
                    int y1 = y0 + 1;

                    x0 = Math.Max(0, Math.Min(x0, width - 1));
                    x1 = Math.Max(0, Math.Min(x1, width - 1));
                    y0 = Math.Max(0, Math.Min(y0, height - 1));
                    y1 = Math.Max(0, Math.Min(y1, height - 1));

                    double dx = srcX - Math.Floor(srcX);
                    double dy = srcY - Math.Floor(srcY);

                    for (int c = 0; c < channels; c++)
                    {
                        double v00 = Convert.ToDouble(image[b, c, y0, x0]);
                        double v01 = Convert.ToDouble(image[b, c, y0, x1]);
                        double v10 = Convert.ToDouble(image[b, c, y1, x0]);
                        double v11 = Convert.ToDouble(image[b, c, y1, x1]);

                        double val = v00 * (1 - dx) * (1 - dy) + v01 * dx * (1 - dy) +
                                     v10 * (1 - dx) * dy + v11 * dx * dy;
                        warped[b, c, h, w] = NumOps.FromDouble(val);
                    }
                }
            }
        }

        return warped;
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

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b) =>
        a.Transform((v, idx) => NumOps.Add(v, b.Data.Span[idx]));

    private Tensor<T> ApplyReLU(Tensor<T> input) =>
        input.Transform((v, _) => NumOps.FromDouble(Math.Max(0, Convert.ToDouble(v))));

    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    {
        var result = new Tensor<T>([1, tensor.Shape[0], tensor.Shape[1], tensor.Shape[2]]);
        tensor.Data.Span.CopyTo(result.Data.Span);
        return result;
    }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    {
        var result = new Tensor<T>([tensor.Shape[1], tensor.Shape[2], tensor.Shape[3]]);
        tensor.Data.Span.Slice(0, result.Data.Length).CopyTo(result.Data.Span);
        return result;
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

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        ModelType = ModelType.OpticalFlow,
        AdditionalInfo = new Dictionary<string, object>
        {
            { "ModelName", "GMFlow" },
            { "Description", "Global Matching Optical Flow" },
            { "InputHeight", _height },
            { "InputWidth", _width },
            { "NumTransformerLayers", _numTransformerLayers }
        },
        ModelData = this.Serialize()
    };

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_height);
        writer.Write(_width);
        writer.Write(_channels);
        writer.Write(_numFeatures);
        writer.Write(_numTransformerLayers);
        writer.Write(_numHeads);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        for (int i = 0; i < 6; i++) _ = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() =>
        new GMFlow<T>(Architecture, _numFeatures, _numTransformerLayers, _numHeads);

    #endregion
}
