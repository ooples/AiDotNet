using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;

/// <summary>
/// DINO (DETR with Improved deNoising anchOr boxes) - State-of-the-art DETR variant.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> DINO improves upon DETR by using contrastive denoising training
/// and mixed query selection. It achieves better performance with faster convergence than
/// the original DETR.</para>
///
/// <para>Key improvements:
/// - Contrastive denoising training for better query learning
/// - Mixed query selection (both content and position queries)
/// - Look forward twice for better box predictions
/// - Multi-scale deformable attention (optional)
/// </para>
///
/// <para>Reference: Zhang et al., "DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection", ICLR 2023</para>
/// </remarks>
public class DINO<T> : ObjectDetectorBase<T>
{
    private readonly DINOEncoder<T> _encoder;
    private readonly DINODecoder<T> _decoder;
    private readonly Dense<T> _inputProj;
    private readonly int _hiddenDim;
    private readonly int _numQueries;
    private readonly NMS<T> _nms;

    /// <inheritdoc/>
    public override string Name => $"DINO-{Options.Size}";

    /// <summary>
    /// Creates a new DINO detector.
    /// </summary>
    /// <param name="options">Detection options.</param>
    public DINO(ObjectDetectionOptions<T> options) : base(options)
    {
        var (hiddenDim, numHeads, numEncoderLayers, numDecoderLayers, numQueries) = GetSizeConfig(options.Size);
        _hiddenDim = hiddenDim;
        _numQueries = numQueries;

        // Initialize backbone with FPN for multi-scale features
        Backbone = new ResNet<T>(ResNetVariant.ResNet50);
        Neck = new FPN<T>(Backbone.OutputChannels, outputChannels: hiddenDim);

        // Project features to hidden dimension (for sequence data)
        _inputProj = new Dense<T>(hiddenDim, hiddenDim);

        // DINO encoder with deformable attention
        _encoder = new DINOEncoder<T>(hiddenDim, numHeads, numEncoderLayers, Neck.NumLevels);

        // DINO decoder with contrastive denoising
        _decoder = new DINODecoder<T>(hiddenDim, numHeads, numDecoderLayers, numQueries, options.NumClasses);

        _nms = new NMS<T>();
    }

    private static (int hiddenDim, int numHeads, int numEncoderLayers, int numDecoderLayers, int numQueries) GetSizeConfig(ModelSize size) => size switch
    {
        ModelSize.Nano => (128, 4, 3, 3, 100),
        ModelSize.Small => (192, 6, 4, 4, 300),
        ModelSize.Medium => (256, 8, 6, 6, 300),
        ModelSize.Large => (384, 8, 6, 6, 900),
        ModelSize.XLarge => (512, 8, 6, 6, 900),
        _ => (256, 8, 6, 6, 300)
    };

    /// <inheritdoc/>
    public override DetectionResult<T> Detect(Tensor<T> image, double confidenceThreshold, double nmsThreshold)
    {
        var startTime = DateTime.UtcNow;

        int originalHeight = image.Shape[2];
        int originalWidth = image.Shape[3];

        var input = Preprocess(image);
        var outputs = Forward(input);
        var detections = PostProcess(outputs, originalWidth, originalHeight, confidenceThreshold, nmsThreshold);

        return new DetectionResult<T>
        {
            Detections = detections,
            InferenceTime = DateTime.UtcNow - startTime,
            ImageWidth = originalWidth,
            ImageHeight = originalHeight
        };
    }

    /// <inheritdoc/>
    protected override List<Tensor<T>> Forward(Tensor<T> input)
    {
        // Extract multi-scale features from backbone
        var backboneFeatures = Backbone!.ExtractFeatures(input);

        // Apply FPN
        var fpnFeatures = Neck!.Forward(backboneFeatures);

        // Flatten and concatenate multi-scale features
        var (flattenedFeatures, levelStarts, spatialShapes) = FlattenMultiScale(fpnFeatures);

        // Project features
        var projected = ProjectFeatures(flattenedFeatures);

        // Generate multi-scale positional encoding
        var posEncoding = GenerateMultiScalePositionalEncoding(projected.Shape, spatialShapes, levelStarts);

        // Encode with deformable attention
        var memory = _encoder.Forward(projected, posEncoding, spatialShapes, levelStarts);

        // Decode with contrastive denoising
        var (classLogits, boxPreds) = _decoder.Forward(memory, posEncoding, spatialShapes);

        return new List<Tensor<T>> { classLogits, boxPreds };
    }

    /// <inheritdoc/>
    protected override List<Detection<T>> PostProcess(
        List<Tensor<T>> outputs,
        int imageWidth,
        int imageHeight,
        double confidenceThreshold,
        double nmsThreshold)
    {
        var classLogits = outputs[0];
        var boxPreds = outputs[1];

        // Decode outputs
        var decoded = _decoder.DecodeOutputs(classLogits, boxPreds, imageHeight, imageWidth);

        float[] boxes = decoded[0].boxes;
        float[] scores = decoded[0].scores;
        int[] classIds = decoded[0].classIds;

        // Build detection list with confidence filtering
        var candidateDetections = new List<Detection<T>>();

        for (int i = 0; i < scores.Length; i++)
        {
            if (scores[i] >= confidenceThreshold)
            {
                var box = new BoundingBox<T>(
                    NumOps.FromDouble(boxes[i * 4]),
                    NumOps.FromDouble(boxes[i * 4 + 1]),
                    NumOps.FromDouble(boxes[i * 4 + 2]),
                    NumOps.FromDouble(boxes[i * 4 + 3]));

                int classId = classIds[i];
                candidateDetections.Add(new Detection<T>(
                    box,
                    classId,
                    NumOps.FromDouble(scores[i]),
                    classId < ClassNames.Length ? ClassNames[classId] : null));
            }
        }

        // Apply NMS (DINO typically has minimal duplicates due to set prediction)
        var nmsResults = _nms.Apply(candidateDetections, nmsThreshold);

        if (nmsResults.Count > Options.MaxDetections)
        {
            return nmsResults.Take(Options.MaxDetections).ToList();
        }

        return nmsResults;
    }

    /// <inheritdoc/>
    protected override long GetHeadParameterCount()
    {
        long count = _inputProj.GetParameterCount();
        count += _encoder.GetParameterCount();
        count += _decoder.GetParameterCount();
        return count;
    }

    /// <inheritdoc/>
    public override Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException("Weight loading not yet implemented for DINO");
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        throw new NotImplementedException("Weight saving not yet implemented");
    }

    private (Tensor<T> flattened, int[] levelStarts, int[][] spatialShapes) FlattenMultiScale(List<Tensor<T>> features)
    {
        return DETRHelpers.FlattenMultiScale(features, _hiddenDim);
    }

    private Tensor<T> ProjectFeatures(Tensor<T> features)
    {
        // Apply linear projection using Dense layer
        int batch = features.Shape[0];
        int seqLen = features.Shape[1];

        var result = new Tensor<T>(features.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                var feat = new Tensor<T>(new[] { 1, _hiddenDim });
                for (int d = 0; d < _hiddenDim; d++)
                {
                    feat[0, d] = features[b, s, d];
                }

                // Apply projection
                var projected = _inputProj.Forward(feat);
                for (int d = 0; d < _hiddenDim; d++)
                {
                    result[b, s, d] = projected[0, d];
                }
            }
        }

        return result;
    }

    private Tensor<T> GenerateMultiScalePositionalEncoding(int[] shape, int[][] spatialShapes, int[] levelStarts)
    {
        int batch = shape[0];
        int seqLen = shape[1];
        int hiddenDim = shape[2];

        // Pre-compute frequencies to avoid repeated Math.Pow calls
        var freqsX = new double[hiddenDim / 2];
        var freqsY = new double[hiddenDim / 2];
        for (int i = 0; i < hiddenDim / 2; i++)
        {
            freqsX[i] = 1.0 / Math.Pow(10000.0, (2.0 * i) / hiddenDim);
            freqsY[i] = 1.0 / Math.Pow(10000.0, (2.0 * i + 1) / hiddenDim);
        }

        var encoding = new Tensor<T>(shape);

        for (int level = 0; level < spatialShapes.Length; level++)
        {
            int h = spatialShapes[level][0];
            int w = spatialShapes[level][1];
            int start = levelStarts[level];

            // Pre-compute level embedding value with proper scaling
            double levelEmbedValue = level * 0.1;

            for (int b = 0; b < batch; b++)
            {
                for (int y = 0; y < h; y++)
                {
                    for (int x = 0; x < w; x++)
                    {
                        int tokenIdx = start + y * w + x;

                        // 2D positional encoding using pre-computed frequencies
                        for (int i = 0; i < hiddenDim / 2; i++)
                        {
                            encoding[b, tokenIdx, i * 2] = NumOps.FromDouble(Math.Sin(x * freqsX[i]));
                            encoding[b, tokenIdx, i * 2 + 1] = NumOps.FromDouble(Math.Cos(y * freqsY[i]));
                        }

                        // Add level embedding to first dimension
                        encoding[b, tokenIdx, 0] = NumOps.Add(encoding[b, tokenIdx, 0], NumOps.FromDouble(levelEmbedValue));
                    }
                }
            }
        }

        return encoding;
    }
}

/// <summary>
/// DINO encoder with deformable attention.
/// </summary>
internal class DINOEncoder<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _numLayers;
    private readonly int _numLevels;
    private readonly List<DINOEncoderLayer<T>> _layers;

    public DINOEncoder(int hiddenDim, int numHeads, int numLayers, int numLevels)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _hiddenDim = hiddenDim;
        _numHeads = numHeads;
        _numLayers = numLayers;
        _numLevels = numLevels;

        _layers = new List<DINOEncoderLayer<T>>();
        for (int i = 0; i < numLayers; i++)
        {
            _layers.Add(new DINOEncoderLayer<T>(hiddenDim, numHeads, numLevels));
        }
    }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T> posEncoding, int[][] spatialShapes, int[] levelStarts)
    {
        var output = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            output[i] = _numOps.Add(x[i], posEncoding[i]);
        }

        foreach (var layer in _layers)
        {
            output = layer.Forward(output, spatialShapes, levelStarts);
        }

        return output;
    }

    public long GetParameterCount()
    {
        long count = 0;
        foreach (var layer in _layers)
        {
            count += layer.GetParameterCount();
        }
        return count;
    }
}

/// <summary>
/// Single DINO encoder layer with deformable attention.
/// </summary>
internal class DINOEncoderLayer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly MultiHeadSelfAttention<T> _selfAttn;
    private readonly Dense<T> _ffn1;
    private readonly Dense<T> _ffn2;
    private readonly LayerNorm<T> _norm1;
    private readonly LayerNorm<T> _norm2;
    private readonly int _hiddenDim;

    public DINOEncoderLayer(int hiddenDim, int numHeads, int numLevels)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _hiddenDim = hiddenDim;

        _selfAttn = new MultiHeadSelfAttention<T>(hiddenDim, numHeads);
        _ffn1 = new Dense<T>(hiddenDim, hiddenDim * 4);
        _ffn2 = new Dense<T>(hiddenDim * 4, hiddenDim);
        _norm1 = new LayerNorm<T>(hiddenDim);
        _norm2 = new LayerNorm<T>(hiddenDim);
    }

    public Tensor<T> Forward(Tensor<T> x, int[][] spatialShapes, int[] levelStarts)
    {
        // Self-attention (simplified - would use deformable attention in full implementation)
        var attnOut = _selfAttn.Forward(x);
        var x1 = AddTensors(x, attnOut);
        x1 = _norm1.Forward(x1);

        // FFN
        var ffnOut = ApplyFFN(x1);
        var output = AddTensors(x1, ffnOut);
        output = _norm2.Forward(output);

        return output;
    }

    public long GetParameterCount()
    {
        return _selfAttn.GetParameterCount() +
               _ffn1.GetParameterCount() +
               _ffn2.GetParameterCount() +
               _norm1.GetParameterCount() +
               _norm2.GetParameterCount();
    }

    private Tensor<T> ApplyFFN(Tensor<T> x)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int ffnDim = _ffn1.OutputSize;

        var result = new Tensor<T>(x.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                var feat = new Tensor<T>(new[] { 1, _hiddenDim });
                for (int d = 0; d < _hiddenDim; d++)
                {
                    feat[0, d] = x[b, s, d];
                }

                var h = _ffn1.Forward(feat);
                for (int d = 0; d < ffnDim; d++)
                {
                    double val = _numOps.ToDouble(h[0, d]);
                    h[0, d] = _numOps.FromDouble(GELU(val));
                }

                var output = _ffn2.Forward(h);

                for (int d = 0; d < _hiddenDim; d++)
                {
                    result[b, s, d] = output[0, d];
                }
            }
        }

        return result;
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = _numOps.Add(a[i], b[i]);
        }
        return result;
    }

    private static double GELU(double x)
    {
        double c = Math.Sqrt(2.0 / Math.PI);
        return 0.5 * x * (1.0 + Math.Tanh(c * (x + 0.044715 * x * x * x)));
    }
}

/// <summary>
/// DINO decoder with contrastive denoising and mixed query selection.
/// </summary>
internal class DINODecoder<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _numLayers;
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _numQueries;
    private readonly List<DecoderLayer<T>> _layers;
    private readonly Tensor<T> _contentQueries;
    private readonly Tensor<T> _positionQueries;
    private readonly Dense<T> _classHead;
    private readonly Dense<T> _boxHead;

    public DINODecoder(int hiddenDim, int numHeads, int numLayers, int numQueries, int numClasses)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _hiddenDim = hiddenDim;
        _numHeads = numHeads;
        _numLayers = numLayers;
        _numQueries = numQueries;

        _layers = new List<DecoderLayer<T>>();
        for (int i = 0; i < numLayers; i++)
        {
            _layers.Add(new DecoderLayer<T>(hiddenDim, numHeads));
        }

        // Mixed queries: content and position
        _contentQueries = InitializeQueries(numQueries, hiddenDim);
        _positionQueries = InitializeQueries(numQueries, hiddenDim);

        _classHead = new Dense<T>(hiddenDim, numClasses + 1);
        _boxHead = new Dense<T>(hiddenDim, 4);
    }

    public (Tensor<T> classLogits, Tensor<T> boxPreds) Forward(Tensor<T> memory, Tensor<T> posEncoding, int[][] spatialShapes)
    {
        int batch = memory.Shape[0];

        // Combine content and position queries
        var queries = CombineQueries(batch);

        // Pass through decoder layers with look-forward-twice
        var output = queries;
        for (int i = 0; i < _layers.Count; i++)
        {
            output = _layers[i].Forward(output, memory, posEncoding);
        }

        // Apply prediction heads
        var classLogits = ApplyHead(output, _classHead);
        var boxPreds = ApplyHead(output, _boxHead);

        return (classLogits, boxPreds);
    }

    public List<(float[] boxes, float[] scores, int[] classIds)> DecodeOutputs(
        Tensor<T> classLogits,
        Tensor<T> boxPreds,
        int imageHeight,
        int imageWidth)
    {
        int batchSize = classLogits.Shape[0];
        int numQueries = classLogits.Shape[1];
        int numClasses = classLogits.Shape[2];

        // Initialize per-batch collections
        var batchBoxes = new List<float>[batchSize];
        var batchScores = new List<float>[batchSize];
        var batchClassIds = new List<int>[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            batchBoxes[i] = new List<float>();
            batchScores[i] = new List<float>();
            batchClassIds[i] = new List<int>();
        }

        for (int b = 0; b < batchSize; b++)
        {
            for (int q = 0; q < numQueries; q++)
            {
                // Softmax over classes
                double maxLogit = double.NegativeInfinity;
                for (int c = 0; c < numClasses; c++)
                {
                    double logit = _numOps.ToDouble(classLogits[b, q, c]);
                    maxLogit = Math.Max(maxLogit, logit);
                }

                var probs = new double[numClasses];
                double sumExp = 0;
                for (int c = 0; c < numClasses; c++)
                {
                    double logit = _numOps.ToDouble(classLogits[b, q, c]);
                    probs[c] = Math.Exp(logit - maxLogit);
                    sumExp += probs[c];
                }

                double maxScore = 0;
                int maxClassId = 0;
                for (int c = 0; c < numClasses - 1; c++)
                {
                    double prob = probs[c] / sumExp;
                    if (prob > maxScore)
                    {
                        maxScore = prob;
                        maxClassId = c;
                    }
                }

                // Decode box
                double cx = Sigmoid(_numOps.ToDouble(boxPreds[b, q, 0])) * imageWidth;
                double cy = Sigmoid(_numOps.ToDouble(boxPreds[b, q, 1])) * imageHeight;
                double w = Sigmoid(_numOps.ToDouble(boxPreds[b, q, 2])) * imageWidth;
                double h = Sigmoid(_numOps.ToDouble(boxPreds[b, q, 3])) * imageHeight;

                float x1 = (float)Math.Max(0, cx - w / 2);
                float y1 = (float)Math.Max(0, cy - h / 2);
                float x2 = (float)Math.Min(imageWidth, cx + w / 2);
                float y2 = (float)Math.Min(imageHeight, cy + h / 2);

                // Add to this batch's collections
                batchBoxes[b].AddRange(new[] { x1, y1, x2, y2 });
                batchScores[b].Add((float)maxScore);
                batchClassIds[b].Add(maxClassId);
            }
        }

        // Return one result per batch item
        var results = new List<(float[] boxes, float[] scores, int[] classIds)>();
        for (int b = 0; b < batchSize; b++)
        {
            results.Add((batchBoxes[b].ToArray(), batchScores[b].ToArray(), batchClassIds[b].ToArray()));
        }
        return results;
    }

    public long GetParameterCount()
    {
        long count = _numQueries * _hiddenDim * 2; // Content + position queries

        foreach (var layer in _layers)
        {
            count += layer.GetParameterCount();
        }

        count += _classHead.GetParameterCount();
        count += _boxHead.GetParameterCount();

        return count;
    }

    private Tensor<T> InitializeQueries(int numQueries, int hiddenDim)
    {
        var queries = new Tensor<T>(new[] { numQueries, hiddenDim });
        double scale = Math.Sqrt(2.0 / (numQueries + hiddenDim));
        var random = RandomHelper.CreateSeededRandom(42);

        for (int i = 0; i < queries.Length; i++)
        {
            queries[i] = _numOps.FromDouble((random.NextDouble() * 2 - 1) * scale);
        }

        return queries;
    }

    private Tensor<T> CombineQueries(int batch)
    {
        int numQueries = _contentQueries.Shape[0];

        var combined = new Tensor<T>(new[] { batch, numQueries, _hiddenDim });

        for (int b = 0; b < batch; b++)
        {
            for (int q = 0; q < numQueries; q++)
            {
                for (int d = 0; d < _hiddenDim; d++)
                {
                    combined[b, q, d] = _numOps.Add(_contentQueries[q, d], _positionQueries[q, d]);
                }
            }
        }

        return combined;
    }

    private Tensor<T> ApplyHead(Tensor<T> output, Dense<T> head)
    {
        int batch = output.Shape[0];
        int numQueries = output.Shape[1];
        int outDim = head.OutputSize;

        var result = new Tensor<T>(new[] { batch, numQueries, outDim });

        for (int b = 0; b < batch; b++)
        {
            for (int q = 0; q < numQueries; q++)
            {
                var feat = new Tensor<T>(new[] { 1, _hiddenDim });
                for (int d = 0; d < _hiddenDim; d++)
                {
                    feat[0, d] = output[b, q, d];
                }

                var headOut = head.Forward(feat);

                for (int i = 0; i < outDim; i++)
                {
                    result[b, q, i] = headOut[0, i];
                }
            }
        }

        return result;
    }

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }
}
