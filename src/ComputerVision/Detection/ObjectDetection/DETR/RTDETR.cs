using System.IO;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;

/// <summary>
/// RT-DETR (Real-Time DEtection TRansformer) - First real-time end-to-end object detector.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> RT-DETR is the first real-time end-to-end transformer-based
/// object detector. It achieves YOLO-level speed while maintaining transformer accuracy
/// by using a hybrid encoder and efficient decoder design.</para>
///
/// <para>Key features:
/// - Hybrid encoder with intra-scale and cross-scale feature interaction
/// - Efficient decoder with uncertainty-minimal query selection
/// - Flexible inference speed/accuracy trade-off
/// - No NMS post-processing required
/// </para>
///
/// <para>Reference: Lv et al., "DETRs Beat YOLOs on Real-time Object Detection", CVPR 2024</para>
/// </remarks>
public class RTDETR<T> : ObjectDetectorBase<T>
{
    private readonly RTDETREncoder<T> _encoder;
    private readonly RTDETRDecoder<T> _decoder;
    private readonly int _hiddenDim;
    private readonly int _numQueries;
    private readonly NMS<T> _nms;

    /// <inheritdoc/>
    public override string Name => $"RT-DETR-{Options.Size}";

    /// <summary>
    /// Creates a new RT-DETR detector.
    /// </summary>
    /// <param name="options">Detection options.</param>
    public RTDETR(ObjectDetectionOptions<T> options) : base(options)
    {
        var (hiddenDim, numHeads, numEncoderLayers, numDecoderLayers, numQueries) = GetSizeConfig(options.Size);
        _hiddenDim = hiddenDim;
        _numQueries = numQueries;

        // RT-DETR uses either ResNet or HGNetV2 backbone
        Backbone = new ResNet<T>(ResNetVariant.ResNet50);

        // Hybrid encoder neck
        Neck = new PANet<T>(Backbone.OutputChannels, outputChannels: hiddenDim);

        // Hybrid encoder with efficient attention
        _encoder = new RTDETREncoder<T>(hiddenDim, numHeads, numEncoderLayers, Neck.NumLevels);

        // Efficient decoder
        _decoder = new RTDETRDecoder<T>(hiddenDim, numHeads, numDecoderLayers, numQueries, options.NumClasses);

        _nms = new NMS<T>();
    }

    private static (int hiddenDim, int numHeads, int numEncoderLayers, int numDecoderLayers, int numQueries) GetSizeConfig(ModelSize size) => size switch
    {
        ModelSize.Nano => (128, 4, 1, 3, 100),
        ModelSize.Small => (192, 6, 1, 4, 300),
        ModelSize.Medium => (256, 8, 1, 4, 300),
        ModelSize.Large => (384, 8, 1, 6, 300),
        ModelSize.XLarge => (512, 8, 1, 6, 300),
        _ => (256, 8, 1, 4, 300)
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
        // Extract multi-scale features
        var backboneFeatures = Backbone!.ExtractFeatures(input);

        // Apply PANet for feature fusion
        var neckFeatures = Neck!.Forward(backboneFeatures);

        // Flatten multi-scale features
        var (flattenedFeatures, levelStarts, spatialShapes) = FlattenMultiScale(neckFeatures);

        // Hybrid encoder
        var memory = _encoder.Forward(flattenedFeatures, spatialShapes, levelStarts);

        // Efficient decoder with query selection
        var (classLogits, boxPreds) = _decoder.Forward(memory, spatialShapes);

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

        // RT-DETR is designed to be NMS-free, but apply with high threshold for safety
        var nmsResults = _nms.Apply(candidateDetections, Math.Max(0.9, nmsThreshold));

        if (nmsResults.Count > Options.MaxDetections)
        {
            return nmsResults.Take(Options.MaxDetections).ToList();
        }

        return nmsResults;
    }

    /// <inheritdoc/>
    protected override long GetHeadParameterCount()
    {
        long count = _encoder.GetParameterCount();
        count += _decoder.GetParameterCount();
        return count;
    }

    /// <inheritdoc/>
    public override Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();

        using var stream = File.OpenRead(pathOrUrl);
        using var reader = new BinaryReader(stream);

        // Read and verify magic number and version
        int magic = reader.ReadInt32();
        if (magic != 0x52544452) // "RTDR" in ASCII
        {
            throw new InvalidDataException("Invalid weight file format: incorrect magic number.");
        }

        int version = reader.ReadInt32();
        if (version != 1)
        {
            throw new InvalidDataException($"Unsupported weight file version: {version}.");
        }

        // Read model configuration
        string modelName = reader.ReadString();
        if (!modelName.StartsWith("RT-DETR"))
        {
            throw new InvalidDataException($"Weight file is for {modelName}, not RT-DETR.");
        }

        // Read backbone parameters
        Backbone!.ReadParameters(reader);

        // Read neck parameters
        Neck!.ReadParameters(reader);

        // Read encoder parameters
        _encoder.ReadParameters(reader);

        // Read decoder parameters
        _decoder.ReadParameters(reader);

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Write magic number and version
        writer.Write(0x52544452); // "RTDR" in ASCII
        writer.Write(1); // Version 1

        // Write model configuration
        writer.Write(Name);

        // Write backbone parameters
        Backbone!.WriteParameters(writer);

        // Write neck parameters
        Neck!.WriteParameters(writer);

        // Write encoder parameters
        _encoder.WriteParameters(writer);

        // Write decoder parameters
        _decoder.WriteParameters(writer);
    }

    private (Tensor<T> flattened, int[] levelStarts, int[][] spatialShapes) FlattenMultiScale(List<Tensor<T>> features)
    {
        return DETRHelpers.FlattenMultiScale(features, _hiddenDim);
    }
}

/// <summary>
/// RT-DETR hybrid encoder with intra-scale and cross-scale attention.
/// </summary>
internal class RTDETREncoder<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _numLayers;
    private readonly int _numLevels;
    private readonly List<RTDETREncoderLayer<T>> _intrascaleLayers;
    private readonly CrossScaleModule<T> _crossScaleModule;

    public RTDETREncoder(int hiddenDim, int numHeads, int numLayers, int numLevels)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _hiddenDim = hiddenDim;
        _numHeads = numHeads;
        _numLayers = numLayers;
        _numLevels = numLevels;

        // Intra-scale self-attention layers
        _intrascaleLayers = new List<RTDETREncoderLayer<T>>();
        for (int i = 0; i < numLayers; i++)
        {
            _intrascaleLayers.Add(new RTDETREncoderLayer<T>(hiddenDim, numHeads));
        }

        // Cross-scale feature fusion
        _crossScaleModule = new CrossScaleModule<T>(hiddenDim, numLevels);
    }

    public Tensor<T> Forward(Tensor<T> x, int[][] spatialShapes, int[] levelStarts)
    {
        var output = x;

        // Apply intra-scale attention within each level
        foreach (var layer in _intrascaleLayers)
        {
            output = layer.Forward(output, spatialShapes, levelStarts);
        }

        // Apply cross-scale fusion
        output = _crossScaleModule.Forward(output, spatialShapes, levelStarts);

        return output;
    }

    public long GetParameterCount()
    {
        long count = 0;
        foreach (var layer in _intrascaleLayers)
        {
            count += layer.GetParameterCount();
        }
        count += _crossScaleModule.GetParameterCount();
        return count;
    }

    /// <summary>
    /// Writes all parameters to a binary writer for serialization.
    /// </summary>
    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_numHeads);
        writer.Write(_numLayers);
        writer.Write(_numLevels);

        foreach (var layer in _intrascaleLayers)
        {
            layer.WriteParameters(writer);
        }

        _crossScaleModule.WriteParameters(writer);
    }

    /// <summary>
    /// Reads parameters from a binary reader for deserialization.
    /// </summary>
    public void ReadParameters(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int numLayers = reader.ReadInt32();
        int numLevels = reader.ReadInt32();

        if (hiddenDim != _hiddenDim || numHeads != _numHeads ||
            numLayers != _numLayers || numLevels != _numLevels)
        {
            throw new InvalidOperationException(
                $"RTDETREncoder configuration mismatch: expected hiddenDim={_hiddenDim}, numHeads={_numHeads}, " +
                $"numLayers={_numLayers}, numLevels={_numLevels}.");
        }

        foreach (var layer in _intrascaleLayers)
        {
            layer.ReadParameters(reader);
        }

        _crossScaleModule.ReadParameters(reader);
    }
}

/// <summary>
/// RT-DETR intra-scale encoder layer.
/// </summary>
internal class RTDETREncoderLayer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly MultiHeadSelfAttention<T> _selfAttn;
    private readonly Dense<T> _ffn1;
    private readonly Dense<T> _ffn2;
    private readonly LayerNorm<T> _norm1;
    private readonly LayerNorm<T> _norm2;
    private readonly int _hiddenDim;

    public RTDETREncoderLayer(int hiddenDim, int numHeads)
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
        // Self-attention with residual
        var attnOut = _selfAttn.Forward(x);
        var x1 = AddTensors(x, attnOut);
        x1 = _norm1.Forward(x1);

        // FFN with residual
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

    /// <summary>
    /// Writes all parameters to a binary writer for serialization.
    /// </summary>
    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);

        _selfAttn.WriteParameters(writer);
        _ffn1.WriteParameters(writer);
        _ffn2.WriteParameters(writer);
        _norm1.WriteParameters(writer);
        _norm2.WriteParameters(writer);
    }

    /// <summary>
    /// Reads parameters from a binary reader for deserialization.
    /// </summary>
    public void ReadParameters(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();

        if (hiddenDim != _hiddenDim)
        {
            throw new InvalidOperationException(
                $"RTDETREncoderLayer configuration mismatch: expected hiddenDim={_hiddenDim}, got {hiddenDim}.");
        }

        _selfAttn.ReadParameters(reader);
        _ffn1.ReadParameters(reader);
        _ffn2.ReadParameters(reader);
        _norm1.ReadParameters(reader);
        _norm2.ReadParameters(reader);
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
/// Cross-scale feature fusion module for RT-DETR.
/// </summary>
internal class CrossScaleModule<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _hiddenDim;
    private readonly int _numLevels;
    private readonly List<Dense<T>> _fusionLayers;

    public CrossScaleModule(int hiddenDim, int numLevels)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _hiddenDim = hiddenDim;
        _numLevels = numLevels;

        _fusionLayers = new List<Dense<T>>();
        for (int i = 0; i < numLevels; i++)
        {
            _fusionLayers.Add(new Dense<T>(hiddenDim * numLevels, hiddenDim));
        }
    }

    public Tensor<T> Forward(Tensor<T> x, int[][] spatialShapes, int[] levelStarts)
    {
        int batch = x.Shape[0];
        int totalTokens = x.Shape[1];

        // Compute global representation for each level
        var levelRepresentations = new List<Tensor<T>>();

        for (int level = 0; level < _numLevels; level++)
        {
            int start = levelStarts[level];
            int numTokens = spatialShapes[level][0] * spatialShapes[level][1];

            var levelRep = new Tensor<T>(new[] { batch, _hiddenDim });

            for (int b = 0; b < batch; b++)
            {
                for (int d = 0; d < _hiddenDim; d++)
                {
                    double sum = 0;
                    for (int t = 0; t < numTokens; t++)
                    {
                        sum += _numOps.ToDouble(x[b, start + t, d]);
                    }
                    levelRep[b, d] = _numOps.FromDouble(sum / numTokens);
                }
            }

            levelRepresentations.Add(levelRep);
        }

        // Concatenate level representations
        var concat = new Tensor<T>(new[] { batch, _hiddenDim * _numLevels });
        for (int b = 0; b < batch; b++)
        {
            int offset = 0;
            for (int level = 0; level < _numLevels; level++)
            {
                for (int d = 0; d < _hiddenDim; d++)
                {
                    concat[b, offset + d] = levelRepresentations[level][b, d];
                }
                offset += _hiddenDim;
            }
        }

        // Fuse and add back to each level
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            result[i] = x[i];
        }

        for (int level = 0; level < _numLevels; level++)
        {
            int start = levelStarts[level];
            int numTokens = spatialShapes[level][0] * spatialShapes[level][1];

            for (int b = 0; b < batch; b++)
            {
                var fused = _fusionLayers[level].Forward(ExtractRow(concat, b));

                for (int t = 0; t < numTokens; t++)
                {
                    for (int d = 0; d < _hiddenDim; d++)
                    {
                        result[b, start + t, d] = _numOps.Add(result[b, start + t, d], fused[0, d]);
                    }
                }
            }
        }

        return result;
    }

    public long GetParameterCount()
    {
        long count = 0;
        foreach (var layer in _fusionLayers)
        {
            count += layer.GetParameterCount();
        }
        return count;
    }

    /// <summary>
    /// Writes all parameters to a binary writer for serialization.
    /// </summary>
    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_numLevels);

        foreach (var layer in _fusionLayers)
        {
            layer.WriteParameters(writer);
        }
    }

    /// <summary>
    /// Reads parameters from a binary reader for deserialization.
    /// </summary>
    public void ReadParameters(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numLevels = reader.ReadInt32();

        if (hiddenDim != _hiddenDim || numLevels != _numLevels)
        {
            throw new InvalidOperationException(
                $"CrossScaleModule configuration mismatch: expected hiddenDim={_hiddenDim}, numLevels={_numLevels}.");
        }

        foreach (var layer in _fusionLayers)
        {
            layer.ReadParameters(reader);
        }
    }

    private Tensor<T> ExtractRow(Tensor<T> x, int row)
    {
        int cols = x.Shape[1];
        var result = new Tensor<T>(new[] { 1, cols });
        for (int c = 0; c < cols; c++)
        {
            result[0, c] = x[row, c];
        }
        return result;
    }
}

/// <summary>
/// RT-DETR decoder with uncertainty-minimal query selection.
/// </summary>
internal class RTDETRDecoder<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _numLayers;
    private readonly int _numQueries;
    private readonly List<DecoderLayer<T>> _layers;
    private readonly Tensor<T> _queryEmbed;
    private readonly Dense<T> _classHead;
    private readonly Dense<T> _boxHead;

    public RTDETRDecoder(int hiddenDim, int numHeads, int numLayers, int numQueries, int numClasses)
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

        _queryEmbed = InitializeQueries(numQueries, hiddenDim);
        _classHead = new Dense<T>(hiddenDim, numClasses + 1);
        _boxHead = new Dense<T>(hiddenDim, 4);
    }

    public (Tensor<T> classLogits, Tensor<T> boxPreds) Forward(Tensor<T> memory, int[][] spatialShapes)
    {
        int batch = memory.Shape[0];

        // Select top-K queries based on uncertainty-minimal principle
        var queries = SelectQueries(memory, batch);

        // Pass through decoder layers
        var output = queries;
        foreach (var layer in _layers)
        {
            output = layer.Forward(output, memory, null);
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
        var results = new List<(float[] boxes, float[] scores, int[] classIds)>();

        int batch = classLogits.Shape[0];
        int numQueries = classLogits.Shape[1];
        int numClasses = classLogits.Shape[2];

        for (int b = 0; b < batch; b++)
        {
            // Separate lists per batch item to maintain batch separation
            var batchBoxes = new List<float>();
            var batchScores = new List<float>();
            var batchClassIds = new List<int>();

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

                batchBoxes.AddRange(new[] { x1, y1, x2, y2 });
                batchScores.Add((float)maxScore);
                batchClassIds.Add(maxClassId);
            }

            // Add results for this batch item
            results.Add((batchBoxes.ToArray(), batchScores.ToArray(), batchClassIds.ToArray()));
        }

        return results;
    }

    public long GetParameterCount()
    {
        long count = _numQueries * _hiddenDim;

        foreach (var layer in _layers)
        {
            count += layer.GetParameterCount();
        }

        count += _classHead.GetParameterCount();
        count += _boxHead.GetParameterCount();

        return count;
    }

    /// <summary>
    /// Writes all parameters to a binary writer for serialization.
    /// </summary>
    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_numHeads);
        writer.Write(_numLayers);
        writer.Write(_numQueries);

        // Write query embeddings
        for (int i = 0; i < _queryEmbed.Length; i++)
        {
            writer.Write(_numOps.ToDouble(_queryEmbed[i]));
        }

        // Write decoder layers
        foreach (var layer in _layers)
        {
            layer.WriteParameters(writer);
        }

        // Write prediction heads
        _classHead.WriteParameters(writer);
        _boxHead.WriteParameters(writer);
    }

    /// <summary>
    /// Reads parameters from a binary reader for deserialization.
    /// </summary>
    public void ReadParameters(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        int numHeads = reader.ReadInt32();
        int numLayers = reader.ReadInt32();
        int numQueries = reader.ReadInt32();

        if (hiddenDim != _hiddenDim || numHeads != _numHeads ||
            numLayers != _numLayers || numQueries != _numQueries)
        {
            throw new InvalidOperationException(
                $"RTDETRDecoder configuration mismatch: expected hiddenDim={_hiddenDim}, numHeads={_numHeads}, " +
                $"numLayers={_numLayers}, numQueries={_numQueries}.");
        }

        // Read query embeddings
        for (int i = 0; i < _queryEmbed.Length; i++)
        {
            _queryEmbed[i] = _numOps.FromDouble(reader.ReadDouble());
        }

        // Read decoder layers
        foreach (var layer in _layers)
        {
            layer.ReadParameters(reader);
        }

        // Read prediction heads
        _classHead.ReadParameters(reader);
        _boxHead.ReadParameters(reader);
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

    private Tensor<T> SelectQueries(Tensor<T> memory, int batch)
    {
        // TODO: Implement uncertainty-minimal query selection as described in RT-DETR paper.
        // Current simplified implementation uses fixed learnable queries.
        // Full implementation should compute uncertainty scores from encoder output
        // and select top-K positions with minimal uncertainty.
        var queries = new Tensor<T>(new[] { batch, _numQueries, _hiddenDim });

        for (int b = 0; b < batch; b++)
        {
            for (int q = 0; q < _numQueries; q++)
            {
                for (int d = 0; d < _hiddenDim; d++)
                {
                    queries[b, q, d] = _queryEmbed[q, d];
                }
            }
        }

        return queries;
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
