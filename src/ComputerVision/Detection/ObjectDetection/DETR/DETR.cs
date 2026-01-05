using System.IO;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;

/// <summary>
/// DETR (DEtection TRansformer) - End-to-end object detection with transformers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> DETR is a revolutionary approach to object detection that uses
/// a transformer architecture instead of traditional anchor-based methods. It treats detection
/// as a set prediction problem, eliminating the need for NMS post-processing.</para>
///
/// <para>Key features:
/// - No anchors needed
/// - No NMS required (uses Hungarian matching for training)
/// - Global reasoning via self-attention
/// - Simple end-to-end architecture
/// </para>
///
/// <para>Reference: Carion et al., "End-to-End Object Detection with Transformers", ECCV 2020</para>
/// </remarks>
public class DETR<T> : ObjectDetectorBase<T>
{
    private readonly DETREncoder<T> _encoder;
    private readonly DETRDecoder<T> _decoder;
    private readonly Conv2D<T> _inputProj;
    private readonly int _hiddenDim;
    private readonly NMS<T> _nms;

    /// <inheritdoc/>
    public override string Name => $"DETR-{Options.Size}";

    /// <summary>
    /// Creates a new DETR detector.
    /// </summary>
    /// <param name="options">Detection options.</param>
    public DETR(ObjectDetectionOptions<T> options) : base(options)
    {
        var (hiddenDim, numHeads, numEncoderLayers, numDecoderLayers, numQueries) = GetSizeConfig(options.Size);
        _hiddenDim = hiddenDim;

        // Initialize backbone (ResNet-50 by default)
        Backbone = new ResNet<T>(ResNetVariant.ResNet50);

        // Project backbone features to hidden dimension
        int backboneChannels = Backbone.OutputChannels[^1];
        _inputProj = new Conv2D<T>(backboneChannels, hiddenDim, kernelSize: 1);

        // Transformer encoder
        _encoder = new DETREncoder<T>(hiddenDim, numHeads, numEncoderLayers);

        // Transformer decoder
        _decoder = new DETRDecoder<T>(hiddenDim, numHeads, numDecoderLayers, numQueries, options.NumClasses);

        _nms = new NMS<T>();
    }

    /// <summary>
    /// Gets configuration based on model size.
    /// </summary>
    private static (int hiddenDim, int numHeads, int numEncoderLayers, int numDecoderLayers, int numQueries) GetSizeConfig(ModelSize size) => size switch
    {
        ModelSize.Nano => (128, 4, 3, 3, 50),
        ModelSize.Small => (192, 6, 4, 4, 100),
        ModelSize.Medium => (256, 8, 6, 6, 100),
        ModelSize.Large => (384, 8, 6, 6, 300),
        ModelSize.XLarge => (512, 8, 6, 6, 500),
        _ => (256, 8, 6, 6, 100)
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
        // Extract backbone features (use the last feature map)
        var backboneFeatures = Backbone!.ExtractFeatures(input);
        var features = backboneFeatures[^1]; // C5 features

        // Project to hidden dimension
        var projected = _inputProj.Forward(features);

        // Flatten spatial dimensions for transformer: [B, C, H, W] -> [B, H*W, C]
        var flattened = FlattenForTransformer(projected);

        // Generate positional encoding
        var posEncoding = GeneratePositionalEncoding(flattened.Shape);

        // Encode features
        var memory = _encoder.Forward(flattened, posEncoding);

        // Decode to get predictions
        var (classLogits, boxPreds) = _decoder.Forward(memory, posEncoding);

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

        // Validate single-image input (batch processing handled by ObjectDetectorBase.DetectBatch)
        int batchSize = classLogits.Shape[0];
        if (batchSize != 1)
        {
            throw new ArgumentException(
                $"PostProcess expects single-image input (batch size 1), got batch size {batchSize}. " +
                "Use DetectBatch for multi-image processing.");
        }

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

        // Note: DETR is designed to not need NMS, but we apply it for safety
        // with a very high IoU threshold
        var nmsResults = _nms.Apply(candidateDetections, Math.Max(0.9, nmsThreshold));

        // Limit to max detections
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
    public override async Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        byte[] data;
        if (pathOrUrl.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
            pathOrUrl.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
        {
            using var client = new System.Net.Http.HttpClient();
            data = await client.GetByteArrayWithCancellationAsync(pathOrUrl, cancellationToken);
        }
        else
        {
            data = await Task.Run(() => File.ReadAllBytes(pathOrUrl), cancellationToken);
        }

        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream);

        // Read and verify magic number and version
        int magic = reader.ReadInt32();
        if (magic != 0x44455452) // "DETR" in ASCII
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
        if (!modelName.StartsWith("DETR"))
        {
            throw new InvalidDataException($"Weight file is for {modelName}, not DETR.");
        }

        // Read backbone parameters
        if (Backbone is not null)
        {
            Backbone.ReadParameters(reader);
        }

        // Read input projection
        _inputProj.ReadParameters(reader);

        // Read encoder parameters
        _encoder.ReadParameters(reader);

        // Read decoder parameters
        _decoder.ReadParameters(reader);
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Write magic number and version
        writer.Write(0x44455452); // "DETR" in ASCII
        writer.Write(1); // Version 1

        // Write model configuration
        writer.Write(Name);

        // Write backbone parameters
        Backbone!.WriteParameters(writer);

        // Write input projection
        _inputProj.WriteParameters(writer);

        // Write encoder parameters
        _encoder.WriteParameters(writer);

        // Write decoder parameters
        _decoder.WriteParameters(writer);
    }

    private Tensor<T> FlattenForTransformer(Tensor<T> x)
    {
        int batch = x.Shape[0];
        int channels = x.Shape[1];
        int height = x.Shape[2];
        int width = x.Shape[3];
        int seqLen = height * width;

        var result = new Tensor<T>(new[] { batch, seqLen, channels });

        for (int b = 0; b < batch; b++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    int seqIdx = h * width + w;
                    for (int c = 0; c < channels; c++)
                    {
                        result[b, seqIdx, c] = x[b, c, h, w];
                    }
                }
            }
        }

        return result;
    }

    private Tensor<T> GeneratePositionalEncoding(int[] shape)
    {
        int batch = shape[0];
        int seqLen = shape[1];
        int hiddenDim = shape[2];

        var encoding = new Tensor<T>(shape);

        for (int b = 0; b < batch; b++)
        {
            for (int pos = 0; pos < seqLen; pos++)
            {
                for (int i = 0; i < hiddenDim; i++)
                {
                    double angle = pos / Math.Pow(10000.0, (2.0 * (i / 2)) / hiddenDim);
                    double value = (i % 2 == 0) ? Math.Sin(angle) : Math.Cos(angle);
                    encoding[b, pos, i] = NumOps.FromDouble(value);
                }
            }
        }

        return encoding;
    }
}

/// <summary>
/// Transformer encoder for DETR.
/// </summary>
internal class DETREncoder<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _hiddenDim;
    private readonly int _numHeads;
    private readonly int _numLayers;
    private readonly List<EncoderLayer<T>> _layers;

    /// <summary>
    /// Gets the encoder layers for weight loading.
    /// </summary>
    public IReadOnlyList<EncoderLayer<T>> Layers => _layers;

    public DETREncoder(int hiddenDim, int numHeads, int numLayers)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _hiddenDim = hiddenDim;
        _numHeads = numHeads;
        _numLayers = numLayers;

        _layers = new List<EncoderLayer<T>>();
        for (int i = 0; i < numLayers; i++)
        {
            _layers.Add(new EncoderLayer<T>(hiddenDim, numHeads));
        }
    }

    public Tensor<T> Forward(Tensor<T> x, Tensor<T> posEncoding)
    {
        // Create a copy of input to avoid mutating the original tensor
        var output = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            output[i] = x[i];
        }

        // Add positional encoding
        for (int i = 0; i < x.Length; i++)
        {
            output[i] = _numOps.Add(output[i], posEncoding[i]);
        }

        foreach (var layer in _layers)
        {
            output = layer.Forward(output);
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

    /// <summary>
    /// Writes all parameters to a binary writer for serialization.
    /// </summary>
    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_numHeads);
        writer.Write(_numLayers);

        foreach (var layer in _layers)
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
        int numHeads = reader.ReadInt32();
        int numLayers = reader.ReadInt32();

        if (hiddenDim != _hiddenDim || numHeads != _numHeads || numLayers != _numLayers)
        {
            throw new InvalidOperationException(
                $"DETREncoder configuration mismatch: expected hiddenDim={_hiddenDim}, numHeads={_numHeads}, numLayers={_numLayers}.");
        }

        foreach (var layer in _layers)
        {
            layer.ReadParameters(reader);
        }
    }
}

/// <summary>
/// Single encoder layer in DETR.
/// </summary>
internal class EncoderLayer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly MultiHeadSelfAttention<T> _selfAttn;
    private readonly Dense<T> _ffn1;
    private readonly Dense<T> _ffn2;
    private readonly LayerNorm<T> _norm1;
    private readonly LayerNorm<T> _norm2;
    private readonly int _hiddenDim;

    // Public properties for weight loading
    public MultiHeadSelfAttention<T> SelfAttn => _selfAttn;
    public Dense<T> FFN1 => _ffn1;
    public Dense<T> FFN2 => _ffn2;
    public LayerNorm<T> Norm1 => _norm1;
    public LayerNorm<T> Norm2 => _norm2;

    public EncoderLayer(int hiddenDim, int numHeads)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _hiddenDim = hiddenDim;

        _selfAttn = new MultiHeadSelfAttention<T>(hiddenDim, numHeads);
        _ffn1 = new Dense<T>(hiddenDim, hiddenDim * 4);
        _ffn2 = new Dense<T>(hiddenDim * 4, hiddenDim);
        _norm1 = new LayerNorm<T>(hiddenDim);
        _norm2 = new LayerNorm<T>(hiddenDim);
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        // Self-attention with residual and layer norm
        var attnOut = _selfAttn.Forward(x);
        var x1 = AddTensors(x, attnOut);
        x1 = _norm1.Forward(x1);

        // FFN with residual and layer norm
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
                $"EncoderLayer configuration mismatch: expected hiddenDim={_hiddenDim}, got {hiddenDim}.");
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

                // FFN1 with GELU
                var h = _ffn1.Forward(feat);
                for (int d = 0; d < ffnDim; d++)
                {
                    double val = _numOps.ToDouble(h[0, d]);
                    h[0, d] = _numOps.FromDouble(GELU(val));
                }

                // FFN2
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
/// Layer normalization with learnable affine parameters (gamma and beta).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
internal class LayerNorm<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _hiddenDim;
    private readonly Tensor<T> _gamma; // Scale parameter
    private readonly Tensor<T> _beta;  // Shift parameter
    private readonly double _eps;

    /// <summary>
    /// Gets the gamma (scale) parameter for weight loading.
    /// </summary>
    public Tensor<T> Gamma => _gamma;

    /// <summary>
    /// Gets the beta (shift) parameter for weight loading.
    /// </summary>
    public Tensor<T> Beta => _beta;

    public LayerNorm(int hiddenDim, double eps = 1e-6)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _hiddenDim = hiddenDim;
        _eps = eps;

        // Initialize gamma to 1 and beta to 0 (standard initialization)
        _gamma = new Tensor<T>(new[] { hiddenDim });
        _beta = new Tensor<T>(new[] { hiddenDim });

        for (int i = 0; i < hiddenDim; i++)
        {
            _gamma[i] = _numOps.FromDouble(1.0);
            _beta[i] = _numOps.FromDouble(0.0);
        }
    }

    public Tensor<T> Forward(Tensor<T> x)
    {
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int hiddenDim = x.Shape[2];

        var result = new Tensor<T>(x.Shape);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                // Compute mean
                double mean = 0;
                for (int d = 0; d < hiddenDim; d++)
                {
                    mean += _numOps.ToDouble(x[b, s, d]);
                }
                mean /= hiddenDim;

                // Compute variance
                double variance = 0;
                for (int d = 0; d < hiddenDim; d++)
                {
                    double diff = _numOps.ToDouble(x[b, s, d]) - mean;
                    variance += diff * diff;
                }
                variance /= hiddenDim;

                // Normalize and apply affine transformation: gamma * (x - mean) / std + beta
                double std = Math.Sqrt(variance + _eps);
                for (int d = 0; d < hiddenDim; d++)
                {
                    double normalized = (_numOps.ToDouble(x[b, s, d]) - mean) / std;
                    double gamma = _numOps.ToDouble(_gamma[d]);
                    double beta = _numOps.ToDouble(_beta[d]);
                    result[b, s, d] = _numOps.FromDouble(gamma * normalized + beta);
                }
            }
        }

        return result;
    }

    public long GetParameterCount()
    {
        return 2 * _hiddenDim; // gamma + beta
    }

    /// <summary>
    /// Writes all parameters to a binary writer for serialization.
    /// </summary>
    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_hiddenDim);
        writer.Write(_eps);

        for (int i = 0; i < _hiddenDim; i++)
        {
            writer.Write(_numOps.ToDouble(_gamma[i]));
        }

        for (int i = 0; i < _hiddenDim; i++)
        {
            writer.Write(_numOps.ToDouble(_beta[i]));
        }
    }

    /// <summary>
    /// Reads parameters from a binary reader for deserialization.
    /// </summary>
    public void ReadParameters(BinaryReader reader)
    {
        int hiddenDim = reader.ReadInt32();
        double eps = reader.ReadDouble();

        if (hiddenDim != _hiddenDim)
        {
            throw new InvalidOperationException(
                $"LayerNorm configuration mismatch: expected hiddenDim={_hiddenDim}, got {hiddenDim}.");
        }

        // Validate eps matches within a small tolerance for floating point comparison
        const double tolerance = 1e-12;
        if (Math.Abs(eps - _eps) > tolerance)
        {
            throw new InvalidOperationException(
                $"LayerNorm configuration mismatch: expected eps={_eps}, got {eps}.");
        }

        for (int i = 0; i < _hiddenDim; i++)
        {
            _gamma[i] = _numOps.FromDouble(reader.ReadDouble());
        }

        for (int i = 0; i < _hiddenDim; i++)
        {
            _beta[i] = _numOps.FromDouble(reader.ReadDouble());
        }
    }
}

