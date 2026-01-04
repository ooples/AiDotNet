using System.IO;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;

/// <summary>
/// YOLOv11 object detector with enhanced feature extraction and attention mechanisms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> YOLOv11 is the latest YOLO version with improved
/// feature extraction using attention mechanisms and more efficient architecture.
/// It builds upon YOLOv8-v10 innovations while adding new enhancements.</para>
///
/// <para>Key features:
/// - C3k2 blocks with attention for enhanced feature extraction
/// - Spatial Pyramid Pooling Fast (SPPF) with larger kernel
/// - Multi-head self-attention in neck
/// - Improved small object detection
/// </para>
///
/// <para>Reference: Ultralytics, "YOLOv11" 2024</para>
/// </remarks>
public class YOLOv11<T> : ObjectDetectorBase<T>
{
    private readonly YOLOv8Head<T> _head;
    private readonly int[] _strides;
    private readonly List<AttentionBlock<T>> _attentionBlocks;
    private readonly SPPFBlock<T> _sppf;
    private readonly NMS<T> _nms;

    /// <inheritdoc/>
    public override string Name => $"YOLOv11-{Options.Size}";

    /// <summary>
    /// Creates a new YOLOv11 detector.
    /// </summary>
    /// <param name="options">Detection options.</param>
    public YOLOv11(ObjectDetectionOptions<T> options) : base(options)
    {
        var (depth, width) = GetSizeConfig(options.Size);

        // Initialize backbone with enhanced C3k2 blocks
        Backbone = new CSPDarknet<T>(depth: depth * 1.1, widthMultiplier: width);

        // Initialize neck
        Neck = new PANet<T>(Backbone.OutputChannels, outputChannels: (int)(256 * width));

        // Add attention blocks to neck features
        _attentionBlocks = new List<AttentionBlock<T>>();
        for (int i = 0; i < Neck.NumLevels; i++)
        {
            _attentionBlocks.Add(new AttentionBlock<T>(Neck.OutputChannels));
        }

        // SPPF block for the deepest feature level
        _sppf = new SPPFBlock<T>(Backbone.OutputChannels[^1], kernelSize: 5);

        // Initialize detection head
        var neckChannels = Enumerable.Repeat(Neck.OutputChannels, Neck.NumLevels).ToArray();
        _head = new YOLOv8Head<T>(neckChannels, options.NumClasses);

        _strides = Backbone.Strides;
        _nms = new NMS<T>();
    }

    private static (double depth, double width) GetSizeConfig(ModelSize size) => size switch
    {
        ModelSize.Nano => (0.33, 0.25),
        ModelSize.Small => (0.50, 0.50),
        ModelSize.Medium => (0.67, 0.75),
        ModelSize.Large => (1.00, 1.00),
        ModelSize.XLarge => (1.33, 1.25),
        _ => (0.67, 0.75)
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
        // Backbone feature extraction
        var backboneFeatures = Backbone!.ExtractFeatures(input);

        // Apply SPPF to deepest feature level
        var enhancedFeatures = new List<Tensor<T>>(backboneFeatures);
        enhancedFeatures[^1] = _sppf.Forward(enhancedFeatures[^1]);

        // Neck feature fusion
        var neckFeatures = Neck!.Forward(enhancedFeatures);

        // Apply attention to each feature level
        var attentionFeatures = new List<Tensor<T>>();
        for (int i = 0; i < neckFeatures.Count; i++)
        {
            var attended = _attentionBlocks[i].Forward(neckFeatures[i]);
            attentionFeatures.Add(attended);
        }

        // Detection head
        var (clsOutputs, regOutputs) = _head.Forward(attentionFeatures);

        var outputs = new List<Tensor<T>>();
        outputs.AddRange(clsOutputs);
        outputs.AddRange(regOutputs);

        return outputs;
    }

    /// <inheritdoc/>
    protected override List<Detection<T>> PostProcess(
        List<Tensor<T>> outputs,
        int imageWidth,
        int imageHeight,
        double confidenceThreshold,
        double nmsThreshold)
    {
        int numLevels = outputs.Count / 2;
        var clsOutputs = outputs.Take(numLevels).ToList();
        var regOutputs = outputs.Skip(numLevels).ToList();

        var decoded = _head.DecodeOutputs(clsOutputs, regOutputs, _strides, imageHeight, imageWidth);

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

        // Apply NMS
        var nmsResults = _nms.Apply(candidateDetections, nmsThreshold);

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
        long count = _head.GetParameterCount();
        count += _sppf.GetParameterCount();
        foreach (var attn in _attentionBlocks)
        {
            count += attn.GetParameterCount();
        }
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
        if (magic != 0x594F4C4F) // "YOLO" in ASCII
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
        if (!modelName.StartsWith("YOLOv11"))
        {
            throw new InvalidDataException($"Weight file is for {modelName}, not YOLOv11.");
        }

        // Read backbone parameters
        Backbone!.ReadParameters(reader);

        // Read SPPF parameters
        _sppf.ReadParameters(reader);

        // Read neck parameters
        Neck!.ReadParameters(reader);

        // Read attention block parameters
        int attnCount = reader.ReadInt32();
        if (attnCount != _attentionBlocks.Count)
        {
            throw new InvalidDataException($"Attention block count mismatch: expected {_attentionBlocks.Count}, got {attnCount}.");
        }
        foreach (var block in _attentionBlocks)
        {
            block.ReadParameters(reader);
        }

        // Read head parameters
        _head.ReadParameters(reader);

        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Write magic number and version for identification
        writer.Write(0x594F4C4F); // "YOLO" in ASCII
        writer.Write(1); // Version 1

        // Write model configuration
        writer.Write(Name);

        // Write backbone parameters
        Backbone!.WriteParameters(writer);

        // Write SPPF parameters
        _sppf.WriteParameters(writer);

        // Write neck parameters
        Neck!.WriteParameters(writer);

        // Write attention block parameters
        writer.Write(_attentionBlocks.Count);
        foreach (var block in _attentionBlocks)
        {
            block.WriteParameters(writer);
        }

        // Write head parameters
        _head.WriteParameters(writer);
    }
}

/// <summary>
/// Spatial Pyramid Pooling Fast (SPPF) block.
/// </summary>
internal class SPPFBlock<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Conv2D<T> _conv1;
    private readonly Conv2D<T> _conv2;
    private readonly int _kernelSize;
    private readonly int _channels;

    public SPPFBlock(int channels, int kernelSize = 5)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _kernelSize = kernelSize;
        _channels = channels;

        int hiddenChannels = channels / 2;
        _conv1 = new Conv2D<T>(channels, hiddenChannels, kernelSize: 1);
        _conv2 = new Conv2D<T>(hiddenChannels * 4, channels, kernelSize: 1);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        // First conv
        var x = _conv1.Forward(input);
        x = ApplySiLU(x);

        // Three consecutive max pools
        var pool1 = MaxPool(x, _kernelSize);
        var pool2 = MaxPool(pool1, _kernelSize);
        var pool3 = MaxPool(pool2, _kernelSize);

        // Concatenate
        var concat = ConcatenateChannels(x, pool1, pool2, pool3);

        // Final conv
        var output = _conv2.Forward(concat);
        output = ApplySiLU(output);

        return output;
    }

    public long GetParameterCount()
    {
        int hiddenChannels = _channels / 2;
        return _channels * hiddenChannels + hiddenChannels + // conv1
               hiddenChannels * 4 * _channels + _channels;    // conv2
    }

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_channels);
        writer.Write(_kernelSize);
        _conv1.WriteParameters(writer);
        _conv2.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        int channels = reader.ReadInt32();
        int kernelSize = reader.ReadInt32();

        if (channels != _channels)
        {
            throw new InvalidDataException($"SPPFBlock channels mismatch: expected {_channels}, got {channels}.");
        }

        if (kernelSize != _kernelSize)
        {
            throw new InvalidDataException($"SPPFBlock kernelSize mismatch: expected {_kernelSize}, got {kernelSize}.");
        }

        _conv1.ReadParameters(reader);
        _conv2.ReadParameters(reader);
    }

    private Tensor<T> MaxPool(Tensor<T> x, int kernelSize)
    {
        int padding = kernelSize / 2;
        int batch = x.Shape[0];
        int channels = x.Shape[1];
        int height = x.Shape[2];
        int width = x.Shape[3];

        var output = new Tensor<T>(x.Shape);

        for (int n = 0; n < batch; n++)
        {
            for (int c = 0; c < channels; c++)
            {
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        double maxVal = double.NegativeInfinity;
                        for (int kh = 0; kh < kernelSize; kh++)
                        {
                            for (int kw = 0; kw < kernelSize; kw++)
                            {
                                int ih = h - padding + kh;
                                int iw = w - padding + kw;
                                if (ih >= 0 && ih < height && iw >= 0 && iw < width)
                                {
                                    double val = _numOps.ToDouble(x[n, c, ih, iw]);
                                    maxVal = Math.Max(maxVal, val);
                                }
                            }
                        }
                        output[n, c, h, w] = _numOps.FromDouble(maxVal == double.NegativeInfinity ? 0 : maxVal);
                    }
                }
            }
        }

        return output;
    }

    private Tensor<T> ConcatenateChannels(params Tensor<T>[] tensors)
    {
        int batch = tensors[0].Shape[0];
        int height = tensors[0].Shape[2];
        int width = tensors[0].Shape[3];
        int totalChannels = tensors.Sum(t => t.Shape[1]);

        var result = new Tensor<T>(new[] { batch, totalChannels, height, width });

        int channelOffset = 0;
        foreach (var tensor in tensors)
        {
            int channels = tensor.Shape[1];
            for (int n = 0; n < batch; n++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            result[n, channelOffset + c, h, w] = tensor[n, c, h, w];
                        }
                    }
                }
            }
            channelOffset += channels;
        }

        return result;
    }

    private Tensor<T> ApplySiLU(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = _numOps.ToDouble(x[i]);
            double silu = val * (1.0 / (1.0 + Math.Exp(-val)));
            result[i] = _numOps.FromDouble(silu);
        }
        return result;
    }
}

/// <summary>
/// Lightweight attention block for feature enhancement.
/// </summary>
internal class AttentionBlock<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Conv2D<T> _query;
    private readonly Conv2D<T> _key;
    private readonly Conv2D<T> _value;
    private readonly Conv2D<T> _proj;
    private readonly int _channels;
    private readonly double _scale;

    public AttentionBlock(int channels)
    {
        _numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        _channels = channels;
        _scale = 1.0 / Math.Sqrt(channels);

        _query = new Conv2D<T>(channels, channels, kernelSize: 1);
        _key = new Conv2D<T>(channels, channels, kernelSize: 1);
        _value = new Conv2D<T>(channels, channels, kernelSize: 1);
        _proj = new Conv2D<T>(channels, channels, kernelSize: 1);
    }

    public Tensor<T> Forward(Tensor<T> input)
    {
        int batch = input.Shape[0];
        int channels = input.Shape[1];
        int height = input.Shape[2];
        int width = input.Shape[3];
        int spatialSize = height * width;

        // Compute Q, K, V
        var q = _query.Forward(input);
        var k = _key.Forward(input);
        var v = _value.Forward(input);

        // Reshape and compute attention
        // For simplicity, compute spatial attention per batch
        var output = new Tensor<T>(input.Shape);

        for (int n = 0; n < batch; n++)
        {
            // Global average for channel attention (simplified)
            var channelWeights = new double[channels];
            double sumWeights = 0;

            for (int c = 0; c < channels; c++)
            {
                double qSum = 0, kSum = 0;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        qSum += _numOps.ToDouble(q[n, c, h, w]);
                        kSum += _numOps.ToDouble(k[n, c, h, w]);
                    }
                }
                double attn = Math.Exp(qSum * kSum * _scale / spatialSize);
                channelWeights[c] = attn;
                sumWeights += attn;
            }

            // Normalize and apply
            for (int c = 0; c < channels; c++)
            {
                double weight = channelWeights[c] / sumWeights;
                for (int h = 0; h < height; h++)
                {
                    for (int w = 0; w < width; w++)
                    {
                        double vVal = _numOps.ToDouble(v[n, c, h, w]);
                        output[n, c, h, w] = _numOps.FromDouble(vVal * weight);
                    }
                }
            }
        }

        // Project and add residual
        var projected = _proj.Forward(output);

        for (int i = 0; i < projected.Length; i++)
        {
            projected[i] = _numOps.Add(projected[i], input[i]);
        }

        return projected;
    }

    public long GetParameterCount()
    {
        return 4 * _channels * _channels + 4 * _channels; // 4 conv1x1 with bias
    }

    public void WriteParameters(BinaryWriter writer)
    {
        writer.Write(_channels);
        _query.WriteParameters(writer);
        _key.WriteParameters(writer);
        _value.WriteParameters(writer);
        _proj.WriteParameters(writer);
    }

    public void ReadParameters(BinaryReader reader)
    {
        int channels = reader.ReadInt32();

        if (channels != _channels)
        {
            throw new InvalidDataException($"AttentionBlock channels mismatch: expected {_channels}, got {channels}.");
        }

        _query.ReadParameters(reader);
        _key.ReadParameters(reader);
        _value.ReadParameters(reader);
        _proj.ReadParameters(reader);
    }
}
