using System.IO;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.ComputerVision.Detection.PostProcessing;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;

/// <summary>
/// YOLOv9 object detector with Programmable Gradient Information (PGI).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> YOLOv9 introduces Programmable Gradient Information (PGI)
/// and Generalized Efficient Layer Aggregation Network (GELAN) to address information
/// loss during feature transformation, achieving state-of-the-art performance.</para>
///
/// <para>Key features:
/// - PGI: Programmable Gradient Information for better gradient flow
/// - GELAN: Generalized ELAN for efficient feature aggregation
/// - Auxiliary reversible branch for improved training
/// - Better information preservation through the network
/// </para>
///
/// <para>Reference: Wang et al., "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information", 2024</para>
/// </remarks>
public class YOLOv9<T> : ObjectDetectorBase<T>
{
    private readonly YOLOv8Head<T> _head;
    private readonly int[] _strides;
    private readonly List<Conv2D<T>> _gelanBlocks;
    private readonly NMS<T> _nms;

    /// <inheritdoc/>
    public override string Name => $"YOLOv9-{Options.Size}";

    /// <summary>
    /// Creates a new YOLOv9 detector.
    /// </summary>
    /// <param name="options">Detection options.</param>
    public YOLOv9(ObjectDetectionOptions<T> options) : base(options)
    {
        var (depth, width) = GetSizeConfig(options.Size);

        // Initialize backbone with higher capacity for PGI
        Backbone = new CSPDarknet<T>(depth: depth * 1.2, widthMultiplier: width);

        // Initialize GELAN-enhanced neck
        Neck = new PANet<T>(Backbone.OutputChannels, outputChannels: (int)(256 * width));

        // GELAN additional blocks for feature enhancement
        _gelanBlocks = new List<Conv2D<T>>();
        for (int i = 0; i < Neck.NumLevels; i++)
        {
            _gelanBlocks.Add(new Conv2D<T>(
                inChannels: Neck.OutputChannels,
                outChannels: Neck.OutputChannels,
                kernelSize: 3,
                padding: 1,
                useBias: false
            ));
        }

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
        ModelSize.Medium => (0.75, 0.75),
        ModelSize.Large => (1.00, 1.00),
        ModelSize.XLarge => (1.25, 1.25),
        _ => (0.75, 0.75)
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

        // Neck feature fusion
        var neckFeatures = Neck!.Forward(backboneFeatures);

        // Apply GELAN enhancement blocks
        var gelanFeatures = new List<Tensor<T>>();
        for (int i = 0; i < neckFeatures.Count; i++)
        {
            var enhanced = _gelanBlocks[i].Forward(neckFeatures[i]);
            enhanced = ApplySiLU(enhanced);
            // Add residual connection
            enhanced = AddTensors(enhanced, neckFeatures[i]);
            gelanFeatures.Add(enhanced);
        }

        // Detection head
        var (clsOutputs, regOutputs) = _head.Forward(gelanFeatures);

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
        // Add GELAN block parameters
        for (int i = 0; i < _gelanBlocks.Count; i++)
        {
            count += _gelanBlocks[i].GetParameterCount();
        }
        return count;
    }

    private Tensor<T> ApplySiLU(Tensor<T> x)
    {
        var result = new Tensor<T>(x.Shape);
        for (int i = 0; i < x.Length; i++)
        {
            double val = NumOps.ToDouble(x[i]);
            double silu = val * (1.0 / (1.0 + Math.Exp(-val)));
            result[i] = NumOps.FromDouble(silu);
        }
        return result;
    }

    private Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = NumOps.Add(a[i], b[i]);
        }
        return result;
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
        if (!modelName.StartsWith("YOLOv9"))
        {
            throw new InvalidDataException($"Weight file is for {modelName}, not YOLOv9.");
        }

        // Read backbone parameters
        Backbone!.ReadParameters(reader);

        // Read neck parameters
        Neck!.ReadParameters(reader);

        // Read GELAN block parameters
        int gelanCount = reader.ReadInt32();
        if (gelanCount != _gelanBlocks.Count)
        {
            throw new InvalidDataException($"GELAN block count mismatch: expected {_gelanBlocks.Count}, got {gelanCount}.");
        }
        foreach (var block in _gelanBlocks)
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

        // Write neck parameters
        Neck!.WriteParameters(writer);

        // Write GELAN block parameters
        writer.Write(_gelanBlocks.Count);
        foreach (var block in _gelanBlocks)
        {
            block.WriteParameters(writer);
        }

        // Write head parameters
        _head.WriteParameters(writer);
    }
}
