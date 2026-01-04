using System.IO;
using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.Enums;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;

/// <summary>
/// SOLOv2 (Segmenting Objects by Locations v2) for instance segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> SOLOv2 is an anchor-free, box-free instance segmentation
/// method. It directly predicts instance masks by location, dividing the image into a grid
/// and predicting masks for each cell. This eliminates the need for ROI operations.</para>
///
/// <para>Key features:
/// - Direct mask prediction without boxes
/// - Dynamic convolution for mask generation
/// - Grid-based location encoding
/// - More efficient than two-stage methods
/// </para>
///
/// <para>Reference: Wang et al., "SOLOv2: Dynamic and Fast Instance Segmentation", NeurIPS 2020</para>
/// </remarks>
public class SOLOv2<T> : InstanceSegmenterBase<T>
{
    private readonly ResNet<T> _backbone;
    private readonly FPN<T> _fpn;
    private readonly Conv2D<T> _categoryHead;
    private readonly Conv2D<T> _kernelHead;
    private readonly Conv2D<T> _maskBranch;
    private readonly int[] _gridSizes;
    private readonly int _kernelDim;

    /// <inheritdoc/>
    public override string Name => "SOLOv2";

    /// <summary>
    /// Creates a new SOLOv2 model.
    /// </summary>
    public SOLOv2(InstanceSegmentationOptions<T> options) : base(options)
    {
        _gridSizes = new[] { 40, 36, 24, 16, 12 }; // Multi-scale grids
        _kernelDim = 256; // Dynamic kernel dimension

        // Backbone
        _backbone = new ResNet<T>(ResNetVariant.ResNet50);

        // FPN
        _fpn = new FPN<T>(new[] { 256, 512, 1024, 2048 }, 256);

        // Category prediction head
        _categoryHead = new Conv2D<T>(256, options.NumClasses, kernelSize: 3, padding: 1);

        // Kernel prediction head (predicts dynamic conv weights)
        _kernelHead = new Conv2D<T>(256, _kernelDim, kernelSize: 3, padding: 1);

        // Mask feature branch
        _maskBranch = new Conv2D<T>(256, _kernelDim, kernelSize: 3, padding: 1);
    }

    /// <inheritdoc/>
    public override InstanceSegmentationResult<T> Segment(Tensor<T> image)
    {
        var startTime = DateTime.UtcNow;

        int imageHeight = image.Shape[2];
        int imageWidth = image.Shape[3];

        // Extract backbone features
        var backboneFeatures = _backbone.ExtractFeatures(image);

        // Apply FPN
        var fpnFeatures = _fpn.Forward(backboneFeatures);

        // Generate mask features from P2 (highest resolution)
        var maskFeatures = _maskBranch.Forward(fpnFeatures[0]);
        int maskH = maskFeatures.Shape[2];
        int maskW = maskFeatures.Shape[3];

        var instances = new List<InstanceMask<T>>();

        // Process each FPN level
        for (int levelIdx = 0; levelIdx < fpnFeatures.Count; levelIdx++)
        {
            var levelFeatures = fpnFeatures[levelIdx];
            int gridSize = _gridSizes[Math.Min(levelIdx, _gridSizes.Length - 1)];

            // Predict categories and kernels for this level
            var categoryPreds = _categoryHead.Forward(levelFeatures);
            var kernelPreds = _kernelHead.Forward(levelFeatures);

            int featH = categoryPreds.Shape[2];
            int featW = categoryPreds.Shape[3];

            // Process each grid cell
            for (int y = 0; y < featH; y++)
            {
                for (int x = 0; x < featW; x++)
                {
                    // Find best category
                    int bestClass = 0;
                    double bestScore = 0;

                    for (int c = 0; c < Options.NumClasses; c++)
                    {
                        double score = Sigmoid(NumOps.ToDouble(categoryPreds[0, c, y, x]));
                        if (score > bestScore)
                        {
                            bestScore = score;
                            bestClass = c;
                        }
                    }

                    if (bestScore < NumOps.ToDouble(Options.ConfidenceThreshold))
                        continue;

                    // Extract kernel weights for this instance
                    var kernel = new Tensor<T>(new[] { 1, _kernelDim, 1, 1 });
                    for (int k = 0; k < _kernelDim; k++)
                    {
                        kernel[0, k, 0, 0] = kernelPreds[0, k, y, x];
                    }

                    // Generate mask using dynamic convolution
                    var mask = GenerateMask(maskFeatures, kernel, maskH, maskW);

                    // Resize mask to image size
                    var fullMask = ResizeMask(mask, imageHeight, imageWidth);

                    // Binarize
                    var binaryMask = BinarizeMask(fullMask, Options.MaskThreshold);

                    // Compute bounding box from mask
                    var box = ComputeBoundingBox(binaryMask);

                    if (box != null)
                    {
                        instances.Add(new InstanceMask<T>(box, binaryMask, bestClass, NumOps.FromDouble(bestScore)));
                    }
                }
            }
        }

        // Apply mask NMS
        instances = ApplyMaskNMS(instances, NumOps.ToDouble(Options.NmsThreshold));

        // Limit to max detections
        instances = instances.Take(Options.MaxDetections).ToList();

        return new InstanceSegmentationResult<T>
        {
            Instances = instances,
            InferenceTime = DateTime.UtcNow - startTime,
            ImageWidth = imageWidth,
            ImageHeight = imageHeight
        };
    }

    private Tensor<T> GenerateMask(Tensor<T> maskFeatures, Tensor<T> kernel, int height, int width)
    {
        // Dynamic convolution: convolve mask features with instance-specific kernel
        int channels = maskFeatures.Shape[1];
        var mask = new Tensor<T>(new[] { height, width });

        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                double sum = 0;
                for (int c = 0; c < channels; c++)
                {
                    double feat = NumOps.ToDouble(maskFeatures[0, c, h, w]);
                    double weight = NumOps.ToDouble(kernel[0, c, 0, 0]);
                    sum += feat * weight;
                }

                // Apply sigmoid
                mask[h, w] = NumOps.FromDouble(Sigmoid(sum));
            }
        }

        return mask;
    }

    private BoundingBox<T>? ComputeBoundingBox(Tensor<T> mask)
    {
        int height = mask.Shape[0];
        int width = mask.Shape[1];

        int minX = width, minY = height, maxX = 0, maxY = 0;
        bool found = false;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (NumOps.ToDouble(mask[y, x]) > 0.5)
                {
                    minX = Math.Min(minX, x);
                    minY = Math.Min(minY, y);
                    maxX = Math.Max(maxX, x);
                    maxY = Math.Max(maxY, y);
                    found = true;
                }
            }
        }

        if (!found || maxX <= minX || maxY <= minY)
            return null;

        return new BoundingBox<T>(
            NumOps.FromDouble(minX), NumOps.FromDouble(minY),
            NumOps.FromDouble(maxX + 1), NumOps.FromDouble(maxY + 1),
            BoundingBoxFormat.XYXY);
    }

    private static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    /// <inheritdoc/>
    public override long GetParameterCount()
    {
        return _backbone.GetParameterCount() +
               _fpn.GetParameterCount() +
               _categoryHead.GetParameterCount() +
               _kernelHead.GetParameterCount() +
               _maskBranch.GetParameterCount();
    }

    /// <inheritdoc/>
    public override async Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default)
    {
        byte[] data;
        if (pathOrUrl.StartsWith("http://", StringComparison.OrdinalIgnoreCase) ||
            pathOrUrl.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
        {
            using var client = new System.Net.Http.HttpClient();
            data = await client.GetByteArrayAsync(pathOrUrl, cancellationToken);
        }
        else
        {
            data = await File.ReadAllBytesAsync(pathOrUrl, cancellationToken);
        }

        using var stream = new MemoryStream(data);
        using var reader = new BinaryReader(stream);

        // Read and verify header
        int magic = reader.ReadInt32();
        if (magic != 0x534F4C32) // "SOL2" in ASCII
        {
            throw new InvalidDataException($"Invalid SOLOv2 model file. Expected magic 0x534F4C32, got 0x{magic:X8}");
        }

        int version = reader.ReadInt32();
        if (version != 1)
        {
            throw new InvalidDataException($"Unsupported SOLOv2 model version: {version}");
        }

        string name = reader.ReadString();
        int kernelDim = reader.ReadInt32();

        if (kernelDim != _kernelDim)
        {
            throw new InvalidOperationException(
                $"SOLOv2 configuration mismatch. Expected kernelDim={_kernelDim}, got {kernelDim}");
        }

        // Read component weights
        _backbone.ReadParameters(reader);
        _fpn.ReadParameters(reader);
        _categoryHead.ReadParameters(reader);
        _kernelHead.ReadParameters(reader);
        _maskBranch.ReadParameters(reader);
    }

    /// <inheritdoc/>
    public override void SaveWeights(string path)
    {
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream);

        // Write header
        writer.Write(0x534F4C32); // "SOL2" in ASCII
        writer.Write(1); // Version 1
        writer.Write(Name);
        writer.Write(_kernelDim);

        // Write component weights
        _backbone.WriteParameters(writer);
        _fpn.WriteParameters(writer);
        _categoryHead.WriteParameters(writer);
        _kernelHead.WriteParameters(writer);
        _maskBranch.WriteParameters(writer);
    }
}
