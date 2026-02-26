using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;

/// <summary>
/// Result of instance segmentation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class InstanceSegmentationResult<T>
{
    /// <summary>
    /// List of detected instances with masks.
    /// </summary>
    public List<InstanceMask<T>> Instances { get; set; } = new();

    /// <summary>
    /// Time taken for inference.
    /// </summary>
    public TimeSpan InferenceTime { get; set; }

    /// <summary>
    /// Width of the input image.
    /// </summary>
    public int ImageWidth { get; set; }

    /// <summary>
    /// Height of the input image.
    /// </summary>
    public int ImageHeight { get; set; }

    /// <summary>
    /// Total number of instances detected.
    /// </summary>
    public int InstanceCount => Instances.Count;
}

/// <summary>
/// Represents a single instance with bounding box and segmentation mask.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class InstanceMask<T>
{
    /// <summary>
    /// Bounding box around the instance.
    /// </summary>
    public BoundingBox<T> Box { get; set; }

    /// <summary>
    /// Binary segmentation mask for this instance [height, width].
    /// </summary>
    public Tensor<T> Mask { get; set; }

    /// <summary>
    /// Class ID of the detected instance.
    /// </summary>
    public int ClassId { get; set; }

    /// <summary>
    /// Class name (if available).
    /// </summary>
    public string? ClassName { get; set; }

    /// <summary>
    /// Detection confidence score.
    /// </summary>
    public T Confidence { get; set; }

    /// <summary>
    /// Mask confidence score (if separate from detection confidence).
    /// </summary>
    public T MaskConfidence { get; set; }

    /// <summary>
    /// Creates a new instance mask.
    /// </summary>
    public InstanceMask(BoundingBox<T> box, Tensor<T> mask, int classId, T confidence)
    {
        Box = box;
        Mask = mask;
        ClassId = classId;
        Confidence = confidence;
        MaskConfidence = confidence;
    }

    /// <summary>
    /// Gets the mask area (number of positive pixels).
    /// </summary>
    public int GetMaskArea(INumericOperations<T> numOps)
    {
        int area = 0;
        for (int i = 0; i < Mask.Length; i++)
        {
            if (numOps.ToDouble(Mask[i]) > 0.5)
                area++;
        }
        return area;
    }

    /// <summary>
    /// Computes IoU with another instance mask.
    /// </summary>
    public double ComputeMaskIoU(InstanceMask<T> other, INumericOperations<T> numOps)
    {
        if (Mask.Shape[0] != other.Mask.Shape[0] || Mask.Shape[1] != other.Mask.Shape[1])
            return 0;

        int intersection = 0;
        int union = 0;

        for (int i = 0; i < Mask.Length; i++)
        {
            bool a = numOps.ToDouble(Mask[i]) > 0.5;
            bool b = numOps.ToDouble(other.Mask[i]) > 0.5;

            if (a && b) intersection++;
            if (a || b) union++;
        }

        return union > 0 ? (double)intersection / union : 0;
    }
}

/// <summary>
/// Options for instance segmentation models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class InstanceSegmentationOptions<T>
{
    private static readonly INumericOperations<T> NumOps =
        Tensors.Helpers.MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Architecture to use for instance segmentation.
    /// </summary>
    public InstanceSegmentationArchitecture Architecture { get; set; } = InstanceSegmentationArchitecture.MaskRCNN;

    /// <summary>
    /// Number of classes to detect.
    /// </summary>
    public int NumClasses { get; set; } = 80;

    /// <summary>
    /// Confidence threshold for detection.
    /// </summary>
    public T ConfidenceThreshold { get; set; } = NumOps.FromDouble(0.5);

    /// <summary>
    /// NMS threshold for suppression.
    /// </summary>
    public T NmsThreshold { get; set; } = NumOps.FromDouble(0.5);

    /// <summary>
    /// Maximum number of detections per image.
    /// </summary>
    public int MaxDetections { get; set; } = 100;

    /// <summary>
    /// Mask resolution (output mask size).
    /// </summary>
    public int MaskResolution { get; set; } = 28;

    /// <summary>
    /// Input image size [height, width].
    /// </summary>
    public int[] InputSize { get; set; } = new[] { 640, 640 };

    /// <summary>
    /// Whether to use pretrained weights.
    /// </summary>
    public bool UsePretrained { get; set; } = true;

    /// <summary>
    /// URL for pretrained weights.
    /// </summary>
    public string? WeightsUrl { get; set; }

    /// <summary>
    /// Mask threshold for binarization.
    /// </summary>
    public T MaskThreshold { get; set; } = NumOps.FromDouble(0.5);
}

/// <summary>
/// Instance segmentation architectures.
/// </summary>
public enum InstanceSegmentationArchitecture
{
    /// <summary>Mask R-CNN architecture.</summary>
    MaskRCNN,
    /// <summary>YOLO-based segmentation.</summary>
    YOLOSeg,
    /// <summary>SOLOv2 architecture.</summary>
    SOLOv2,
    /// <summary>Mask2Former architecture.</summary>
    Mask2Former
}

/// <summary>
/// Base class for instance segmentation models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This base class provides the common functionality for all instance segmentation models, which detect and mask individual objects in images.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class InstanceSegmenterBase<T>
{
    protected readonly INumericOperations<T> NumOps;
    protected readonly InstanceSegmentationOptions<T> Options;

    /// <summary>
    /// Name of this segmentation model.
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Creates a new instance segmenter.
    /// </summary>
    protected InstanceSegmenterBase(InstanceSegmentationOptions<T> options)
    {
        NumOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        Options = options;
    }

    /// <summary>
    /// Performs instance segmentation on an image.
    /// </summary>
    /// <param name="image">Input image tensor [batch, channels, height, width].</param>
    /// <returns>Instance segmentation result with masks.</returns>
    public abstract InstanceSegmentationResult<T> Segment(Tensor<T> image);

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public abstract long GetParameterCount();

    /// <summary>
    /// Loads pretrained weights.
    /// </summary>
    public abstract Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default);

    /// <summary>
    /// Saves model weights.
    /// </summary>
    public abstract void SaveWeights(string path);

    /// <summary>
    /// Resizes a mask to target size using bilinear interpolation.
    /// </summary>
    protected Tensor<T> ResizeMask(Tensor<T> mask, int targetHeight, int targetWidth)
    {
        int srcH = mask.Shape[0];
        int srcW = mask.Shape[1];

        var resized = new Tensor<T>(new[] { targetHeight, targetWidth });

        for (int h = 0; h < targetHeight; h++)
        {
            for (int w = 0; w < targetWidth; w++)
            {
                double srcY = (double)h / targetHeight * srcH;
                double srcX = (double)w / targetWidth * srcW;

                int y0 = (int)Math.Floor(srcY);
                int x0 = (int)Math.Floor(srcX);
                int y1 = Math.Min(y0 + 1, srcH - 1);
                int x1 = Math.Min(x0 + 1, srcW - 1);

                double wy1 = srcY - y0;
                double wy0 = 1.0 - wy1;
                double wx1 = srcX - x0;
                double wx0 = 1.0 - wx1;

                double v00 = NumOps.ToDouble(mask[y0, x0]);
                double v01 = NumOps.ToDouble(mask[y0, x1]);
                double v10 = NumOps.ToDouble(mask[y1, x0]);
                double v11 = NumOps.ToDouble(mask[y1, x1]);

                double val = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
                resized[h, w] = NumOps.FromDouble(val);
            }
        }

        return resized;
    }

    /// <summary>
    /// Binarizes a mask using threshold.
    /// </summary>
    protected Tensor<T> BinarizeMask(Tensor<T> mask, T threshold)
    {
        var binarized = new Tensor<T>(mask.Shape);
        double threshVal = NumOps.ToDouble(threshold);

        for (int i = 0; i < mask.Length; i++)
        {
            binarized[i] = NumOps.ToDouble(mask[i]) > threshVal
                ? NumOps.FromDouble(1.0)
                : NumOps.FromDouble(0.0);
        }

        return binarized;
    }

    /// <summary>
    /// Crops a mask to bounding box region.
    /// </summary>
    protected Tensor<T> CropMaskToBox(Tensor<T> mask, BoundingBox<T> box, int imageHeight, int imageWidth)
    {
        int x1 = Math.Max(0, (int)NumOps.ToDouble(box.X1));
        int y1 = Math.Max(0, (int)NumOps.ToDouble(box.Y1));
        int x2 = Math.Min(imageWidth, (int)NumOps.ToDouble(box.X2));
        int y2 = Math.Min(imageHeight, (int)NumOps.ToDouble(box.Y2));

        int cropW = x2 - x1;
        int cropH = y2 - y1;

        if (cropW <= 0 || cropH <= 0)
            return new Tensor<T>(new[] { 1, 1 });

        var cropped = new Tensor<T>(new[] { cropH, cropW });

        for (int h = 0; h < cropH; h++)
        {
            for (int w = 0; w < cropW; w++)
            {
                int srcH = y1 + h;
                int srcW = x1 + w;

                if (srcH < mask.Shape[0] && srcW < mask.Shape[1])
                {
                    cropped[h, w] = mask[srcH, srcW];
                }
            }
        }

        return cropped;
    }

    /// <summary>
    /// Applies sigmoid activation to tensor.
    /// </summary>
    protected void ApplySigmoid(Tensor<T> tensor)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            double x = NumOps.ToDouble(tensor[i]);
            tensor[i] = NumOps.FromDouble(1.0 / (1.0 + Math.Exp(-x)));
        }
    }

    /// <summary>
    /// Applies NMS to filter overlapping instances.
    /// </summary>
    protected List<InstanceMask<T>> ApplyMaskNMS(List<InstanceMask<T>> instances, double iouThreshold)
    {
        if (instances.Count <= 1)
            return instances;

        // Sort by confidence descending
        var sorted = instances.OrderByDescending(i => NumOps.ToDouble(i.Confidence)).ToList();
        var keep = new List<InstanceMask<T>>();

        while (sorted.Count > 0)
        {
            var best = sorted[0];
            keep.Add(best);
            sorted.RemoveAt(0);

            sorted = sorted.Where(inst =>
            {
                double iou = best.ComputeMaskIoU(inst, NumOps);
                return iou < iouThreshold;
            }).ToList();
        }

        return keep;
    }
}
