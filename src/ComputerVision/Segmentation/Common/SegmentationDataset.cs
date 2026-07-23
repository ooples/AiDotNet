namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// Represents a segmentation dataset sample with image, masks, and metadata.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A segmentation dataset contains pairs of images and their
/// ground-truth masks. This class represents a single sample that can be used for
/// training, evaluation, or visualization. It supports all segmentation task types
/// including semantic labels, instance masks, and panoptic annotations.
/// </para>
/// </remarks>
public class SegmentationSample<T>
{
    /// <summary>
    /// Input image tensor [C, H, W].
    /// </summary>
    public Tensor<T>? Image { get; set; }

    /// <summary>
    /// Ground-truth semantic label map [H, W] (for semantic/panoptic).
    /// </summary>
    public Tensor<T>? SemanticLabels { get; set; }

    /// <summary>
    /// Ground-truth instance masks [numInstances, H, W] (for instance/panoptic).
    /// </summary>
    public Tensor<T>? InstanceMasks { get; set; }

    /// <summary>
    /// Instance class IDs [numInstances].
    /// </summary>
    public int[]? InstanceClassIds { get; set; }

    /// <summary>
    /// Instance bounding boxes [numInstances, 4] as (x1, y1, x2, y2).
    /// </summary>
    public Tensor<T>? InstanceBoxes { get; set; }

    /// <summary>
    /// 3D point cloud [N, featureDim] (for point cloud segmentation).
    /// </summary>
    public Tensor<T>? PointCloud { get; set; }

    /// <summary>
    /// 3D volume [C, D, H, W] (for medical volumetric segmentation).
    /// </summary>
    public Tensor<T>? Volume { get; set; }

    /// <summary>
    /// Video frames [numFrames, C, H, W] (for video segmentation).
    /// </summary>
    public Tensor<T>? VideoFrames { get; set; }

    /// <summary>
    /// Per-frame masks [numFrames, numObjects, H, W] (for video segmentation).
    /// </summary>
    public Tensor<T>? VideoMasks { get; set; }

    /// <summary>
    /// Text annotations associated with this sample (for referring segmentation).
    /// </summary>
    public List<string>? TextAnnotations { get; set; }

    /// <summary>
    /// Sample identifier (e.g., file name or dataset ID).
    /// </summary>
    public string? SampleId { get; set; }

    /// <summary>
    /// Dataset name this sample belongs to.
    /// </summary>
    public string? DatasetName { get; set; }

    /// <summary>
    /// Image height in pixels.
    /// </summary>
    public int Height { get; set; }

    /// <summary>
    /// Image width in pixels.
    /// </summary>
    public int Width { get; set; }

    /// <summary>
    /// Number of classes in the dataset.
    /// </summary>
    public int NumClasses { get; set; }

    /// <summary>
    /// Class name mapping.
    /// </summary>
    public Dictionary<int, string>? ClassNames { get; set; }
}

/// <summary>
/// Configuration for standard segmentation datasets.
/// </summary>
public static class SegmentationDatasets
{
    /// <summary>
    /// ADE20K dataset configuration (150 semantic classes, 20K train / 2K val images).
    /// </summary>
    public static SegmentationDatasetConfig ADE20K => new()
    {
        Name = "ADE20K",
        NumClasses = 150,
        DefaultInputSize = (512, 512),
        TaskTypes = [SegmentationTaskType.Semantic],
        IgnoreIndex = 255
    };

    /// <summary>
    /// COCO Panoptic dataset configuration (133 classes: 80 things + 53 stuff).
    /// </summary>
    public static SegmentationDatasetConfig COCOPanoptic => new()
    {
        Name = "COCO-Panoptic",
        NumClasses = 133,
        NumThingClasses = 80,
        NumStuffClasses = 53,
        DefaultInputSize = (640, 640),
        TaskTypes = [SegmentationTaskType.Instance, SegmentationTaskType.Panoptic, SegmentationTaskType.Semantic],
        IgnoreIndex = 255
    };

    /// <summary>
    /// Cityscapes dataset configuration (19 semantic classes, 8 instance classes).
    /// </summary>
    public static SegmentationDatasetConfig Cityscapes => new()
    {
        Name = "Cityscapes",
        NumClasses = 19,
        NumThingClasses = 8,
        NumStuffClasses = 11,
        DefaultInputSize = (1024, 2048),
        TaskTypes = [SegmentationTaskType.Semantic, SegmentationTaskType.Instance, SegmentationTaskType.Panoptic],
        IgnoreIndex = 255
    };

    /// <summary>
    /// DAVIS 2017 video object segmentation dataset.
    /// </summary>
    public static SegmentationDatasetConfig DAVIS2017 => new()
    {
        Name = "DAVIS-2017",
        NumClasses = 0,
        DefaultInputSize = (480, 854),
        TaskTypes = [SegmentationTaskType.Video],
        IgnoreIndex = 255
    };

    /// <summary>
    /// SA-1B Segment Anything dataset (11M images, 1B+ masks).
    /// </summary>
    public static SegmentationDatasetConfig SA1B => new()
    {
        Name = "SA-1B",
        NumClasses = 0,
        DefaultInputSize = (1024, 1024),
        TaskTypes = [SegmentationTaskType.Promptable],
        IgnoreIndex = 255
    };

    /// <summary>
    /// BTCV multi-organ CT segmentation dataset (13 organs).
    /// </summary>
    public static SegmentationDatasetConfig BTCV => new()
    {
        Name = "BTCV",
        NumClasses = 14,
        DefaultInputSize = (512, 512),
        TaskTypes = [SegmentationTaskType.Medical],
        IgnoreIndex = 0,
        Is3D = true
    };
}

/// <summary>
/// Configuration for a segmentation dataset.
/// </summary>
public class SegmentationDatasetConfig
{
    /// <summary>Dataset name.</summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>Number of classes.</summary>
    public int NumClasses { get; set; }

    /// <summary>Number of "thing" classes (for panoptic datasets).</summary>
    public int NumThingClasses { get; set; }

    /// <summary>Number of "stuff" classes (for panoptic datasets).</summary>
    public int NumStuffClasses { get; set; }

    /// <summary>Default input size (height, width).</summary>
    public (int Height, int Width) DefaultInputSize { get; set; }

    /// <summary>Supported task types.</summary>
    public SegmentationTaskType[] TaskTypes { get; set; } = [];

    /// <summary>Label index to ignore during evaluation (typically 255).</summary>
    public int IgnoreIndex { get; set; } = 255;

    /// <summary>Whether this dataset is 3D volumetric.</summary>
    public bool Is3D { get; set; }
}
