namespace AiDotNet.ComputerVision.Detection;

/// <summary>Owns an immutable, unpadded target list for every image in a training batch.</summary>
/// <typeparam name="T">The detector's numeric type.</typeparam>
public sealed class DetectionTrainingBatch<T>
{
    private readonly IReadOnlyList<DetectionTrainingTarget<T>>[] _images;

    /// <summary>Copies the supplied lists. Empty images are valid; an empty batch is not.</summary>
    public DetectionTrainingBatch(IEnumerable<IEnumerable<DetectionTrainingTarget<T>>> images)
    {
        if (images is null) throw new ArgumentNullException(nameof(images));
        var owned = new List<IReadOnlyList<DetectionTrainingTarget<T>>>();
        int total = 0;
        foreach (var image in images)
        {
            if (image is null) throw new ArgumentException("Each image must have a target list, even when empty.", nameof(images));
            var targets = image.ToArray();
            if (targets.Any(target => target is null))
                throw new ArgumentException("Target lists cannot contain null entries.", nameof(images));
            total = checked(total + targets.Length);
            owned.Add(Array.AsReadOnly(targets));
        }
        if (owned.Count == 0) throw new ArgumentException("A training batch must contain at least one image.", nameof(images));
        _images = owned.ToArray();
        TargetCount = total;
    }

    /// <summary>Gets the number of images, including images with no objects.</summary>
    public int ImageCount => _images.Length;
    /// <summary>Gets the total foreground target count across all images.</summary>
    public int TargetCount { get; }
    /// <summary>Gets the immutable targets for one image.</summary>
    public IReadOnlyList<DetectionTrainingTarget<T>> this[int imageIndex] => _images[imageIndex];

    /// <summary>
    /// Converts the COCO loader's [batch, objects, 5] normalized top-left xywh labels.
    /// A completely zero row is padding; all other rows must describe a valid foreground box.
    /// </summary>
    public static DetectionTrainingBatch<T> FromPaddedCoco(Tensor<T> labels)
    {
        ValidatePaddedShape(labels, exactWidth: true);
        var ops = MathHelper.GetNumericOperations<T>();
        var images = new List<DetectionTrainingTarget<T>>[labels.Shape[0]];
        for (int image = 0; image < images.Length; image++)
        {
            var targets = new List<DetectionTrainingTarget<T>>();
            for (int item = 0; item < labels.Shape[1]; item++)
            {
                bool padding = true;
                for (int coordinate = 0; coordinate < 5; coordinate++)
                    padding &= ops.Equals(labels[image, item, coordinate], ops.Zero);
                if (padding) continue;
                int label = ReadClass(labels[image, item, 0], nameof(labels));
                targets.Add(DetectionTrainingTarget<T>.FromNormalizedXywh(label,
                    labels[image, item, 1], labels[image, item, 2], labels[image, item, 3], labels[image, item, 4]));
            }
            images[image] = targets;
        }
        return new DetectionTrainingBatch<T>(images);
    }

    internal static DetectionTrainingBatch<T> FromPaddedDetr(Tensor<T> labels)
    {
        ValidatePaddedShape(labels, exactWidth: false);
        var ops = MathHelper.GetNumericOperations<T>();
        var images = new List<DetectionTrainingTarget<T>>[labels.Shape[0]];
        for (int image = 0; image < images.Length; image++)
        {
            var targets = new List<DetectionTrainingTarget<T>>();
            bool sawPadding = false;
            for (int item = 0; item < labels.Shape[1]; item++)
            {
                if (ops.Equals(labels[image, item, 0], ops.FromDouble(-1)))
                {
                    sawPadding = true;
                    continue;
                }
                if (sawPadding)
                    throw new ArgumentException("DETR targets cannot follow a -1 padding row.", nameof(labels));
                int label = ReadClass(labels[image, item, 0], nameof(labels));
                targets.Add(new DetectionTrainingTarget<T>(label,
                    labels[image, item, 1], labels[image, item, 2], labels[image, item, 3], labels[image, item, 4]));
            }
            images[image] = targets;
        }
        return new DetectionTrainingBatch<T>(images);
    }

    internal void ValidateForModel(int imageCount, int foregroundClasses, int queries)
    {
        if (ImageCount != imageCount)
            throw new ArgumentException("The target batch must contain one list per input image.", "targets");
        foreach (var image in _images)
        {
            if (image.Count > queries)
                throw new ArgumentException("An image has more targets than detection queries; targets cannot be silently discarded.", "targets");
            foreach (var target in image)
                if (target.ClassId >= foregroundClasses)
                    throw new ArgumentException("Every target class must be a foreground class supported by the model.", "targets");
        }
    }

    private static int ReadClass(T value, string parameterName)
    {
        double label = MathHelper.GetNumericOperations<T>().ToDouble(value);
        if (double.IsNaN(label) || double.IsInfinity(label) || label < 0 || label > int.MaxValue || label != Math.Truncate(label))
            throw new ArgumentException("Target classes must be finite nonnegative integers.", parameterName);
        return (int)label;
    }

    private static void ValidatePaddedShape(Tensor<T> labels, bool exactWidth)
    {
        if (labels is null) throw new ArgumentNullException(nameof(labels));
        if (labels.Rank != 3 || labels.Shape[0] <= 0 || labels.Shape[2] < 5)
            throw new ArgumentException("Padded labels must have shape [positive batch, objects, at least 5].", nameof(labels));
        if (exactWidth && labels.Shape[2] != 5)
            throw new ArgumentException("COCO labels must have exactly five values per row.", nameof(labels));
    }
}
