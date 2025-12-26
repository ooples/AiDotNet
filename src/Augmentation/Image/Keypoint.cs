namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Represents a keypoint annotation for pose estimation and landmark detection.
/// </summary>
/// <remarks>
/// <para>
/// Keypoints are specific points of interest on objects, typically used for:
/// - Human pose estimation (joints, facial landmarks)
/// - Animal pose estimation
/// - Object landmark detection (car wheels, furniture corners)
/// </para>
/// <para><b>For Beginners:</b> Think of keypoints as dots marking important locations
/// on an object. For a human, these might be the shoulders, elbows, wrists, etc.
/// When you flip or rotate an image, these points need to move accordingly.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for coordinates.</typeparam>
public class Keypoint<T>
{
    /// <summary>
    /// Gets or sets the X coordinate.
    /// </summary>
    public T X { get; set; }

    /// <summary>
    /// Gets or sets the Y coordinate.
    /// </summary>
    public T Y { get; set; }

    /// <summary>
    /// Gets or sets the visibility state.
    /// </summary>
    /// <remarks>
    /// Standard values:
    /// - 0: Not labeled (missing from annotation)
    /// - 1: Labeled but not visible (occluded)
    /// - 2: Labeled and visible
    /// </remarks>
    public int Visibility { get; set; }

    /// <summary>
    /// Gets or sets the confidence score (if from a detector).
    /// </summary>
    public T? Confidence { get; set; }

    /// <summary>
    /// Gets or sets the keypoint name/label.
    /// </summary>
    public string? Name { get; set; }

    /// <summary>
    /// Gets or sets the keypoint index within the skeleton.
    /// </summary>
    public int Index { get; set; }

    /// <summary>
    /// Gets or sets the parent keypoint index (for skeleton hierarchy).
    /// </summary>
    /// <remarks>
    /// -1 indicates no parent (root keypoint).
    /// </remarks>
    public int ParentIndex { get; set; } = -1;

    /// <summary>
    /// Gets or sets the image width (for normalized coordinates).
    /// </summary>
    public int ImageWidth { get; set; }

    /// <summary>
    /// Gets or sets the image height (for normalized coordinates).
    /// </summary>
    public int ImageHeight { get; set; }

    /// <summary>
    /// Gets or sets whether coordinates are normalized to [0, 1].
    /// </summary>
    public bool IsNormalized { get; set; }

    /// <summary>
    /// Gets or sets additional metadata.
    /// </summary>
    public IDictionary<string, object>? Metadata { get; set; }

    /// <summary>
    /// Creates an empty keypoint.
    /// </summary>
    public Keypoint()
    {
        X = default!;
        Y = default!;
        Visibility = 2; // Default to visible
    }

    /// <summary>
    /// Creates a keypoint with the specified coordinates.
    /// </summary>
    /// <param name="x">The X coordinate.</param>
    /// <param name="y">The Y coordinate.</param>
    /// <param name="visibility">The visibility state (default: 2 = visible).</param>
    /// <param name="name">The keypoint name.</param>
    /// <param name="index">The keypoint index.</param>
    public Keypoint(T x, T y, int visibility = 2, string? name = null, int index = 0)
    {
        X = x;
        Y = y;
        Visibility = visibility;
        Name = name;
        Index = index;
    }

    /// <summary>
    /// Creates a deep copy of this keypoint.
    /// </summary>
    /// <returns>A new keypoint with the same values.</returns>
    public Keypoint<T> Clone()
    {
        return new Keypoint<T>
        {
            X = X,
            Y = Y,
            Visibility = Visibility,
            Confidence = Confidence,
            Name = Name,
            Index = Index,
            ParentIndex = ParentIndex,
            ImageWidth = ImageWidth,
            ImageHeight = ImageHeight,
            IsNormalized = IsNormalized,
            Metadata = Metadata is not null ? new Dictionary<string, object>(Metadata) : null
        };
    }

    /// <summary>
    /// Converts to absolute pixel coordinates.
    /// </summary>
    /// <returns>The (x, y) coordinates in pixels.</returns>
    public (double x, double y) ToAbsolute()
    {
        double x = Convert.ToDouble(X);
        double y = Convert.ToDouble(Y);

        if (IsNormalized)
        {
            if (ImageWidth <= 0 || ImageHeight <= 0)
            {
                throw new InvalidOperationException("ImageWidth and ImageHeight must be set for normalized coordinates.");
            }

            return (x * ImageWidth, y * ImageHeight);
        }

        return (x, y);
    }

    /// <summary>
    /// Converts to normalized coordinates [0, 1].
    /// </summary>
    /// <returns>The (x, y) coordinates normalized to [0, 1].</returns>
    public (double x, double y) ToNormalized()
    {
        if (ImageWidth <= 0 || ImageHeight <= 0)
        {
            throw new InvalidOperationException("ImageWidth and ImageHeight must be set for normalization.");
        }

        double x = Convert.ToDouble(X);
        double y = Convert.ToDouble(Y);

        if (IsNormalized)
        {
            return (x, y);
        }

        return (x / ImageWidth, y / ImageHeight);
    }

    /// <summary>
    /// Checks if this keypoint is visible.
    /// </summary>
    /// <returns>True if visibility is 2 (labeled and visible).</returns>
    public bool IsVisible()
    {
        return Visibility == 2;
    }

    /// <summary>
    /// Checks if this keypoint is labeled (visible or occluded).
    /// </summary>
    /// <returns>True if visibility is 1 or 2.</returns>
    public bool IsLabeled()
    {
        return Visibility >= 1;
    }

    /// <summary>
    /// Checks if this keypoint is within image boundaries.
    /// </summary>
    /// <param name="width">The image width.</param>
    /// <param name="height">The image height.</param>
    /// <returns>True if the keypoint is within bounds.</returns>
    public bool IsWithinBounds(int width, int height)
    {
        var (x, y) = ToAbsolute();
        return x >= 0 && x < width && y >= 0 && y < height;
    }

    /// <summary>
    /// Calculates the Euclidean distance to another keypoint.
    /// </summary>
    /// <param name="other">The other keypoint.</param>
    /// <returns>The distance in pixels.</returns>
    public double DistanceTo(Keypoint<T> other)
    {
        var (x1, y1) = ToAbsolute();
        var (x2, y2) = other.ToAbsolute();

        double dx = x2 - x1;
        double dy = y2 - y1;

        return Math.Sqrt(dx * dx + dy * dy);
    }

    /// <summary>
    /// Calculates the Object Keypoint Similarity (OKS) score.
    /// </summary>
    /// <param name="groundTruth">The ground truth keypoint.</param>
    /// <param name="scale">The object scale (usually sqrt of bbox area).</param>
    /// <param name="kappa">The per-keypoint constant (standard COCO values range 0.025-0.107).</param>
    /// <returns>The OKS score between 0 and 1.</returns>
    public double OKS(Keypoint<T> groundTruth, double scale, double kappa = 0.05)
    {
        if (!IsLabeled() || !groundTruth.IsLabeled())
        {
            return 0;
        }

        double distance = DistanceTo(groundTruth);
        double denominator = 2 * scale * scale * kappa * kappa;

        if (denominator <= 0)
        {
            return 0;
        }

        return Math.Exp(-distance * distance / denominator);
    }
}

/// <summary>
/// Represents a skeleton definition for pose estimation.
/// </summary>
/// <remarks>
/// A skeleton defines the structure of keypoints and their connections.
/// </remarks>
public class SkeletonDefinition
{
    /// <summary>
    /// Gets or sets the keypoint names in order.
    /// </summary>
    public IList<string> KeypointNames { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets the skeleton connections as pairs of keypoint indices.
    /// </summary>
    public IList<(int from, int to)> Connections { get; set; } = new List<(int, int)>();

    /// <summary>
    /// Gets or sets the per-keypoint OKS constants (for COCO-style evaluation).
    /// </summary>
    public IList<double>? KeypointOKSConstants { get; set; }

    /// <summary>
    /// Gets or sets the left-right symmetric keypoint pairs for horizontal flip.
    /// </summary>
    /// <remarks>
    /// When horizontally flipping, left_shoulder should swap with right_shoulder, etc.
    /// </remarks>
    public IList<(int left, int right)> SymmetricPairs { get; set; } = new List<(int, int)>();

    /// <summary>
    /// Creates a standard COCO human pose skeleton with 17 keypoints.
    /// </summary>
    /// <returns>The COCO skeleton definition.</returns>
    public static SkeletonDefinition CreateCOCO()
    {
        return new SkeletonDefinition
        {
            KeypointNames = new List<string>
            {
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            },
            Connections = new List<(int, int)>
            {
                (0, 1), (0, 2), (1, 3), (2, 4),  // Face
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  // Arms
                (5, 11), (6, 12), (11, 12),  // Torso
                (11, 13), (13, 15), (12, 14), (14, 16)  // Legs
            },
            SymmetricPairs = new List<(int, int)>
            {
                (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)
            },
            KeypointOKSConstants = new List<double>
            {
                0.026, 0.025, 0.025, 0.035, 0.035,
                0.079, 0.079, 0.072, 0.072, 0.062, 0.062,
                0.107, 0.107, 0.087, 0.087, 0.089, 0.089
            }
        };
    }

    /// <summary>
    /// Creates a MPII human pose skeleton with 16 keypoints.
    /// </summary>
    /// <returns>The MPII skeleton definition.</returns>
    public static SkeletonDefinition CreateMPII()
    {
        return new SkeletonDefinition
        {
            KeypointNames = new List<string>
            {
                "right_ankle", "right_knee", "right_hip", "left_hip",
                "left_knee", "left_ankle", "pelvis", "thorax",
                "upper_neck", "head_top", "right_wrist", "right_elbow",
                "right_shoulder", "left_shoulder", "left_elbow", "left_wrist"
            },
            Connections = new List<(int, int)>
            {
                (0, 1), (1, 2), (2, 6), (6, 3), (3, 4), (4, 5),  // Legs
                (6, 7), (7, 8), (8, 9),  // Spine
                (10, 11), (11, 12), (12, 7), (7, 13), (13, 14), (14, 15)  // Arms
            },
            SymmetricPairs = new List<(int, int)>
            {
                (0, 5), (1, 4), (2, 3), (10, 15), (11, 14), (12, 13)
            }
        };
    }
}
