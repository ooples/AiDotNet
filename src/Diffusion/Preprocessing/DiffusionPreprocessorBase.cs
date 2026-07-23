using AiDotNet.Diffusion.Control;
using AiDotNet.Interfaces;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.Preprocessing;

/// <summary>
/// Base class for diffusion model condition preprocessors that convert input images
/// into control signals (edge maps, depth maps, pose skeletons, etc.).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Diffusion preprocessors transform input images into condition maps that guide
/// controlled generation. For example, a Canny edge preprocessor converts a photo
/// into an edge map that ControlNet uses to preserve structure.
/// </para>
/// <para>
/// <b>For Beginners:</b> These preprocessors are the "preparation step" before using
/// ControlNet. They extract specific features from your image:
/// - Edge detection: finds outlines and boundaries
/// - Depth estimation: estimates how far each pixel is
/// - Pose detection: finds body keypoints
/// - Segmentation: identifies object regions
///
/// The output becomes the "blueprint" that guides image generation.
/// </para>
/// </remarks>
public abstract class DiffusionPreprocessorBase<T> : IDataTransformer<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc />
    public bool IsFitted => true;

    /// <inheritdoc />
    public int[]? ColumnIndices => null;

    /// <inheritdoc />
    public bool SupportsInverseTransform => false;

    /// <summary>
    /// Gets the control type this preprocessor produces.
    /// </summary>
    public abstract ControlType OutputControlType { get; }

    /// <summary>
    /// Gets the number of output channels for the control signal.
    /// </summary>
    public abstract int OutputChannels { get; }

    /// <inheritdoc />
    public void Fit(Tensor<T> data)
    {
        // Preprocessors are stateless — no fitting required
    }

    /// <inheritdoc />
    public abstract Tensor<T> Transform(Tensor<T> data);

    /// <inheritdoc />
    public Tensor<T> FitTransform(Tensor<T> data) => Transform(data);

    /// <inheritdoc />
    public Tensor<T> InverseTransform(Tensor<T> data) =>
        throw new NotSupportedException("Diffusion preprocessors do not support inverse transformation.");

    /// <inheritdoc />
    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null) =>
        new[] { $"{OutputControlType}_control_map" };

    /// <summary>
    /// Clamps a value between min and max.
    /// </summary>
    protected static T Clamp(T value, T min, T max)
    {
        if (NumOps.LessThan(value, min)) return min;
        if (NumOps.GreaterThan(value, max)) return max;
        return value;
    }

    /// <summary>
    /// Converts an RGB pixel to grayscale using luminance weights.
    /// </summary>
    protected static T ToGrayscale(T r, T g, T b)
    {
        // ITU-R BT.601: 0.299R + 0.587G + 0.114B
        var result = NumOps.Multiply(NumOps.FromDouble(0.299), r);
        result = NumOps.Add(result, NumOps.Multiply(NumOps.FromDouble(0.587), g));
        result = NumOps.Add(result, NumOps.Multiply(NumOps.FromDouble(0.114), b));
        return result;
    }
}
