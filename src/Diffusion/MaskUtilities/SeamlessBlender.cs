using AiDotNet.Helpers;
using AiDotNet.Models;
using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.Diffusion.MaskUtilities;

/// <summary>
/// Seamless blending for panoramic and tiled diffusion generation with overlap regions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Handles the smooth blending of overlapping patches in panoramic, tiled, or outpainting
/// generation pipelines. Supports linear ramp, cosine, and Gaussian blending profiles for
/// the overlap regions. Ensures seamless transitions without visible seams at patch boundaries.
/// </para>
/// <para>
/// <b>For Beginners:</b> When creating large images by stitching together smaller patches
/// (like a panorama), the edges of each patch need to blend smoothly together. SeamlessBlender
/// handles this transition, making sure there are no visible lines or color jumps where
/// patches meet — similar to how panorama photo apps stitch multiple photos together.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SeamlessBlender<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly BlendProfile _blendProfile;
    private readonly int _overlapSize;

    /// <summary>
    /// Gets the blend profile type.
    /// </summary>
    public BlendProfile Profile => _blendProfile;

    /// <summary>
    /// Gets the overlap size in pixels/elements.
    /// </summary>
    public int OverlapSize => _overlapSize;

    /// <summary>
    /// Initializes a new seamless blender.
    /// </summary>
    /// <param name="blendProfile">Profile for overlap blending (default: Cosine).</param>
    /// <param name="overlapSize">Size of overlap region in elements (default: 32).</param>
    public SeamlessBlender(BlendProfile blendProfile = BlendProfile.Cosine, int overlapSize = 32)
    {
        _blendProfile = blendProfile;
        _overlapSize = overlapSize;
    }

    /// <summary>
    /// Blends two overlapping patches in the overlap region.
    /// </summary>
    /// <param name="leftPatch">Left/first patch values in overlap region.</param>
    /// <param name="rightPatch">Right/second patch values in overlap region.</param>
    /// <returns>Blended values for the overlap region.</returns>
    public Vector<T> BlendOverlap(Vector<T> leftPatch, Vector<T> rightPatch)
    {
        int len = Math.Min(leftPatch.Length, rightPatch.Length);
        return MaskVectorOps.Lerp(
            MaskVectorOps.Fit(leftPatch, len, NumOps.Zero),
            MaskVectorOps.Fit(rightPatch, len, NumOps.Zero),
            ComputeBlendWeights(len));
    }

    /// <summary>
    /// Generates a 1D blend weight ramp for the configured overlap size.
    /// </summary>
    /// <returns>Array of blend weights from 0.0 to 1.0.</returns>
    public T[] GenerateBlendWeights()
    {
        return ComputeBlendWeights(_overlapSize).ToArray();
    }

    /// <summary>
    /// Right-patch weights at positions i / (length - 1) (0.5 for a single element) for the blend profile.
    /// </summary>
    private Vector<T> ComputeBlendWeights(int length)
    {
        if (length <= 0)
            return new Vector<T>(0);

        var engine = AiDotNetEngine.Current;
        var position = length > 1
            ? engine.TensorLinspace(NumOps.Zero, NumOps.One, length)
            : new Tensor<T>(new[] { 1 }, new Vector<T>(new[] { NumOps.FromDouble(0.5) }));

        var weights = _blendProfile switch
        {
            // 0.5 * (1 - cos(pi * p))
            BlendProfile.Cosine => engine.TensorMultiplyScalar(
                engine.ScalarMinusTensor(NumOps.One, engine.TensorCos(engine.TensorMultiplyScalar(position, NumOps.FromDouble(Math.PI)))),
                NumOps.FromDouble(0.5)),
            // 1 - exp(-4 p^2)
            BlendProfile.Gaussian => engine.ScalarMinusTensor(NumOps.One,
                engine.TensorExp(engine.TensorMultiplyScalar(engine.TensorSquare(position), NumOps.FromDouble(-4.0)))),
            _ => position
        };
        return weights.ToVector();
    }
}

/// <summary>
/// Specifies the blending profile for overlap region transitions.
/// </summary>
public enum BlendProfile
{
    /// <summary>Linear ramp from 0 to 1.</summary>
    Linear,
    /// <summary>Smooth cosine curve (recommended for most cases).</summary>
    Cosine,
    /// <summary>Gaussian-based smooth transition.</summary>
    Gaussian
}
