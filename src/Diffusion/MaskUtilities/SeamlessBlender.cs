using AiDotNet.Helpers;
using AiDotNet.Models;

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
        var result = new Vector<T>(len);

        for (int i = 0; i < len; i++)
        {
            double position = len > 1 ? (double)i / (len - 1) : 0.5;
            double rightWeight = ComputeBlendWeight(position);
            double leftWeight = 1.0 - rightWeight;

            result[i] = NumOps.Add(
                NumOps.Multiply(NumOps.FromDouble(leftWeight), leftPatch[i]),
                NumOps.Multiply(NumOps.FromDouble(rightWeight), rightPatch[i]));
        }

        return result;
    }

    /// <summary>
    /// Generates a 1D blend weight ramp for the configured overlap size.
    /// </summary>
    /// <returns>Array of blend weights from 0.0 to 1.0.</returns>
    public T[] GenerateBlendWeights()
    {
        var weights = new T[_overlapSize];
        for (int i = 0; i < _overlapSize; i++)
        {
            double position = _overlapSize > 1 ? (double)i / (_overlapSize - 1) : 0.5;
            weights[i] = NumOps.FromDouble(ComputeBlendWeight(position));
        }
        return weights;
    }

    private double ComputeBlendWeight(double position)
    {
        return _blendProfile switch
        {
            BlendProfile.Linear => position,
            BlendProfile.Cosine => 0.5 * (1.0 - Math.Cos(Math.PI * position)),
            BlendProfile.Gaussian => 1.0 - Math.Exp(-4.0 * position * position),
            _ => position
        };
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
