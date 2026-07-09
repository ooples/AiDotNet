using System;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralRadianceFields.Data;

/// <summary>
/// A batch of pixels sampled from one or more <see cref="ImageView{T}"/> views — the target
/// side of image-space radiance-field training (#1834). Each sampled pixel carries a ray
/// origin + direction (so the model can render exactly that pixel through volume rendering)
/// and the ground-truth RGB the loss compares against.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
public sealed class PixelBatch<T>
{
    /// <summary>
    /// Ray origins in world coordinates [N, 3]. One origin per sampled pixel.
    /// </summary>
    public Tensor<T> RayOrigins { get; }

    /// <summary>
    /// Ray directions (unit vectors) in world coordinates [N, 3]. Points into the scene from
    /// the corresponding <see cref="RayOrigins"/> row.
    /// </summary>
    public Tensor<T> RayDirections { get; }

    /// <summary>
    /// Ground-truth pixel colors [N, 3] — the target the photometric loss compares the model's
    /// rendered output against.
    /// </summary>
    public Tensor<T> TargetColors { get; }

    public PixelBatch(Tensor<T> rayOrigins, Tensor<T> rayDirections, Tensor<T> targetColors)
    {
        RayOrigins    = rayOrigins    ?? throw new ArgumentNullException(nameof(rayOrigins));
        RayDirections = rayDirections ?? throw new ArgumentNullException(nameof(rayDirections));
        TargetColors  = targetColors  ?? throw new ArgumentNullException(nameof(targetColors));

        int n = rayOrigins.Shape[0];
        if (rayOrigins.Shape.Length != 2 || rayOrigins.Shape[1] != 3)
        {
            throw new ArgumentException("RayOrigins must be [N, 3].", nameof(rayOrigins));
        }
        if (rayDirections.Shape.Length != 2 || rayDirections.Shape[0] != n || rayDirections.Shape[1] != 3)
        {
            throw new ArgumentException(
                $"RayDirections must be [N, 3] with N matching RayOrigins ({n}); got " +
                $"[{string.Join(",", rayDirections.Shape)}].",
                nameof(rayDirections));
        }
        if (targetColors.Shape.Length != 2 || targetColors.Shape[0] != n || targetColors.Shape[1] != 3)
        {
            throw new ArgumentException(
                $"TargetColors must be [N, 3] with N matching RayOrigins ({n}); got " +
                $"[{string.Join(",", targetColors.Shape)}].",
                nameof(targetColors));
        }
    }

    /// <summary>Number of pixels / rays in this batch.</summary>
    public int Count => RayOrigins.Shape[0];
}
