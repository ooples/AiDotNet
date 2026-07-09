using System;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralRadianceFields.Data;

/// <summary>
/// A single photo + camera pose used as input to image-space (photometric) radiance-field
/// training (#1834). Reference NeRF / GS implementations require the caller to hand-assemble
/// ray samples from photos; this type is the canonical shape a radiance-field loader emits so
/// the model receives everything it needs to sample rays and compute a photometric loss.
/// </summary>
/// <typeparam name="T">Numeric type (float or double).</typeparam>
/// <remarks>
/// <para>
/// Reference impls fragment this data across a config file (poses), a directory of image
/// files (photos), and hard-coded assumptions (focal length must be passed by hand).
/// <see cref="ImageView{T}"/> unifies them into one strongly-typed record with sensible
/// nullable knobs (<see cref="FocalLength"/> is nullable — a null value asks the loader /
/// facade to auto-detect from EXIF or fall back to a paper default derived from image size).
/// </para>
/// </remarks>
public sealed class ImageView<T>
{
    /// <summary>
    /// Photo tensor of shape [H, W, 3] (RGB) or [H, W, 4] (RGBA — alpha ignored by the model,
    /// preserved for compositing).
    /// </summary>
    public Tensor<T> Photo { get; init; }

    /// <summary>
    /// Camera origin in world coordinates [3].
    /// </summary>
    public Vector<T> CameraPosition { get; init; }

    /// <summary>
    /// Camera-to-world rotation matrix [3, 3]. Column vectors are the world-space directions
    /// of the camera's right / up / forward axes (nerfstudio + Instant-NGP convention).
    /// </summary>
    public Matrix<T> CameraRotation { get; init; }

    /// <summary>
    /// Focal length in pixels. Nullable — null asks the loader to auto-detect from EXIF or
    /// fall back to a size-based default (<c>max(H, W) * 0.7</c> is a paper-standard fallback
    /// used by nerfstudio).
    /// </summary>
    public T? FocalLength { get; init; }

    public ImageView(
        Tensor<T> photo,
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        T? focalLength = default)
    {
        Photo          = photo          ?? throw new ArgumentNullException(nameof(photo));
        CameraPosition = cameraPosition ?? throw new ArgumentNullException(nameof(cameraPosition));
        CameraRotation = cameraRotation ?? throw new ArgumentNullException(nameof(cameraRotation));
        FocalLength    = focalLength;

        if (photo.Shape.Length != 3 || (photo.Shape[2] != 3 && photo.Shape[2] != 4))
        {
            throw new ArgumentException(
                $"Photo must be [H, W, 3] or [H, W, 4]; got [{string.Join(",", photo.Shape)}].",
                nameof(photo));
        }
        if (cameraPosition.Length != 3)
        {
            throw new ArgumentException(
                $"CameraPosition must be a 3-vector; got length {cameraPosition.Length}.",
                nameof(cameraPosition));
        }
        if (cameraRotation.Rows != 3 || cameraRotation.Columns != 3)
        {
            throw new ArgumentException(
                $"CameraRotation must be 3x3; got {cameraRotation.Rows}x{cameraRotation.Columns}.",
                nameof(cameraRotation));
        }
    }

    /// <summary>Image height (pixels).</summary>
    public int Height => Photo.Shape[0];

    /// <summary>Image width (pixels).</summary>
    public int Width => Photo.Shape[1];
}
