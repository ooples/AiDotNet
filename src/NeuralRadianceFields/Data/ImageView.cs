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

    /// <summary>
    /// Optional EXIF metadata bundle (35mm-equivalent focal length in mm + sensor width). When
    /// present, <see cref="ResolveFocalLengthInPixels"/> prefers this over the size fallback,
    /// giving physically-correct intrinsics for photos exported straight from a camera. Reference
    /// NeRF impls require the caller to pass focal in pixels — this excellence path adds the
    /// EXIF derivation as a first-class ImageView concern.
    /// </summary>
    public ExifIntrinsics? Exif { get; init; }

    /// <summary>
    /// Optional single-image reconstruction prior (Zero123 / TripoSR-style). When non-null, the
    /// radiance-field training loop composes the prior's novel-view hallucination with the
    /// caller's real photos so single-photo reconstruction produces walkable 3D scenes.
    /// </summary>
    public LearnedPrior<T>? Prior { get; init; }

    public ImageView(
        Tensor<T> photo,
        Vector<T> cameraPosition,
        Matrix<T> cameraRotation,
        T? focalLength = default,
        ExifIntrinsics? exif = null,
        LearnedPrior<T>? prior = null)
    {
        Photo          = photo          ?? throw new ArgumentNullException(nameof(photo));
        CameraPosition = cameraPosition ?? throw new ArgumentNullException(nameof(cameraPosition));
        CameraRotation = cameraRotation ?? throw new ArgumentNullException(nameof(cameraRotation));
        FocalLength    = focalLength;
        Exif           = exif;
        Prior          = prior;

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

    /// <summary>
    /// Resolves the effective focal length in pixels using the precedence order the loader +
    /// facade rely on: (1) explicit <see cref="FocalLength"/> the caller passed, (2) EXIF-derived
    /// focal computed from <see cref="Exif"/> and the photo's pixel dimensions, (3) nerfstudio-
    /// standard <c>max(H, W) * 0.7</c> fallback. Reference impls require the caller to pass focal
    /// in pixels explicitly — surfacing this three-tier auto-detect as an API concern is #1834's
    /// beyond-industry excellence goal.
    /// </summary>
    public double ResolveFocalLengthInPixels()
    {
        if (FocalLength is not null)
        {
            var numOps = AiDotNet.Helpers.MathHelper.GetNumericOperations<T>();
            if (!numOps.Equals(FocalLength!, numOps.Zero))
            {
                return numOps.ToDouble(FocalLength!);
            }
        }
        if (Exif is not null)
        {
            // f_px = (f_mm * pixel_width) / sensor_width_mm
            return Exif.FocalLengthMm * Width / Exif.SensorWidthMm;
        }
        return System.Math.Max(Height, Width) * 0.7;
    }
}

/// <summary>
/// EXIF intrinsics bundle used to derive per-photo focal-in-pixels when the loader was populated
/// from a camera JPEG. Both values are required together — focal length in millimeters combined
/// with sensor width in millimeters yields an unambiguous pixel-space focal via the pinhole
/// intrinsic formula.
/// </summary>
public sealed class ExifIntrinsics
{
    /// <summary>Focal length in millimeters (as reported by EXIF).</summary>
    public double FocalLengthMm { get; init; }

    /// <summary>Sensor width in millimeters (camera-body constant; look up per model).</summary>
    public double SensorWidthMm { get; init; }

    public ExifIntrinsics(double focalLengthMm, double sensorWidthMm)
    {
        if (focalLengthMm <= 0.0)
        {
            throw new System.ArgumentOutOfRangeException(nameof(focalLengthMm), "Focal length must be positive.");
        }
        if (sensorWidthMm <= 0.0)
        {
            throw new System.ArgumentOutOfRangeException(nameof(sensorWidthMm), "Sensor width must be positive.");
        }
        FocalLengthMm = focalLengthMm;
        SensorWidthMm = sensorWidthMm;
    }
}
