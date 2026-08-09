using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// One calibrated image paired with a point cloud, supplying the inputs Concerto's cross-modal
/// joint-embedding objective needs (arXiv:2510.23607).
/// </summary>
/// <remarks>
/// <para>Concerto aligns 3D point features with 2D image-patch features taken from a frozen
/// DINOv2 encoder. Establishing which point corresponds to which patch requires more than the
/// image itself: the point cloud has to be projected into the camera and then depth-checked for
/// visibility, so the camera's intrinsics and its pose relative to the cloud are both required.
/// Without them, occluded points would be paired with whatever surface is drawn in front of
/// them.</para>
/// <para><b>For Beginners:</b> To teach the 3D model from a photo, you first have to work out
/// which pixel each 3D point lands on. That needs the camera's lens parameters (intrinsics), where
/// the camera was standing (extrinsics), and a depth image to tell whether the point was actually
/// visible or hidden behind something.</para>
/// </remarks>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
public sealed class ConcertoPairedView<T>
{
    /// <summary>
    /// Patch features produced by the frozen 2D image encoder for this view.
    /// </summary>
    /// <remarks>
    /// Shape <c>[numPatches, featureDim]</c>. The paper uses DINOv2-L at 518x518 and keeps it
    /// frozen throughout pretraining, so these are supplied by the caller rather than trained
    /// here.
    /// </remarks>
    public required Tensor<T> ImagePatchFeatures { get; init; }

    /// <summary>
    /// Camera intrinsics as a row-major 3x3 matrix, used to project camera-space points onto the
    /// image plane.
    /// </summary>
    public required Tensor<T> Intrinsics { get; init; }

    /// <summary>
    /// Camera extrinsics as a row-major 4x4 world-to-camera transform.
    /// </summary>
    public required Tensor<T> Extrinsics { get; init; }

    /// <summary>
    /// Rendered depth for this view, used to verify that a projected point is actually visible
    /// rather than occluded.
    /// </summary>
    /// <remarks>
    /// Shape <c>[height, width]</c>, in metres. A projected point counts as visible only when
    /// <c>|d_camera - d_depthMap| &lt; VisibilityDepthToleranceMeters</c>.
    /// </remarks>
    public required Tensor<T> DepthMap { get; init; }

    /// <summary>Number of patches along each axis of the image encoder's patch grid.</summary>
    /// <remarks>
    /// Needed to map a projected pixel to the patch whose feature it should be pooled into.
    /// DINOv2-L at 518x518 with 14-pixel patches gives 37.
    /// </remarks>
    public required int PatchGridSize { get; init; }
}
