using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// One point cloud together with the camera views paired to it, forming a single training example
/// for <see cref="Concerto{T}.Pretrain"/>.
/// </summary>
/// <typeparam name="T">Numeric type (float / double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Concerto learns without labels by comparing two things that describe the
/// same scene: the point cloud itself, and photographs taken of it. Each sample here is one scene —
/// the points, where they sit in the world, and the pictures with the camera information needed to
/// work out which pixel each point falls on.
/// </para>
/// <para>
/// <see cref="PointCoordinates"/> is deliberately separate from <see cref="Input"/>. The network
/// consumes whatever tensor layout its encoder expects, while the cross-modal objective needs true
/// world coordinates to project points into each camera — the two are not interchangeable, and
/// conflating them silently produces projections that land nowhere.
/// </para>
/// </remarks>
public sealed class ConcertoPretrainingSample<T>
{
    /// <summary>
    /// The tensor fed to the network, in the layout its encoder expects.
    /// </summary>
    public required Tensor<T> Input { get; init; }

    /// <summary>
    /// World-space coordinates of each point, shape [points, 3].
    /// </summary>
    /// <remarks>
    /// The point count must equal the spatial positions produced by the decoder level the
    /// cross-modal objective reads, because each position supplies one point's feature vector.
    /// </remarks>
    public required Tensor<T> PointCoordinates { get; init; }

    /// <summary>
    /// Camera views paired with this point cloud.
    /// </summary>
    /// <remarks>
    /// <c>ConcertoOptions.ImagesPerPointCloud</c> is the published number of views per cloud; a
    /// sample carrying a different count is accepted, and <see cref="Concerto{T}.Pretrain"/> reports
    /// the discrepancy rather than silently training on a thinner signal than configured.
    /// </remarks>
    public required IReadOnlyList<ConcertoPairedView<T>> Views { get; init; }
}
