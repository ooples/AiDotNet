using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>
/// Reconstructs a triangle mesh from a point cloud by one of the published surface reconstruction algorithms.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="SurfaceReconstructionMethod.Poisson"/>: screened Poisson reconstruction (Kazhdan &amp; Hoppe 2013), a
/// smooth watertight surface. <see cref="SurfaceReconstructionMethod.BallPivoting"/>: the ball-pivoting algorithm
/// (Bernardini et al. 1999), a mesh that interpolates the samples. <see cref="SurfaceReconstructionMethod.MarchingCubes"/>:
/// marching cubes over the signed distance to the samples' tangent planes (Hoppe et al. 1992).
/// <see cref="SurfaceReconstructionMethod.AlphaShape"/>: the regularised alpha shape (Edelsbrunner &amp; Muecke 1994)
/// of the Delaunay tetrahedralisation.
/// </para>
/// <para>
/// Poisson, ball pivoting and the signed distance need oriented normals. When none are given they are estimated by
/// principal component analysis over the k nearest neighbours and oriented along a minimum spanning tree (Hoppe et al.
/// 1992), which points them outward for a closed surface.
/// </para>
/// <para>
/// The geometry runs in double precision (its neighbour searches, predicates and linear solves need it); the input is
/// read once and the output converted to <typeparamref name="T"/> once.
/// </para>
/// <para>
/// <b>For Beginners:</b> A point cloud is just dots in space. This joins the dots into a surface made of triangles,
/// which you can render, measure, or export.
/// </para>
/// </remarks>
public static class PointCloudSurfaceReconstruction
{
    /// <summary>
    /// Reconstructs a surface mesh from a point cloud.
    /// </summary>
    /// <typeparam name="T">The numeric type of the points and of the mesh.</typeparam>
    /// <param name="pointCloud">The points, [N, C] or [1, N, C] with C &gt;= 3; the first three channels are x, y, z
    /// (further channels such as colours are ignored).</param>
    /// <param name="method">The reconstruction algorithm.</param>
    /// <param name="options">Algorithm settings; null uses the defaults.</param>
    /// <param name="normals">Optional outward unit normals, the same layout as the points with C = 3; null estimates
    /// them.</param>
    /// <returns>The reconstructed mesh with per-vertex normals.</returns>
    public static Mesh3D<T> Reconstruct<T>(
        Tensor<T> pointCloud,
        SurfaceReconstructionMethod method,
        SurfaceReconstructionOptions? options = null,
        Tensor<T>? normals = null)
    {
        var settings = options ?? new SurfaceReconstructionOptions();
        double[] points = ReadPoints(pointCloud, nameof(pointCloud));
        int count = points.Length / 3;
        if (count < 4)
            throw new ArgumentException("Surface reconstruction needs at least four points.", nameof(pointCloud));
        if (settings.NormalNeighborCount < 3)
            throw new ArgumentOutOfRangeException(nameof(options), "NormalNeighborCount must be at least 3.");

        int k = Math.Min(settings.NormalNeighborCount, count - 1);
        var spacing = MeasureSpacing(points, k);
        var grid = new PointGrid(points, spacing.MeanNearest);

        if (method == SurfaceReconstructionMethod.AlphaShape)
        {
            double alpha = settings.AlphaShapeRadius ?? (2.0 * spacing.MeanKth);
            return AlphaShapeReconstruction.Reconstruct(points, alpha).ToMesh<T>();
        }

        double[] unitNormals;
        double[] centers;
        if (normals is null)
        {
            (unitNormals, centers) = PointCloudNormals.Estimate(points, Math.Max(3, k), grid);
        }
        else
        {
            unitNormals = ReadPoints(normals, nameof(normals));
            if (unitNormals.Length != points.Length)
                throw new ArgumentException("There must be one normal per point.", nameof(normals));
            Normalize(unitNormals);
            centers = TangentCenters(points, Math.Max(3, k), grid);
        }

        switch (method)
        {
            case SurfaceReconstructionMethod.Poisson:
                return ScreenedPoissonReconstruction.Reconstruct(points, unitNormals, spacing.Areas, settings).ToMesh<T>();
            case SurfaceReconstructionMethod.BallPivoting:
            {
                double[] radii = settings.BallPivotingRadii ?? new[] { 1.5 * spacing.MeanNearest, 3.0 * spacing.MeanNearest };
                return BallPivotingReconstruction.Reconstruct(points, unitNormals, radii).ToMesh<T>();
            }

            case SurfaceReconstructionMethod.MarchingCubes:
            {
                double support = settings.SignedDistanceSupportRadius ?? (2.0 * spacing.MeanKth);
                return SignedDistanceReconstruction.Reconstruct(
                    points, unitNormals, centers, support, settings.SignedDistanceResolution).ToMesh<T>();
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(method), method, "Unknown surface reconstruction method.");
        }
    }

    /// <summary>
    /// Estimates outward unit normals for a point cloud (Hoppe et al. 1992: k-nearest-neighbour PCA, oriented along a
    /// minimum spanning tree).
    /// </summary>
    /// <typeparam name="T">The numeric type of the points.</typeparam>
    /// <param name="pointCloud">The points, [N, C] or [1, N, C] with C &gt;= 3.</param>
    /// <param name="neighborCount">The neighbourhood size k.</param>
    /// <returns>The normals, [N, 3].</returns>
    public static Tensor<T> EstimateNormals<T>(Tensor<T> pointCloud, int neighborCount = 10)
    {
        double[] points = ReadPoints(pointCloud, nameof(pointCloud));
        int count = points.Length / 3;
        if (count < 3)
            throw new ArgumentException("Normal estimation needs at least three points.", nameof(pointCloud));

        var spacing = MeasureSpacing(points, Math.Min(Math.Max(1, neighborCount), count - 1));
        var grid = new PointGrid(points, spacing.MeanNearest);
        var (estimated, _) = PointCloudNormals.Estimate(points, neighborCount, grid);
        return new Tensor<double>(new[] { count, 3 }, new Vector<double>(estimated)).Cast<T>();
    }

    /// <summary>
    /// Reads the x, y, z channels of a [N, C] or [1, N, C] tensor (C &gt;= 3) into one double array.
    /// </summary>
    internal static double[] ReadPoints<T>(Tensor<T> tensor, string parameterName)
    {
        if (tensor is null)
            throw new ArgumentNullException(parameterName);

        var shape = tensor.Shape;
        int rows, channels;
        if (shape.Length == 2)
        {
            rows = shape[0];
            channels = shape[1];
        }
        else if (shape.Length == 3 && shape[0] == 1)
        {
            rows = shape[1];
            channels = shape[2];
        }
        else
        {
            throw new ArgumentException("Points must have shape [N, C] or [1, N, C] with C >= 3.", parameterName);
        }

        if (channels < 3)
            throw new ArgumentException("Points need at least three channels (x, y, z).", parameterName);

        double[] all = tensor.Cast<double>().ToArray();
        var points = new double[3 * rows];
        for (int i = 0; i < rows; i++)
        {
            for (int axis = 0; axis < 3; axis++)
            {
                double value = all[(i * channels) + axis];
                if (double.IsNaN(value) || double.IsInfinity(value))
                    throw new ArgumentException("Points must be finite.", parameterName);
                points[(3 * i) + axis] = value;
            }
        }

        return points;
    }

    /// <summary>
    /// The sampling density: the mean nearest-neighbour distance, the mean distance to the k-th neighbour, and each
    /// sample's surface area pi d_k^2 / (k + 1) (the area of the disc that holds its k + 1 nearest samples, divided
    /// among them).
    /// </summary>
    internal static (double MeanNearest, double MeanKth, double[] Areas) MeasureSpacing(double[] points, int k)
    {
        int count = points.Length / 3;
        var (_, size) = ScreenedPoissonReconstruction.BoundingCube(points, 1.0);
        var grid = new PointGrid(points, size / Math.Max(1.0, Math.Pow(count, 1.0 / 3.0)));
        double nearestSum = 0.0, kthSum = 0.0;
        var areas = new double[count];
        for (int i = 0; i < count; i++)
        {
            grid.Nearest(points[3 * i], points[(3 * i) + 1], points[(3 * i) + 2], k, i, out double[] distanceSquared);
            double nearest = Math.Sqrt(distanceSquared[0]);
            double kth = Math.Sqrt(distanceSquared[distanceSquared.Length - 1]);
            nearestSum += nearest;
            kthSum += kth;
            areas[i] = Math.PI * kth * kth / (distanceSquared.Length + 1);
        }

        double meanNearest = nearestSum / count;
        double meanKth = kthSum / count;
        if (!(meanNearest > 0.0))
            meanNearest = Math.Max(meanKth, size * 1e-6);
        if (!(meanKth > 0.0))
            meanKth = meanNearest;
        return (meanNearest, meanKth, areas);
    }

    /// <summary>The centroid of each sample's k-neighbourhood (Hoppe et al. 1992's tangent-plane centres).</summary>
    private static double[] TangentCenters(double[] points, int k, PointGrid grid)
    {
        int count = points.Length / 3;
        var centers = new double[points.Length];
        for (int i = 0; i < count; i++)
        {
            int[] near = grid.Nearest(points[3 * i], points[(3 * i) + 1], points[(3 * i) + 2], k, -1, out _);
            foreach (int j in near)
            {
                for (int axis = 0; axis < 3; axis++)
                    centers[(3 * i) + axis] += points[(3 * j) + axis] / near.Length;
            }
        }

        return centers;
    }

    private static void Normalize(double[] normals)
    {
        for (int i = 0; i < normals.Length; i += 3)
        {
            double length = Math.Sqrt((normals[i] * normals[i]) + (normals[i + 1] * normals[i + 1]) + (normals[i + 2] * normals[i + 2]));
            if (!(length > 0.0))
                throw new ArgumentException("Every normal must be non-zero.", nameof(normals));
            normals[i] /= length;
            normals[i + 1] /= length;
            normals[i + 2] /= length;
        }
    }
}
