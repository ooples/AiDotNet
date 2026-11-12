using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Geometry.Preprocessing;

/// <summary>
/// Provides standardization and normalization utilities for 3D point cloud data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public static class PointCloudNormalization<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Centers a point cloud at the origin by subtracting the centroid.
    /// </summary>
    public static PointCloudData<T> Center(PointCloudData<T> pointCloud)
    {
        int numPoints = pointCloud.NumPoints;

        if (numPoints == 0)
        {
            return pointCloud;
        }

        // Compute centroid (only for XYZ coordinates)
        var engine = AiDotNetEngine.Current;
        var positions = GeometryEngineOps<T>.Positions(pointCloud.Points);
        var centroid = engine.TensorDivideScalar(
            engine.ReduceSum(positions, new[] { 0 }, true), NumOps.FromDouble(numPoints));

        // Create centered point cloud
        var centered = engine.TensorSubtract(positions, engine.TensorBroadcastTo(centroid, positions.Shape.ToArray()));
        var tensor = GeometryEngineOps<T>.WithPositions(pointCloud.Points, centered);
        return new PointCloudData<T>(tensor, pointCloud.Labels);
    }

    /// <summary>
    /// Scales a point cloud to fit within a unit sphere (radius = 1).
    /// </summary>
    public static PointCloudData<T> ScaleToUnitSphere(PointCloudData<T> pointCloud, bool center = true)
    {
        var workingCloud = center ? Center(pointCloud) : pointCloud;
        int numPoints = workingCloud.NumPoints;

        if (numPoints == 0)
        {
            return workingCloud;
        }

        // Find maximum distance from origin; the running maximum started at zero and skipped NaN.
        var engine = AiDotNetEngine.Current;
        var positions = GeometryEngineOps<T>.Positions(workingCloud.Points);
        var distancesSq = GeometryEngineOps<T>.RowSquaredNorms(positions);
        var finiteDistancesSq = engine.TensorWhere(
            engine.TensorIsNan(distancesSq), engine.TensorZerosLike(distancesSq), distancesSq);
        T maxDistSq = engine.TensorMaxValue(engine.TensorMax(finiteDistancesSq, NumOps.Zero));

        T maxDist = NumOps.Sqrt(maxDistSq);
        if (NumOps.LessThan(maxDist, NumOps.FromDouble(1e-10)))
        {
            return workingCloud;
        }

        T scale = NumOps.Divide(NumOps.One, maxDist);
        var scaled = engine.TensorMultiplyScalar(positions, scale);
        var tensor = GeometryEngineOps<T>.WithPositions(workingCloud.Points, scaled);
        return new PointCloudData<T>(tensor, workingCloud.Labels);
    }

    /// <summary>
    /// Scales a point cloud to fit within a unit cube [-0.5, 0.5]^3.
    /// </summary>
    public static PointCloudData<T> ScaleToUnitCube(PointCloudData<T> pointCloud, bool center = true)
    {
        var workingCloud = center ? Center(pointCloud) : pointCloud;
        int numPoints = workingCloud.NumPoints;

        if (numPoints == 0)
        {
            return workingCloud;
        }

        var engine = AiDotNetEngine.Current;
        var positions = GeometryEngineOps<T>.Positions(workingCloud.Points);
        var (min, max) = GeometryEngineOps<T>.RunningColumnBounds(positions);

        // Math.Max propagates NaN, so a NaN range makes the largest range NaN (and the guard below false).
        var ranges = engine.TensorSubtract(max, min);
        T maxRange = GeometryEngineOps<T>.PropagatingMax(ranges);

        if (NumOps.LessThan(maxRange, NumOps.FromDouble(1e-10)))
        {
            return workingCloud;
        }

        // Re-center at bounding box midpoint so data is symmetric around 0,
        // then scale by 1/maxRange to guarantee all points fit in [-0.5, 0.5]^3
        var mid = engine.TensorDivideScalar(engine.TensorAdd(min, max), NumOps.FromDouble(2.0)).Reshape(1, 3);
        T scale = NumOps.Divide(NumOps.One, maxRange);

        var shifted = engine.TensorSubtract(positions, engine.TensorBroadcastTo(mid, positions.Shape.ToArray()));
        var scaled = engine.TensorMultiplyScalar(shifted, scale);
        var tensor = GeometryEngineOps<T>.WithPositions(workingCloud.Points, scaled);
        return new PointCloudData<T>(tensor, workingCloud.Labels);
    }

    /// <summary>
    /// Normalizes color values to the range [0, 1].
    /// </summary>
    public static PointCloudData<T> NormalizeColors(
        PointCloudData<T> pointCloud,
        int colorOffset = 3,
        double maxColorValue = 255.0)
    {
        int numPoints = pointCloud.NumPoints;
        int numFeatures = pointCloud.NumFeatures;

        if (colorOffset + 3 > numFeatures)
        {
            return pointCloud;
        }

        if (numPoints == 0)
        {
            return new PointCloudData<T>(new Tensor<T>(new[] { 0, numFeatures }), pointCloud.Labels);
        }

        var engine = AiDotNetEngine.Current;
        T scale = NumOps.FromDouble(1.0 / maxColorValue);
        var points = pointCloud.Points;
        var parts = new List<Tensor<T>>();
        if (colorOffset > 0)
        {
            parts.Add(engine.TensorSlice(points, new[] { 0, 0 }, new[] { numPoints, colorOffset }));
        }

        var colors = engine.TensorSlice(points, new[] { 0, colorOffset }, new[] { numPoints, 3 });
        parts.Add(engine.TensorMultiplyScalar(colors, scale));

        int trailing = numFeatures - colorOffset - 3;
        if (trailing > 0)
        {
            parts.Add(engine.TensorSlice(points, new[] { 0, colorOffset + 3 }, new[] { numPoints, trailing }));
        }

        var tensor = engine.TensorConcatenate(parts.ToArray(), 1);
        return new PointCloudData<T>(tensor, pointCloud.Labels);
    }
}
