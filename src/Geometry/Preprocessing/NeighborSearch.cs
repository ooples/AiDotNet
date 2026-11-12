using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Geometry.Preprocessing;

/// <summary>
/// Provides neighbor search utilities for point cloud data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public static class NeighborSearch<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Finds the k nearest neighbors for each point in the cloud.
    /// </summary>
    public static int[,] KNearestNeighbors(PointCloudData<T> pointCloud, int k)
    {
        int numPoints = pointCloud.NumPoints;
        k = Math.Min(k, numPoints);

        var result = new int[numPoints, k];
        if (numPoints == 0)
        {
            return result;
        }

        var keys = GeometryEngineOps<T>.DenseRowRanks(
            GeometryEngineOps<T>.PairwiseSquaredDistances(pointCloud.Points, pointCloud.Points));
        var indices = new int[numPoints];

        for (int i = 0; i < numPoints; i++)
        {
            for (int j = 0; j < numPoints; j++)
            {
                indices[j] = j;
            }

            PartialSort(indices, keys, i * numPoints, k);

            for (int n = 0; n < k; n++)
            {
                result[i, n] = indices[n];
            }
        }

        return result;
    }

    /// <summary>
    /// Finds all points within a given radius of each query point.
    /// </summary>
    public static List<int>[] RadiusSearch(PointCloudData<T> pointCloud, PointCloudData<T> queryPoints, T radius)
    {
        int numPoints = pointCloud.NumPoints;
        int numQueries = queryPoints.NumPoints;

        var result = new List<int>[numQueries];
        for (int i = 0; i < numQueries; i++)
        {
            result[i] = [];
        }

        if (numPoints == 0 || numQueries == 0)
        {
            return result;
        }

        var engine = AiDotNetEngine.Current;
        var distancesSq = GeometryEngineOps<T>.PairwiseSquaredDistances(queryPoints.Points, pointCloud.Points);
        var radiusSq = engine.TensorFullLike(distancesSq, NumOps.Multiply(radius, radius));

        // Nonzero pairs come back row-major, so each query's neighbours are listed in point order.
        var pairs = GeometryEngineOps<T>.NonzeroPairs(engine.TensorLessEqual(distancesSq, radiusSq));
        for (int p = 0; p + 1 < pairs.Length; p += 2)
        {
            result[pairs[p]].Add(pairs[p + 1]);
        }

        return result;
    }

    /// <summary>
    /// Performs ball query for PointNet++ style grouping.
    /// </summary>
    public static int[,] BallQuery(
        PointCloudData<T> pointCloud,
        PointCloudData<T> centroids,
        T radius,
        int maxSamples)
    {
        int numPoints = pointCloud.NumPoints;
        int numCentroids = centroids.NumPoints;

        var result = new int[numCentroids, maxSamples];
        var neighbors = new List<(int Index, int Key)>[numCentroids];
        for (int i = 0; i < numCentroids; i++)
        {
            neighbors[i] = [];
        }

        if (numPoints > 0 && numCentroids > 0)
        {
            var engine = AiDotNetEngine.Current;
            var distancesSq = GeometryEngineOps<T>.PairwiseSquaredDistances(centroids.Points, pointCloud.Points);
            var radiusSq = engine.TensorFullLike(distancesSq, NumOps.Multiply(radius, radius));
            var keys = GeometryEngineOps<T>.DenseRowRanks(distancesSq);
            var pairs = GeometryEngineOps<T>.NonzeroPairs(engine.TensorLessEqual(distancesSq, radiusSq));
            for (int p = 0; p + 1 < pairs.Length; p += 2)
            {
                int centroid = pairs[p];
                int point = pairs[p + 1];
                neighbors[centroid].Add((point, keys[centroid * numPoints + point]));
            }
        }

        for (int i = 0; i < numCentroids; i++)
        {
            // The keys order exactly like the squared distances, so List.Sort produces the same permutation.
            var candidates = neighbors[i];
            candidates.Sort((a, b) => a.Key.CompareTo(b.Key));
            int count = Math.Min(candidates.Count, maxSamples);

            if (count == 0)
            {
                for (int n = 0; n < maxSamples; n++)
                {
                    result[i, n] = 0;
                }
            }
            else
            {
                for (int n = 0; n < count; n++)
                {
                    result[i, n] = candidates[n].Index;
                }
                for (int n = count; n < maxSamples; n++)
                {
                    result[i, n] = candidates[0].Index;
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Computes a distance matrix between all pairs of points.
    /// </summary>
    public static Tensor<T> ComputeDistanceMatrix(PointCloudData<T> pointCloud)
    {
        int numPoints = pointCloud.NumPoints;
        if (numPoints == 0)
        {
            return new Tensor<T>(new[] { 0, 0 });
        }

        var engine = AiDotNetEngine.Current;
        var distances = engine.TensorSqrt(
            GeometryEngineOps<T>.PairwiseSquaredDistances(pointCloud.Points, pointCloud.Points));

        // The diagonal is zero by definition, even for points whose coordinates are NaN.
        return engine.TensorWhere(engine.TensorEye<T>(numPoints), engine.TensorZerosLike(distances), distances);
    }

    private static void PartialSort(int[] indices, int[] keys, int keyOffset, int k)
    {
        // Selection sort on order-preserving keys: a NaN distance (key -1) neither wins nor is beaten,
        // exactly like the "distance < best" comparison on the values.
        for (int i = 0; i < k; i++)
        {
            int minIdx = i;
            int minKey = keys[keyOffset + indices[i]];

            for (int j = i + 1; j < indices.Length; j++)
            {
                int key = keys[keyOffset + indices[j]];
                if (key >= 0 && minKey >= 0 && key < minKey)
                {
                    minIdx = j;
                    minKey = key;
                }
            }

            if (minIdx != i)
            {
                (indices[i], indices[minIdx]) = (indices[minIdx], indices[i]);
            }
        }
    }
}
