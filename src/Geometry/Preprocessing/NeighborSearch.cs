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
        var distances = new double[numPoints];

        for (int i = 0; i < numPoints; i++)
        {
            double qx = NumOps.ToDouble(pointCloud.Points[i, 0]);
            double qy = NumOps.ToDouble(pointCloud.Points[i, 1]);
            double qz = NumOps.ToDouble(pointCloud.Points[i, 2]);

            for (int j = 0; j < numPoints; j++)
            {
                double x = NumOps.ToDouble(pointCloud.Points[j, 0]);
                double y = NumOps.ToDouble(pointCloud.Points[j, 1]);
                double z = NumOps.ToDouble(pointCloud.Points[j, 2]);

                double dx = x - qx;
                double dy = y - qy;
                double dz = z - qz;
                distances[j] = dx * dx + dy * dy + dz * dz;
            }

            var indices = Enumerable.Range(0, numPoints).ToArray();
            PartialSort(indices, distances, k);

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
    public static List<int>[] RadiusSearch(PointCloudData<T> pointCloud, PointCloudData<T> queryPoints, double radius)
    {
        int numPoints = pointCloud.NumPoints;
        int numQueries = queryPoints.NumPoints;
        double radiusSq = radius * radius;

        var result = new List<int>[numQueries];

        for (int i = 0; i < numQueries; i++)
        {
            result[i] = [];
            double qx = NumOps.ToDouble(queryPoints.Points[i, 0]);
            double qy = NumOps.ToDouble(queryPoints.Points[i, 1]);
            double qz = NumOps.ToDouble(queryPoints.Points[i, 2]);

            for (int j = 0; j < numPoints; j++)
            {
                double x = NumOps.ToDouble(pointCloud.Points[j, 0]);
                double y = NumOps.ToDouble(pointCloud.Points[j, 1]);
                double z = NumOps.ToDouble(pointCloud.Points[j, 2]);

                double dx = x - qx;
                double dy = y - qy;
                double dz = z - qz;
                double distSq = dx * dx + dy * dy + dz * dz;

                if (distSq <= radiusSq)
                {
                    result[i].Add(j);
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Performs ball query for PointNet++ style grouping.
    /// </summary>
    public static int[,] BallQuery(
        PointCloudData<T> pointCloud,
        PointCloudData<T> centroids,
        double radius,
        int maxSamples)
    {
        int numPoints = pointCloud.NumPoints;
        int numCentroids = centroids.NumPoints;
        double radiusSq = radius * radius;

        var result = new int[numCentroids, maxSamples];

        for (int i = 0; i < numCentroids; i++)
        {
            double cx = NumOps.ToDouble(centroids.Points[i, 0]);
            double cy = NumOps.ToDouble(centroids.Points[i, 1]);
            double cz = NumOps.ToDouble(centroids.Points[i, 2]);

            var neighbors = new List<(int Index, double DistSq)>();

            for (int j = 0; j < numPoints; j++)
            {
                double x = NumOps.ToDouble(pointCloud.Points[j, 0]);
                double y = NumOps.ToDouble(pointCloud.Points[j, 1]);
                double z = NumOps.ToDouble(pointCloud.Points[j, 2]);

                double dx = x - cx;
                double dy = y - cy;
                double dz = z - cz;
                double distSq = dx * dx + dy * dy + dz * dz;

                if (distSq <= radiusSq)
                {
                    neighbors.Add((j, distSq));
                }
            }

            neighbors.Sort((a, b) => a.DistSq.CompareTo(b.DistSq));
            int count = Math.Min(neighbors.Count, maxSamples);

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
                    result[i, n] = neighbors[n].Index;
                }
                for (int n = count; n < maxSamples; n++)
                {
                    result[i, n] = neighbors[0].Index;
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Computes a distance matrix between all pairs of points.
    /// </summary>
    public static double[,] ComputeDistanceMatrix(PointCloudData<T> pointCloud)
    {
        int numPoints = pointCloud.NumPoints;
        var result = new double[numPoints, numPoints];

        for (int i = 0; i < numPoints; i++)
        {
            double xi = NumOps.ToDouble(pointCloud.Points[i, 0]);
            double yi = NumOps.ToDouble(pointCloud.Points[i, 1]);
            double zi = NumOps.ToDouble(pointCloud.Points[i, 2]);

            result[i, i] = 0;

            for (int j = i + 1; j < numPoints; j++)
            {
                double xj = NumOps.ToDouble(pointCloud.Points[j, 0]);
                double yj = NumOps.ToDouble(pointCloud.Points[j, 1]);
                double zj = NumOps.ToDouble(pointCloud.Points[j, 2]);

                double dx = xi - xj;
                double dy = yi - yj;
                double dz = zi - zj;
                double dist = Math.Sqrt(dx * dx + dy * dy + dz * dz);

                result[i, j] = dist;
                result[j, i] = dist;
            }
        }

        return result;
    }

    private static void PartialSort(int[] indices, double[] distances, int k)
    {
        for (int i = 0; i < k; i++)
        {
            int minIdx = i;
            double minDist = distances[indices[i]];

            for (int j = i + 1; j < indices.Length; j++)
            {
                if (distances[indices[j]] < minDist)
                {
                    minIdx = j;
                    minDist = distances[indices[j]];
                }
            }

            if (minIdx != i)
            {
                (indices[i], indices[minIdx]) = (indices[minIdx], indices[i]);
            }
        }
    }
}
