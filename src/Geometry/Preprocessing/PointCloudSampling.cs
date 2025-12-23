using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Geometry.Preprocessing;

/// <summary>
/// Provides sampling utilities for point cloud data.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public static class PointCloudSampling<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Performs uniform random sampling on a point cloud.
    /// </summary>
    public static PointCloudData<T> UniformSample(PointCloudData<T> pointCloud, int numSamples, int? seed = null)
    {
        int numPoints = pointCloud.NumPoints;
        int numFeatures = pointCloud.NumFeatures;

        if (numSamples >= numPoints)
        {
            return pointCloud;
        }

        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : new Random();
        
        var indices = Enumerable.Range(0, numPoints).ToArray();
        for (int i = 0; i < numSamples; i++)
        {
            int j = random.Next(i, numPoints);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var sampledData = new T[numSamples * numFeatures];
        Vector<T>? sampledLabels = null;
        
        if (pointCloud.Labels != null)
        {
            var labelData = new T[numSamples];
            for (int i = 0; i < numSamples; i++)
            {
                labelData[i] = pointCloud.Labels[indices[i]];
            }
            sampledLabels = new Vector<T>(labelData);
        }

        for (int i = 0; i < numSamples; i++)
        {
            int srcIdx = indices[i];
            int dstBase = i * numFeatures;
            
            for (int f = 0; f < numFeatures; f++)
            {
                sampledData[dstBase + f] = pointCloud.Points[srcIdx, f];
            }
        }

        var tensor = new Tensor<T>(sampledData, [numSamples, numFeatures]);
        return new PointCloudData<T>(tensor, sampledLabels);
    }

    /// <summary>
    /// Performs farthest point sampling (FPS) on a point cloud.
    /// </summary>
    public static PointCloudData<T> FarthestPointSample(PointCloudData<T> pointCloud, int numSamples, int? seed = null)
    {
        int numPoints = pointCloud.NumPoints;
        int numFeatures = pointCloud.NumFeatures;

        if (numSamples >= numPoints)
        {
            return pointCloud;
        }

        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : new Random();
        
        var selectedIndices = new List<int>(numSamples);
        var minDistances = new double[numPoints];
        Array.Fill(minDistances, double.MaxValue);

        int firstIdx = random.Next(numPoints);
        selectedIndices.Add(firstIdx);
        UpdateDistances(pointCloud, firstIdx, minDistances);

        for (int i = 1; i < numSamples; i++)
        {
            int farthestIdx = 0;
            double maxMinDist = -1;

            for (int j = 0; j < numPoints; j++)
            {
                if (minDistances[j] > maxMinDist)
                {
                    maxMinDist = minDistances[j];
                    farthestIdx = j;
                }
            }

            selectedIndices.Add(farthestIdx);
            UpdateDistances(pointCloud, farthestIdx, minDistances);
        }

        var sampledData = new T[numSamples * numFeatures];
        Vector<T>? sampledLabels = null;
        
        if (pointCloud.Labels != null)
        {
            var labelData = new T[numSamples];
            for (int i = 0; i < numSamples; i++)
            {
                labelData[i] = pointCloud.Labels[selectedIndices[i]];
            }
            sampledLabels = new Vector<T>(labelData);
        }

        for (int i = 0; i < numSamples; i++)
        {
            int srcIdx = selectedIndices[i];
            int dstBase = i * numFeatures;

            for (int f = 0; f < numFeatures; f++)
            {
                sampledData[dstBase + f] = pointCloud.Points[srcIdx, f];
            }
        }

        var tensor = new Tensor<T>(sampledData, [numSamples, numFeatures]);
        return new PointCloudData<T>(tensor, sampledLabels);
    }

    /// <summary>
    /// Performs Poisson disk sampling on a point cloud.
    /// </summary>
    public static PointCloudData<T> PoissonDiskSample(
        PointCloudData<T> pointCloud,
        double minDistance,
        int maxSamples = 0,
        int? seed = null)
    {
        int numPoints = pointCloud.NumPoints;
        int numFeatures = pointCloud.NumFeatures;

        if (numPoints == 0)
        {
            return pointCloud;
        }

        var random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : new Random();
        double minDistSq = minDistance * minDistance;

        var indices = Enumerable.Range(0, numPoints).ToArray();
        for (int i = numPoints - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var selectedIndices = new List<int>();
        var selectedPositions = new List<(double X, double Y, double Z)>();

        foreach (int idx in indices)
        {
            if (maxSamples > 0 && selectedIndices.Count >= maxSamples)
            {
                break;
            }

            double x = NumOps.ToDouble(pointCloud.Points[idx, 0]);
            double y = NumOps.ToDouble(pointCloud.Points[idx, 1]);
            double z = NumOps.ToDouble(pointCloud.Points[idx, 2]);

            bool tooClose = false;
            foreach (var pos in selectedPositions)
            {
                double dx = x - pos.X;
                double dy = y - pos.Y;
                double dz = z - pos.Z;
                double distSq = dx * dx + dy * dy + dz * dz;

                if (distSq < minDistSq)
                {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose)
            {
                selectedIndices.Add(idx);
                selectedPositions.Add((x, y, z));
            }
        }

        int numSamples = selectedIndices.Count;
        var sampledData = new T[numSamples * numFeatures];
        Vector<T>? sampledLabels = null;
        
        if (pointCloud.Labels != null)
        {
            var labelData = new T[numSamples];
            for (int i = 0; i < numSamples; i++)
            {
                labelData[i] = pointCloud.Labels[selectedIndices[i]];
            }
            sampledLabels = new Vector<T>(labelData);
        }

        for (int i = 0; i < numSamples; i++)
        {
            int srcIdx = selectedIndices[i];
            int dstBase = i * numFeatures;

            for (int f = 0; f < numFeatures; f++)
            {
                sampledData[dstBase + f] = pointCloud.Points[srcIdx, f];
            }
        }

        var tensor = new Tensor<T>(sampledData, [numSamples, numFeatures]);
        return new PointCloudData<T>(tensor, sampledLabels);
    }

    /// <summary>
    /// Samples points by voxel grid downsampling.
    /// </summary>
    public static PointCloudData<T> VoxelGridSample(PointCloudData<T> pointCloud, double voxelSize)
    {
        int numPoints = pointCloud.NumPoints;
        int numFeatures = pointCloud.NumFeatures;

        if (numPoints == 0 || voxelSize <= 0)
        {
            return pointCloud;
        }

        var voxelMap = new Dictionary<(int, int, int), List<int>>();

        for (int i = 0; i < numPoints; i++)
        {
            double x = NumOps.ToDouble(pointCloud.Points[i, 0]);
            double y = NumOps.ToDouble(pointCloud.Points[i, 1]);
            double z = NumOps.ToDouble(pointCloud.Points[i, 2]);

            int vx = (int)Math.Floor(x / voxelSize);
            int vy = (int)Math.Floor(y / voxelSize);
            int vz = (int)Math.Floor(z / voxelSize);

            var key = (vx, vy, vz);
            if (!voxelMap.TryGetValue(key, out var indices))
            {
                indices = [];
                voxelMap[key] = indices;
            }
            indices.Add(i);
        }

        int numSamples = voxelMap.Count;
        var sampledData = new T[numSamples * numFeatures];
        Vector<T>? sampledLabels = null;
        T[]? labelData = pointCloud.Labels != null ? new T[numSamples] : null;

        int sampleIdx = 0;
        foreach (var (_, indices) in voxelMap)
        {
            int dstBase = sampleIdx * numFeatures;
            
            for (int f = 0; f < numFeatures; f++)
            {
                double sum = 0;
                foreach (int idx in indices)
                {
                    sum += NumOps.ToDouble(pointCloud.Points[idx, f]);
                }
                sampledData[dstBase + f] = NumOps.FromDouble(sum / indices.Count);
            }

            if (labelData != null && pointCloud.Labels != null)
            {
                var labelCounts = new Dictionary<double, int>();
                foreach (int idx in indices)
                {
                    double label = NumOps.ToDouble(pointCloud.Labels[idx]);
                    labelCounts.TryGetValue(label, out int count);
                    labelCounts[label] = count + 1;
                }
                double majorityLabel = labelCounts.OrderByDescending(kv => kv.Value).First().Key;
                labelData[sampleIdx] = NumOps.FromDouble(majorityLabel);
            }

            sampleIdx++;
        }

        if (labelData != null)
        {
            sampledLabels = new Vector<T>(labelData);
        }

        var tensor = new Tensor<T>(sampledData, [numSamples, numFeatures]);
        return new PointCloudData<T>(tensor, sampledLabels);
    }

    private static void UpdateDistances(PointCloudData<T> pointCloud, int newPointIdx, double[] minDistances)
    {
        int numPoints = pointCloud.NumPoints;
        double nx = NumOps.ToDouble(pointCloud.Points[newPointIdx, 0]);
        double ny = NumOps.ToDouble(pointCloud.Points[newPointIdx, 1]);
        double nz = NumOps.ToDouble(pointCloud.Points[newPointIdx, 2]);

        for (int i = 0; i < numPoints; i++)
        {
            double x = NumOps.ToDouble(pointCloud.Points[i, 0]);
            double y = NumOps.ToDouble(pointCloud.Points[i, 1]);
            double z = NumOps.ToDouble(pointCloud.Points[i, 2]);

            double dx = x - nx;
            double dy = y - ny;
            double dz = z - nz;
            double dist = dx * dx + dy * dy + dz * dz;

            if (dist < minDistances[i])
            {
                minDistances[i] = dist;
            }
        }

        minDistances[newPointIdx] = 0;
    }
}
