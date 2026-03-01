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
        int numFeatures = pointCloud.NumFeatures;

        if (numPoints == 0)
        {
            return pointCloud;
        }

        // Compute centroid (only for XYZ coordinates)
        T sumX = NumOps.Zero;
        T sumY = NumOps.Zero;
        T sumZ = NumOps.Zero;

        for (int i = 0; i < numPoints; i++)
        {
            sumX = NumOps.Add(sumX, pointCloud.Points[i, 0]);
            sumY = NumOps.Add(sumY, pointCloud.Points[i, 1]);
            sumZ = NumOps.Add(sumZ, pointCloud.Points[i, 2]);
        }

        T count = NumOps.FromDouble(numPoints);
        T centroidX = NumOps.Divide(sumX, count);
        T centroidY = NumOps.Divide(sumY, count);
        T centroidZ = NumOps.Divide(sumZ, count);

        // Create centered point cloud
        var centeredData = new T[numPoints * numFeatures];
        for (int i = 0; i < numPoints; i++)
        {
            int baseIdx = i * numFeatures;
            centeredData[baseIdx] = NumOps.Subtract(pointCloud.Points[i, 0], centroidX);
            centeredData[baseIdx + 1] = NumOps.Subtract(pointCloud.Points[i, 1], centroidY);
            centeredData[baseIdx + 2] = NumOps.Subtract(pointCloud.Points[i, 2], centroidZ);

            for (int f = 3; f < numFeatures; f++)
            {
                centeredData[baseIdx + f] = pointCloud.Points[i, f];
            }
        }

        var tensor = new Tensor<T>(centeredData, [numPoints, numFeatures]);
        return new PointCloudData<T>(tensor, pointCloud.Labels);
    }

    /// <summary>
    /// Scales a point cloud to fit within a unit sphere (radius = 1).
    /// </summary>
    public static PointCloudData<T> ScaleToUnitSphere(PointCloudData<T> pointCloud, bool center = true)
    {
        var workingCloud = center ? Center(pointCloud) : pointCloud;
        int numPoints = workingCloud.NumPoints;
        int numFeatures = workingCloud.NumFeatures;

        if (numPoints == 0)
        {
            return workingCloud;
        }

        // Find maximum distance from origin
        T maxDistSq = NumOps.Zero;
        for (int i = 0; i < numPoints; i++)
        {
            T x = workingCloud.Points[i, 0];
            T y = workingCloud.Points[i, 1];
            T z = workingCloud.Points[i, 2];
            T distSq = NumOps.Add(NumOps.Add(
                NumOps.Multiply(x, x),
                NumOps.Multiply(y, y)),
                NumOps.Multiply(z, z));

            if (NumOps.GreaterThan(distSq, maxDistSq))
            {
                maxDistSq = distSq;
            }
        }

        double maxDist = Math.Sqrt(NumOps.ToDouble(maxDistSq));
        if (maxDist < 1e-10)
        {
            return workingCloud;
        }

        T scale = NumOps.FromDouble(1.0 / maxDist);

        var scaledData = new T[numPoints * numFeatures];
        for (int i = 0; i < numPoints; i++)
        {
            int baseIdx = i * numFeatures;
            scaledData[baseIdx] = NumOps.Multiply(workingCloud.Points[i, 0], scale);
            scaledData[baseIdx + 1] = NumOps.Multiply(workingCloud.Points[i, 1], scale);
            scaledData[baseIdx + 2] = NumOps.Multiply(workingCloud.Points[i, 2], scale);

            for (int f = 3; f < numFeatures; f++)
            {
                scaledData[baseIdx + f] = workingCloud.Points[i, f];
            }
        }

        var tensor = new Tensor<T>(scaledData, [numPoints, numFeatures]);
        return new PointCloudData<T>(tensor, workingCloud.Labels);
    }

    /// <summary>
    /// Scales a point cloud to fit within a unit cube [-0.5, 0.5]^3.
    /// </summary>
    public static PointCloudData<T> ScaleToUnitCube(PointCloudData<T> pointCloud, bool center = true)
    {
        var workingCloud = center ? Center(pointCloud) : pointCloud;
        int numPoints = workingCloud.NumPoints;
        int numFeatures = workingCloud.NumFeatures;

        if (numPoints == 0)
        {
            return workingCloud;
        }

        T minX = workingCloud.Points[0, 0];
        T minY = workingCloud.Points[0, 1];
        T minZ = workingCloud.Points[0, 2];
        T maxX = minX;
        T maxY = minY;
        T maxZ = minZ;

        for (int i = 1; i < numPoints; i++)
        {
            T x = workingCloud.Points[i, 0];
            T y = workingCloud.Points[i, 1];
            T z = workingCloud.Points[i, 2];

            if (NumOps.LessThan(x, minX)) minX = x;
            if (NumOps.LessThan(y, minY)) minY = y;
            if (NumOps.LessThan(z, minZ)) minZ = z;
            if (NumOps.GreaterThan(x, maxX)) maxX = x;
            if (NumOps.GreaterThan(y, maxY)) maxY = y;
            if (NumOps.GreaterThan(z, maxZ)) maxZ = z;
        }

        double rangeX = NumOps.ToDouble(NumOps.Subtract(maxX, minX));
        double rangeY = NumOps.ToDouble(NumOps.Subtract(maxY, minY));
        double rangeZ = NumOps.ToDouble(NumOps.Subtract(maxZ, minZ));
        double maxRange = Math.Max(Math.Max(rangeX, rangeY), rangeZ);

        if (maxRange < 1e-10)
        {
            return workingCloud;
        }

        // Re-center at bounding box midpoint so data is symmetric around 0,
        // then scale by 1/maxRange to guarantee all points fit in [-0.5, 0.5]^3
        T midX = NumOps.FromDouble((NumOps.ToDouble(minX) + NumOps.ToDouble(maxX)) / 2.0);
        T midY = NumOps.FromDouble((NumOps.ToDouble(minY) + NumOps.ToDouble(maxY)) / 2.0);
        T midZ = NumOps.FromDouble((NumOps.ToDouble(minZ) + NumOps.ToDouble(maxZ)) / 2.0);
        T scale = NumOps.FromDouble(1.0 / maxRange);

        var scaledData = new T[numPoints * numFeatures];
        for (int i = 0; i < numPoints; i++)
        {
            int baseIdx = i * numFeatures;
            scaledData[baseIdx] = NumOps.Multiply(NumOps.Subtract(workingCloud.Points[i, 0], midX), scale);
            scaledData[baseIdx + 1] = NumOps.Multiply(NumOps.Subtract(workingCloud.Points[i, 1], midY), scale);
            scaledData[baseIdx + 2] = NumOps.Multiply(NumOps.Subtract(workingCloud.Points[i, 2], midZ), scale);

            for (int f = 3; f < numFeatures; f++)
            {
                scaledData[baseIdx + f] = workingCloud.Points[i, f];
            }
        }

        var tensor = new Tensor<T>(scaledData, [numPoints, numFeatures]);
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

        T scale = NumOps.FromDouble(1.0 / maxColorValue);
        var normalizedData = new T[numPoints * numFeatures];

        for (int i = 0; i < numPoints; i++)
        {
            int baseIdx = i * numFeatures;

            for (int f = 0; f < colorOffset; f++)
            {
                normalizedData[baseIdx + f] = pointCloud.Points[i, f];
            }

            for (int c = 0; c < 3; c++)
            {
                T colorVal = pointCloud.Points[i, colorOffset + c];
                normalizedData[baseIdx + colorOffset + c] = NumOps.Multiply(colorVal, scale);
            }

            for (int f = colorOffset + 3; f < numFeatures; f++)
            {
                normalizedData[baseIdx + f] = pointCloud.Points[i, f];
            }
        }

        var tensor = new Tensor<T>(normalizedData, [numPoints, numFeatures]);
        return new PointCloudData<T>(tensor, pointCloud.Labels);
    }
}
