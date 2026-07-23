using AiDotNet.PointCloud.Data;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;

namespace AiDotNet.Geometry.Preprocessing;

/// <summary>
/// Provides quality metrics for comparing 3D geometry (point clouds, meshes).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public static class GeometryMetrics<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Computes the Chamfer Distance between two point clouds.
    /// </summary>
    public static double ChamferDistance(PointCloudData<T> source, PointCloudData<T> target, bool squared = false)
    {
        int numSource = source.NumPoints;
        int numTarget = target.NumPoints;

        if (numSource == 0 || numTarget == 0)
        {
            return 0;
        }

        double sumSourceToTarget = 0;
        for (int i = 0; i < numSource; i++)
        {
            double sx = NumOps.ToDouble(source.Points[i, 0]);
            double sy = NumOps.ToDouble(source.Points[i, 1]);
            double sz = NumOps.ToDouble(source.Points[i, 2]);

            double minDistSq = double.MaxValue;
            for (int j = 0; j < numTarget; j++)
            {
                double tx = NumOps.ToDouble(target.Points[j, 0]);
                double ty = NumOps.ToDouble(target.Points[j, 1]);
                double tz = NumOps.ToDouble(target.Points[j, 2]);

                double dx = sx - tx;
                double dy = sy - ty;
                double dz = sz - tz;
                double distSq = dx * dx + dy * dy + dz * dz;

                if (distSq < minDistSq)
                {
                    minDistSq = distSq;
                }
            }
            sumSourceToTarget += squared ? minDistSq : Math.Sqrt(minDistSq);
        }

        double sumTargetToSource = 0;
        for (int i = 0; i < numTarget; i++)
        {
            double tx = NumOps.ToDouble(target.Points[i, 0]);
            double ty = NumOps.ToDouble(target.Points[i, 1]);
            double tz = NumOps.ToDouble(target.Points[i, 2]);

            double minDistSq = double.MaxValue;
            for (int j = 0; j < numSource; j++)
            {
                double sx = NumOps.ToDouble(source.Points[j, 0]);
                double sy = NumOps.ToDouble(source.Points[j, 1]);
                double sz = NumOps.ToDouble(source.Points[j, 2]);

                double dx = tx - sx;
                double dy = ty - sy;
                double dz = tz - sz;
                double distSq = dx * dx + dy * dy + dz * dz;

                if (distSq < minDistSq)
                {
                    minDistSq = distSq;
                }
            }
            sumTargetToSource += squared ? minDistSq : Math.Sqrt(minDistSq);
        }

        return (sumSourceToTarget / numSource + sumTargetToSource / numTarget) / 2;
    }

    /// <summary>
    /// Computes the F-Score between two point clouds at a given threshold.
    /// </summary>
    public static (double FScore, double Precision, double Recall) FScore(
        PointCloudData<T> prediction,
        PointCloudData<T> groundTruth,
        double threshold)
    {
        int numPred = prediction.NumPoints;
        int numGT = groundTruth.NumPoints;
        double thresholdSq = threshold * threshold;

        if (numPred == 0 || numGT == 0)
        {
            return (0, 0, 0);
        }

        int truePositivesPrecision = 0;
        for (int i = 0; i < numPred; i++)
        {
            double px = NumOps.ToDouble(prediction.Points[i, 0]);
            double py = NumOps.ToDouble(prediction.Points[i, 1]);
            double pz = NumOps.ToDouble(prediction.Points[i, 2]);

            for (int j = 0; j < numGT; j++)
            {
                double gx = NumOps.ToDouble(groundTruth.Points[j, 0]);
                double gy = NumOps.ToDouble(groundTruth.Points[j, 1]);
                double gz = NumOps.ToDouble(groundTruth.Points[j, 2]);

                double dx = px - gx;
                double dy = py - gy;
                double dz = pz - gz;
                double distSq = dx * dx + dy * dy + dz * dz;

                if (distSq <= thresholdSq)
                {
                    truePositivesPrecision++;
                    break;
                }
            }
        }

        int truePositivesRecall = 0;
        for (int i = 0; i < numGT; i++)
        {
            double gx = NumOps.ToDouble(groundTruth.Points[i, 0]);
            double gy = NumOps.ToDouble(groundTruth.Points[i, 1]);
            double gz = NumOps.ToDouble(groundTruth.Points[i, 2]);

            for (int j = 0; j < numPred; j++)
            {
                double px = NumOps.ToDouble(prediction.Points[j, 0]);
                double py = NumOps.ToDouble(prediction.Points[j, 1]);
                double pz = NumOps.ToDouble(prediction.Points[j, 2]);

                double dx = gx - px;
                double dy = gy - py;
                double dz = gz - pz;
                double distSq = dx * dx + dy * dy + dz * dz;

                if (distSq <= thresholdSq)
                {
                    truePositivesRecall++;
                    break;
                }
            }
        }

        double precision = (double)truePositivesPrecision / numPred;
        double recall = (double)truePositivesRecall / numGT;

        double fScore = 0;
        if (precision + recall > 0)
        {
            fScore = 2 * precision * recall / (precision + recall);
        }

        return (fScore, precision, recall);
    }

    /// <summary>
    /// Computes the Hausdorff Distance between two point clouds.
    /// </summary>
    public static double HausdorffDistance(PointCloudData<T> source, PointCloudData<T> target)
    {
        int numSource = source.NumPoints;
        int numTarget = target.NumPoints;

        if (numSource == 0 || numTarget == 0)
        {
            return 0;
        }

        double maxMinSourceToTarget = 0;
        for (int i = 0; i < numSource; i++)
        {
            double sx = NumOps.ToDouble(source.Points[i, 0]);
            double sy = NumOps.ToDouble(source.Points[i, 1]);
            double sz = NumOps.ToDouble(source.Points[i, 2]);

            double minDist = double.MaxValue;
            for (int j = 0; j < numTarget; j++)
            {
                double tx = NumOps.ToDouble(target.Points[j, 0]);
                double ty = NumOps.ToDouble(target.Points[j, 1]);
                double tz = NumOps.ToDouble(target.Points[j, 2]);

                double dx = sx - tx;
                double dy = sy - ty;
                double dz = sz - tz;
                double dist = Math.Sqrt(dx * dx + dy * dy + dz * dz);

                if (dist < minDist)
                {
                    minDist = dist;
                }
            }

            if (minDist > maxMinSourceToTarget)
            {
                maxMinSourceToTarget = minDist;
            }
        }

        double maxMinTargetToSource = 0;
        for (int i = 0; i < numTarget; i++)
        {
            double tx = NumOps.ToDouble(target.Points[i, 0]);
            double ty = NumOps.ToDouble(target.Points[i, 1]);
            double tz = NumOps.ToDouble(target.Points[i, 2]);

            double minDist = double.MaxValue;
            for (int j = 0; j < numSource; j++)
            {
                double sx = NumOps.ToDouble(source.Points[j, 0]);
                double sy = NumOps.ToDouble(source.Points[j, 1]);
                double sz = NumOps.ToDouble(source.Points[j, 2]);

                double dx = tx - sx;
                double dy = ty - sy;
                double dz = tz - sz;
                double dist = Math.Sqrt(dx * dx + dy * dy + dz * dz);

                if (dist < minDist)
                {
                    minDist = dist;
                }
            }

            if (minDist > maxMinTargetToSource)
            {
                maxMinTargetToSource = minDist;
            }
        }

        return Math.Max(maxMinSourceToTarget, maxMinTargetToSource);
    }
}
