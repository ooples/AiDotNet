using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.MetaLearning.Data;

/// <summary>
/// Estimates the difficulty of a meta-learning task based on geometric properties of the data:
/// inter-class separation, intra-class variance, and support/query alignment.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This utility looks at the support and query data and estimates
/// how hard the task is. Tasks where classes overlap a lot are harder than tasks where classes
/// are clearly separated. This estimate can feed into curriculum or dynamic samplers.</para>
/// <para><b>Reference:</b> Adaptive Task Sampling for Meta-Learning (2024).</para>
/// </remarks>
public static class TaskDifficultyEstimator<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Estimates difficulty of a task in [0, 1] based on support set geometry.
    /// 0 = trivially easy (well-separated classes), 1 = very hard (overlapping classes).
    /// </summary>
    /// <param name="supportX">Support set features as a flat vector (row-major, numExamples x featureDim).</param>
    /// <param name="supportY">Support set labels.</param>
    /// <param name="numWays">Number of classes.</param>
    /// <param name="numShots">Number of shots per class.</param>
    /// <returns>Difficulty score in [0, 1].</returns>
    public static double EstimateDifficulty(Vector<T> supportX, Vector<T> supportY, int numWays, int numShots)
    {
        if (supportX.Length == 0 || supportY.Length == 0 || numWays < 2)
            return 0.5;

        int numExamples = supportY.Length;
        int featureDim = Math.Max(1, supportX.Length / numExamples);

        // 1. Compute class centroids
        var centroids = new double[numWays][];
        var classCounts = new int[numWays];
        for (int c = 0; c < numWays; c++)
            centroids[c] = new double[featureDim];

        for (int i = 0; i < numExamples; i++)
        {
            int classIdx = (int)Math.Round(NumOps.ToDouble(supportY[i])) % numWays;
            if (classIdx < 0) classIdx += numWays;
            classCounts[classIdx]++;
            int start = i * featureDim;
            for (int d = 0; d < featureDim && start + d < supportX.Length; d++)
                centroids[classIdx][d] += NumOps.ToDouble(supportX[start + d]);
        }

        for (int c = 0; c < numWays; c++)
        {
            if (classCounts[c] > 0)
                for (int d = 0; d < featureDim; d++)
                    centroids[c][d] /= classCounts[c];
        }

        // 2. Inter-class distance (average pairwise centroid distance)
        double interClassDist = 0;
        int pairCount = 0;
        for (int i = 0; i < numWays; i++)
        {
            for (int j = i + 1; j < numWays; j++)
            {
                double dist = 0;
                for (int d = 0; d < featureDim; d++)
                {
                    double diff = centroids[i][d] - centroids[j][d];
                    dist += diff * diff;
                }
                interClassDist += Math.Sqrt(dist);
                pairCount++;
            }
        }
        if (pairCount > 0) interClassDist /= pairCount;

        // 3. Intra-class variance (average distance from examples to their centroid)
        double intraClassVar = 0;
        int totalIntra = 0;
        for (int i = 0; i < numExamples; i++)
        {
            int classIdx = (int)Math.Round(NumOps.ToDouble(supportY[i])) % numWays;
            if (classIdx < 0) classIdx += numWays;
            int start = i * featureDim;
            double dist = 0;
            for (int d = 0; d < featureDim && start + d < supportX.Length; d++)
            {
                double diff = NumOps.ToDouble(supportX[start + d]) - centroids[classIdx][d];
                dist += diff * diff;
            }
            intraClassVar += Math.Sqrt(dist);
            totalIntra++;
        }
        if (totalIntra > 0) intraClassVar /= totalIntra;

        // 4. Difficulty = intra-class variance / (inter-class distance + epsilon)
        //    High intra / low inter = hard; low intra / high inter = easy
        double ratio = intraClassVar / (interClassDist + 1e-8);

        // Map through sigmoid to [0, 1]
        double difficulty = 1.0 / (1.0 + Math.Exp(-(ratio - 1.0) * 3.0));
        return difficulty;
    }

    /// <summary>
    /// Estimates difficulty using the Fisher discriminant ratio:
    /// difficulty = 1 - (between-class variance / total variance).
    /// </summary>
    /// <param name="supportX">Support features (flat vector).</param>
    /// <param name="supportY">Support labels.</param>
    /// <param name="numWays">Number of classes.</param>
    /// <returns>Difficulty score in [0, 1].</returns>
    public static double EstimateFisherDifficulty(Vector<T> supportX, Vector<T> supportY, int numWays)
    {
        if (supportX.Length == 0 || supportY.Length == 0 || numWays < 2)
            return 0.5;

        int n = supportY.Length;
        int featureDim = Math.Max(1, supportX.Length / n);

        // Grand mean
        var grandMean = new double[featureDim];
        for (int i = 0; i < n; i++)
        {
            int start = i * featureDim;
            for (int d = 0; d < featureDim && start + d < supportX.Length; d++)
                grandMean[d] += NumOps.ToDouble(supportX[start + d]);
        }
        for (int d = 0; d < featureDim; d++) grandMean[d] /= n;

        // Total variance
        double totalVar = 0;
        for (int i = 0; i < n; i++)
        {
            int start = i * featureDim;
            for (int d = 0; d < featureDim && start + d < supportX.Length; d++)
            {
                double diff = NumOps.ToDouble(supportX[start + d]) - grandMean[d];
                totalVar += diff * diff;
            }
        }

        // Between-class variance (same centroid computation as above)
        var centroids = new double[numWays][];
        var counts = new int[numWays];
        for (int c = 0; c < numWays; c++) centroids[c] = new double[featureDim];

        for (int i = 0; i < n; i++)
        {
            int c = (int)Math.Round(NumOps.ToDouble(supportY[i])) % numWays;
            if (c < 0) c += numWays;
            counts[c]++;
            int start = i * featureDim;
            for (int d = 0; d < featureDim && start + d < supportX.Length; d++)
                centroids[c][d] += NumOps.ToDouble(supportX[start + d]);
        }
        for (int c = 0; c < numWays; c++)
            if (counts[c] > 0)
                for (int d = 0; d < featureDim; d++) centroids[c][d] /= counts[c];

        double betweenVar = 0;
        for (int c = 0; c < numWays; c++)
        {
            for (int d = 0; d < featureDim; d++)
            {
                double diff = centroids[c][d] - grandMean[d];
                betweenVar += counts[c] * diff * diff;
            }
        }

        // Fisher ratio: higher between/total = easier
        double fisherRatio = totalVar > 1e-12 ? betweenVar / totalVar : 0.5;
        return 1.0 - Math.Max(0, Math.Min(1, fisherRatio));
    }
}
