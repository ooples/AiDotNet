using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Shared helper methods for all model family test base classes.
/// </summary>
internal static class ModelTestHelpers
{
    /// <summary>
    /// Generates linear regression data: y = coeff[0]*x[0] + coeff[1]*x[1] + ... + intercept + noise.
    /// </summary>
    public static (Matrix<double> X, Vector<double> Y) GenerateLinearData(
        int samples, int features, Random rng, double noise = 0.1)
    {
        var x = new Matrix<double>(samples, features);
        var y = new Vector<double>(samples);

        // Generate random coefficients
        var coefficients = new double[features];
        for (int j = 0; j < features; j++)
            coefficients[j] = (j + 1) * 2.0; // 2, 4, 6, ...

        double intercept = 1.0;

        for (int i = 0; i < samples; i++)
        {
            double target = intercept;
            for (int j = 0; j < features; j++)
            {
                x[i, j] = rng.NextDouble() * 10.0;
                target += coefficients[j] * x[i, j];
            }
            y[i] = target + NextGaussian(rng) * noise;
        }

        return (x, y);
    }

    /// <summary>
    /// Generates linearly separable classification data with Gaussian clusters.
    /// </summary>
    public static (Matrix<double> X, Vector<double> Y) GenerateClassificationData(
        int samples, int features, int nClasses, Random rng)
    {
        var x = new Matrix<double>(samples, features);
        var y = new Vector<double>(samples);

        int samplesPerClass = samples / nClasses;

        for (int c = 0; c < nClasses; c++)
        {
            // Cluster center offset for separation
            double centerOffset = c * 4.0;

            int startIdx = c * samplesPerClass;
            int endIdx = c == nClasses - 1 ? samples : startIdx + samplesPerClass;

            for (int i = startIdx; i < endIdx; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    x[i, j] = centerOffset + NextGaussian(rng) * 0.5;
                }
                y[i] = c;
            }
        }

        // Shuffle to avoid ordered data bias
        ShuffleRows(x, y, rng);

        return (x, y);
    }

    /// <summary>
    /// Generates well-separated cluster data (blobs).
    /// </summary>
    public static (Matrix<double> X, Vector<double> Y) GenerateClusterData(
        int samples, int nClusters, int features, Random rng)
    {
        var x = new Matrix<double>(samples, features);
        var y = new Vector<double>(samples);

        int samplesPerCluster = samples / nClusters;

        for (int c = 0; c < nClusters; c++)
        {
            int startIdx = c * samplesPerCluster;
            int endIdx = c == nClusters - 1 ? samples : startIdx + samplesPerCluster;

            for (int i = startIdx; i < endIdx; i++)
            {
                for (int j = 0; j < features; j++)
                {
                    x[i, j] = c * 10.0 + NextGaussian(rng) * 0.5;
                }
                y[i] = c;
            }
        }

        ShuffleRows(x, y, rng);

        return (x, y);
    }

    /// <summary>
    /// Generates time series data: linear trend + sinusoidal seasonal + noise.
    /// </summary>
    public static (Matrix<double> X, Vector<double> Y) GenerateTimeSeriesData(
        int length, Random rng, double noise = 0.1)
    {
        var x = new Matrix<double>(length, 1);
        var y = new Vector<double>(length);

        for (int i = 0; i < length; i++)
        {
            double t = i;
            x[i, 0] = t;
            y[i] = 0.5 * t + 3.0 * Math.Sin(2.0 * Math.PI * t / 20.0) + NextGaussian(rng) * noise;
        }

        return (x, y);
    }

    /// <summary>
    /// Calculates R-squared (coefficient of determination).
    /// </summary>
    public static double CalculateR2(Vector<double> actual, Vector<double> predicted)
    {
        double mean = 0;
        for (int i = 0; i < actual.Length; i++)
            mean += actual[i];
        mean /= actual.Length;

        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            ssRes += Math.Pow(actual[i] - predicted[i], 2);
            ssTot += Math.Pow(actual[i] - mean, 2);
        }

        return ssTot == 0 ? 0 : 1.0 - ssRes / ssTot;
    }

    /// <summary>
    /// Calculates classification accuracy.
    /// </summary>
    public static double CalculateAccuracy(Vector<double> actual, Vector<double> predicted)
    {
        int correct = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            if (Math.Abs(Math.Round(predicted[i]) - actual[i]) < 0.5)
                correct++;
        }
        return (double)correct / actual.Length;
    }

    /// <summary>
    /// Box-Muller transform for Gaussian random numbers.
    /// </summary>
    public static double NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>
    /// Checks that all values in a vector are finite (no NaN or Infinity).
    /// </summary>
    public static bool AllFinite(Vector<double> v)
    {
        for (int i = 0; i < v.Length; i++)
        {
            if (double.IsNaN(v[i]) || double.IsInfinity(v[i]))
                return false;
        }
        return true;
    }

    /// <summary>
    /// Creates a deterministic, cryptographically secure random instance.
    /// </summary>
    public static Random CreateSeededRandom(int seed = 42)
    {
        return RandomHelper.CreateSeededRandom(seed);
    }

    /// <summary>
    /// Calculates Mean Squared Error.
    /// </summary>
    public static double CalculateMSE(Vector<double> actual, Vector<double> predicted)
    {
        double mse = 0;
        for (int i = 0; i < actual.Length; i++)
            mse += Math.Pow(actual[i] - predicted[i], 2);
        return mse / actual.Length;
    }

    /// <summary>
    /// Adds a pure-noise feature column to a matrix.
    /// </summary>
    public static Matrix<double> AddNoiseFeature(Matrix<double> x, Random rng)
    {
        var result = new Matrix<double>(x.Rows, x.Columns + 1);
        for (int i = 0; i < x.Rows; i++)
        {
            for (int j = 0; j < x.Columns; j++)
                result[i, j] = x[i, j];
            result[i, x.Columns] = rng.NextDouble() * 100.0;
        }
        return result;
    }

    /// <summary>
    /// Calculates Adjusted Rand Index between two clustering assignments.
    /// ARI = 0 for random clustering, ARI = 1 for perfect agreement,
    /// ARI &lt; 0 for anti-correlated clustering.
    /// </summary>
    public static double CalculateAdjustedRandIndex(Vector<double> labels1, Vector<double> labels2)
    {
        int n = labels1.Length;
        if (n != labels2.Length || n < 2) return 0;

        // Build contingency table
        var map1 = new Dictionary<int, List<int>>();
        var map2 = new Dictionary<int, List<int>>();

        for (int i = 0; i < n; i++)
        {
            int l1 = (int)Math.Round(labels1[i]);
            int l2 = (int)Math.Round(labels2[i]);

            if (!map1.ContainsKey(l1)) map1[l1] = new List<int>();
            if (!map2.ContainsKey(l2)) map2[l2] = new List<int>();
            map1[l1].Add(i);
            map2[l2].Add(i);
        }

        // Compute contingency table n_ij
        long sumNijC2 = 0; // sum of C(n_ij, 2)
        long[] a = new long[map1.Count]; // row sums
        long[] b = new long[map2.Count]; // col sums

        var keys1 = map1.Keys.ToArray();
        var keys2 = map2.Keys.ToArray();

        for (int i = 0; i < keys1.Length; i++)
        {
            var set1 = new HashSet<int>(map1[keys1[i]]);
            for (int j = 0; j < keys2.Length; j++)
            {
                int nij = 0;
                foreach (var idx in map2[keys2[j]])
                {
                    if (set1.Contains(idx)) nij++;
                }
                sumNijC2 += Choose2(nij);
                b[j] += nij;
            }
            a[i] = set1.Count;
        }

        long sumAC2 = 0;
        foreach (var ai in a) sumAC2 += Choose2((int)ai);
        long sumBC2 = 0;
        foreach (var bi in b) sumBC2 += Choose2((int)bi);

        long nC2 = Choose2(n);
        if (nC2 == 0) return 0;

        double expectedIndex = (double)sumAC2 * sumBC2 / nC2;
        double maxIndex = 0.5 * (sumAC2 + sumBC2);
        double denominator = maxIndex - expectedIndex;

        if (Math.Abs(denominator) < 1e-12) return 0;

        return (sumNijC2 - expectedIndex) / denominator;
    }

    private static long Choose2(int n)
    {
        return n < 2 ? 0 : (long)n * (n - 1) / 2;
    }

    private static void ShuffleRows(Matrix<double> x, Vector<double> y, Random rng)
    {
        int n = x.Rows;
        int cols = x.Columns;
        for (int i = n - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            // Swap rows i and j
            for (int c = 0; c < cols; c++)
            {
                (x[i, c], x[j, c]) = (x[j, c], x[i, c]);
            }
            (y[i], y[j]) = (y[j], y[i]);
        }
    }
}
