using System;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Regression tests for the element-wise sweep of the meta-learning algorithms (#2499). The references here are
/// the scalar loops the sweep replaced, not the vectorized code.
/// </summary>
public class Elementwise11bRegressionTests
{
    private static (Tensor<double> Features, Tensor<double> OneHot, int[] Labels) Episode(int n, int d, int classes, int seed)
    {
        var rng = new Random(seed);
        var features = new Tensor<double>(new[] { n, d });
        var oneHot = new Tensor<double>(new[] { n, classes });
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            labels[i] = i % classes;
            oneHot[i * classes + labels[i]] = 1.0;
            for (int j = 0; j < d; j++) features[i * d + j] = rng.NextDouble() * 2.0 - 1.0 + (j == labels[i] ? 0.5 : 0.0);
        }

        return (features, oneHot, labels);
    }

    /// <summary>The loop the sweep replaced: sum_i (diag(p_i) - p_i p_i') (x) (z_i z_i') / n + lambda I.</summary>
    private static double[,] LoopLogisticHessian(Tensor<double> features, Tensor<double> weights, double lambda)
    {
        int n = features.Shape[0], d = features.Shape[1], classes = weights.Shape[0], size = classes * d;
        var hessian = new double[size, size];
        for (int i = 0; i < n; i++)
        {
            var p = new double[classes];
            double largest = double.NegativeInfinity, total = 0;
            for (int c = 0; c < classes; c++)
            {
                for (int j = 0; j < d; j++) p[c] += features[i * d + j] * weights[c * d + j];
                largest = Math.Max(largest, p[c]);
            }

            for (int c = 0; c < classes; c++) { p[c] = Math.Exp(p[c] - largest); total += p[c]; }
            for (int c = 0; c < classes; c++) p[c] /= total;

            for (int c = 0; c < classes; c++)
                for (int e = 0; e < classes; e++)
                {
                    double weight = p[c] * ((c == e ? 1.0 : 0.0) - p[e]) / n;
                    for (int j = 0; j < d; j++)
                        for (int l = 0; l < d; l++)
                            hessian[c * d + j, e * d + l] += weight * features[i * d + j] * features[i * d + l];
                }
        }

        for (int i = 0; i < size; i++) hessian[i, i] += lambda;
        return hessian;
    }

    [Fact]
    public void Logistic_HessianInverse_InvertsTheLoopHessian()
    {
        const int n = 7, d = 4, classes = 3;
        var (features, oneHot, _) = Episode(n, d, classes, seed: 5);

        var (weights, hessianInverse) = ConvexBaseLearners<double>.SolveLogistic(features, oneHot, 0.5, 30, 1e-12);

        var hessian = LoopLogisticHessian(features, weights, 0.5);
        int size = classes * d;
        for (int r = 0; r < size; r++)
            for (int c = 0; c < size; c++)
            {
                double product = 0;
                for (int k = 0; k < size; k++) product += hessianInverse[r * size + k] * hessian[k, c];
                Assert.Equal(r == c ? 1.0 : 0.0, product, 8);
            }
    }

    [SkippableFact]
    public void Logistic_DoesNotMaterialiseEveryExamplesKroneckerBlock()
    {
#if !NETCOREAPP3_0_OR_GREATER
        // net471 has no per-thread allocation counter, and a process-wide one counts parallel tests too.
        Skip.If(true, "GC.GetAllocatedBytesForCurrentThread needs .NET Core 3.0 or later.");
#else
        // The Hessian is [128, 128] doubles (131 KB). Building it from a [n, c, d, c, d] block tensor took
        // n = 64 times that per operand, 180 MB over this solve; at the default 512-wide embedding with 25
        // support examples the same tensor is 1.3 GB per operand.
        const int n = 64, d = 32, classes = 4;
        var (features, oneHot, _) = Episode(n, d, classes, seed: 3);
        ConvexBaseLearners<double>.SolveLogistic(features, oneHot, 1.0, 1, 1e-9);

        long before = GC.GetAllocatedBytesForCurrentThread();
        ConvexBaseLearners<double>.SolveLogistic(features, oneHot, 1.0, 3, 1e-12);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.True(allocated < 40_000_000, $"SolveLogistic allocated {allocated / 1e6:F1} MB for a 131 KB Hessian.");
#endif
    }

    /// <summary>The scalar solver the sweep replaced: coordinate descent with a bisected block projection.</summary>
    private static double[,] LoopCrammerSinger(double[,] kernel, int[] labels, int classes, double cost, int sweeps, double tolerance)
    {
        int n = labels.Length;
        var alpha = new double[n, classes];
        var gradient = new double[n, classes];
        for (int i = 0; i < n; i++) gradient[i, labels[i]] = -1.0;

        for (int sweep = 0; sweep < sweeps; sweep++)
        {
            double largestMove = 0;
            for (int i = 0; i < n; i++)
            {
                double diagonal = kernel[i, i];
                if (diagonal < 1e-12) continue;
                var linear = new double[classes];
                for (int k = 0; k < classes; k++) linear[k] = gradient[i, k] - diagonal * alpha[i, k];

                double Sum(double nu)
                {
                    double total = 0;
                    for (int k = 0; k < classes; k++)
                        total += Math.Min(k == labels[i] ? cost : 0.0, -(linear[k] + nu) / diagonal);
                    return total;
                }

                double low = -1.0, high = 1.0;
                for (int s = 0; s < 200 && Sum(low) < 0; s++) low *= 2.0;
                for (int s = 0; s < 200 && Sum(high) > 0; s++) high *= 2.0;
                for (int s = 0; s < 200; s++)
                {
                    double middle = 0.5 * (low + high);
                    if (Sum(middle) > 0) low = middle; else high = middle;
                }

                double nuStar = 0.5 * (low + high);
                for (int k = 0; k < classes; k++)
                {
                    double updated = Math.Min(k == labels[i] ? cost : 0.0, -(linear[k] + nuStar) / diagonal);
                    double delta = updated - alpha[i, k];
                    if (delta == 0) continue;
                    largestMove = Math.Max(largestMove, Math.Abs(delta));
                    alpha[i, k] = updated;
                    for (int j = 0; j < n; j++) gradient[j, k] += kernel[j, i] * delta;
                }
            }

            if (largestMove < tolerance) break;
        }

        return alpha;
    }

    [Fact]
    public void CrammerSingerDual_MatchesTheBisectedLoopSolver_ForFourClasses()
    {
        const int n = 12, d = 5, classes = 4;
        const double cost = 0.7;
        var (features, oneHot, labels) = Episode(n, d, classes, seed: 11);

        var kernel = new double[n, n];
        var kernelTensor = new Tensor<double>(new[] { n, n });
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
            {
                double dot = 0;
                for (int k = 0; k < d; k++) dot += features[i * d + k] * features[j * d + k];
                kernel[i, j] = dot;
                kernelTensor[i * n + j] = dot;
            }

        var expected = LoopCrammerSinger(kernel, labels, classes, cost, sweeps: 200, tolerance: 1e-12);
        var actual = ConvexBaseLearners<double>.SolveCrammerSingerDual(kernelTensor, oneHot, cost, 200, 1e-12);

        bool anyAtBound = false, anyFree = false;
        for (int i = 0; i < n; i++)
        {
            double rowSum = 0;
            for (int k = 0; k < classes; k++)
            {
                Assert.Equal(expected[i, k], actual[i * classes + k], 7);
                rowSum += actual[i * classes + k];
                double upper = k == labels[i] ? cost : 0.0;
                if (upper - actual[i * classes + k] < 1e-9) anyAtBound = true; else anyFree = true;
            }

            Assert.Equal(0.0, rowSum, 9);
        }

        // The episode exercises both sides of the projection.
        Assert.True(anyAtBound && anyFree);
    }

    private static bool SameParameters<T>(Vector<T> a, Vector<T> b)
    {
        var method = typeof(MbPAAdaptedModel<T, Matrix<T>, Vector<T>>)
            .GetMethod("SameParameters", BindingFlags.NonPublic | BindingFlags.Static);
        Assert.NotNull(method);
        return (bool)(method.Invoke(null, new object[] { a, b }) ?? false);
    }

    [Fact]
    public void MbPASameParameters_SeesOneChangedEntry_WhenTheLengthIsNotExactInT()
    {
        // 4097 is not a Half (the neighbours are 4096 and 4100), so a count of 4096 matches compared equal to
        // the length. The same happens in float from 16777217 parameters on.
        const int length = 4097;
        var a = new Vector<Half>(length);
        for (int i = 0; i < length; i++) a[i] = (Half)(i % 5);
        var b = a.Clone();

        Assert.True(SameParameters(a, b));

        b[1234] = (Half)9.0;
        Assert.False(SameParameters(a, b));
    }

    [Fact]
    public void MbPAHeadLoss_RejectsOneOutOfRangeClassIndex_WhenTheBatchSizeIsNotExactInT()
    {
        const int featureDim = 1, classes = 2, batch = 4097;
        var head = new Vector<Half>((featureDim + 1) * classes);
        var loss = new MbPAHeadLoss<Half>(head, featureDim, classes, MbPAOutputDistribution.Categorical);
        var embeddings = new Vector<Half>(batch * featureDim);
        var indices = new Vector<Half>(batch);
        for (int i = 0; i < batch; i++) indices[i] = (Half)(i % classes);

        _ = loss.CalculateLoss(embeddings, indices);

        indices[77] = (Half)classes;
        Assert.Throws<ArgumentOutOfRangeException>(() => loss.CalculateLoss(embeddings, indices));
    }
}
