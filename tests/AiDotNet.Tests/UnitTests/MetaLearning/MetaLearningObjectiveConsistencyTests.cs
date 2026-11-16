using System;
using System.Collections.Generic;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Data;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.IntegrationTests.MetaLearning;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Reference tests for two objective bugs: Meta-FDMixup's alignment gradient was identically zero, and MPTS's SPSA
/// objective dropped the group regularizer that MetaTrain trains.
/// </summary>
public class MetaLearningObjectiveConsistencyTests
{
    /// <summary>A regression task whose targets follow its own weight vector, so tasks disagree in direction.</summary>
    private static MetaLearningTask<double, Matrix<double>, Vector<double>> CreateTask(int seed, double[] weights)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        const int rows = 6;
        var supportX = new Matrix<double>(rows, 3);
        var supportY = new Vector<double>(rows);
        var queryX = new Matrix<double>(rows, 3);
        var queryY = new Vector<double>(rows);
        for (int r = 0; r < rows; r++)
        {
            double s = 0, q = 0;
            for (int c = 0; c < 3; c++)
            {
                supportX[r, c] = rng.NextDouble() * 2.0 - 1.0;
                queryX[r, c] = rng.NextDouble() * 2.0 - 1.0;
                s += weights[c] * supportX[r, c];
                q += weights[c] * queryX[r, c];
            }

            supportY[r] = s;
            queryY[r] = q;
        }

        return new MetaLearningTask<double, Matrix<double>, Vector<double>>
        {
            SupportSetX = supportX, SupportSetY = supportY, QuerySetX = queryX, QuerySetY = queryY,
            NumWays = 2, NumShots = 3, NumQueryPerClass = 3, Name = $"objective-{seed}",
        };
    }

    private static IMetaLearningTask<double, Matrix<double>, Vector<double>>[] NonAlignedPair()
        => new IMetaLearningTask<double, Matrix<double>, Vector<double>>[]
        {
            CreateTask(11, new[] { 2.0, -1.0, 0.5 }),
            CreateTask(23, new[] { -1.5, 0.5, 2.0 }),
        };

    private static double Norm(Vector<double> v)
    {
        double s = 0;
        for (int i = 0; i < v.Length; i++) s += v[i] * v[i];
        return Math.Sqrt(s);
    }

    private static Vector<double> Axpy(Vector<double> x, double a, Vector<double> y)
    {
        var r = new Vector<double>(x.Length);
        for (int i = 0; i < x.Length; i++) r[i] = x[i] + a * y[i];
        return r;
    }

    [Fact]
    public void MetaFDMixup_AlignmentGradient_IsNonZeroAndMatchesTheAlignmentLoss()
    {
        var model = new LinearVectorModel(3);
        var algorithm = new MetaFDMixupAlgorithm<double, Matrix<double>, Vector<double>>(
            new MetaFDMixupOptions<double, Matrix<double>, Vector<double>>(model)
            {
                LossFunction = new MeanSquaredErrorLoss<double>(),
                GradientClipThreshold = null,
            });
        var tasks = NonAlignedPair();
        var theta = model.GetParameters();

        double before = algorithm.AlignmentLossAt(tasks, theta);
        Assert.True(before > 1e-6, $"The toy pair should not be aligned (L_align = {before}).");

        var gradient = algorithm.ComputeAlignmentGradient(algorithm.SupportGradientsAt(tasks, theta), tasks, theta);
        Assert.NotNull(gradient);
        var g = gradient ?? new Vector<double>(0);
        double gNorm = Norm(g);
        Assert.True(gNorm > 1e-6, $"The alignment gradient must not vanish (|g| = {gNorm}).");

        // Directional derivative along g / |g| from a central difference of the loss equals |g|.
        const double h = 1e-5;
        double plus = algorithm.AlignmentLossAt(tasks, Axpy(theta, h / gNorm, g));
        double minus = algorithm.AlignmentLossAt(tasks, Axpy(theta, -h / gNorm, g));
        double directional = (plus - minus) / (2 * h);
        Assert.True(Math.Abs(directional - gNorm) <= 1e-4 * Math.Max(1.0, gNorm),
            $"Finite-difference slope {directional} should equal |g| = {gNorm}.");

        // A small step against the gradient lowers the alignment loss.
        double eta = 0.01 * before / (gNorm * gNorm);
        double after = algorithm.AlignmentLossAt(tasks, Axpy(theta, -eta, g));
        Assert.True(after < before, $"Stepping along -g should lower L_align ({before} -> {after}).");

        // The probes leave the model where it was.
        var restored = model.GetParameters();
        for (int i = 0; i < theta.Length; i++) Assert.Equal(theta[i], restored[i]);
    }

    [Fact]
    public void MPTS_SpsaObjective_MatchesTheTrainedObjective()
    {
        MPTSAlgorithm<double, Matrix<double>, Vector<double>> Create(double groupRegWeight)
            => new MPTSAlgorithm<double, Matrix<double>, Vector<double>>(
                new MPTSOptions<double, Matrix<double>, Vector<double>>(new LinearVectorModel(3))
                {
                    LossFunction = new MeanSquaredErrorLoss<double>(),
                    NumParamGroups = 2,
                    GroupRegWeight = groupRegWeight,
                    InnerLearningRate = 0.1,
                    OuterLearningRate = 0.01,
                    GradientClipThreshold = null,
                });

        var batch = new TaskBatch<double, Matrix<double>, Vector<double>>(NonAlignedPair());

        var algorithm = Create(10.0);
        double objective = algorithm.ComputeMPTSObjective(batch);
        double trained = algorithm.MetaTrain(batch);
        Assert.Equal(trained, objective, 10);

        // The regularizer is active here, so an objective without it would not have matched.
        double unregularized = Create(0.0).ComputeMPTSObjective(batch);
        Assert.True(Math.Abs(objective - unregularized) > 1e-8,
            $"The group regularizer should contribute ({objective} vs {unregularized}).");
    }
}
