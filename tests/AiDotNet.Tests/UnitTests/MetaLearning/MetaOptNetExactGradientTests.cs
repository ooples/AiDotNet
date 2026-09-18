using System;
using System.Linq;
using AiDotNet.Data.Structures;
using AiDotNet.Enums;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// MetaOptNet as Lee et al. 2019 state it: a regularized linear classifier solved on the support set, with the
/// embedding trained through the solver by the implicit function theorem on its KKT conditions (#2155).
/// </summary>
public class MetaOptNetExactGradientTests
{
    private const int Features = 3;
    private const int Width = 3;
    private const int Classes = 2;

    private static MetaLearningTask<double, Matrix<double>, Tensor<double>> CreateTask(int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var centres = Enumerable.Range(0, Classes)
            .Select(_ => Enumerable.Range(0, Features).Select(_ => rng.NextDouble() * 2.0 - 1.0).ToArray()).ToArray();

        (Matrix<double> X, Tensor<double> Y) Rows(int perClass)
        {
            int rows = perClass * Classes;
            var x = new Matrix<double>(rows, Features);
            var y = new Tensor<double>(new[] { rows });
            for (int r = 0; r < rows; r++)
            {
                int label = r % Classes;
                for (int f = 0; f < Features; f++) x[r, f] = centres[label][f] + 0.3 * (rng.NextDouble() - 0.5);
                y[r] = label;
            }

            return (x, y);
        }

        var support = Rows(2);
        var query = Rows(2);
        return new MetaLearningTask<double, Matrix<double>, Tensor<double>>
        {
            SupportSetX = support.X, SupportSetY = support.Y, QuerySetX = query.X, QuerySetY = query.Y,
            NumWays = Classes, NumShots = 2, NumQueryPerClass = 2, Name = $"metaoptnet-{seed}",
        };
    }

    private static MetaOptNetAlgorithm<double, Matrix<double>, Tensor<double>> CreateLearner(
        Action<MetaOptNetOptions<double, Matrix<double>, Tensor<double>>>? configure = null)
    {
        var options = new MetaOptNetOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(Features, Width))
        {
            NumClasses = Classes,
            EmbeddingDimension = Width,
            OuterLearningRate = 0.05,
            GradientClipThreshold = null,
            RandomSeed = 9,
        };
        configure?.Invoke(options);
        return new MetaOptNetAlgorithm<double, Matrix<double>, Tensor<double>>(options);
    }

    private static double CentralDifference(Func<double> loss, Action<double> shift, double h)
    {
        shift(h);
        double plus = loss();
        shift(-2 * h);
        double minus = loss();
        shift(h);
        return (plus - minus) / (2 * h);
    }

    private static void CheckVector(Func<double> loss, double[] analytic, Func<double[]> get, Action<double[]> set,
        string name, System.Collections.Generic.List<string> failures)
    {
        for (int i = 0; i < analytic.Length; i++)
        {
            int index = i;
            double numeric = CentralDifference(loss, h =>
            {
                var shifted = get();
                shifted[index] += h;
                set(shifted);
            }, 1e-6);
            if (Math.Abs(analytic[i] - numeric) > 1e-5 + 1e-3 * Math.Abs(numeric))
            {
                failures.Add($"dL/d{name}[{i}]: analytic {analytic[i]:G10} vs central difference {numeric:G10}.");
            }
        }
    }

    [Theory]
    [InlineData(ConvexSolverType.SVM, false, 0.0)]
    [InlineData(ConvexSolverType.RidgeRegression, false, 0.0)]
    [InlineData(ConvexSolverType.LogisticRegression, false, 0.0)]
    [InlineData(ConvexSolverType.SVM, true, 0.1)]
    [InlineData(ConvexSolverType.RidgeRegression, true, 0.0)]
    public void EpisodeGradient_MatchesCentralDifferences(ConvexSolverType solver, bool normalize, double smoothing)
    {
        var learner = CreateLearner(o =>
        {
            o.SolverType = solver;
            o.NormalizeEmbeddings = normalize;
            o.LabelSmoothing = smoothing;
            o.RidgeLambda = 0.7;
            o.SvmCost = 0.5;
            o.LogisticLambda = 0.3;
            o.MaxSolverIterations = 200;
            o.SolverTolerance = 1e-12;
        });
        var task = CreateTask(seed: 3);
        learner.LogitScaleForTesting = new Vector<double>(new[] { 1.3 });

        var (_, body, scale) = learner.EpisodeGradientForTesting(task);
        double Loss() => learner.EpisodeLossForTesting(task);

        var model = learner.GetMetaModel();
        var failures = new System.Collections.Generic.List<string>();
        CheckVector(Loss, body.ToArray(), () => model.GetParameters().ToArray(),
            v => model.SetParameters(new Vector<double>(v)), "body", failures);
        CheckVector(Loss, scale.ToArray(), () => learner.LogitScaleForTesting.ToArray(),
            v => learner.LogitScaleForTesting = new Vector<double>(v), "scale", failures);

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void RidgeCoefficients_AreTheDualClosedForm()
    {
        // M = (K + lambda I)^-1 Y, the reference implementation's ridge head.
        var learner = CreateLearner(o => { o.SolverType = ConvexSolverType.RidgeRegression; o.RidgeLambda = 2.0; });
        var task = CreateTask(seed: 4);
        var coefficients = learner.CoefficientsForTesting(task).ToArray();

        var z = learner.GetMetaModel().Predict(task.SupportSetX);
        int n = task.SupportSetX.Rows;
        var system = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double dot = 0;
                for (int f = 0; f < Width; f++) dot += z[i * Width + f] * z[j * Width + f];
                system[i, j] = dot + (i == j ? 2.0 : 0.0);
            }
        }

        for (int c = 0; c < Classes; c++)
        {
            for (int i = 0; i < n; i++)
            {
                double row = 0;
                for (int j = 0; j < n; j++) row += system[i, j] * coefficients[j * Classes + c];
                Assert.Equal(task.SupportSetY[i] == c ? 1.0 : 0.0, row, 8);
            }
        }
    }

    [Fact]
    public void SvmCoefficients_SatisfyTheDualConstraints()
    {
        // Eq. 10: alpha_nk <= C for the true class and <= 0 otherwise, and each example's block sums to zero.
        var learner = CreateLearner(o =>
        {
            o.SolverType = ConvexSolverType.SVM;
            o.SvmCost = 0.4;
            o.MaxSolverIterations = 300;
            o.SolverTolerance = 1e-12;
        });
        var task = CreateTask(seed: 5);
        var alpha = learner.CoefficientsForTesting(task).ToArray();

        for (int i = 0; i < task.SupportSetX.Rows; i++)
        {
            double sum = 0;
            for (int c = 0; c < Classes; c++)
            {
                double upper = task.SupportSetY[i] == c ? 0.4 : 0.0;
                Assert.True(alpha[i * Classes + c] <= upper + 1e-9, $"alpha[{i},{c}] = {alpha[i * Classes + c]} exceeds {upper}.");
                sum += alpha[i * Classes + c];
            }

            Assert.Equal(0.0, sum, 8);
        }
    }

    [Fact]
    public void Defaults_AreThePapers()
    {
        var options = new MetaOptNetOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(Features, Width));

        Assert.Equal(ConvexSolverType.SVM, options.SolverType);
        Assert.Equal(0.1, options.SvmCost);
        Assert.Equal(50.0, options.RidgeLambda);
        Assert.Equal(15, options.MaxSolverIterations);
        Assert.False(options.NormalizeEmbeddings);
        Assert.True(options.UseLearnedTemperature);
        Assert.Equal(1.0, options.InitialTemperature);
        Assert.Equal(0.0, options.LabelSmoothing);
    }

    [Fact]
    public void MetaTrain_LowersTheQueryLoss()
    {
        var learner = CreateLearner();
        var tasks = new[] { 100, 101, 102, 103 }.Select(CreateTask).ToArray();
        double Evaluate() => tasks.Average(t => learner.EpisodeLossForTesting(t));

        double before = Evaluate();
        for (int step = 0; step < 20; step++)
        {
            learner.MetaTrain(new TaskBatch<double, Matrix<double>, Tensor<double>>(tasks));
        }

        Assert.True(Evaluate() < before, $"Meta-training left the query loss at {Evaluate()} from {before}.");
    }

    [Fact]
    public void LogitScale_IsTrained()
    {
        var learner = CreateLearner();
        double before = learner.LogitScaleForTesting[0];

        learner.MetaTrain(new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { CreateTask(8), CreateTask(9) }));

        Assert.NotEqual(before, learner.LogitScaleForTesting[0]);
    }

    [Fact]
    public void Adapt_ScoresEveryClassForEachExample()
    {
        var learner = CreateLearner();
        var task = CreateTask(seed: 6);

        var scores = learner.Adapt(task).Predict(task.QuerySetX);

        Assert.Equal(new[] { task.QuerySetX.Rows, Classes }, scores.Shape.ToArray());
        for (int i = 0; i < scores.Length; i++) Assert.False(double.IsNaN(scores[i]));
    }

    [Fact]
    public void RejectsAnEmbeddingWhoseWidthIsNotEmbeddingDimension()
    {
        var learner = CreateLearner(o => o.EmbeddingDimension = Width + 1);

        Assert.Throws<InvalidOperationException>(() => learner.MetaTrain(
            new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { CreateTask(7) })));
    }
}
