using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Data.Structures;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Models;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// AWGIM (Guo &amp; Cheung, CVPR 2020; #1929): the generator's objective differentiated exactly, the
/// information-maximization terms training their own networks, the adapted model writing a different
/// classifier per query, and meta-training lowering a held-out objective.
/// </summary>
public class AWGIMTests
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
            NumWays = Classes, NumShots = 2, NumQueryPerClass = 2, Name = $"awgim-{seed}",
        };
    }

    private static AWGIMAlgorithm<double, Matrix<double>, Tensor<double>> CreateLearner(double reconstructionWeight)
        => new(new AWGIMOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(Features, Width))
        {
            EmbeddingDimension = Width,
            LatentDimension = 4,
            NumHeads = 2,
            NumClasses = Classes,
            DecoderLayers = 2,
            DropoutRate = 0.2,
            SupportClassificationWeight = 1.0,
            // Larger than the paper's 0.001 so the reconstruction gradients sit well above the
            // finite-difference noise floor; these tests check gradients, not the weighting.
            ContextReconstructionWeight = reconstructionWeight,
            QueryReconstructionWeight = reconstructionWeight,
            RandomSeed = 11,
        });

    /// <summary>Central differences of the objective against the analytic gradient over one weight range.</summary>
    private static List<string> CheckGradient(AWGIMAlgorithm<double, Matrix<double>, Tensor<double>> learner,
        MetaLearningTask<double, Matrix<double>, Tensor<double>> task, int? noiseSeed, int offset, int length)
    {
        var analytic = learner.WeightGradientForTesting(task, noiseSeed);
        var weights = learner.WeightsForTesting;
        var failures = new List<string>();
        const double h = 1e-6;
        for (int i = offset; i < offset + length; i++)
        {
            var shifted = new Vector<double>(weights.Length);
            for (int j = 0; j < weights.Length; j++) shifted[j] = weights[j];
            shifted[i] += h;
            learner.WeightsForTesting = shifted;
            double plus = learner.ObjectiveForTesting(task, noiseSeed);
            shifted[i] -= 2 * h;
            learner.WeightsForTesting = shifted;
            double minus = learner.ObjectiveForTesting(task, noiseSeed);
            learner.WeightsForTesting = weights;

            double numeric = (plus - minus) / (2 * h);
            if (Math.Abs(analytic[i] - numeric) > 1e-6 + 1e-4 * Math.Abs(numeric) && failures.Count < 5)
            {
                failures.Add($"dL/dw[{i}]: analytic {analytic[i]:G10} vs central difference {numeric:G10}.");
            }
        }

        return failures;
    }

    [Theory]
    [InlineData(null)]
    [InlineData(17)]
    public void ClassificationObjective_GradientMatchesCentralDifferences(int? noiseSeed)
    {
        // Query and support classification through the whole generator - both encoders, the three
        // attention blocks, the decoder and the weight sampling - on every weight, in evaluation mode and
        // in training mode with its dropout and weight noise fixed.
        var learner = CreateLearner(reconstructionWeight: 0);
        var task = CreateTask(3);
        var failures = CheckGradient(learner, task, noiseSeed, 0, learner.WeightsForTesting.Length);
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [Theory]
    [InlineData("reconstruct.context")]
    [InlineData("reconstruct.query")]
    public void ReconstructionNetworks_GradientMatchesCentralDifferences(string network)
    {
        // The reconstruction targets are stop-gradient codes (the paper's surrogate reconstructs sg(code)),
        // so a central difference of the full objective also moves the target wherever a weight feeds the
        // code. The reconstruction networks' own weights cannot, so there the two must agree exactly.
        var learner = CreateLearner(reconstructionWeight: 0.5);
        var task = CreateTask(3);
        var (offset, length) = learner.WeightRangeForTesting(network);
        var failures = CheckGradient(learner, task, 17, offset, length);
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [Theory]
    [InlineData("reconstruct.context")]
    [InlineData("reconstruct.query")]
    public void ReconstructionNetworks_TrainOnlyThroughTheirTerm(string network)
    {
        var task = CreateTask(3);
        var on = CreateLearner(reconstructionWeight: 0.5);
        var off = CreateLearner(reconstructionWeight: 0);
        var (offset, length) = on.WeightRangeForTesting(network);

        var withTerm = on.WeightGradientForTesting(task, 17);
        var withoutTerm = off.WeightGradientForTesting(task, 17);
        Assert.All(Enumerable.Range(offset, length), i => Assert.Equal(0.0, withoutTerm[i]));
        Assert.Contains(Enumerable.Range(offset, length), i => Math.Abs(withTerm[i]) > 1e-8);
    }

    [Fact]
    public void AdaptedModel_WritesADifferentClassifierPerQuery()
    {
        // The attentive path makes the classifier a function of the query. A fixed linear classifier scales
        // its logits exactly with the input, so if a query's log-odds are not exactly doubled by doubling
        // the query, the weights moved with it.
        var learner = CreateLearner(reconstructionWeight: 0.001);
        var task = CreateTask(5);
        var model = Assert.IsType<AWGIMModel<double, Matrix<double>, Tensor<double>>>(learner.Adapt(task));

        var single = new Matrix<double>(1, Features);
        var doubled = new Matrix<double>(1, Features);
        for (int f = 0; f < Features; f++)
        {
            single[0, f] = task.QuerySetX[0, f];
            doubled[0, f] = 2 * task.QuerySetX[0, f];
        }

        double LogOdds(Tensor<double> p) => Math.Log(p[0] / p[1]);
        double a = LogOdds(model.Predict(single)), b = LogOdds(model.Predict(doubled));
        Assert.True(Math.Abs(b - 2 * a) > 1e-6, $"Log-odds scaled exactly ({a} -> {b}): the classifier did not depend on the query.");
    }

    [Fact]
    public void MetaTraining_ReducesTheEvaluationObjective()
    {
        var learner = CreateLearner(reconstructionWeight: 0.001);
        var evaluation = CreateTask(21);
        double before = learner.ObjectiveForTesting(evaluation, null);
        for (int step = 0; step < 30; step++)
        {
            learner.MetaTrain(new TaskBatch<double, Matrix<double>, Tensor<double>>(
                new[] { CreateTask(100 + 2 * step), CreateTask(101 + 2 * step) }));
        }

        double after = learner.ObjectiveForTesting(evaluation, null);
        Assert.True(after < before, $"Thirty meta-updates did not lower the held-out objective ({before} -> {after}).");
    }
}
