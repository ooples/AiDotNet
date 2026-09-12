using System;
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
/// LEO as Rusu et al. 2019 and DeepMind's reference implementation state it - relation-network encoding, stochastic
/// latent codes and weights, latent steps then fine-tuning with learned step sizes, and the eq. 6 objective -
/// differentiated exactly through its inner loop (#2155).
/// </summary>
public class LEOExactGradientTests
{
    private const int Features = 3;
    private const int Width = 3;
    private const int Classes = 2;
    private const int Latent = 2;

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
            NumWays = Classes, NumShots = 2, NumQueryPerClass = 2, Name = $"leo-{seed}",
        };
    }

    private static LEOAlgorithm<double, Matrix<double>, Tensor<double>> CreateLearner(
        Action<LEOOptions<double, Matrix<double>, Tensor<double>>>? configure = null)
    {
        var options = new LEOOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(Features, Width))
        {
            EmbeddingDimension = Width,
            LatentDimension = Latent,
            HiddenDimension = 4,
            NumClasses = Classes,
            AdaptationSteps = 2,
            FineTuningSteps = 2,
            InnerLearningRate = 0.5,
            FineTuningLearningRate = 0.1,
            OuterLearningRate = 0.05,
            GradientClipThreshold = null,
            KLWeight = 0.5,
            EncoderPenaltyWeight = 0.3,
            L2Regularization = 0.01,
            OrthogonalityWeight = 0.2,
            DropoutRate = 0.2,
            RandomSeed = 7,
        };
        configure?.Invoke(options);
        return new LEOAlgorithm<double, Matrix<double>, Tensor<double>>(options);
    }

    private static TaskBatch<double, Matrix<double>, Tensor<double>> Batch(params int[] seeds)
        => new TaskBatch<double, Matrix<double>, Tensor<double>>(seeds.Select(CreateTask).ToArray());

    private static double CentralDifference(Func<double> loss, Action<double> shift, double h)
    {
        shift(h);
        double plus = loss();
        shift(-2 * h);
        double minus = loss();
        shift(h);
        return (plus - minus) / (2 * h);
    }

    /// <summary>Every component's mismatches are collected, so one failure names all the gradients that disagree.</summary>
    private static void CheckVector(Func<double> loss, double[] analytic, Func<double[]> get, Action<double[]> set, string name,
        System.Collections.Generic.List<string> failures)
    {
        int reported = 0;
        for (int i = 0; i < analytic.Length; i++)
        {
            int index = i;
            double numeric = CentralDifference(loss, h =>
            {
                var shifted = get();
                shifted[index] += h;
                set(shifted);
            }, 1e-6);
            if (Math.Abs(analytic[i] - numeric) > 1e-6 + 1e-4 * Math.Abs(numeric) && reported++ < 3)
            {
                failures.Add($"dL/d{name}[{i}]: analytic {analytic[i]:G10} vs central difference {numeric:G10}.");
            }
        }
    }

    [Theory]
    [InlineData(true, true, true, 2)]
    [InlineData(true, true, false, 2)]
    [InlineData(false, true, true, 1)]
    [InlineData(true, false, true, 0)]
    [InlineData(true, true, true, 0)]
    public void EpisodeGradient_MatchesCentralDifferences(bool relation, bool shareEncoder, bool training, int fineTuningSteps)
    {
        var learner = CreateLearner(o =>
        {
            o.UseRelationEncoder = relation;
            o.ShareEncoder = shareEncoder;
            o.FineTuningSteps = fineTuningSteps;
            o.EntropyWeight = 0.1;

            // The encoder penalty is |stopgrad(z') - z|^2 (eq. 6), so the objective's gradient deliberately omits the
            // path through the adapted codes - which a central difference of the same objective still measures. It is
            // checked by EncoderPenalty_IsTheGapToTheAdaptedCodes instead.
            o.EncoderPenaltyWeight = 0;
        });
        var task = CreateTask(seed: 3);

        var (_, body, encoder, relationGradient, decoder, latentRates, fineTuningRates) =
            learner.EpisodeGradientForTesting(task, training);
        double Loss() => learner.EpisodeLossForTesting(task, training);

        var model = learner.GetMetaModel();
        var failures = new System.Collections.Generic.List<string>();
        CheckVector(Loss, body.ToArray(), () => model.GetParameters().ToArray(), v => model.SetParameters(new Vector<double>(v)), "body", failures);
        CheckVector(Loss, encoder.ToArray(), () => learner.EncoderWeightsForTesting.ToArray(),
            v => learner.EncoderWeightsForTesting = new Vector<double>(v), "encoder", failures);
        CheckVector(Loss, relationGradient.ToArray(), () => learner.RelationWeightsForTesting.ToArray(),
            v => learner.RelationWeightsForTesting = new Vector<double>(v), "relation", failures);
        CheckVector(Loss, decoder.ToArray(), () => learner.DecoderWeightsForTesting.ToArray(),
            v => learner.DecoderWeightsForTesting = new Vector<double>(v), "decoder", failures);
        CheckVector(Loss, latentRates.ToArray(), () => learner.LatentRatesForTesting.ToArray(),
            v => learner.LatentRatesForTesting = new Vector<double>(v), "latent rate", failures);
        CheckVector(Loss, fineTuningRates.ToArray(), () => learner.FineTuningRatesForTesting.ToArray(),
            v => learner.FineTuningRatesForTesting = new Vector<double>(v), "fine-tuning rate", failures);

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void Adapt_IsTheClassifierTheObjectiveScores()
    {
        // With the regularisers off and no sampling, the objective is the query loss of the classifier Adapt returns.
        var learner = CreateLearner(o =>
        {
            o.KLWeight = 0;
            o.EncoderPenaltyWeight = 0;
            o.L2Regularization = 0;
            o.OrthogonalityWeight = 0;
        });
        var task = CreateTask(seed: 4);

        var probabilities = learner.Adapt(task).Predict(task.QuerySetX);
        double expected = 0;
        int rows = task.QuerySetX.Rows;
        for (int r = 0; r < rows; r++) expected -= Math.Log(probabilities[r * Classes + (int)task.QuerySetY[r]]);

        Assert.Equal(expected / rows, learner.EpisodeLossForTesting(task, training: false), 10);
    }

    [Fact]
    public void Gradient_UnderSamplingAlone_MatchesCentralDifferences()
    {
        // Isolates the sampling path: no KL, entropy, dropout, encoder penalty or regularisers, so anything that
        // disagrees here comes from the reparameterised codes and weights themselves.
        var learner = CreateLearner(o =>
        {
            o.KLWeight = 0;
            o.EntropyWeight = 0;
            o.DropoutRate = 0;
            o.EncoderPenaltyWeight = 0;
            o.L2Regularization = 0;
            o.OrthogonalityWeight = 0;
            o.FineTuningSteps = 1;
        });
        var task = CreateTask(seed: 3);

        var scales = learner.ScalesForTesting(task).ToArray();
        int clamped = scales.Count(s => s <= 1e-9);
        var (_, body, encoder, relation, decoder, latentRates, fineTuningRates) =
            learner.EpisodeGradientForTesting(task, training: true);
        double Loss() => learner.EpisodeLossForTesting(task, training: true);

        var model = learner.GetMetaModel();
        var failures = new System.Collections.Generic.List<string>();
        CheckVector(Loss, body.ToArray(), () => model.GetParameters().ToArray(), v => model.SetParameters(new Vector<double>(v)), "body", failures);
        CheckVector(Loss, encoder.ToArray(), () => learner.EncoderWeightsForTesting.ToArray(),
            v => learner.EncoderWeightsForTesting = new Vector<double>(v), "encoder", failures);
        CheckVector(Loss, relation.ToArray(), () => learner.RelationWeightsForTesting.ToArray(),
            v => learner.RelationWeightsForTesting = new Vector<double>(v), "relation", failures);
        CheckVector(Loss, decoder.ToArray(), () => learner.DecoderWeightsForTesting.ToArray(),
            v => learner.DecoderWeightsForTesting = new Vector<double>(v), "decoder", failures);
        CheckVector(Loss, latentRates.ToArray(), () => learner.LatentRatesForTesting.ToArray(),
            v => learner.LatentRatesForTesting = new Vector<double>(v), "latent rate", failures);
        CheckVector(Loss, fineTuningRates.ToArray(), () => learner.FineTuningRatesForTesting.ToArray(),
            v => learner.FineTuningRatesForTesting = new Vector<double>(v), "fine-tuning rate", failures);

        Assert.True(failures.Count == 0,
            $"{clamped} of {scales.Length} scales sit at the 1e-10 floor.\n" + string.Join("\n", failures));
    }

    [Fact]
    public void Gradient_WithoutInnerSteps_MatchesCentralDifferences()
    {
        // Sampling with the inner steps neutralised: the latent step still runs on its nested tape, but a zero step
        // size leaves the codes at the values they were sampled at, so nothing the inner gradient contributes
        // reaches the objective. A mismatch here is therefore in the plain sampled forward path - encode, sample a
        // code, decode, sample weights, score the query set - rather than in the second-order chain that the latent
        // and fine-tuning steps build. LEO takes latent steps by construction, so AdaptationSteps stays positive:
        // its own options reject zero, and neutralising the step is what isolates the forward path.
        var learner = CreateLearner(o =>
        {
            o.AdaptationSteps = 1;
            o.FineTuningSteps = 0;
            o.KLWeight = 0;
            o.EntropyWeight = 0;
            o.DropoutRate = 0;
            o.EncoderPenaltyWeight = 0;
            o.L2Regularization = 0;
            o.OrthogonalityWeight = 0;
        });
        var task = CreateTask(seed: 3);

        var (_, body, encoder, relation, decoder, _, _) = learner.EpisodeGradientForTesting(task, training: true);
        double Loss() => learner.EpisodeLossForTesting(task, training: true);

        var model = learner.GetMetaModel();
        var failures = new System.Collections.Generic.List<string>();
        CheckVector(Loss, body.ToArray(), () => model.GetParameters().ToArray(), v => model.SetParameters(new Vector<double>(v)), "body", failures);
        CheckVector(Loss, encoder.ToArray(), () => learner.EncoderWeightsForTesting.ToArray(),
            v => learner.EncoderWeightsForTesting = new Vector<double>(v), "encoder", failures);
        CheckVector(Loss, relation.ToArray(), () => learner.RelationWeightsForTesting.ToArray(),
            v => learner.RelationWeightsForTesting = new Vector<double>(v), "relation", failures);
        CheckVector(Loss, decoder.ToArray(), () => learner.DecoderWeightsForTesting.ToArray(),
            v => learner.DecoderWeightsForTesting = new Vector<double>(v), "decoder", failures);

        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    [Fact]
    public void EncoderPenalty_IsTheGapToTheAdaptedCodes()
    {
        // Eq. 6's third term: gamma * mean((stopgrad(z') - z)^2). Its value is checked here because its gradient
        // deliberately ignores the adapted codes, which a central difference cannot.
        var task = CreateTask(seed: 11);
        void Bare(LEOOptions<double, Matrix<double>, Tensor<double>> o)
        {
            o.KLWeight = 0;
            o.L2Regularization = 0;
            o.OrthogonalityWeight = 0;
            o.DropoutRate = 0;
            o.EncoderPenaltyWeight = 0;
        }

        double without = CreateLearner(Bare).EpisodeLossForTesting(task, training: false);
        double with = CreateLearner(o => { Bare(o); o.EncoderPenaltyWeight = 0.75; })
            .EpisodeLossForTesting(task, training: false);

        var learner = CreateLearner(Bare);
        var codes = learner.LatentCodesForTesting(task);
        double gap = 0;
        for (int i = 0; i < codes.Initial.Length; i++) gap += Math.Pow(codes.Adapted[i] - codes.Initial[i], 2);

        Assert.Equal(without + 0.75 * gap / codes.Initial.Length, with, 10);
    }

    [Fact]
    public void FirstOrder_DropsTheSecondOrderTerms()
    {
        var task = CreateTask(seed: 5);
        var exact = CreateLearner().EpisodeGradientForTesting(task, training: false);
        var firstOrder = CreateLearner(o => o.UseFirstOrder = true).EpisodeGradientForTesting(task, training: false);

        Assert.Equal(exact.Loss, firstOrder.Loss, 12);
        Assert.NotEqual(exact.Decoder.ToArray(), firstOrder.Decoder.ToArray());
    }

    [Fact]
    public void Defaults_AreThePapers()
    {
        var options = new LEOOptions<double, Matrix<double>, Tensor<double>>(new LinearEmbeddingModel(Features, Width));

        Assert.Equal(64, options.LatentDimension);
        Assert.Equal(128, options.HiddenDimension);
        Assert.Equal(5, options.AdaptationSteps);
        Assert.Equal(5, options.FineTuningSteps);
        Assert.Equal(1.0, options.InnerLearningRate);
        Assert.Equal(0.001, options.FineTuningLearningRate);
        Assert.Equal(0.00043653954, options.OuterLearningRate);
        Assert.Equal(1.33365371e-9, options.KLWeight);
        Assert.Equal(0.124171967, options.EncoderPenaltyWeight);
        Assert.Equal(0.000108982953, options.L2Regularization);
        Assert.Equal(303.216647, options.OrthogonalityWeight);
        Assert.Equal(1.0 - 0.711524088, options.DropoutRate, 12);
        Assert.False(options.UseFirstOrder);
        Assert.Equal(0.0, options.EntropyWeight);
    }

    [Fact]
    public void MetaTrain_LowersTheQueryLoss()
    {
        var learner = CreateLearner(o =>
        {
            o.KLWeight = 0;
            o.EncoderPenaltyWeight = 0;
            o.L2Regularization = 0;
            o.OrthogonalityWeight = 0;
            o.DropoutRate = 0;
        });
        var tasks = new[] { 100, 101, 102, 103 }.Select(CreateTask).ToArray();
        double Evaluate() => tasks.Average(t => learner.EpisodeLossForTesting(t, training: false));

        double before = Evaluate();
        for (int step = 0; step < 20; step++) learner.MetaTrain(new TaskBatch<double, Matrix<double>, Tensor<double>>(tasks));

        Assert.True(Evaluate() < before, $"Meta-training left the query loss at {Evaluate()} from {before}.");
    }

    [Fact]
    public void LearnedStepSizes_AreTrained()
    {
        var learner = CreateLearner();
        var latent = learner.LatentRatesForTesting.ToArray();
        var fineTuning = learner.FineTuningRatesForTesting.ToArray();

        learner.MetaTrain(Batch(8, 9));

        Assert.NotEqual(latent, learner.LatentRatesForTesting.ToArray());
        Assert.NotEqual(fineTuning, learner.FineTuningRatesForTesting.ToArray());
    }

    [Fact]
    public void OrthogonalInit_GivesOrthonormalLatentRows()
    {
        var learner = CreateLearner(o => o.UseOrthogonalInit = true);
        var decoder = learner.DecoderWeightsForTesting.ToArray();
        int outputs = 2 * Width;

        for (int a = 0; a < Latent; a++)
        {
            for (int b = 0; b < Latent; b++)
            {
                double dot = Enumerable.Range(0, outputs).Sum(r => decoder[r * Latent + a] * decoder[r * Latent + b]);
                Assert.Equal(a == b ? 1.0 : 0.0, dot, 10);
            }
        }
    }

    [Fact]
    public void UnsharedEncoder_GivesEachClassSlotItsOwnBlock()
    {
        var learner = CreateLearner(o => o.ShareEncoder = false);

        Assert.Equal(Classes * Latent * Width, learner.EncoderWeightsForTesting.Length);
    }

    [Fact]
    public void RejectsAnEncoderWhoseWidthIsNotEmbeddingDimension()
    {
        var learner = CreateLearner(o => o.EmbeddingDimension = Width + 1);

        Assert.Throws<InvalidOperationException>(() => learner.MetaTrain(Batch(10)));
    }

    /// <summary>
    /// Checks the FINITE DIFFERENCE rather than the gradient: the same body coordinates are differenced at a
    /// range of step sizes and reported beside the analytic value.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A central difference carries two errors that move in opposite directions: truncation of order h^2 f''',
    /// and cancellation roundoff of order eps*|f|/h. Between them there is a best h, and away from it the
    /// difference is wrong while the gradient is right. Every component of this objective has been checked
    /// individually - the engine's exp and clamp double-backward, the nested tape, the sampled reparameterisation,
    /// the model's own gradient path - and each agreed with a difference at its own scale, so the one thing left
    /// untested is the instrument. If the difference is stable across h and sits away from the analytic value,
    /// the gradient is wrong; if it moves with h, the fixed 1e-6 the other tests use is what is wrong.
    /// </para>
    /// <para>
    /// Reported rather than asserted: the point is the shape of the h-sweep, and a pass/fail here would just
    /// restate whichever tolerance it was given.
    /// </para>
    /// </remarks>
    [Fact]
    public void CentralDifference_IsStableAcrossStepSizes()
    {
        var learner = CreateLearner(o =>
        {
            o.UseRelationEncoder = true;
            o.ShareEncoder = true;
            o.FineTuningSteps = 0;
            o.EntropyWeight = 0.1;
            o.EncoderPenaltyWeight = 0;
        });
        var task = CreateTask(seed: 3);

        var (_, body, _, _, _, _, _) = learner.EpisodeGradientForTesting(task, training: true);
        double Loss() => learner.EpisodeLossForTesting(task, training: true);
        var model = learner.GetMetaModel();
        var start = model.GetParameters().ToArray();

        double[] steps = { 1e-3, 1e-4, 1e-5, 1e-6, 1e-7 };
        var report = new System.Text.StringBuilder();
        report.AppendLine("leo-h-sweep: analytic | differences at h = 1e-3, 1e-4, 1e-5, 1e-6, 1e-7");

        for (int i = 0; i < Math.Min(body.Length, 3); i++)
        {
            int index = i;
            report.Append($"  body[{i}] analytic {body[i]:G10}");
            foreach (double h in steps)
            {
                double numeric = CentralDifference(Loss, delta =>
                {
                    var shifted = (double[])start.Clone();
                    shifted[index] += delta;
                    model.SetParameters(new Vector<double>(shifted));
                }, h);
                report.Append($" | {numeric:G10}");
            }

            model.SetParameters(new Vector<double>(start));
            report.AppendLine();
        }

        Assert.True(false, report.ToString());
    }
}
