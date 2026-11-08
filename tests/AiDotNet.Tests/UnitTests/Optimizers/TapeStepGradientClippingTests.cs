using System;
using System.Collections.Generic;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// The clipping options act on the tape step of every optimizer, as they do on the vector path. Before, only Adam and
/// AdamW clipped there (Adam by norm only), so the same options trained clipped through Optimize and unclipped through
/// Train, and a by-value clip was silently ignored. Each case steps one optimizer on a raw gradient with clipping on
/// and a twin with clipping off on the gradient clipped by hand; the parameters must land in the same place.
/// </summary>
public class TapeStepGradientClippingTests
{
    private static readonly double[] RawGradient = { 3.0, -0.1, 0.5, -2.0 };
    private static readonly double[] Start = { 0.4, -0.3, 0.2, 0.1 };

    public static TheoryData<string> Optimizers => new() { "SGD", "Momentum", "RMSProp", "Adagrad", "Adam", "AdamW" };

    private static GradientBasedOptimizerBase<double, Tensor<double>, Tensor<double>> Create(
        string name, Action<GradientBasedOptimizerOptions<double, Tensor<double>, Tensor<double>>> configure)
    {
        T Configured<T>(T options) where T : GradientBasedOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            options.InitialLearningRate = 0.05;
            configure(options);
            return options;
        }

        return name switch
        {
            "SGD" => new StochasticGradientDescentOptimizer<double, Tensor<double>, Tensor<double>>(null,
                Configured(new StochasticGradientDescentOptimizerOptions<double, Tensor<double>, Tensor<double>>())),
            "Momentum" => new MomentumOptimizer<double, Tensor<double>, Tensor<double>>(null,
                Configured(new MomentumOptimizerOptions<double, Tensor<double>, Tensor<double>>())),
            "RMSProp" => new RootMeanSquarePropagationOptimizer<double, Tensor<double>, Tensor<double>>(null,
                Configured(new RootMeanSquarePropagationOptimizerOptions<double, Tensor<double>, Tensor<double>>())),
            "Adagrad" => new AdagradOptimizer<double, Tensor<double>, Tensor<double>>(null,
                Configured(new AdagradOptimizerOptions<double, Tensor<double>, Tensor<double>>())),
            "Adam" => new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null,
                Configured(new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>())),
            "AdamW" => new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(null,
                Configured(new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>())),
            _ => throw new ArgumentOutOfRangeException(nameof(name)),
        };
    }

    // Two steps on the same gradient, so a clip that only took effect on the first would show.
    private static double[] TwoSteps(GradientBasedOptimizerBase<double, Tensor<double>, Tensor<double>> optimizer, double[] gradient)
    {
        var parameter = new Tensor<double>((double[])Start.Clone(), new[] { Start.Length });
        for (int step = 0; step < 2; step++)
        {
            var gradients = new Dictionary<Tensor<double>, Tensor<double>>(TensorReferenceComparer<Tensor<double>>.Instance)
            {
                [parameter] = new Tensor<double>((double[])gradient.Clone(), new[] { gradient.Length }),
            };
            optimizer.Step(new TapeStepContext<double>(new[] { parameter }, gradients, loss: 0.0));
        }

        return parameter.ToArray();
    }

    private static void AssertSameStep(string name, double[] clipped, double[] byHand)
    {
        for (int i = 0; i < clipped.Length; i++)
            Assert.True(Math.Abs(clipped[i] - byHand[i]) <= 1e-12,
                $"{name}, element {i}: clipped by the options {clipped[i]:R}, by hand {byHand[i]:R}");
    }

    [Theory]
    [MemberData(nameof(Optimizers))]
    public void ValueClip_ClampsEachGradientElement(string name)
    {
        const double bound = 0.25;
        var clipped = TwoSteps(Create(name, o =>
        {
            o.EnableGradientClipping = true;
            o.GradientClippingMethod = GradientClippingMethod.ByValue;
            o.MaxGradientValue = bound;
        }), RawGradient);
        var byHand = TwoSteps(Create(name, o => o.EnableGradientClipping = false),
            Array.ConvertAll(RawGradient, g => Math.Max(-bound, Math.Min(bound, g))));
        AssertSameStep(name, clipped, byHand);

        var unclipped = TwoSteps(Create(name, o => o.EnableGradientClipping = false), RawGradient);
        Assert.NotEqual(unclipped, clipped);
    }

    [Theory]
    [MemberData(nameof(Optimizers))]
    public void NormClip_ScalesTheGradientToTheMaximumNorm(string name)
    {
        const double maxNorm = 1.0;
        double norm = 0;
        foreach (var g in RawGradient) norm += g * g;
        norm = Math.Sqrt(norm);
        var clipped = TwoSteps(Create(name, o =>
        {
            o.EnableGradientClipping = true;
            o.GradientClippingMethod = GradientClippingMethod.ByNorm;
            o.MaxGradientNorm = maxNorm;
        }), RawGradient);
        // torch.nn.utils.clip_grad_norm_'s coefficient: max_norm / (total_norm + 1e-6).
        var byHand = TwoSteps(Create(name, o => o.EnableGradientClipping = false),
            Array.ConvertAll(RawGradient, g => g * maxNorm / (norm + 1e-6)));
        AssertSameStep(name, clipped, byHand);
    }

    /// <summary>
    /// The fused plan clips by global norm only, so it takes the optimizer's norm and refuses a by-value clip (NaN),
    /// for every optimizer rather than only Adam and AdamW.
    /// </summary>
    [Theory]
    [MemberData(nameof(Optimizers))]
    public void FusedClipThreshold_FollowsTheOptions(string name)
    {
        var byNorm = Create(name, o => { o.EnableGradientClipping = true; o.GradientClippingMethod = GradientClippingMethod.ByNorm; o.MaxGradientNorm = 0.7; });
        var byValue = Create(name, o => { o.EnableGradientClipping = true; o.GradientClippingMethod = GradientClippingMethod.ByValue; o.MaxGradientValue = 0.3; });
        var off = Create(name, o => o.EnableGradientClipping = false);
        Assert.Equal(0.7, byNorm.TapeStepGradientClipNorm);
        Assert.Equal(0.0, byNorm.TapeStepGradientClipValue);
        Assert.True(double.IsNaN(byValue.TapeStepGradientClipNorm));
        Assert.Equal(0.3, byValue.TapeStepGradientClipValue);
        Assert.Equal(0.0, off.TapeStepGradientClipNorm);
        Assert.Equal(0.0, off.TapeStepGradientClipValue);

        // A negative norm clips nothing on the fused path rather than flipping the gradient's sign.
        var negative = Create(name, o => { o.EnableGradientClipping = true; o.GradientClippingMethod = GradientClippingMethod.ByNorm; o.MaxGradientNorm = -1.0; });
        Assert.Equal(0.0, negative.TapeStepGradientClipNorm);
    }

    /// <summary>
    /// An embedding table's gradient arrives as sparse rows. Rows that occur once are clipped in place and the sparse
    /// step still runs; repeated rows are densified first, since their gradient row is the sum. Either way the table
    /// must land where a dense step on the hand-clipped dense gradient puts it.
    /// </summary>
    [Theory]
    [InlineData(new[] { 1, 4 })]
    [InlineData(new[] { 1, 1, 4 })]
    public void NormClip_CoversSparseEmbeddingGradients(int[] ids)
    {
        const int vocab = 6, dim = 3;
        const double maxNorm = 0.5;
        var engine = AiDotNetEngine.Current;
        var start = new double[vocab * dim];
        for (int i = 0; i < start.Length; i++) start[i] = Math.Sin(0.37 * i) + 0.1;
        var weight = new Tensor<double>(new[] { ids.Length, dim });
        for (int i = 0; i < weight.Length; i++) weight[i] = 0.2 + 0.3 * i;

        // The dense gradient of sum(rows * weight): each looked-up row adds its weight row to its table row.
        var dense = new double[vocab * dim];
        for (int n = 0; n < ids.Length; n++)
            for (int c = 0; c < dim; c++) dense[ids[n] * dim + c] += weight[n * dim + c];
        double norm = 0;
        foreach (double g in dense) norm += g * g;
        norm = Math.Sqrt(norm);
        Assert.True(norm > maxNorm, "the case must actually clip");

        var table = new Tensor<double>((double[])start.Clone(), new[] { vocab, dim });
        var optimizer = Create("SGD", o =>
        {
            o.EnableGradientClipping = true;
            o.GradientClippingMethod = GradientClippingMethod.ByNorm;
            o.MaxGradientNorm = maxNorm;
        });
        bool sawSparse = false;
        using (var tape = new GradientTape<double>())
        {
            var rows = engine.TensorEmbeddingLookup<double, int>(table, new Tensor<int>(ids, new[] { ids.Length }));
            var loss = engine.ReduceSum(engine.TensorMultiply(rows, weight), null);
            tape.ComputeGradientsStreaming(loss, new[] { table }, (source, grad) =>
            {
                sawSparse |= SparseEmbeddingOptimizerHelpers.HasSparseEmbeddingGrad(source);
                optimizer.Step(new TapeStepContext<double>(new[] { source }, new Dictionary<Tensor<double>, Tensor<double>>(), 0.0));
            });
        }

        Assert.True(sawSparse, "the embedding backward produced no sparse gradient, so the sparse clip was not exercised");
        var byHand = new Tensor<double>((double[])start.Clone(), new[] { vocab, dim });
        var gradients = new Dictionary<Tensor<double>, Tensor<double>>(TensorReferenceComparer<Tensor<double>>.Instance)
        {
            [byHand] = new Tensor<double>(Array.ConvertAll(dense, g => g * maxNorm / (norm + 1e-6)), new[] { vocab, dim }),
        };
        Create("SGD", o => o.EnableGradientClipping = false)
            .Step(new TapeStepContext<double>(new[] { byHand }, gradients, loss: 0.0));
        AssertSameStep($"sparse rows [{string.Join(", ", ids)}]", table.ToArray(), byHand.ToArray());
    }
}
