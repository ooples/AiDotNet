using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Losses;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.SelfSupervisedLearning;

/// <summary>
/// Every self-supervised objective must be DIFFERENTIABLE, not merely computable.
/// </summary>
/// <remarks>
/// <para>
/// These objectives used to return a bare scalar assembled from host loops over tensor indexers.
/// That number looks like a loss and reads like a loss, but it carries no tape history, so calling
/// it produced no gradient and nothing could train on it. The whole family — InfoNCE, NT-Xent,
/// BYOL, DINO, Barlow Twins, MAE — was affected, which is why models reaching for a published
/// contrastive objective silently fell back to a pointwise loss instead.
/// </para>
/// <para>
/// A test asserting only that the loss is a finite number would have passed the entire time. These
/// assert what actually matters: that a gradient exists, is finite, is non-zero, and agrees with
/// central finite differences of the objective itself.
/// </para>
/// </remarks>
public class ContrastiveObjectiveGradientTests
{
    private static Tensor<double> Sample(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(new[] { rows, cols });
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    public static TheoryData<string> Objectives => new()
    {
        "InfoNCE", "NTXent", "BYOL", "BarlowTwins", "DINO", "MAE",
    };

    private static IContrastiveLoss<double> Create(string name, int dim) => name switch
    {
        "InfoNCE" => new InfoNCELoss<double>(temperature: 0.07),
        "NTXent" => new NTXentLoss<double>(temperature: 0.5),
        "BYOL" => new BYOLLoss<double>(),
        "BarlowTwins" => new BarlowTwinsLoss<double>(),
        "DINO" => new DINOLoss<double>(outputDim: dim),
        "MAE" => new MAEReconstructionLoss<double>(),
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    [Theory]
    [MemberData(nameof(Objectives))]
    public void Objective_ProducesFiniteNonZeroGradient(string name)
    {
        const int rows = 4, cols = 3;
        var view1 = Sample(rows, cols, 11);
        var view2 = Sample(rows, cols, 29);
        var loss = Create(name, cols);

        using var tape = new GradientTape<double>();
        var value = loss.ComputeLoss(view1, view2);
        var grads = tape.ComputeGradients(value, new[] { view1 });

        Assert.True(grads.ContainsKey(view1),
            $"{name} produced no gradient for its input. The objective is not on the tape — it was " +
            "almost certainly assembled with host arithmetic instead of IEngine operations.");

        var g = grads[view1];
        double magnitude = 0;
        for (int i = 0; i < g.Length; i++)
        {
            Assert.True(double.IsFinite(g[i]), $"{name} gradient[{i}] is not finite: {g[i]}");
            magnitude += Math.Abs(g[i]);
        }

        Assert.True(magnitude > 1e-12,
            $"{name} produced an all-zero gradient (sum |g| = {magnitude}). A loss that cannot move " +
            "its input cannot train the model that owns it.");
    }

    [Theory]
    [MemberData(nameof(Objectives))]
    public void Objective_GradientMatchesFiniteDifferences(string name)
    {
        const int rows = 4, cols = 3;
        var view1 = Sample(rows, cols, 7);
        var view2 = Sample(rows, cols, 13);
        var loss = Create(name, cols);

        using var tape = new GradientTape<double>();
        var value = loss.ComputeLoss(view1, view2);
        var analytic = tape.ComputeGradients(value, new[] { view1 })[view1];

        // Central differences on a few coordinates. DINO carries EMA centering state that updates
        // on every call, so it is compared against its own re-evaluated loss the same way.
        const double eps = 1e-6;
        int probes = Math.Min(5, view1.Length);
        for (int k = 0; k < probes; k++)
        {
            double original = view1[k];

            view1[k] = original + eps;
            double plus = Create(name, cols).ComputeLoss(view1, view2)[0];
            view1[k] = original - eps;
            double minus = Create(name, cols).ComputeLoss(view1, view2)[0];
            view1[k] = original;

            double numeric = (plus - minus) / (2 * eps);
            double tolerance = 1e-4 * Math.Max(1.0, Math.Abs(numeric));

            Assert.True(Math.Abs(numeric - analytic[k]) <= tolerance,
                $"{name} gradient[{k}] disagrees with finite differences: analytic {analytic[k]:G10} " +
                $"vs numeric {numeric:G10}. The objective is differentiable but computes the wrong " +
                "derivative, which trains the model in the wrong direction.");
        }
    }

    /// <summary>
    /// The contrastive objectives must stay finite at the small temperatures their papers use.
    /// </summary>
    /// <remarks>
    /// MoCo's default temperature is 0.07, which scales logits by ~14x. A naive
    /// <c>log(softmax(x))</c> overflows there; the stable form subtracts the row max first.
    /// </remarks>
    [Theory]
    [InlineData(0.07)]
    [InlineData(0.01)]
    public void InfoNce_StaysFiniteAtSmallTemperature(double temperature)
    {
        var view1 = Sample(4, 3, 5);
        var view2 = Sample(4, 3, 6);

        // Large-magnitude embeddings amplify the overflow risk.
        for (int i = 0; i < view1.Length; i++) view1[i] *= 50.0;
        for (int i = 0; i < view2.Length; i++) view2[i] *= 50.0;

        var value = new InfoNCELoss<double>(temperature: temperature, normalize: false)
            .ComputeLoss(view1, view2)[0];

        Assert.True(double.IsFinite(value),
            $"InfoNCE overflowed at temperature {temperature}: {value}.");
    }
}
