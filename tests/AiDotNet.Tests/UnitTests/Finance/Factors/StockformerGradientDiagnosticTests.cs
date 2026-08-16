using System;
using System.Text;
using AiDotNet.Finance.Trading.Factors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.UnitTests.Finance.Factors;

/// <summary>
/// Distinguishes the three causes the family's training diagnostics name when loss rises: a wrong
/// gradient SIGN, optimizer OSCILLATION, or a first step that overshoots from a near-optimal start.
/// </summary>
/// <remarks>
/// <para>
/// Stockformer's <c>Training_ShouldReduceLoss</c> and <c>LossStrictlyDecreasesOnMemorizationTask</c>
/// both fail with loss GROWING (step 1 = 5.0e-5, step 100 = 4.95e-3). Those three causes need opposite
/// responses — a sign error is a bug to fix, overshoot is a step-size property — so guessing between
/// them and lowering the learning rate until the tests pass would hide a real defect.
/// </para>
/// <para>
/// The discriminator is the learning-rate sweep below. For a descent direction that is CORRECT, there
/// must exist a step size small enough that one step does not increase the loss; that is what "descent
/// direction" means. If loss rises at every step size down to 1e-8, the direction itself is wrong.
/// </para>
/// </remarks>
public class StockformerGradientDiagnosticTests
{
    private readonly ITestOutputHelper _output;

    public StockformerGradientDiagnosticTests(ITestOutputHelper output) => _output = output;

    private const int Assets = 4;
    private const int Window = 16;
    private const int Features = 6;

    private static StockformerOptions<double> Options(double learningRate) => new()
    {
        NumAssets = Assets,
        NumFeatures = Features,
        HiddenDimension = 8,
        SequenceLength = Window,
        NumLayers = 1,
        NumDirectionClasses = 2,
        LearningRate = learningRate,
        // Set, but MEASURED NOT TO WORK: pinning it produced byte-identical `before` losses to the
        // unseeded run, i.e. it changes nothing. The layers are built with DenseLayer's own default
        // initialization, which does not consult ModelOptions.Seed — so Seed joins LearningRate,
        // NumLayers and DropoutRate as an option this model declares and nothing consumes. Left here
        // deliberately as the marker for that defect.
        //
        // Consequence for this harness: rows are NOT comparable to one another, because each builds
        // different initial weights. Only the within-row before -> after comparison is valid, and that
        // is what the verdict rests on.
        Seed = 1,
    };

    private static Tensor<double> Input(int seed = 11)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(new[] { Assets, Window, Features });
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() - 0.5;
        return t;
    }

    private static Tensor<double> Target(int seed = 12)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>(new[] { Assets, 1 });
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() - 0.5;
        return t;
    }

    private static double Mse(Tensor<double> prediction, Tensor<double> target)
    {
        double sum = 0.0;
        int n = Math.Min(prediction.Length, target.Length);
        for (int i = 0; i < n; i++)
        {
            double d = prediction[i] - target[i];
            sum += d * d;
        }
        return n == 0 ? 0.0 : sum / n;
    }

    [Fact]
    public void SomeStepSizeMustNotIncreaseTheLoss()
    {
        var input = Input();
        var target = Target();

        var report = new StringBuilder();
        report.AppendLine("learning-rate sweep, one training step each (before -> after):");

        bool anyImproved = false;
        double bestRatio = double.PositiveInfinity;

        foreach (double lr in new[] { 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8 })
        {
            using var model = new Stockformer<double>(Options(lr));

            double before = Mse(model.Predict(input), target);
            model.Train(input, target);
            double after = Mse(model.Predict(input), target);

            double ratio = before > 0 ? after / before : double.PositiveInfinity;
            if (ratio < bestRatio) bestRatio = ratio;
            if (after <= before) anyImproved = true;

            report.AppendLine(
                $"  lr={lr,-8:G3} {before:E6} -> {after:E6}   ratio={ratio:F4}" +
                (after <= before ? "   IMPROVED" : ""));
        }

        report.AppendLine();
        report.AppendLine(anyImproved
            ? "VERDICT: a small enough step reduces the loss, so the descent DIRECTION is correct and the "
              + "failures are step-size overshoot from a near-zero starting loss."
            : "VERDICT: loss rises at EVERY step size down to 1e-8. That cannot be overshoot — the "
              + "descent direction itself is wrong (gradient sign or accumulation bug).");
        report.AppendLine($"best after/before ratio observed: {bestRatio:F4}");

        // Emitted unconditionally: the sweep is the evidence, and a passing assert would hide it.
        _output.WriteLine(report.ToString());

        Assert.True(anyImproved, report.ToString());
    }
}
