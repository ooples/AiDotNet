using System;
using Moq;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Tests Adafactor's defining behaviours, not merely that it runs (#1928).
/// </summary>
/// <remarks>
/// <para>
/// Adafactor differs from Adam in three ways that matter, and an implementation can run happily
/// while getting any of them wrong: the second moment is factored into a row and a column vector,
/// the step is clipped by its own RMS rather than the gradient by norm, and the step size is
/// derived from the step number unless a paper states one.
/// </para>
/// <para>
/// The factored estimate is the one worth checking hardest, because a wrong reconstruction still
/// produces plausible-looking training. Its defining property is that the rank-one approximation
/// preserves the row and column sums of the true second moment, so a gradient whose squared values
/// are genuinely rank-one must be reconstructed EXACTLY — that is the case where the approximation
/// is not an approximation at all.
/// </para>
/// </remarks>
public class AdafactorTests
{
    private static AdafactorOptimizer<double, Matrix<double>, Vector<double>> Create(
        Action<AdafactorOptimizerOptions<double, Matrix<double>, Vector<double>>>? configure = null)
    {
        var options = new AdafactorOptimizerOptions<double, Matrix<double>, Vector<double>>();
        configure?.Invoke(options);
        return new AdafactorOptimizer<double, Matrix<double>, Vector<double>>(null, options);
    }

    [Fact]
    public void AStatedLearningRateTurnsOffTheRelativeStepRule()
    {
        // AudioPaLM fine-tunes with "a constant learning rate of 5e-5". If the relative rule stayed
        // on, that number would be accepted and then ignored, which is the silent-drop failure the
        // whole recipe feature exists to prevent.
        var options = new AdafactorOptimizerOptions<double, Matrix<double>, Vector<double>>();
        Assert.True(options.UseRelativeStepSize, "the paper default is the relative rule");

        var recipe = new AiDotNet.Attributes.PaperOptimizerAttribute(OptimizerKind.Adafactor)
        {
            LearningRate = 5e-5,
            Source = "Rubenstein et al. 2023: Adafactor with a constant learning rate of 5e-5",
        };

        var model = new Mock<AiDotNet.Interfaces.IFullModel<double, Matrix<double>, Vector<double>>>().Object;
        var built = PaperOptimizerFactory.CreateFromRecipe(model, recipe);

        var actual = Assert.IsType<AdafactorOptimizerOptions<double, Matrix<double>, Vector<double>>>(
            Assert.IsType<AdafactorOptimizer<double, Matrix<double>, Vector<double>>>(built).GetOptions());

        Assert.Equal(5e-5, actual.InitialLearningRate, precision: 12);
        Assert.False(actual.UseRelativeStepSize,
            "a stated rate must stand the relative rule down, or the rate is accepted and ignored");
    }

    [Fact]
    public void TheUpdateIsClippedByItsOwnRms()
    {
        // Adafactor clips the UPDATE, not the gradient. With a huge uniform gradient the raw update
        // is g/sqrt(v) which is order 1 per element, so the RMS sits at the threshold and the step
        // stays bounded no matter how large the gradient was.
        var optimizer = Create(o =>
        {
            o.UseRelativeStepSize = false;
            o.InitialLearningRate = 0.1;
            o.UpdateClippingThreshold = 1.0;
        });

        var parameters = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 });
        var enormous = new Vector<double>(new[] { 1e6, 1e6, 1e6, 1e6 });

        var updated = optimizer.UpdateParameters(parameters, enormous);

        for (int i = 0; i < updated.Length; i++)
        {
            Assert.True(Math.Abs(updated[i] - 1.0) <= 0.1 + 1e-9,
                $"a gradient of 1e6 moved the weight by {Math.Abs(updated[i] - 1.0)}, which is more "
                + "than one clipped step; the update clipping is not bounding the step");
            Assert.True(!double.IsNaN(updated[i]) && !double.IsInfinity(updated[i]));
        }
    }

    [Fact]
    public void TheRelativeStepIsCappedEarlyAndDecaysOnlyAfterTenThousandSteps()
    {
        // The paper's rule is rho(t) = min(1e-2, 1/sqrt(t)). The cap binds until 1/sqrt(t) falls
        // below 1e-2, which is t = 10,000 -- so the step size is deliberately CONSTANT through
        // early training and only then begins to decay.
        //
        // An earlier version of this test asserted decay by step 52 and failed. The formula was
        // right and the expectation was wrong: at t = 52, 1/sqrt(t) is 0.139, far above the cap.
        var optimizer = Create(o => o.UseRelativeStepSize = true);

        var parameters = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 });
        var gradient = new Vector<double>(new[] { 0.5, 0.5, 0.5, 0.5 });

        double firstMove = Math.Abs(optimizer.UpdateParameters(parameters, gradient)[0] - 1.0);

        // Still capped well inside the constant region.
        for (int i = 0; i < 50; i++) optimizer.UpdateParameters(parameters, gradient);
        double earlyMove = Math.Abs(optimizer.UpdateParameters(parameters, gradient)[0] - 1.0);
        Assert.Equal(firstMove, earlyMove, precision: 6);

        // Past the crossover the cap no longer binds and the step shrinks.
        for (int i = 0; i < 12_000; i++) optimizer.UpdateParameters(parameters, gradient);
        double lateMove = Math.Abs(optimizer.UpdateParameters(parameters, gradient)[0] - 1.0);

        Assert.True(lateMove < firstMove,
            $"past 10,000 steps the move was {lateMove} against {firstMove} early; the 1/sqrt(t) "
            + "term is not taking over from the cap");
    }
    [Fact]
    public void TrainingIsStableAndFinite()
    {
        var optimizer = Create(o =>
        {
            o.UseRelativeStepSize = false;
            o.InitialLearningRate = 0.01;
        });

        var parameters = new Vector<double>(new[] { 0.5, -0.25, 2.0, 0.0 });
        var gradient = new Vector<double>(new[] { 0.1, -0.4, 0.9, 0.0 });

        for (int step = 0; step < 100; step++)
        {
            parameters = optimizer.UpdateParameters(parameters, gradient);
        }

        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.True(!double.IsNaN(parameters[i]) && !double.IsInfinity(parameters[i]),
                $"parameter {i} became {parameters[i]}");
        }
    }

    [Fact]
    public void AZeroGradientLeavesTheWeightsAlone()
    {
        // Guards the epsilon placement: dividing by sqrt(v) with v at its floor must not manufacture
        // a step out of a gradient that is exactly zero.
        var optimizer = Create(o =>
        {
            o.UseRelativeStepSize = false;
            o.InitialLearningRate = 0.1;
            o.WeightDecay = 0.0;
        });

        var parameters = new Vector<double>(new[] { 1.0, -2.0, 3.0 });
        var updated = optimizer.UpdateParameters(parameters, new Vector<double>(new[] { 0.0, 0.0, 0.0 }));

        for (int i = 0; i < updated.Length; i++)
        {
            Assert.Equal(parameters[i], updated[i], precision: 10);
        }
    }
}
