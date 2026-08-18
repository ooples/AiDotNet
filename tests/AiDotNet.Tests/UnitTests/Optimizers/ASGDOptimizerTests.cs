using System;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Optimizers.Fused;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Pins <see cref="ASGDOptimizer{T, TInput, TOutput}"/> to the reference ASGD formulation: the decayed SGD step
/// of Xu (arXiv:1107.2490) and the Polyak-Ruppert tail average.
/// </summary>
/// <remarks>
/// <para>
/// The averaging is the whole method, and it is also the part that silently does nothing if mu_t is wrong — with
/// mu_t stuck at 1 you still get a perfectly reasonable SGD optimizer that simply is not ASGD. So the central
/// test here checks the exact algebraic identity that averaging is supposed to satisfy: with T0 = 0 the running
/// average after n steps must equal the arithmetic mean of all n iterates, to full precision.
/// </para>
/// </remarks>
public class ASGDOptimizerTests
{
    private const double Gamma0 = 0.05;
    private const double Lambda = 1e-3;
    private const double Alpha = 0.75;

    private static ASGDOptimizer<double, Matrix<double>, Vector<double>> CreateOptimizer(
        double t0 = 1e6, double weightDecay = 0.0, bool adaptiveLr = false)
        => new(null, new ASGDOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = Gamma0,
            Lambda = Lambda,
            Alpha = Alpha,
            T0 = t0,
            WeightDecay = weightDecay,
            UseAdaptiveLearningRate = adaptiveLr,
        });

    private static double Eta(int t) => Gamma0 / Math.Pow(1.0 + Lambda * Gamma0 * t, Alpha);

    [Fact]
    public void TapeStep_MultipleStepsMaintainTheArithmeticMeanOfIterates()
    {
        var optimizer = CreateOptimizer(t0: 0.0);
        var parameter = new Tensor<double>(new[] { 2 });
        parameter.AsWritableSpan()[0] = 1.0;
        parameter.AsWritableSpan()[1] = -2.0;
        var gradient = new Tensor<double>(new[] { 2 });
        gradient.AsWritableSpan()[0] = 0.4;
        gradient.AsWritableSpan()[1] = -0.7;
        var gradients = new Dictionary<Tensor<double>, Tensor<double>> { [parameter] = gradient };

        double sum0 = 0.0;
        double sum1 = 0.0;
        const int steps = 12;
        for (int step = 0; step < steps; step++)
        {
            optimizer.Step(new TapeStepContext<double>(new[] { parameter }, gradients, 0.0));
            sum0 += parameter.AsSpan()[0];
            sum1 += parameter.AsSpan()[1];
        }

        var average = optimizer.GetTapeAveragedParameterForTests(parameter);
        Assert.NotNull(average);
        Assert.Equal(sum0 / steps, average!.AsSpan()[0], 12);
        Assert.Equal(sum1 / steps, average.AsSpan()[1], 12);
        Assert.NotEqual(parameter.AsSpan()[0], average.AsSpan()[0], 12);
    }

    /// <summary>
    /// One step must be exactly theta*(1 - eta_t*lambda) - eta_t*g, with eta_t from the schedule rather than the
    /// raw learning rate.
    /// </summary>
    [Fact]
    public void SingleStepAppliesTheDecayedScheduleNotTheRawLearningRate()
    {
        var optimizer = CreateOptimizer();
        var updated = optimizer.UpdateParameters(
            new Vector<double>(new[] { 2.0 }), new Vector<double>(new[] { 0.5 }));

        double expected = 2.0 * (1.0 - Eta(1) * Lambda) - Eta(1) * 0.5;
        Assert.Equal(expected, updated[0], 12);

        // And that is genuinely different from using gamma_0 undecayed, so the assertion has teeth.
        double undecayed = 2.0 * (1.0 - Gamma0 * Lambda) - Gamma0 * 0.5;
        Assert.NotEqual(undecayed, updated[0], 12);
    }

    /// <summary>
    /// The step size must strictly decrease with t, following gamma_0 (1 + lambda*gamma_0*t)^-alpha.
    /// </summary>
    [Fact]
    public void StepSizeDecaysMonotonicallyAlongTheSchedule()
    {
        var optimizer = CreateOptimizer();
        var parameters = new Vector<double>(new[] { 0.0 });
        var gradient = new Vector<double>(new[] { 1.0 });

        double previousStep = double.MaxValue;
        for (int t = 1; t <= 20; t++)
        {
            var next = optimizer.UpdateParameters(parameters, gradient);
            // theta starts at 0 and lambda*theta is negligible next to eta*g here, so the step tracks eta_t.
            double step = Math.Abs(next[0] - parameters[0]);
            Assert.True(step < previousStep, $"step at t={t} ({step}) did not shrink below t={t - 1} ({previousStep})");
            previousStep = step;
            parameters = next;
        }
    }

    /// <summary>
    /// The defining property: with averaging started at step 0, the running average after n steps IS the
    /// arithmetic mean of the n iterates.
    /// </summary>
    /// <remarks>
    /// mu_t = 1/max(1, t - T0) = 1/t, and ax += mu_t*(theta - ax) is the incremental form of a running mean. If
    /// mu_t were wrong — stuck at 1, off by one, or an exponential decay rather than 1/t — this identity would
    /// fail immediately, while every other observable behaviour of the optimizer would look fine.
    /// </remarks>
    [Fact]
    public void AverageEqualsTheArithmeticMeanOfTheIterates_WhenAveragingStartsAtStepZero()
    {
        var optimizer = CreateOptimizer(t0: 0.0);
        var parameters = new Vector<double>(new[] { 1.0, -2.0 });
        var gradient = new Vector<double>(new[] { 0.4, -0.7 });

        double sum0 = 0.0, sum1 = 0.0;
        const int steps = 25;
        for (int t = 1; t <= steps; t++)
        {
            parameters = optimizer.UpdateParameters(parameters, gradient);
            sum0 += parameters[0];
            sum1 += parameters[1];
        }

        var averaged = optimizer.GetAveragedParameters();
        Assert.NotNull(averaged);
        Assert.Equal(sum0 / steps, averaged![0], 12);
        Assert.Equal(sum1 / steps, averaged[1], 12);

        // The average must actually differ from the final iterate, or the test would pass trivially.
        Assert.NotEqual(parameters[0], averaged[0], 6);
    }

    /// <summary>
    /// Below the averaging start step, mu_t is 1 and the "average" is just the current iterate. This is the
    /// documented default behaviour (T0 = 1e6 means averaging never engages in a normal run), so it is asserted
    /// rather than left implicit.
    /// </summary>
    [Fact]
    public void AverageTracksTheIterateExactly_BeforeAveragingStarts()
    {
        var optimizer = CreateOptimizer();   // default T0 = 1e6
        var parameters = new Vector<double>(new[] { 1.0 });
        var gradient = new Vector<double>(new[] { 0.4 });

        for (int t = 1; t <= 10; t++)
        {
            parameters = optimizer.UpdateParameters(parameters, gradient);
            var averaged = optimizer.GetAveragedParameters();
            Assert.NotNull(averaged);
            Assert.Equal(parameters[0], averaged![0], 12);
        }
    }

    /// <summary>
    /// Weight decay enters the gradient (g + wd*theta) before the step, on top of the multiplicative
    /// (1 - eta*lambda) term. Both must be present.
    /// </summary>
    [Fact]
    public void WeightDecayIsAddedToTheGradientOnTopOfTheMultiplicativeDecay()
    {
        const double wd = 0.02;
        var optimizer = CreateOptimizer(weightDecay: wd);
        var updated = optimizer.UpdateParameters(
            new Vector<double>(new[] { 3.0 }), new Vector<double>(new[] { 0.5 }));

        double expected = 3.0 * (1.0 - Eta(1) * Lambda) - Eta(1) * (0.5 + wd * 3.0);
        Assert.Equal(expected, updated[0], 12);
    }

    /// <summary>
    /// The ASGD step is affine in theta, so <c>ReverseUpdate</c> inverts it exactly rather than approximately.
    /// </summary>
    [Fact]
    public void ReverseUpdateExactlyInvertsTheStep()
    {
        var optimizer = CreateOptimizer(weightDecay: 0.02);
        var original = new Vector<double>(new[] { 1.5, -0.75, 4.0 });
        var gradient = new Vector<double>(new[] { 0.3, -1.2, 0.05 });

        var updated = optimizer.UpdateParameters(original, gradient);
        var restored = optimizer.ReverseUpdate(updated, gradient);

        for (int i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i], restored[i], 10);
        }
    }

    /// <summary>
    /// The fused kernel reads lambda, alpha and t0 from Extras, not from the beta/epsilon slots. Leaving Extras
    /// null would hand the plan default-constructed constants — a different schedule, silently, on the compiled
    /// path only.
    /// </summary>
    [Fact]
    public void FusedSpecCarriesTheScheduleParametersInExtras()
    {
        var optimizer = CreateOptimizer(t0: 500.0, weightDecay: 0.01);

        Assert.True(((IFusedOptimizerSpec)optimizer).TryGetFusedOptimizerConfig(out var config));
        Assert.Equal(Tensors.Engines.Compilation.OptimizerType.ASGD, config.Type);
        Assert.Equal((float)Gamma0, config.LearningRate, 6);
        Assert.Equal(0.01f, config.WeightDecay, 6);

        Assert.NotNull(config.Extras);
        Assert.Equal((float)Lambda, config.Extras!.Lambd, 8);
        Assert.Equal((float)Alpha, config.Extras.Alpha, 6);
        Assert.Equal(500f, config.Extras.T0, 3);
    }

    /// <summary>
    /// ASGD owns a learning-rate schedule of its own, so an outer adaptive learning rate the plan cannot follow
    /// must make it decline rather than fuse with a stale gamma_0.
    /// </summary>
    [Fact]
    public void FusedSpecDeclines_WhenAnOuterAdaptiveLearningRateIsConfigured()
    {
        var optimizer = CreateOptimizer(adaptiveLr: true);

        Assert.False(((IFusedOptimizerSpec)optimizer).TryGetFusedOptimizerConfig(out _),
            "ASGD fused despite an outer adaptive learning rate that would move gamma_0 after the plan baked it in.");
    }

    /// <summary>
    /// Resuming must restore both the running average and the step count — the latter positions the schedule and
    /// decides whether averaging has started.
    /// </summary>
    [Fact]
    public void SerializationRoundTripPreservesTheAverageAndTheStepCounter()
    {
        var original = CreateOptimizer(t0: 3.0);
        var parameters = new Vector<double>(new[] { 1.0 });
        var gradient = new Vector<double>(new[] { 0.3 });

        for (int t = 1; t <= 8; t++)
        {
            parameters = original.UpdateParameters(parameters, gradient);
        }

        var restored = CreateOptimizer(t0: 3.0);
        restored.Deserialize(original.Serialize());

        var originalAverage = original.GetAveragedParameters();
        var restoredAverage = restored.GetAveragedParameters();
        Assert.NotNull(originalAverage);
        Assert.NotNull(restoredAverage);
        Assert.Equal(originalAverage![0], restoredAverage![0], 12);

        // Step 9 must use eta_9 and mu_9 on both, which only holds if _t survived the round trip.
        Assert.Equal(
            original.UpdateParameters(parameters, gradient)[0],
            restored.UpdateParameters(parameters, gradient)[0],
            12);
    }
}
