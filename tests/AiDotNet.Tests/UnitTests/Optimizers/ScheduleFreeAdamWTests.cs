using System;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Tests the three sequences that make Schedule-Free what it is (#1928).
/// </summary>
/// <remarks>
/// <para>
/// Defazio et al. 2024 keep a fast iterate z, an equal-weighted running average x, and take
/// gradients at y = (1 - beta) z + beta x. The averaging replaces a learning-rate schedule, which
/// is what lets the method run without knowing the length of training in advance.
/// </para>
/// <para>
/// An implementation can run and even train while collapsing this to plain AdamW — if y is never
/// distinguished from z, the averaging does nothing and the method silently becomes the thing it
/// was designed to replace. The beta endpoints are the sharpest check available: at 0 the exposed
/// parameters must equal the fast iterate, at 1 they must equal the average, and those two are
/// different sequences.
/// </para>
/// </remarks>
public class ScheduleFreeAdamWTests
{
    private static ScheduleFreeAdamWOptimizer<double, Matrix<double>, Vector<double>> Create(
        Action<ScheduleFreeAdamWOptimizerOptions<double, Matrix<double>, Vector<double>>>? configure = null)
    {
        var options = new ScheduleFreeAdamWOptimizerOptions<double, Matrix<double>, Vector<double>>();
        configure?.Invoke(options);
        return new ScheduleFreeAdamWOptimizer<double, Matrix<double>, Vector<double>>(null, options);
    }

    private static Vector<double> Run(
        ScheduleFreeAdamWOptimizer<double, Matrix<double>, Vector<double>> optimizer,
        Vector<double> start, Vector<double> gradient, int steps)
    {
        var parameters = start;
        for (int i = 0; i < steps; i++) parameters = optimizer.UpdateParameters(parameters, gradient);
        return parameters;
    }

    [Fact]
    public void AtInterpolationZeroTheExposedPointIsTheFastIterate()
    {
        // beta = 0 is Polyak-Ruppert: gradients are taken at z, so the parameters ARE z and must
        // have moved further from the start than the average has.
        var optimizer = Create(o => { o.Interpolation = 0.0; o.InitialLearningRate = 0.01; });

        var start = new Vector<double>(new[] { 1.0, 1.0 });
        var moved = Run(optimizer, start, new Vector<double>(new[] { 1.0, 1.0 }), 20);
        var averaged = optimizer.AveragedParameters();

        // z leads, x trails behind it.
        Assert.True(moved[0] < averaged[0],
            $"with beta 0 the exposed point {moved[0]} should be the fast iterate, ahead of the "
            + $"average {averaged[0]}");
    }

    [Fact]
    public void AtInterpolationOneTheExposedPointIsTheAverage()
    {
        // beta = 1 is primal averaging: gradients are taken at x, so the parameters ARE the average.
        var optimizer = Create(o => { o.Interpolation = 1.0; o.InitialLearningRate = 0.01; });

        var moved = Run(optimizer, new Vector<double>(new[] { 1.0, 1.0 }),
            new Vector<double>(new[] { 1.0, 1.0 }), 20);
        var averaged = optimizer.AveragedParameters();

        for (int i = 0; i < moved.Length; i++)
        {
            Assert.Equal(averaged[i], moved[i], precision: 10);
        }
    }

    [Fact]
    public void TheAverageTrailsTheFastIterate()
    {
        // The defining relationship. If these ever coincide the averaging is not happening and the
        // method has quietly become plain AdamW.
        var optimizer = Create(o => { o.Interpolation = 0.9; o.InitialLearningRate = 0.01; });

        Run(optimizer, new Vector<double>(new[] { 1.0 }), new Vector<double>(new[] { 1.0 }), 30);
        var averaged = optimizer.AveragedParameters();

        Assert.True(averaged[0] < 1.0, "the average should have moved from the start");
        Assert.True(averaged[0] > 0.0, "the average should trail, not overshoot");
    }

    [Fact]
    public void ItTrainsStablyWithNoScheduleOverALongRun()
    {
        // The whole claim of the method: a constant rate, no decay, and it still settles. A
        // scheduled optimizer run this way would still be at full step size at the end.
        var optimizer = Create(o => { o.InitialLearningRate = 0.01; o.Interpolation = 0.9; });

        var parameters = new Vector<double>(new[] { 2.0, -1.5, 0.75 });
        var gradient = new Vector<double>(new[] { 0.3, -0.2, 0.1 });

        parameters = Run(optimizer, parameters, gradient, 500);

        for (int i = 0; i < parameters.Length; i++)
        {
            Assert.True(double.IsFinite(parameters[i]), $"parameter {i} became {parameters[i]}");
        }
    }

    [Fact]
    public void WarmupRampsAndThenStopsMattering()
    {
        // The one schedule the method keeps, because averaging cannot stabilise the first updates
        // when there is nothing yet to average. Moonshine ramps to 1.4e-3 over 8192 steps.
        var optimizer = Create(o =>
        {
            o.InitialLearningRate = 0.01;
            o.WarmupSteps = 10;
            o.Interpolation = 0.0;
        });

        var start = new Vector<double>(new[] { 1.0 });
        var gradient = new Vector<double>(new[] { 1.0 });

        double firstMove = Math.Abs(optimizer.UpdateParameters(start, gradient)[0] - 1.0);

        var current = start;
        for (int i = 0; i < 20; i++) current = optimizer.UpdateParameters(current, gradient);
        double afterWarmup = Math.Abs(optimizer.UpdateParameters(current, gradient)[0] - current[0]);

        Assert.True(firstMove < afterWarmup,
            $"the first step moved {firstMove} and a post-warmup step moved {afterWarmup}; the ramp "
            + "is not taking effect");
    }
}
