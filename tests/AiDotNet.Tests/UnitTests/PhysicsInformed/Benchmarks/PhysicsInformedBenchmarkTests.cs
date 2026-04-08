using System;
using AiDotNet.PhysicsInformed.Benchmarks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PhysicsInformed.Benchmarks;

public class PhysicsInformedBenchmarkTests
{
    [Fact]
    public void BurgersBenchmark_BaselinePredictorMatchesFiniteDifference()
    {
        var options = new BurgersBenchmarkOptions
        {
            SpatialPoints = 32,
            TimeSteps = 50,
            FinalTime = 0.1,
            Viscosity = 0.01
        };

        var baseline = FiniteDifferenceBaseline.SolveBurgers(options, out _);
        double dx = (options.DomainEnd - options.DomainStart) / (options.SpatialPoints - 1);

        double Predictor(double x, double t)
        {
            int index = (int)Math.Round((x - options.DomainStart) / dx);
            index = Math.Max(0, Math.Min(baseline.Length - 1, index));
            return baseline[index];
        }

        var result = PdeBenchmarkSuite.RunBurgers(options, Predictor);

        Assert.InRange(result.L2Error, 0.0, 1e-8);
        Assert.InRange(result.MaxError, 0.0, 1e-8);
    }

    [Fact]
    public void BurgersBenchmark_NaivePredictorWorseThanBaseline()
    {
        var options = new BurgersBenchmarkOptions
        {
            SpatialPoints = 32,
            TimeSteps = 40,
            FinalTime = 0.2,
            Viscosity = 0.01
        };

        var baseline = FiniteDifferenceBaseline.SolveBurgers(options, out _);
        double dx = (options.DomainEnd - options.DomainStart) / (options.SpatialPoints - 1);

        double BaselinePredictor(double x, double t)
        {
            int index = (int)Math.Round((x - options.DomainStart) / dx);
            index = Math.Max(0, Math.Min(baseline.Length - 1, index));
            return baseline[index];
        }

        var baselineResult = PdeBenchmarkSuite.RunBurgers(options, BaselinePredictor);
        var naiveResult = PdeBenchmarkSuite.RunBurgers(options, (_, _) => 0.0);

        Assert.True(naiveResult.L2Error > baselineResult.L2Error);
        Assert.True(naiveResult.MaxError > baselineResult.MaxError);
    }

    [Fact]
    public void AllenCahnBenchmark_BaselinePredictorMatchesFiniteDifference()
    {
        var options = new AllenCahnBenchmarkOptions
        {
            SpatialPoints = 32,
            TimeSteps = 40,
            FinalTime = 0.1,
            Epsilon = 0.01
        };

        var baseline = FiniteDifferenceBaseline.SolveAllenCahn(options, out _);
        double dx = (options.DomainEnd - options.DomainStart) / (options.SpatialPoints - 1);

        double Predictor(double x, double t)
        {
            int index = (int)Math.Round((x - options.DomainStart) / dx);
            index = Math.Max(0, Math.Min(baseline.Length - 1, index));
            return baseline[index];
        }

        var result = PdeBenchmarkSuite.RunAllenCahn(options, Predictor);

        Assert.InRange(result.L2Error, 0.0, 1e-8);
        Assert.InRange(result.MaxError, 0.0, 1e-8);
    }

    [Fact]
    public void AllenCahnBenchmark_NaivePredictorWorseThanBaseline()
    {
        var options = new AllenCahnBenchmarkOptions
        {
            SpatialPoints = 32,
            TimeSteps = 40,
            FinalTime = 0.2,
            Epsilon = 0.01
        };

        var baseline = FiniteDifferenceBaseline.SolveAllenCahn(options, out _);
        double dx = (options.DomainEnd - options.DomainStart) / (options.SpatialPoints - 1);

        double BaselinePredictor(double x, double t)
        {
            int index = (int)Math.Round((x - options.DomainStart) / dx);
            index = Math.Max(0, Math.Min(baseline.Length - 1, index));
            return baseline[index];
        }

        var baselineResult = PdeBenchmarkSuite.RunAllenCahn(options, BaselinePredictor);
        var naiveResult = PdeBenchmarkSuite.RunAllenCahn(options, (_, _) => 0.0);

        Assert.True(naiveResult.L2Error > baselineResult.L2Error);
        Assert.True(naiveResult.MaxError > baselineResult.MaxError);
    }

    [Fact]
    public void OperatorBenchmark_MovingAverageBeatsIdentity()
    {
        var options = new OperatorBenchmarkOptions
        {
            SpatialPoints = 32,
            SampleCount = 8,
            MaxFrequency = 2,
            SmoothingWindow = 5,
            Seed = 11
        };

        var baseline = OperatorBenchmarkSuite.RunSmoothingOperatorBenchmark(
            options,
            input => OperatorBenchmarkSuite.ApplyMovingAverage(input, options.SmoothingWindow));

        var identity = OperatorBenchmarkSuite.RunSmoothingOperatorBenchmark(
            options,
            input => (double[])input.Clone());

        Assert.InRange(baseline.L2Error, 0.0, 1e-12);
        Assert.InRange(baseline.MaxError, 0.0, 1e-12);
        Assert.True(baseline.L2Error < identity.L2Error);
        Assert.True(baseline.MaxError < identity.MaxError);
    }

    [Fact]
    public void PoissonOperatorBenchmark_BaselineSolverMatchesDataset()
    {
        var options = new PoissonOperatorBenchmarkOptions
        {
            GridSize = 8,
            SampleCount = 3,
            MaxFrequency = 2,
            MaxIterations = 200,
            Tolerance = 1e-8,
            Seed = 7
        };

        var dataset = OperatorBenchmarkSuite.GeneratePoissonDataset(options);

        Assert.Equal(options.SampleCount, dataset.Inputs.GetLength(0));
        Assert.Equal(options.GridSize, dataset.Inputs.GetLength(1));
        Assert.Equal(options.GridSize, dataset.Inputs.GetLength(2));
        Assert.Equal(options.SampleCount, dataset.Outputs.GetLength(0));

        var result = OperatorBenchmarkSuite.RunPoissonOperatorBenchmark(
            options,
            input => OperatorBenchmarkSuite.SolvePoisson(input, options.MaxIterations, options.Tolerance));

        Assert.InRange(result.L2Error, 0.0, 1e-10);
        Assert.InRange(result.RelativeL2Error, 0.0, 1e-10);
    }

    [Fact]
    public void PoissonOperatorBenchmark_NaivePredictorWorseThanBaseline()
    {
        var options = new PoissonOperatorBenchmarkOptions
        {
            GridSize = 8,
            SampleCount = 3,
            MaxFrequency = 2,
            MaxIterations = 200,
            Tolerance = 1e-8,
            Seed = 17
        };

        var baseline = OperatorBenchmarkSuite.RunPoissonOperatorBenchmark(
            options,
            input => OperatorBenchmarkSuite.SolvePoisson(input, options.MaxIterations, options.Tolerance));

        var naive = OperatorBenchmarkSuite.RunPoissonOperatorBenchmark(options, input => input);

        Assert.True(naive.L2Error > baseline.L2Error);
        Assert.True(naive.RelativeL2Error > baseline.RelativeL2Error);
    }

    [Fact]
    public void DarcyOperatorBenchmark_BaselineSolverMatchesDataset()
    {
        var options = new DarcyOperatorBenchmarkOptions
        {
            GridSize = 8,
            SampleCount = 2,
            MaxFrequency = 2,
            MaxIterations = 200,
            Tolerance = 1e-8,
            ForcingValue = 1.0,
            LogPermeabilityScale = 0.4,
            Seed = 13
        };

        var dataset = OperatorBenchmarkSuite.GenerateDarcyDataset(options);

        Assert.Equal(options.SampleCount, dataset.Inputs.GetLength(0));
        Assert.Equal(options.GridSize, dataset.Inputs.GetLength(1));
        Assert.Equal(options.GridSize, dataset.Inputs.GetLength(2));
        Assert.Equal(options.SampleCount, dataset.Outputs.GetLength(0));

        var result = OperatorBenchmarkSuite.RunDarcyOperatorBenchmark(
            options,
            input => OperatorBenchmarkSuite.SolveDarcy(
                input,
                options.ForcingValue,
                options.MaxIterations,
                options.Tolerance));

        Assert.InRange(result.L2Error, 0.0, 1e-10);
        Assert.InRange(result.RelativeL2Error, 0.0, 1e-10);
    }

    [Fact]
    public void DarcyOperatorBenchmark_NaivePredictorWorseThanBaseline()
    {
        var options = new DarcyOperatorBenchmarkOptions
        {
            GridSize = 8,
            SampleCount = 2,
            MaxFrequency = 2,
            MaxIterations = 200,
            Tolerance = 1e-8,
            ForcingValue = 1.0,
            LogPermeabilityScale = 0.4,
            Seed = 19
        };

        var baseline = OperatorBenchmarkSuite.RunDarcyOperatorBenchmark(
            options,
            input => OperatorBenchmarkSuite.SolveDarcy(
                input,
                options.ForcingValue,
                options.MaxIterations,
                options.Tolerance));

        var naive = OperatorBenchmarkSuite.RunDarcyOperatorBenchmark(options, input => input);

        Assert.True(naive.L2Error > baseline.L2Error);
        Assert.True(naive.RelativeL2Error > baseline.RelativeL2Error);
    }
}
