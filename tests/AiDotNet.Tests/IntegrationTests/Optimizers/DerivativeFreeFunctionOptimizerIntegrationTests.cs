#nullable disable
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Optimizers;

/// <summary>
/// Integration tests for <c>IDerivativeFreeFunctionOptimizer&lt;T&gt;</c> — minimizing a plain
/// function from values alone, with no gradient anywhere.
/// </summary>
/// <remarks>
/// CRITICAL: These check answers against KNOWN optima. If one fails, FIX THE OPTIMIZER, do not
/// relax the assertion.
///
/// Every method here is stochastic, so every test seeds it. A run that cannot be repeated cannot
/// be debugged, and a budget assertion on an unseeded run is a coin toss dressed as a test.
/// </remarks>
public class DerivativeFreeFunctionOptimizerIntegrationTests
{
    private const int MaxIterations = 2000;

    /// <summary>Sum of squares: minimum 0 at the origin, and about as easy as a problem gets.</summary>
    private static double Sphere(Vector<double> point)
    {
        double total = 0.0;
        for (int i = 0; i < point.Length; i++) total += point[i] * point[i];
        return total;
    }

    /// <summary>Rosenbrock: minimum 0 at (1, 1), inside a narrow curved valley.</summary>
    private static double Rosenbrock(Vector<double> point)
    {
        double flat = 1.0 - point[0];
        double curve = point[1] - point[0] * point[0];
        return flat * flat + 100.0 * curve * curve;
    }

    /// <summary>
    /// Every method, seeded, so a failure names the one that broke.
    /// </summary>
    public static IEnumerable<object[]> Methods()
    {
        yield return Named("NelderMead", () =>
            new NelderMeadOptimizer<double, Tensor<double>, Tensor<double>>(
                new NelderMeadOptimizerOptions<double, Tensor<double>, Tensor<double>> { Seed = 7 }));

        yield return Named("SimulatedAnnealing", () =>
            SimulatedAnnealingOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction(
                new SimulatedAnnealingOptions<double, Tensor<double>, Tensor<double>> { Seed = 7 }));

        yield return Named("ParticleSwarm", () =>
            ParticleSwarmOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction(
                new ParticleSwarmOptimizationOptions<double, Tensor<double>, Tensor<double>> { Seed = 7 }));

        yield return Named("DifferentialEvolution", () =>
            DifferentialEvolutionOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction(
                new DifferentialEvolutionOptions<double, Tensor<double>, Tensor<double>> { Seed = 7 }));

        yield return Named("GeneticAlgorithm", () =>
            GeneticAlgorithmOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction(
                new GeneticAlgorithmOptimizerOptions<double, Tensor<double>, Tensor<double>> { Seed = 7 }));

        yield return Named("CMAES", () =>
            CMAESOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction(
                new CMAESOptimizerOptions<double, Tensor<double>, Tensor<double>> { Seed = 7 }));

        yield return Named("Powell", () =>
            PowellOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction(
                new PowellOptimizerOptions<double, Tensor<double>, Tensor<double>> { Seed = 7 }));

        yield return Named("TabuSearch", () =>
            TabuSearchOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction(
                new TabuSearchOptions<double, Tensor<double>, Tensor<double>> { Seed = 7 }));

        yield return Named("AntColony", () =>
            AntColonyOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction(
                new AntColonyOptimizationOptions<double, Tensor<double>, Tensor<double>> { Seed = 7 }));
    }

    private static object[] Named(string name, Func<IDerivativeFreeFunctionOptimizer<double>> build)
        => new object[] { name, build };

    /// <summary>
    /// Every method drives the sphere's value down by orders of magnitude from a start where it is
    /// 180. The threshold is deliberately loose: this asks whether the search works at all, not
    /// how well, and the two global explorers below are not local refiners.
    /// </summary>
    [Theory]
    [MemberData(nameof(Methods))]
    public void EveryMethod_MinimizesTheSphere(
        string name, Func<IDerivativeFreeFunctionOptimizer<double>> build)
    {
        var start = new Vector<double>(new[] { 10.0, -8.0, 4.0 });

        Vector<double> answer = build().Minimize(start, Sphere, MaxIterations, 1e-10);

        Assert.True(
            Sphere(answer) < 1e-2,
            $"{name} left the sphere at {Sphere(answer)}, from a start of {Sphere(start)}.");
    }

    /// <summary>
    /// And on Rosenbrock, whose narrow curved valley defeats a method that can only move along
    /// coordinate axes.
    /// </summary>
    [Theory]
    [MemberData(nameof(Methods))]
    public void EveryMethod_MinimizesRosenbrock(
        string name, Func<IDerivativeFreeFunctionOptimizer<double>> build)
    {
        var start = new Vector<double>(new[] { -1.2, 1.0 });

        Vector<double> answer = build().Minimize(start, Rosenbrock, MaxIterations, 1e-10);

        Assert.True(
            Rosenbrock(answer) < 1e-2,
            $"{name} left Rosenbrock at {Rosenbrock(answer)}, from a start of {Rosenbrock(start)}.");
    }

    /// <summary>
    /// A seeded run must be exactly repeatable. Without this a budget assertion is a coin toss and
    /// a failure cannot be reproduced to be diagnosed.
    /// </summary>
    [Theory]
    [MemberData(nameof(Methods))]
    public void EveryMethod_IsReproducibleFromItsSeed(
        string name, Func<IDerivativeFreeFunctionOptimizer<double>> build)
    {
        var start = new Vector<double>(new[] { 10.0, -8.0, 4.0 });

        Vector<double> first = build().Minimize(start, Sphere, 200, 1e-10);
        Vector<double> again = build().Minimize(start, Sphere, 200, 1e-10);

        for (int i = 0; i < first.Length; i++)
        {
            Assert.Equal(first[i], again[i], 12);
        }
    }

    /// <summary>
    /// The answer must never be worse than the starting point. Several of these methods
    /// deliberately accept worse points as they go, so "where it ended" and "the best it found"
    /// are different questions — and it is the second one that gets returned.
    /// </summary>
    [Theory]
    [MemberData(nameof(Methods))]
    public void EveryMethod_NeverReturnsWorseThanItsStart(
        string name, Func<IDerivativeFreeFunctionOptimizer<double>> build)
    {
        var start = new Vector<double>(new[] { 0.05, 0.05 });
        double startValue = Rosenbrock(start);

        Vector<double> answer = build().Minimize(start, Rosenbrock, 50, 1e-10);

        Assert.True(
            Rosenbrock(answer) <= startValue,
            $"{name} returned {Rosenbrock(answer)}, worse than the start's {startValue}.");
    }

    [Theory]
    [MemberData(nameof(Methods))]
    public void EveryMethod_RejectsAnEmptyStartingPoint(
        string name, Func<IDerivativeFreeFunctionOptimizer<double>> build)
    {
        Assert.Throws<ArgumentException>(
            () => build().Minimize(new Vector<double>(0), Sphere, MaxIterations, 1e-10));
    }

    [Theory]
    [MemberData(nameof(Methods))]
    public void EveryMethod_RejectsANonPositiveBudget(
        string name, Func<IDerivativeFreeFunctionOptimizer<double>> build)
    {
        Assert.Throws<ArgumentException>(
            () => build().Minimize(new Vector<double>(new[] { 1.0 }), Sphere, 0, 1e-10));
    }

    /// <summary>
    /// The methods that are genuinely local refiners reach the optimum to machine precision.
    /// Simulated annealing and the genetic algorithm are not on this list, by design: they explore
    /// rather than polish, and both stop several digits short.
    /// </summary>
    [Theory]
    [InlineData("NelderMead")]
    [InlineData("ParticleSwarm")]
    [InlineData("DifferentialEvolution")]
    [InlineData("CMAES")]
    [InlineData("Powell")]
    [InlineData("TabuSearch")]
    [InlineData("AntColony")]
    public void TheRefiningMethods_ReachRosenbrockExactly(string name)
    {
        var build = (Func<IDerivativeFreeFunctionOptimizer<double>>)Methods()
            .Single(row => (string)row[0] == name)[1];

        Vector<double> answer = build().Minimize(
            new Vector<double>(new[] { -1.2, 1.0 }), Rosenbrock, MaxIterations, 1e-10);

        Assert.True(
            Rosenbrock(answer) < 1e-12,
            $"{name} reached only {Rosenbrock(answer)}.");
    }

    /// <summary>
    /// CMA-ES learns the shape of the surface, so it should handle a badly scaled bowl that would
    /// cost a fixed-step method dearly — that adaptation is the whole point of the method.
    /// </summary>
    [Fact]
    public void Cmaes_HandlesABadlyScaledBowl()
    {
        static double Stretched(Vector<double> point)
            => point[0] * point[0] + 100.0 * point[1] * point[1] + 10_000.0 * point[2] * point[2];

        var optimizer = CMAESOptimizer<double, Tensor<double>, Tensor<double>>.CreateForFunction(
            new CMAESOptimizerOptions<double, Tensor<double>, Tensor<double>> { Seed = 7 });

        Vector<double> answer = optimizer.Minimize(
            new Vector<double>(new[] { 1.0, 1.0, 1.0 }), Stretched, MaxIterations, 1e-12);

        Assert.True(Stretched(answer) < 1e-12, $"CMA-ES reached only {Stretched(answer)}.");
    }
}
