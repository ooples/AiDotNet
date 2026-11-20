using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Reference tests for <see cref="NEAT{T}"/> against Stanley &amp; Miikkulainen (2002), "Evolving Neural
/// Networks through Augmenting Topologies".
/// </summary>
public class NEATPaperFidelityTests
{
    private static NEAT<double> Create(int inputs, int outputs, int population, NEATOptions? options = null)
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: AiDotNet.Enums.InputType.OneDimensional,
            taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            inputSize: inputs,
            outputSize: outputs)
        {
            RandomSeed = 7
        };
        return new NEAT<double>(architecture, population, options: options);
    }

    private static Genome<double> Genome(params (int Innovation, double Weight, bool Enabled)[] genes)
    {
        var genome = new Genome<double>(1, 1);
        foreach (var (innovation, weight, enabled) in genes)
        {
            genome.AddConnection(innovation, 100 + innovation, weight, enabled, innovation);
        }

        return genome;
    }

    /// <summary>
    /// Equation 1: delta = c1 E / N + c2 D / N + c3 W. Genome A has innovations 1-5, B has 1-3, 6, 7:
    /// E = 2 (6, 7), D = 2 (4, 5), W = (0.5 + 0 + 0.5) / 3. With N = 1 (both under 20 genes):
    /// 2 + 2 + 0.4 / 3 = 4.1333; with N = 5 (larger genome): 4 / 5 + 0.4 / 3 = 0.9333.
    /// </summary>
    [Fact]
    public void CompatibilityDistance_MatchesEquation1()
    {
        var a = Genome((1, 0.5, true), (2, -0.5, true), (3, 1.0, false), (4, 0.2, true), (5, 0.3, true));
        var b = Genome((1, 0.0, true), (2, -0.5, true), (3, 0.5, true), (6, 0.1, true), (7, 0.4, true));

        var small = Create(1, 1, 2);
        Assert.Equal(4.133333333333334, small.CompatibilityDistance(a, b), 12);

        var normalized = Create(1, 1, 2, new NEATOptions { GenomeSizeNormalizationThreshold = 1 });
        Assert.Equal(0.9333333333333333, normalized.CompatibilityDistance(a, b), 12);
    }

    /// <summary>
    /// Crossover: disjoint and excess genes come only from the fitter parent; with equal fitness they
    /// come from both (Figure 4). Matching genes take either parent's weight.
    /// </summary>
    [Fact]
    public void Crossover_InheritsDisjointAndExcessFromFitterParent()
    {
        var neat = Create(1, 1, 2);
        var a = Genome((1, 0.5, true), (2, -0.5, true), (4, 0.2, true));
        var b = Genome((1, 0.0, true), (2, -0.5, true), (3, 0.1, true), (5, 0.4, true));

        a.Fitness = 2.0;
        b.Fitness = 1.0;
        var child = neat.Crossover(a, b);
        Assert.Equal(new[] { 1, 2, 4 }, child.Connections.Select(c => c.Innovation).OrderBy(i => i).ToArray());
        Assert.Contains(child.Connections.Single(c => c.Innovation == 1).Weight, new[] { 0.5, 0.0 });

        b.Fitness = 2.0;
        var union = neat.Crossover(a, b);
        Assert.Equal(new[] { 1, 2, 3, 4, 5 }, union.Connections.Select(c => c.Innovation).OrderBy(i => i).ToArray());
    }

    /// <summary>A gene disabled in either parent is disabled in the child 75% of the time.</summary>
    [Fact]
    public void Crossover_KeepsDisabledGenesDisabledWithProbability075()
    {
        var neat = Create(1, 1, 2);
        var a = Genome((1, 0.5, false));
        var b = Genome((1, 0.5, true));
        a.Fitness = 1.0;
        b.Fitness = 1.0;

        const int trials = 4000;
        int disabled = Enumerable.Range(0, trials).Count(_ => !neat.Crossover(a, b).Connections[0].IsEnabled);
        Assert.InRange(disabled / (double)trials, 0.72, 0.78);
    }

    /// <summary>
    /// The initial population is uniform: every genome has the same genes (inputs and bias to every
    /// output) with the same innovation numbers, so crossover can line them up.
    /// </summary>
    [Fact]
    public void InitialPopulation_SharesHistoricalMarkings()
    {
        var neat = Create(3, 2, 10);
        var reference = neat.Population[0].Connections.Select(c => (c.Innovation, c.FromNode, c.ToNode)).ToArray();
        Assert.Equal(8, reference.Length);
        Assert.Contains(reference, g => g.FromNode == 5);
        foreach (var genome in neat.Population)
        {
            Assert.Equal(reference, genome.Connections.Select(c => (c.Innovation, c.FromNode, c.ToNode)).ToArray());
        }
    }

    /// <summary>
    /// Historical markings stay consistent through evolution: an innovation number always denotes the
    /// same (from, to) gene in every genome, because the same split within a generation reuses its numbers.
    /// </summary>
    [Fact]
    public void Evolution_InnovationNumbersIdentifyTheSameGene()
    {
        var neat = Create(2, 1, 30, new NEATOptions { AddNodeRate = 1.0, AddConnectionRate = 0.5 });
        neat.EvolvePopulation(g => 1.0 + g.Connections.Count * 0.01, 3);

        var genes = neat.Population.SelectMany(g => g.Connections)
            .GroupBy(c => c.Innovation)
            .Select(group => group.Select(c => (c.FromNode, c.ToNode)).Distinct().Count());
        Assert.All(genes, count => Assert.Equal(1, count));
        Assert.Contains(neat.Population, g => g.Connections.Any(c => c.FromNode > 3 || c.ToNode > 3));
    }

    /// <summary>
    /// Explicit fitness sharing: offspring are proportional to the sum of adjusted fitness f / |S|, i.e.
    /// species mean fitness. Four members of fitness 1 (mean 1) against one member of fitness 2 (mean 2)
    /// with 9 offspring gives 3 and 6; a stagnant species gets none; quotas always sum to the population.
    /// </summary>
    [Fact]
    public void FitnessSharing_AllocatesOffspringByMeanFitness()
    {
        var a = new Tensor<double>(new[] { 1.0, 1.0, 1.0, 1.0 }, new[] { 4 });
        var b = new Tensor<double>(new[] { 2.0 }, new[] { 1 });

        Assert.Equal(new[] { 3, 6 }, NEAT<double>.ShareOffspring(new[] { a, b }, new[] { true, true }, 9));
        Assert.Equal(new[] { 9, 0 }, NEAT<double>.ShareOffspring(new[] { a, b }, new[] { true, false }, 9));

        var c = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var thirds = NEAT<double>.ShareOffspring(new[] { c, c, c }, new[] { true, true, true }, 10);
        Assert.Equal(10, thirds.Sum());
        Assert.All(thirds, q => Assert.InRange(q, 3, 4));
    }

    /// <summary>
    /// Activation uses the steepened sigmoid 1 / (1 + e^(-4.9 x)) and squashes hidden nodes before they
    /// feed forward: input 0.5 -> hidden sigma(0.5) = 0.92056 -> output sigma(0.92056) = 0.98913.
    /// </summary>
    [Fact]
    public void Activation_IsLayeredSteepenedSigmoid()
    {
        var neat = Create(1, 1, 2);
        var genome = new Genome<double>(1, 1);
        genome.AddConnection(0, 1, 1.0, false, 0);
        genome.AddConnection(0, 3, 1.0, true, 1);
        genome.AddConnection(3, 1, 1.0, true, 2);

        var activations = neat.ActivateGenome(genome, new Tensor<double>(new[] { 0.5 }, new[] { 1, 1 }));
        Assert.Equal(0.9205614508160216, activations[0, 3], 12);
        Assert.Equal(0.9891292692938102, activations[0, 1], 12);
        Assert.Equal(1.0, activations[0, 2], 12);

        genome.AddConnection(2, 1, -0.5, true, 3);
        var withBias = neat.ActivateGenome(genome, new Tensor<double>(new[] { 0.5 }, new[] { 1, 1 }));
        Assert.Equal(0.8870294590901284, withBias[0, 1], 12);
    }

    [Fact]
    public void Options_HavePaperDefaults()
    {
        var o = new NEATOptions();
        Assert.Equal(1.0, o.ExcessCoefficient);
        Assert.Equal(1.0, o.DisjointCoefficient);
        Assert.Equal(0.4, o.WeightDifferenceCoefficient);
        Assert.Equal(3.0, o.CompatibilityThreshold);
        Assert.Equal(0.8, o.WeightMutationRate);
        Assert.Equal(0.1, o.WeightReplaceRate);
        Assert.Equal(0.03, o.AddNodeRate);
        Assert.Equal(0.05, o.AddConnectionRate);
        Assert.Equal(0.75, o.DisableInheritedRate);
        Assert.Equal(0.001, o.InterspeciesMatingRate);
        Assert.Equal(0.75, o.CrossoverRate);
        Assert.Equal(15, o.SpeciesStagnationLimit);
        Assert.Equal(4.9, o.SigmoidSlope);
    }

    /// <summary>
    /// NaN and the infinities pass a one-sided comparison, so each compatibility and mutation setting is checked
    /// for finiteness: any of them non-finite makes every compatibility distance or perturbation non-finite.
    /// </summary>
    [Theory]
    [InlineData(nameof(NEATOptions.CompatibilityThreshold), double.PositiveInfinity)]
    [InlineData(nameof(NEATOptions.WeightCap), double.PositiveInfinity)]
    [InlineData(nameof(NEATOptions.WeightPerturbPower), double.NaN)]
    [InlineData(nameof(NEATOptions.WeightPerturbPower), double.PositiveInfinity)]
    [InlineData(nameof(NEATOptions.ExcessCoefficient), double.NaN)]
    [InlineData(nameof(NEATOptions.DisjointCoefficient), double.PositiveInfinity)]
    [InlineData(nameof(NEATOptions.WeightDifferenceCoefficient), double.NegativeInfinity)]
    [InlineData(nameof(NEATOptions.SigmoidSlope), double.NaN)]
    public void Validate_RejectsNonFiniteSettings(string property, double value)
    {
        var options = new NEATOptions();
        var setter = typeof(NEATOptions).GetProperty(property)
            ?? throw new InvalidOperationException($"NEATOptions has no property {property}.");
        setter.SetValue(options, value);
        Assert.Throws<ArgumentException>(() => options.Validate());
    }
}
