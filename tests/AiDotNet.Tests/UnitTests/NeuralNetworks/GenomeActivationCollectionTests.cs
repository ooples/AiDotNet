using System;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Regression tests for <see cref="Genome{T}.Activate"/>. Its per-node activation map used to be
/// declared but never filled, so the documented activation step was a no-op and the encoded network
/// computed a purely linear weighted sum.
/// </summary>
public class GenomeActivationCollectionTests
{
    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    [Fact(Timeout = 60000)]
    public async Task Activate_DirectConnection_SquashesOutputWithSigmoid()
    {
        var genome = new Genome<double>(inputSize: 1, outputSize: 1);
        genome.AddConnection(fromNode: 0, toNode: 1, weight: 2.0, isEnabled: true, innovation: 0);

        var output = genome.Activate(new Vector<double>(new[] { 1.0 }));

        // Previously the raw weighted sum 2.0 was returned.
        Assert.Equal(Sigmoid(2.0), output[0], 12);

        await Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task Activate_HiddenNode_IsActivatedBeforeItFeedsTheOutput()
    {
        // input 0 -> hidden 3 -> output 1
        var genome = new Genome<double>(inputSize: 1, outputSize: 1);
        genome.AddConnection(fromNode: 0, toNode: 3, weight: 1.0, isEnabled: true, innovation: 0);
        genome.AddConnection(fromNode: 3, toNode: 1, weight: 1.0, isEnabled: true, innovation: 1);

        var output = genome.Activate(new Vector<double>(new[] { 0.0 }));

        // hidden = sigmoid(0) = 0.5 feeds the output, which is then sigmoid(0.5).
        // Previously the linear network returned 0 (0 * 1 * 1).
        Assert.Equal(Sigmoid(Sigmoid(0.0)), output[0], 12);

        await Task.CompletedTask;
    }

    [Fact(Timeout = 60000)]
    public async Task Activate_InputPassesThroughUnchangedIntoTheWeightedSum()
    {
        var genome = new Genome<double>(inputSize: 2, outputSize: 1);
        genome.AddConnection(fromNode: 0, toNode: 2, weight: 0.5, isEnabled: true, innovation: 0);
        genome.AddConnection(fromNode: 1, toNode: 2, weight: -1.5, isEnabled: true, innovation: 1);

        var output = genome.Activate(new Vector<double>(new[] { 2.0, 1.0 }));

        // Inputs are not squashed: output = sigmoid(0.5 * 2 - 1.5 * 1) = sigmoid(-0.5).
        Assert.Equal(Sigmoid(-0.5), output[0], 12);

        await Task.CompletedTask;
    }
}
