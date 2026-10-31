using System;
using System.Linq;
using System.Reflection;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// The distributional dueling head (Hessel et al. 2018, Sec. 3): V has one output per atom and the advantage mean is
/// taken over actions for each atom. A scalar V, or a mean over every (action, atom) logit, adds one constant to all
/// of an action's logits, which the per-action softmax over atoms cancels, so the value stream never learns.
/// </summary>
public class DuelingCombinationLayerDistributionalTests
{
    private const int FeatureDim = 4;
    private const int Actions = 2;
    private const int Atoms = 3;

    private static Tensor<double> Features()
    {
        var input = new Tensor<double>(new[] { 1, FeatureDim });
        for (int i = 0; i < FeatureDim; i++) input[i] = Math.Sin(1.0 + i);
        return input;
    }

    private static Tensor<double> Field(DuelingCombinationLayer<double> layer, string name) =>
        (Tensor<double>)(typeof(DuelingCombinationLayer<double>)
            .GetField(name, BindingFlags.NonPublic | BindingFlags.Instance)?.GetValue(layer)
            ?? throw new InvalidOperationException($"DuelingCombinationLayer has no {name} field."));

    private static double[] SoftmaxOverAtoms(Tensor<double> logits, int action)
    {
        var row = Enumerable.Range(0, Atoms).Select(z => logits[0, action * Atoms + z]).ToArray();
        double max = row.Max();
        var exp = row.Select(x => Math.Exp(x - max)).ToArray();
        double sum = exp.Sum();
        return exp.Select(x => x / sum).ToArray();
    }

    [Fact]
    public void ValueOfOneAtom_ShiftsThatAtomForEveryAction_AndChangesTheDistribution()
    {
        var layer = new DuelingCombinationLayer<double>(FeatureDim, Actions, Atoms, new[] { 5 }, new[] { 6 }, seed: 3);
        var before = layer.Forward(Features());
        var probabilitiesBefore = SoftmaxOverAtoms(before, 0);

        Field(layer, "_valueBias")[0] += 0.75;
        var after = layer.Forward(Features());

        for (int a = 0; a < Actions; a++)
        {
            Assert.Equal(before[0, a * Atoms] + 0.75, after[0, a * Atoms], 12);
            for (int z = 1; z < Atoms; z++) Assert.Equal(before[0, a * Atoms + z], after[0, a * Atoms + z], 12);
        }
        // The value stream reaches the categorical distribution, so it can receive gradient through it.
        var probabilitiesAfter = SoftmaxOverAtoms(after, 0);
        Assert.True(Math.Abs(probabilitiesAfter[0] - probabilitiesBefore[0]) > 1e-3);
    }

    [Fact]
    public void AdvantageIsCenteredOverActions_ForEachAtom()
    {
        var layer = new DuelingCombinationLayer<double>(FeatureDim, Actions, Atoms, Array.Empty<int>(), new[] { 6 }, seed: 5);
        var valueWeights = Field(layer, "_valueWeights");
        for (int i = 0; i < valueWeights.Length; i++) valueWeights[i] = 0.0;

        var logits = layer.Forward(Features());
        // With V = 0 the output is A − mean_a A, which sums to zero over actions for every atom separately.
        for (int z = 0; z < Atoms; z++)
        {
            double sum = Enumerable.Range(0, Actions).Sum(a => logits[0, a * Atoms + z]);
            Assert.Equal(0.0, sum, 12);
        }
    }

    [Fact]
    public void StreamWidths_SizeTheParameters()
    {
        var layer = new DuelingCombinationLayer<double>(FeatureDim, Actions, Atoms, new[] { 5, 7 }, new[] { 6 }, seed: 1);
        int value = FeatureDim * 5 + 5 + 5 * 7 + 7 + 7 * Atoms + Atoms;
        int advantage = FeatureDim * 6 + 6 + 6 * Actions * Atoms + Actions * Atoms;
        Assert.Equal(value + advantage, layer.ParameterCount);
        Assert.Equal(new[] { Actions * Atoms }, layer.GetOutputShape());
    }

    [Fact]
    public void ScalarHead_KeepsItsParameterLayout()
    {
        // The two-argument constructor is the head without streams or atoms; a saved model built with it must
        // keep its parameter count.
        var layer = new DuelingCombinationLayer<double>(FeatureDim, Actions, seed: 9);
        Assert.Equal(FeatureDim + 1 + FeatureDim * Actions + Actions, layer.ParameterCount);
    }
}
