using System;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Correctness contract for the copy-on-write <see cref="NeuralNetworkBase{T}.Clone"/> lever (#1624):
/// a clone must be <b>observationally identical</b> to its source yet <b>independent under mutation</b>.
/// The COW fast path shares weight storage via <c>Tensor&lt;T&gt;.CloneShared()</c> (O(1)-until-write),
/// so the load-bearing guarantee is that the first in-place write to either model privatizes that tensor
/// — never silently corrupting the other. (If the share never engages, the eager fallback satisfies the
/// same contract, so this test pins the behavior regardless of which path runs.)
/// </summary>
public class CopyOnWriteCloneTests
{
    private static FeedForwardNeuralNetwork<double> BuildModel()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);
        return new FeedForwardNeuralNetwork<double>(arch);
    }

    private static Tensor<double> Input() =>
        new(new Vector<double>(new[] { 0.1, -0.2, 0.3, -0.4 }), new[] { 1, 4 });

    [Fact]
    public void Clone_IsObservationallyIdentical_AndIndependentUnderMutation()
    {
        var model = BuildModel();
        var input = Input();
        _ = model.Predict(input); // materialize lazy weights so Clone sees the full parameter set

        var sourceParamsBefore = model.GetParameters().Clone();
        var sourcePredBefore = model.Predict(input);

        var clone = (FeedForwardNeuralNetwork<double>)model.Clone();

        // 1) The clone is observationally identical to the source right after cloning.
        var clonePred = clone.Predict(input);
        Assert.Equal(sourcePredBefore.Length, clonePred.Length);
        for (int i = 0; i < sourcePredBefore.Length; i++)
            Assert.Equal(sourcePredBefore[i], clonePred[i], 10);

        // 2) Mutate the clone's weights. Copy-on-write must privatize them so the SOURCE is untouched.
        var mutated = clone.GetParameters().Clone();
        for (int i = 0; i < mutated.Length; i++) mutated[i] += 0.5;
        clone.SetParameters(mutated);

        var sourceParamsAfter = model.GetParameters();
        double maxDrift = 0;
        for (int i = 0; i < sourceParamsBefore.Length; i++)
            maxDrift = Math.Max(maxDrift, Math.Abs(sourceParamsAfter[i] - sourceParamsBefore[i]));
        Assert.True(maxDrift < 1e-12,
            $"Mutating the clone changed the source by {maxDrift:E3} — copy-on-write failed to privatize " +
            "the shared weight storage (the clone and source are aliasing the same tensors).");

        // 3) The source's prediction is unchanged; the clone's actually changed (mutation took effect).
        var sourcePredAfter = model.Predict(input);
        for (int i = 0; i < sourcePredBefore.Length; i++)
            Assert.Equal(sourcePredBefore[i], sourcePredAfter[i], 10);

        var clonePredAfter = clone.Predict(input);
        double cloneDelta = 0;
        for (int i = 0; i < clonePredAfter.Length; i++)
            cloneDelta = Math.Max(cloneDelta, Math.Abs(clonePredAfter[i] - sourcePredBefore[i]));
        Assert.True(cloneDelta > 1e-6, "Mutating the clone's parameters did not change its prediction.");
    }
}
