using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LoRA;

/// <summary>
/// Tests for <see cref="LoRAAdapterMerger"/>: baking LoRA adapter layers into plain layers must preserve
/// the adapted model's behavior exactly (merged inference == adapter inference) while removing the adapters.
/// </summary>
public class LoRAAdapterMergerTests
{
    [Fact]
    public void MergeInPlace_PreservesBehavior_AndRemovesAdapters()
    {
        // Base model: a single Dense 4 -> 3 (identity activation so we compare linear outputs directly).
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 3,
            layers: new List<ILayer<double>>
            {
                new DenseLayer<double>(3, activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<double>())
            });
        var net = new NeuralNetwork<double>(arch);

        // Wrap the Dense layer with a LoRA adapter and give it a non-zero adaptation.
        int di = net.Layers.FindIndex(l => l is DenseLayer<double>);
        Assert.True(di >= 0);
        // Pass the resolved input size (4) so the LoRA decomposition uses the real dimension rather than the
        // outSize×2 fallback heuristic that applies when the base layer is still lazy.
        var adapter = new DenseLoRAAdapter<double>((LayerBase<double>)net.Layers[di], inputSize: 4, rank: 2, alpha: 2.0);
        var p = adapter.GetParameters();
        var perturbed = new Vector<double>(p.Length);
        for (int i = 0; i < p.Length; i++) perturbed[i] = 0.1 * ((i % 7) + 1); // non-zero A and B => non-zero delta
        adapter.SetParameters(perturbed);
        net.Layers[di] = adapter;

        var input = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(new[] { 0.5, -0.3, 0.8, 0.2 }));
        var before = net.Predict(input);

        // Merge.
        Assert.True(LoRAAdapterMerger.HasAdapters(net));
        int merged = LoRAAdapterMerger.MergeInPlace(net);
        Assert.Equal(1, merged);
        Assert.False(LoRAAdapterMerger.HasAdapters(net));

        // Behavior preserved: the merged plain layer produces the same output as the adapter did.
        var after = net.Predict(input);
        Assert.Equal(before.Length, after.Length);
        for (int i = 0; i < before.Length; i++)
        {
            Assert.Equal(before[i], after[i], 6);
        }
    }

    [Fact]
    public void MergeInPlace_NoAdapters_IsNoOp()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 3,
            layers: new List<ILayer<double>>
            {
                new DenseLayer<double>(3, activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<double>())
            });
        var net = new NeuralNetwork<double>(arch);

        Assert.False(LoRAAdapterMerger.HasAdapters(net));
        Assert.Equal(0, LoRAAdapterMerger.MergeInPlace(net));
    }
}
