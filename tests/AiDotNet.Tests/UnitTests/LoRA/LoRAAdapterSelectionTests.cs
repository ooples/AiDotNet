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
/// Tests for <see cref="LoRAAdapterSelection"/> — the shared primitive behind S-LoRA-style serving. One
/// shared base model carries several LoRA adapters (tasks); selecting a task switches every adapter layer,
/// so a single base can serve many fine-tunes by swapping the small adapter rather than reloading the model.
/// </summary>
public class LoRAAdapterSelectionTests
{
    [Fact]
    public void SelectTask_SwitchesEveryAdapterLayer_AndChangesOutput()
    {
        // Base model: a single Dense 4 -> 3 (identity activation).
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

        var input = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(new[] { 0.5, -0.3, 0.8, 0.2 }));
        var baseOutput = net.Predict(input); // also resolves the Dense layer's input shape to 4

        // Wrap the Dense with a shared-base multi-task adapter: a zero base task + two distinct adapters.
        int di = net.Layers.FindIndex(l => l is DenseLayer<double>);
        var multi = new MultiLoRAAdapter<double>(net.Layers[di], "__base__", defaultRank: 2, alpha: 2.0);
        multi.AddTask("A", rank: 2, alpha: 2.0);
        multi.AddTask("B", rank: 2, alpha: 2.0);
        SetTaskWeights(multi, "A", 0.15);
        SetTaskWeights(multi, "B", -0.25);
        net.Layers[di] = multi;

        Assert.True(LoRAAdapterSelection.HasMultiLoRAAdapters(net));

        // Base task => original behavior (zero adapter).
        Assert.Equal(1, LoRAAdapterSelection.SelectTask(net, "__base__"));
        var outBase = net.Predict(input);
        AssertClose(baseOutput, outBase);

        // Two different adapters => two different, non-base outputs.
        Assert.Equal(1, LoRAAdapterSelection.SelectTask(net, "A"));
        var outA = net.Predict(input);
        Assert.Equal(1, LoRAAdapterSelection.SelectTask(net, "B"));
        var outB = net.Predict(input);

        Assert.False(AreClose(outA, outBase), "adapter A must change the output vs base");
        Assert.False(AreClose(outB, outBase), "adapter B must change the output vs base");
        Assert.False(AreClose(outA, outB), "adapters A and B must produce different outputs");
    }

    private static void SetTaskWeights(MultiLoRAAdapter<double> multi, string task, double scale)
    {
        var lora = multi.TaskAdapters[task];
        var p = lora.GetParameters();
        var v = new Vector<double>(p.Length);
        for (int i = 0; i < p.Length; i++) v[i] = scale * ((i % 5) + 1);
        lora.SetParameters(v);
    }

    private static bool AreClose(Tensor<double> a, Tensor<double> b, double tol = 1e-9)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
        {
            if (System.Math.Abs(a[i] - b[i]) > tol) return false;
        }
        return true;
    }

    private static void AssertClose(Tensor<double> a, Tensor<double> b)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++) Assert.Equal(a[i], b[i], 9);
    }
}
