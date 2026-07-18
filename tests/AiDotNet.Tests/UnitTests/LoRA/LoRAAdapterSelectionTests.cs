using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Inference;
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

    [Fact]
    public void SelectTask_SwitchesEveryAdapterLayer_NotJustTheFirst()
    {
        // Two stacked Dense layers (4 -> 3 -> 3) so the "switch EVERY adapter layer" contract is actually
        // exercised: a regression that only switched the first layer would return 1 here and fail.
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 3,
            layers: new List<ILayer<double>>
            {
                new DenseLayer<double>(3, activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<double>()),
                new DenseLayer<double>(3, activationFunction: new AiDotNet.ActivationFunctions.IdentityActivation<double>())
            });
        var net = new NeuralNetwork<double>(arch);
        var input = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(new[] { 0.5, -0.3, 0.8, 0.2 }));
        net.Predict(input); // resolve both Dense layers' input shapes

        // Wrap BOTH dense layers with independent multi-task adapters, each carrying tasks "A" and "B".
        int wrapped = 0;
        for (int i = 0; i < net.Layers.Count; i++)
        {
            if (net.Layers[i] is DenseLayer<double>)
            {
                var multi = new MultiLoRAAdapter<double>(net.Layers[i], "A", defaultRank: 2, alpha: 2.0);
                multi.AddTask("B", rank: 2, alpha: 2.0);
                SetTaskWeights(multi, "A", 0.15 + 0.05 * wrapped);
                SetTaskWeights(multi, "B", -0.25 - 0.05 * wrapped);
                net.Layers[i] = multi;
                wrapped++;
            }
        }
        Assert.Equal(2, wrapped);

        // SelectTask must switch BOTH adapter layers and report 2 switched — not 1.
        Assert.Equal(2, LoRAAdapterSelection.SelectTask(net, "A"));
        var outA = net.Predict(input);
        Assert.Equal(2, LoRAAdapterSelection.SelectTask(net, "B"));
        var outB = net.Predict(input);
        Assert.False(AreClose(outA, outB), "switching every adapter layer must change the network output");
    }

    [Fact]
    public void ForwardWithContext_RoutesAdapterPerRow_WithoutMutatingSharedState()
    {
        // Build a shared-base MultiLoRAAdapter over a resolved Dense 4 -> 3 with two distinct adapters.
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
        var probe = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(new[] { 0.5, -0.3, 0.8, 0.2 }));
        net.Predict(probe); // resolves the Dense layer's input shape to 4

        int di = net.Layers.FindIndex(l => l is DenseLayer<double>);
        var multi = new MultiLoRAAdapter<double>(net.Layers[di], "A", defaultRank: 2, alpha: 2.0);
        multi.AddTask("B", rank: 2, alpha: 2.0);
        SetTaskWeights(multi, "A", 0.15);
        SetTaskWeights(multi, "B", -0.25);

        var ctxLayer = (IContextAwareInferenceLayer<double>)multi;

        // Per-adapter reference outputs for the SAME single-row input, selected via the context (no mutation).
        var row = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(new[] { 0.5, -0.3, 0.8, 0.2 }));
        var outA = ctxLayer.ForwardWithContext(row, new InferenceForwardContext(1L, 0) { LoraTask = "A" });
        var outB = ctxLayer.ForwardWithContext(row, new InferenceForwardContext(1L, 0) { LoraTask = "B" });
        Assert.False(AreClose(outA, outB), "adapters A and B must produce different single-row outputs");

        // A BATCHED forward with row 0 -> A and row 1 -> B (same features in both rows) must reproduce each
        // adapter's single-row output in the corresponding row — proving true per-row adapter routing in one
        // co-batched forward rather than a single shared adapter applied to the whole batch.
        var batched = new Tensor<double>(new[] { 2, 4 },
            new Vector<double>(new[] { 0.5, -0.3, 0.8, 0.2, 0.5, -0.3, 0.8, 0.2 }));
        var batchedCtx = new InferenceForwardContext(new[] { 1L, 2L }, new[] { 0, 0 })
        {
            LoraTasks = new string?[] { "A", "B" }
        };
        var outBatched = ctxLayer.ForwardWithContext(batched, batchedCtx);

        Assert.Equal(6, outBatched.Length); // [2, 3]
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(outA[i], outBatched[i], 9);       // row 0 == adapter A
            Assert.Equal(outB[i], outBatched[3 + i], 9);   // row 1 == adapter B
        }

        // The context path must NOT have mutated the shared adapter selection.
        Assert.Equal("A", multi.CurrentTask);
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
