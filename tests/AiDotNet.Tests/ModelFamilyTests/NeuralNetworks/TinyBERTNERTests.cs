using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NER.TransformerBased;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using System.Threading.Tasks;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

public class TinyBERTNERTests : TransformerNERTestBase
{
    // TinyBERT overrides TransformerNEROptions.HiddenDimension to 312
    // (vs the 768 default used by BERT-base). The generator's default
    // [8, 768] InputShape would fail MultiHeadAttention weight matching,
    // so we pin it to 312 here and construct the model via its
    // architecture-only constructor (which wires up CreateTinyBERTDefaults
    // with the 312-dim attention weights).
    protected override int[] InputShape => [8, 312];

    protected override INeuralNetworkModel<double> CreateNetwork()
        => new TinyBERTNER<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 128,
                outputSize: 4));

    /// <summary>
    /// Override with varied random inputs instead of base-class
    /// <c>CreateConstantTensor(0.1)</c> vs <c>CreateConstantTensor(0.9)</c>.
    /// LayerNorm + self-attention on a uniform [8, 312] input mathematically
    /// collapses to a uniform output (Q/K/V uniform → QK^T uniform →
    /// softmax uniform → output uniform). That's a pre-training architectural
    /// artifact, not a model bug. Mirrors the override the auto-scaffold
    /// emits for the TransformerNER family (commit 5d81cacb9); needed here
    /// too because TinyBERTNERTests is a manual scaffold that bypasses
    /// the generator's per-family overrides.
    /// </summary>
    [Fact(Timeout = 120000)]
    public override async Task DifferentInputs_ShouldProduceDifferentOutputs()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        using var network = CreateNetwork();
        var rng1 = ModelTestHelpers.CreateSeededRandom();
        var rng2 = ModelTestHelpers.CreateSeededRandom(seed: 1729);
        var input1 = CreateRandomTensor(InputShape, rng1);
        var input2 = CreateRandomTensor(InputShape, rng2);
        var output1 = network.Predict(input1);
        var output2 = network.Predict(input2);
        bool anyDifferent = false;
        int minLen = System.Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (System.Math.Abs(output1[i] - output2[i]) > 1e-12) { anyDifferent = true; break; }
        }
        Assert.True(anyDifferent,
            "TinyBERT encoder produces identical output for distinct random " +
            "inputs. Attention may be broken or all attention weights collapsed.");
    }

    /// <summary>
    /// Same uniform-input-collapse rationale as
    /// <see cref="DifferentInputs_ShouldProduceDifferentOutputs"/>. The
    /// NER base test uses the same constant-tensor pattern that produces
    /// uniform attention output regardless of input value.
    /// </summary>
    [Fact(Timeout = 120000)]
    public override async Task DifferentInputs_DifferentLabels()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();
        var rng1 = ModelTestHelpers.CreateSeededRandom();
        var rng2 = ModelTestHelpers.CreateSeededRandom(seed: 1729);
        var input1 = CreateRandomTensor(InputShape, rng1);
        var input2 = CreateRandomTensor(InputShape, rng2);
        var labels1 = network.Predict(input1);
        var labels2 = network.Predict(input2);
        bool anyDifferent = false;
        int minLen = System.Math.Min(labels1.Length, labels2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (System.Math.Abs(labels1[i] - labels2[i]) > 1e-12) { anyDifferent = true; break; }
        }
        Assert.True(anyDifferent,
            "TinyBERT NER produces identical labels for distinct random inputs — model may be degenerate.");
    }

    /// <summary>
    /// Same rationale as <see cref="DifferentInputs_ShouldProduceDifferentOutputs"/>:
    /// uniform-input collapse is a LayerNorm + self-attention artifact, not
    /// a real degenerate-solution bug. Use varied random inputs to exercise
    /// the per-position attention routing that the assertion intends to test.
    /// </summary>
    [Fact(Timeout = 120000)]
    public override async Task DifferentInputs_AfterTraining_ShouldProduceDifferentOutputs()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        var trainInput = CreateRandomTensor(InputShape, rng);
        var trainTarget = CreateRandomTargetTensor(EffectiveOutputShape, rng);
        for (int i = 0; i < TrainingIterations; i++)
            network.Train(trainInput, trainTarget);

        var rng1 = ModelTestHelpers.CreateSeededRandom();
        var rng2 = ModelTestHelpers.CreateSeededRandom(seed: 1729);
        var input1 = CreateRandomTensor(InputShape, rng1);
        var input2 = CreateRandomTensor(InputShape, rng2);
        var output1 = network.Predict(input1);
        var output2 = network.Predict(input2);
        double sumSquared = 0;
        int minLen = System.Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            double d = output1[i] - output2[i];
            sumSquared += d * d;
        }
        double l2 = System.Math.Sqrt(sumSquared);
        Assert.True(l2 > 1e-9,
            $"TinyBERT produces identical output for distinct random inputs AFTER " +
            $"training: L2 = {l2:E3}. Attention or output projection collapsed.");
    }
}
