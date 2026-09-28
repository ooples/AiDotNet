using System;
using System.Reflection;
using AiDotNet.ActivationFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// DenseLayer's inference-only Q8_0 weight copy (installed from GGUF) must never outlive the fp32 weights it was made
/// from. It used to be kept forever, so after fine-tuning a GGUF-loaded model, small-batch inference on VNNI hardware
/// (token-by-token decode) kept running the pre-training weights.
/// </summary>
public class DenseLayerQuantizedWeightTests
{
    private const int In = 32;
    private const int Out = 4;

    private static DenseLayer<float> MaterializedLayer()
    {
        var dense = new DenseLayer<float>(Out, activationFunction: new IdentityActivation<float>());
        dense.SetTrainingMode(false);
        dense.Forward(new Tensor<float>(new[] { 1, In }));
        return dense;
    }

    private static void InstallQ8(DenseLayer<float> dense)
    {
        var qs = new sbyte[In * Out];
        for (int i = 0; i < qs.Length; i++) qs[i] = (sbyte)((i % 7) - 3);
        var scales = new float[Out * (In / 32)];
        for (int i = 0; i < scales.Length; i++) scales[i] = 0.01f;
        dense.SetQuantizedWeightsQ8_0(qs, scales, In, Out);
    }

    private static bool HasQ8(DenseLayer<float> dense)
    {
        var field = typeof(DenseLayer<float>).GetField("_weightsQ8", BindingFlags.NonPublic | BindingFlags.Instance)
            ?? throw new InvalidOperationException("DenseLayer has no _weightsQ8 field");
        return field.GetValue(dense) is not null;
    }

    [Fact]
    public void InferenceWithUnchangedWeights_KeepsTheQuantizedCopy()
    {
        // Control: the checks below are about weight changes, not about any forward pass dropping the copy.
        var dense = MaterializedLayer();
        InstallQ8(dense);
        dense.Forward(new Tensor<float>(new[] { 1, In }));
        Assert.True(HasQ8(dense));
    }

    [Fact]
    public void Training_RetiresTheQuantizedCopy()
    {
        var dense = MaterializedLayer();
        InstallQ8(dense);
        dense.SetTrainingMode(true);
        dense.Forward(new Tensor<float>(new[] { 1, In }));
        Assert.False(HasQ8(dense));
    }

    [Fact]
    public void LoadingNewWeights_RetiresTheQuantizedCopy()
    {
        var dense = MaterializedLayer();
        InstallQ8(dense);
        var values = new float[dense.GetParameters().Length];
        for (int i = 0; i < values.Length; i++) values[i] = 0.5f;
        dense.SetParameters(new Vector<float>(values));
        dense.Forward(new Tensor<float>(new[] { 1, In }));
        Assert.False(HasQ8(dense));
    }
}