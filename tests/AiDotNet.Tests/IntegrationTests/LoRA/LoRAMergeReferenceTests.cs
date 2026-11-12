using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LoRA;

/// <summary>
/// Reference tests: merging an adapter into its base layer must give the same outputs as the adapter's own
/// forward pass, on non-square layers (where an untransposed or misaligned delta cannot hide).
/// </summary>
public class LoRAMergeReferenceTests
{
    private const int InputSize = 12;
    private const int OutputSize = 8;
    private const int Rank = 3;

    private static DenseLayer<double> CreateBaseLayer(int inputSize, int outputSize)
    {
        // Identity activation: the merged layer applies the activation to (W + ΔW)x + b,
        // which only equals the adapter's base(x) + ΔWx for a linear base layer.
        var layer = new DenseLayer<double>(outputSize, (IActivationFunction<double>)new IdentityActivation<double>());
        layer.ResolveFromShape(new[] { inputSize });
        return layer;
    }

    private static Tensor<double> CreateInput(int batchSize, int featureSize)
    {
        var data = new Vector<double>(batchSize * featureSize);
        for (int i = 0; i < data.Length; i++)
        {
            data[i] = Math.Sin(0.37 * i + 0.5);
        }
        return new Tensor<double>(new[] { batchSize, featureSize }, data);
    }

    /// <summary>
    /// Gives every trainable parameter a distinct non-zero value so the low-rank delta is not the zero matrix.
    /// </summary>
    private static void SetNonZeroParameters(LoRAAdapterBase<double> adapter)
    {
        var payload = new Vector<double>(adapter.GetParameters().Length);
        for (int i = 0; i < payload.Length; i++)
        {
            payload[i] = 0.05 * Math.Cos(0.61 * i + 0.2);
        }
        adapter.SetParameters(payload);
    }

    public static IEnumerable<object[]> Adapters()
    {
        yield return new object[] { "Standard" };
        yield return new object[] { "PiSSA" };
        yield return new object[] { "ReLoRA" };
        yield return new object[] { "LoKr" };
        yield return new object[] { "MoRA" };
    }

    private static LoRAAdapterBase<double> Create(string name)
    {
        return name switch
        {
            "Standard" => new StandardLoRAAdapter<double>(CreateBaseLayer(InputSize, OutputSize), Rank, 2.0 * Rank),
            "PiSSA" => new PiSSAAdapter<double>(CreateBaseLayer(InputSize, OutputSize), Rank, 2.0 * Rank),
            "ReLoRA" => new ReLoRAAdapter<double>(CreateBaseLayer(InputSize, OutputSize), Rank, 2.0 * Rank),
            "LoKr" => new LoKrAdapter<double>(CreateBaseLayer(InputSize, OutputSize), Rank, 2.0 * Rank),
            // MoRA needs a square layer; a rank below the width exercises the compress/decompress layout
            "MoRA" => new MoRAAdapter<double>(CreateBaseLayer(OutputSize, OutputSize), Rank, 1.0),
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, "Unknown adapter")
        };
    }

    [Theory(Timeout = 120000)]
    [MemberData(nameof(Adapters))]
    public async Task MergeToOriginalLayer_MatchesAdapterForward(string name)
    {
        await Task.Yield();
        var adapter = Create(name);
        SetNonZeroParameters(adapter);
        int inputSize = name == "MoRA" ? OutputSize : InputSize;
        var input = CreateInput(3, inputSize);

        var adapterOutput = adapter.Forward(input);
        var mergedOutput = adapter.MergeToOriginalLayer().Forward(input);

        Assert.Equal(adapterOutput.Length, mergedOutput.Length);
        for (int i = 0; i < adapterOutput.Length; i++)
        {
            Assert.True(
                Math.Abs(adapterOutput[i] - mergedOutput[i]) <= 1e-9 + 1e-7 * Math.Abs(adapterOutput[i]),
                $"{name} output {i}: adapter {adapterOutput[i]:G12} vs merged {mergedOutput[i]:G12}");
        }
    }
}
