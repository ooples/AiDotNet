// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.MixedPrecision;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.MixedPrecision;

/// <summary>
/// An eager mixed-precision step on a GPU engine autocasts its matrix products, as the fused compiled step does
/// (PyTorch's <c>torch.autocast</c>): the FP32 master weights are not rounded, every product inside the step rounds its
/// operands to 16 bits, and restoring the masters closes the scope so the optimizer update runs outside it. On the CPU
/// engine, which has no matmul autocast, the step keeps rounding its working weights.
/// </summary>
// Sets AiDotNetEngine.Current, a process-wide static: run apart from every other test class.
[Collection("EngineCurrentGlobalState")]
public class EagerMixedPrecisionAutocastTests
{
    private readonly ITestOutputHelper _output;
    public EagerMixedPrecisionAutocastTests(ITestOutputHelper output) => _output = output;

    private static NeuralNetwork<float> BuildNet()
    {
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 8,
            outputSize: 4,
            layers: new List<ILayer<float>>
            {
                new DenseLayer<float>(4, activationFunction: new IdentityActivation<float>()),
            })
        {
            RandomSeed = 7
        };
        var net = new NeuralNetwork<float>(arch, lossFunction: new MeanSquaredErrorLoss<float>());
        _ = net.Predict(new Tensor<float>(new[] { 1, 8 }));
        // Weights with bits below BF16's 8-bit mantissa, so a rounded copy differs from the master.
        foreach (var p in net.GetParameterChunks())
            for (int i = 0; i < p.Length; i++) p.SetFlat(i, 1f + (i + 1) * 1e-4f);
        return net;
    }

    private static float RoundToBf16(float v)
    {
        uint bits = BitConverter.ToUInt32(BitConverter.GetBytes(v), 0);
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        return BitConverter.ToSingle(BitConverter.GetBytes(bits & 0xFFFF0000u), 0);
    }

    [SkippableFact]
    public void OnTheGpu_TheStepAutocastsItsProducts_AndLeavesTheMasterWeightsUnrounded()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) { _output.WriteLine($"No GPU engine: {ex.GetType().Name}: {ex.Message}"); }
        var previous = AiDotNetEngine.Current;
        try
        {
            Skip.If(gpu is null || !gpu.SupportsGpu || !gpu.IsGpuAvailable, "No GPU backend available.");
            AiDotNetEngine.Current = gpu!;
            var net = BuildNet();
            var parameters = net.GetParameterChunks().Where(p => p is not null && p.Length > 0).ToList();
            var masters = parameters.Select(p => p.ToArray()).ToArray();

            using var context = new MixedPrecisionContext(new MixedPrecisionConfig { PrecisionType = MixedPrecisionType.BF16 });
            bool restored = false;
            net.PrepareMixedPrecisionWorkingWeights(context, parameters, out var snapshots, out var targets, ref restored);
            try
            {
                Assert.False(restored);
                Assert.IsType<AutocastWorkingWeights>(snapshots);
                for (int k = 0; k < parameters.Count; k++)
                    Assert.Equal(masters[k], parameters[k].ToArray());
                Assert.True(gpu!.MatMulAutocastActive, "The eager step did not open the matmul autocast scope.");

                // A product inside the step multiplies the BF16-rounded operands (exact in FP32 here: 1x1 inner dim).
                var a = new Tensor<float>(new[] { 1, 1 }, new Vector<float>(new[] { 1.001f }));
                var b = new Tensor<float>(new[] { 1, 1 }, new Vector<float>(new[] { 3.0007f }));
                float product = gpu.TensorMatMul(a, b).ToArray()[0];
                Assert.Equal(RoundToBf16(1.001f) * RoundToBf16(3.0007f), product);
            }
            finally
            {
                NeuralNetworkBase<float>.RestoreMixedPrecisionMasterWeights(snapshots, targets, ref restored);
            }

            Assert.True(restored);
            Assert.False(gpu!.MatMulAutocastActive, "Restoring the master weights must close the autocast scope.");
            for (int k = 0; k < parameters.Count; k++)
                Assert.Equal(masters[k], parameters[k].ToArray());
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            gpu?.Dispose();
        }
    }

    [Fact]
    public void OnTheCpu_TheStepRoundsItsWorkingWeights_AndRestoresTheMasters()
    {
        var previous = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            var net = BuildNet();
            var parameters = net.GetParameterChunks().Where(p => p is not null && p.Length > 0).ToList();
            var masters = parameters.Select(p => p.ToArray()).ToArray();

            using var context = new MixedPrecisionContext(new MixedPrecisionConfig { PrecisionType = MixedPrecisionType.BF16 });
            bool restored = false;
            net.PrepareMixedPrecisionWorkingWeights(context, parameters, out var snapshots, out var targets, ref restored);
            Assert.IsNotType<AutocastWorkingWeights>(snapshots);
            Assert.Contains(Enumerable.Range(0, parameters.Count),
                k => !masters[k].SequenceEqual(parameters[k].ToArray()));
            NeuralNetworkBase<float>.RestoreMixedPrecisionMasterWeights(snapshots, targets, ref restored);
            for (int k = 0; k < parameters.Count; k++)
                Assert.Equal(masters[k], parameters[k].ToArray());
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
