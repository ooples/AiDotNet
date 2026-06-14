// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Streaming;

/// <summary>
/// End-to-end transparent int8 weight-streaming INFERENCE: a Dense network configured with
/// <see cref="StreamingStoreDtype.Int8"/> computes its Linear layers directly on int8 weights
/// (FusedLinear routes the int8-streamed weight to the int8 weight-only GEMM, no upcast to fp32)
/// without any layer-code change. The int8 output matches the fp32 output of the same weights
/// within quantization tolerance, proving the path is both wired and correct.
/// </summary>
public class StreamingInt8InferenceIntegrationTests
{
    private static NeuralNetwork<float> BuildNet()
    {
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(outputSize: 32, activationFunction: null),
            new DenseLayer<float>(outputSize: 16, activationFunction: null),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 64,
            outputSize: 16,
            layers: layers);
        return new NeuralNetwork<float>(arch);
    }

    [Fact]
    public void Int8StreamingInference_TransparentlyRoutesAndMatchesFp32()
    {
        var net = BuildNet();
        net.SetTrainingMode(false);

        var input = new Tensor<float>(new[] { 1, 64 });
        var rng = new Random(7);
        for (int i = 0; i < 64; i++) input[i] = (float)(rng.NextDouble() - 0.5);

        // fp32 baseline (also materializes the lazily-initialized Dense weights).
        var fp32 = net.Predict(input);

        // Configure int8 weight streaming; ConfigureWeightLifetime registers the now-initialized
        // weights with the pool. A tiny budget guarantees they page out.
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-int8infer-{Guid.NewGuid():N}");
        net.ConfigureWeightLifetime(new GpuOffloadOptions
        {
            StreamingBackingStorePath = dir,
            StreamingPoolMaxResidentBytes = 64L * 1024,
            StreamingStoreDtype = StreamingStoreDtype.Int8,
        });
        net.SetTrainingMode(false); // inference → int8 no-upcast routing in FusedLinear

        try
        {
            var int8 = net.Predict(input);

            double sum2 = 0, ref2 = 0;
            int diffs = 0;
            for (int i = 0; i < 16; i++)
            {
                double e = int8[i] - fp32[i];
                sum2 += e * e;
                ref2 += (double)fp32[i] * fp32[i];
                if (int8[i] != fp32[i]) diffs++;
            }
            double rel = Math.Sqrt(sum2 / Math.Max(1e-30, ref2));

            // Same weights, int8-quantized per output channel across two Dense layers → output
            // tracks fp32 within a few %.
            Assert.True(rel < 0.08, $"int8 streaming inference should match fp32 within ~8% (rel {rel:E3}).");
            // ...and it actually quantized (not a silent fp32 fallback).
            Assert.True(diffs > 0, "int8 path should differ from fp32 (proves quantization ran, not a fallback).");
        }
        finally
        {
            WeightRegistry.SetStreamingExecutionTraining(null);
            WeightRegistry.Reset();
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }
}
