using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// One AdamW training step on a convolutional network must update every parameter tensor the same way whether it
/// runs through the fused compiled plan or the eager tape. The PyTorch parity harness's step-1 check found the fused
/// plan moving the CNN's conv weights in a different direction from both the eager tape and PyTorch (update cosine
/// 0.21-0.50) while the dense head matched exactly; the variants bisect which layer's fused backward is wrong.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class FusedConvolutionalStepParityTests
{
    public enum CnnVariant
    {
        /// <summary>Conv(1→8, 3x3, pad 1) + ReLU + Flatten + Dense.</summary>
        ConvOnly,
        /// <summary>Conv + ReLU + MaxPool(2) + Flatten + Dense.</summary>
        ConvMaxPool,
        /// <summary>Conv + ReLU + AdaptiveAvgPool 7x7→4x4 (non-dividing) + Flatten + Dense.</summary>
        ConvAdaptiveAvgPool,
        /// <summary>The parity harness CNN: Conv+ReLU+MaxPool+Conv+ReLU+AdaptiveAvgPool(4x4)+Flatten+Dense.</summary>
        ParityHarnessCnn,
    }

    [Theory(Timeout = 180000)]
    [InlineData(CnnVariant.ConvOnly)]
    [InlineData(CnnVariant.ConvMaxPool)]
    [InlineData(CnnVariant.ConvAdaptiveAvgPool)]
    [InlineData(CnnVariant.ParityHarnessCnn)]
    public async System.Threading.Tasks.Task FusedStep_MatchesEagerStep_PerParameterTensor(CnnVariant variant)
    {
        await System.Threading.Tasks.Task.CompletedTask;
        int side = variant == CnnVariant.ConvAdaptiveAvgPool ? 7 : 14;
        var input = RandomTensor([4, 1, side, side], seed: 11);
        var labels = OneHot([3, 0, 7, 9], classes: 10);

        float[]? initial = null;
        (float[] Before, float[] After, int[] TensorLengths, long FusedSteps) Step(bool compile)
        {
            var network = Build(variant, side);
            network.Predict(RandomTensor([1, 1, side, side], seed: 5));
            if (initial is null) initial = network.GetParameters().ToArray();
            else network.UpdateParameters(new Vector<float>(initial));

            var original = TensorCodecOptions.Current;
            try
            {
                TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
                CompiledTapeTrainingStep<float>.Invalidate();
                CompiledTapeTrainingStep<float>.ResetFusedStepCount();
                network.Train(input, labels);
                var lengths = network.Layers.OfType<LayerBase<float>>()
                    .SelectMany(l => l.GetTrainableParameters()).Select(t => t.Length).ToArray();
                return (initial, network.GetParameters().ToArray(), lengths, CompiledTapeTrainingStep<float>.GetFusedStepCount());
            }
            finally
            {
                TensorCodecOptions.SetCurrent(original);
                CompiledTapeTrainingStep<float>.Invalidate();
                CompiledTapeTrainingStep<float>.ResetFusedStepCount();
            }
        }

        var fused = Step(compile: true);
        var eager = Step(compile: false);
        Assert.Equal(1, fused.FusedSteps);
        Assert.Equal(0, eager.FusedSteps);

        var failures = new List<string>();
        int offset = 0;
        for (int t = 0; t < fused.TensorLengths.Length; t++)
        {
            double dot = 0, fn = 0, en = 0, diff = 0;
            for (int k = offset; k < offset + fused.TensorLengths[t]; k++)
            {
                double f = fused.After[k] - fused.Before[k], e = eager.After[k] - eager.Before[k];
                dot += f * e; fn += f * f; en += e * e; diff += (f - e) * (f - e);
            }
            offset += fused.TensorLengths[t];
            Assert.True(en > 0, $"tensor {t}: the eager step did not move it.");
            double cosine = dot / Math.Sqrt(fn * en), relError = Math.Sqrt(diff / en);
            if (cosine < 0.9999 || relError > 1e-3)
                failures.Add($"tensor {t} [{fused.TensorLengths[t]}]: update cosine {cosine:F5}, relative error {relError:E2}");
        }

        Assert.True(failures.Count == 0,
            $"{variant}: the fused step diverged from the eager step:{Environment.NewLine}{string.Join(Environment.NewLine, failures)}");
    }

    private static ConvolutionalNeuralNetwork<float> Build(CnnVariant variant, int side)
    {
        IActivationFunction<float> relu = new ReLUActivation<float>();
        var layers = new List<ILayer<float>> { new ConvolutionalLayer<float>(outputDepth: 8, kernelSize: 3, stride: 1, padding: 1, activationFunction: relu) };
        switch (variant)
        {
            case CnnVariant.ConvMaxPool:
                layers.Add(new MaxPoolingLayer<float>(poolSize: 2, stride: 2));
                break;
            case CnnVariant.ConvAdaptiveAvgPool:
                layers.Add(new AdaptiveAveragePoolingLayer<float>(outputHeight: 4, outputWidth: 4));
                break;
            case CnnVariant.ParityHarnessCnn:
                layers.Add(new MaxPoolingLayer<float>(poolSize: 2, stride: 2));
                layers.Add(new ConvolutionalLayer<float>(outputDepth: 16, kernelSize: 3, stride: 1, padding: 1, activationFunction: new ReLUActivation<float>()));
                layers.Add(new AdaptiveAveragePoolingLayer<float>(outputHeight: 4, outputWidth: 4));
                break;
        }
        layers.Add(new FlattenLayer<float>());
        layers.Add(new DenseLayer<float>(10, activationFunction: (IActivationFunction<float>?)null));

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: side, inputWidth: side, inputDepth: 1,
            outputSize: 10,
            layers: layers);
        var network = new ConvolutionalNeuralNetwork<float>(architecture, lossFunction: new CrossEntropyWithLogitsLoss<float>());
        network.SetBaseTrainOptimizer(new AdamWOptimizer<float, Tensor<float>, Tensor<float>>(null,
            new AdamWOptimizerOptions<float, Tensor<float>, Tensor<float>>
            {
                InitialLearningRate = 1e-3, Beta1 = 0.9, Beta2 = 0.999, Epsilon = 1e-8, WeightDecay = 0.01,
            }));
        return network;
    }

    private static Tensor<float> RandomTensor(int[] shape, int seed)
    {
        var random = new Random(seed);
        var data = new float[shape.Aggregate(1, (a, b) => a * b)];
        for (int i = 0; i < data.Length; i++) data[i] = (float)random.NextDouble();
        return new Tensor<float>(data, shape);
    }

    private static Tensor<float> OneHot(int[] labels, int classes)
    {
        var data = new float[labels.Length * classes];
        for (int b = 0; b < labels.Length; b++) data[b * classes + labels[b]] = 1f;
        return new Tensor<float>(data, [labels.Length, classes]);
    }
}
