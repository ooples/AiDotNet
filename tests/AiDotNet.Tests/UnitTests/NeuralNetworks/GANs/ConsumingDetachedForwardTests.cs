using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.GANs;

/// <summary>
/// The GAN builds its discriminator batch from the generator's detached forward inside that forward
/// (<c>ForwardForTrainingDetached(input, consume)</c>): no copy of the samples, and the batch comes from the forward's
/// arena, so a steady loop reuses it instead of allocating it on the large-object heap every iteration. The samples
/// must be the ones the copying overload returns, and the generator's running statistics must stay where they were.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class ConsumingDetachedForwardTests
{
    private const int Batch = 4;
    private const int Width = 4096;   // [4, 4096] doubles: 128 KB, a large-object-heap buffer the arena pools

    private static NeuralNetworkBase<double> BuildGenerator()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Generative, inputSize: 8, outputSize: Width,
            layers: new List<ILayer<double>>
            {
                new DenseLayer<double>(16, (IActivationFunction<double>)new LeakyReLUActivation<double>()),
                new BatchNormalizationLayer<double>(),
                new DenseLayer<double>(Width, (IActivationFunction<double>)new TanhActivation<double>()),
            });
        var discriminator = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.BinaryClassification, inputSize: Width,
            outputSize: 1,
            layers: new List<ILayer<double>>
            {
                new DenseLayer<double>(1, (IActivationFunction<double>)new IdentityActivation<double>()),
            });
        var gan = new GenerativeAdversarialNetwork<double>(architecture, discriminator, InputType.OneDimensional);
        return (NeuralNetworkBase<double>)gan.Generator;
    }

    private static Tensor<double> Data(int width, int step)
    {
        var tensor = new Tensor<double>([Batch, width]);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = Math.Sin(0.37 * i + 1.3 * step);
        return tensor;
    }

    private static double[] RunningStatistics(NeuralNetworkBase<double> generator)
    {
        var values = new List<double>();
        foreach (var layer in generator.Layers)
        {
            if (layer is not LayerBase<double> concrete) continue;
            foreach (var (_, buffer) in concrete.GetRegisteredBuffers())
                for (int i = 0; i < buffer.Length; i++) values.Add(buffer[i]);
        }
        return values.ToArray();
    }

    [Fact]
    public void TheConsumer_SeesTheCopiedSamples_AndItsBatchIsReusedNextCall()
    {
        var previousEngine = AiDotNetEngine.Current;
        try
        {
            AiDotNetEngine.Current = new CpuEngine();
            var engine = AiDotNetEngine.Current;
            var generator = BuildGenerator();
            generator.SetTrainingMode(true);
            var real = Data(Width, 50);

            var x1 = Data(8, 1);
            var x2 = Data(8, 2);
            var copied1 = generator.ForwardForTrainingDetached(x1).ToArray();   // the first call also sizes the layers
            var copied2 = generator.ForwardForTrainingDetached(x2).ToArray();
            var statistics = RunningStatistics(generator);

            var batches = new List<double[]>();
            var seen = new List<double[]>();
            var combined = new List<double[]>();
            foreach (var x in new[] { x1, x2 })
            {
                generator.ForwardForTrainingDetached(x, fake =>
                {
                    seen.Add(fake.ToArray());
                    var batch = engine.TensorConcatenate(new[] { real, fake }, axis: 0);
                    combined.Add(batch.ToArray());
                    batches.Add(batch.GetDataArray());
                });
            }

            Assert.Equal(copied1, seen[0]);
            Assert.Equal(copied2, seen[1]);
            Assert.Equal(real.ToArray().Concat(copied2).ToArray(), combined[1]);
            Assert.Equal(statistics, RunningStatistics(generator));
            // The second call's batch is the first call's buffer, handed back to the arena's pool when it returned.
            Assert.Same(batches[0], batches[1]);
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
        }
    }
}
