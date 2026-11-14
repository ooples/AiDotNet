using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// LayerBase.ApplyActivation keeps each pre-activation for an eager Backward to pop. A compiled plan's trace runs the
/// forward in training mode with no tape, so it pushed one batch-sized tensor per activating layer, and the plan's
/// backward never pops: DDPM at batch 64 held about 1.4 GB of them for the model's life.
/// </summary>
[Collection("FusedTrainingSerial")]
public class PreActivationCacheTraceTests
{
    private static readonly FieldInfo s_cache = typeof(LayerBase<float>).GetField(
        "_preActivationCache", BindingFlags.Instance | BindingFlags.NonPublic)
        ?? throw new InvalidOperationException("LayerBase._preActivationCache was not found.");

    private static int Cached(LayerBase<float> layer) => ((ICollection)s_cache.GetValue(layer)!).Count;

    private static Tensor<float> Batch(int step, int rows, int cols)
    {
        var t = new Tensor<float>(new[] { rows, cols });
        for (int i = 0; i < t.Length; i++) t[i] = (float)Math.Sin(0.7 * i + 1.3 * step);
        return t;
    }

    [Fact(Timeout = 120000)]
    public async System.Threading.Tasks.Task ACompiledTrace_LeavesNoPreActivationCached()
    {
        await System.Threading.Tasks.Task.CompletedTask;
        var optimizer = new AdamOptimizer<float, Tensor<float>, Tensor<float>>(null,
            new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>> { InitialLearningRate = 0.01, UseAdaptiveLearningRate = false });
        var model = new FeedForwardNeuralNetwork<float>(
            new NeuralNetworkArchitecture<float>(inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
                complexity: NetworkComplexity.Simple, inputSize: 6, outputSize: 2),
            optimizer, new MeanSquaredErrorLoss<float>());
        var options = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current;
        bool saved = options.EnableCompilation;
        try
        {
            options.EnableCompilation = true;
            for (int s = 0; s < 4; s++) model.Train(Batch(s, 4, 6), Batch(s + 100, 4, 2));
        }
        finally
        {
            options.EnableCompilation = saved;
        }

        var dense = model.Layers.OfType<LayerBase<float>>().Where(l => l is DenseLayer<float> or FullyConnectedLayer<float>).ToList();
        Assert.NotEmpty(dense);
        foreach (var layer in dense)
            Assert.True(Cached(layer) == 0, $"{layer.GetType().Name} kept {Cached(layer)} pre-activations after compiled training.");

        // The push itself is live: a training-mode forward with no tape and no trace keeps one for its Backward.
        var probe = dense[0];
        probe.SetTrainingMode(true);
        probe.Forward(Batch(0, 4, 6));
        Assert.Equal(1, Cached(probe));
    }
}
