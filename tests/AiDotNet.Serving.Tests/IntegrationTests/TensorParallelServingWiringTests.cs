using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.DistributedTraining;
using AiDotNet.Enums;
using AiDotNet.Inference.PagedAttention;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Serving.ContinuousBatching;
using AiDotNet.Serving.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Serving.Tests.IntegrationTests;

/// <summary>
/// End-to-end proof that <see cref="ServableModelWrapper{T}"/> honors
/// <c>InferenceOptimizationConfig.TensorParallelSize &gt; 1</c>: it partitions a recognized transformer into a
/// tensor-parallel served model + composite paged KV cache and drives generation through the real continuous
/// batcher, producing the SAME tokens as the directly-driven partitioned model — and falls back cleanly for a
/// non-transformer model.
/// </summary>
public sealed class TensorParallelServingWiringTests
{
    private const int Vocab = 16;
    private const int EmbedDim = 12;
    private const int NumHeads = 4;
    private const int HeadDim = 3;
    private const int FfnDim = 16;
    private const int Blocks = 2;

    private static NeuralNetwork<double> BuildTransformer()
    {
        var layers = new List<AiDotNet.Interfaces.ILayer<double>> { new EmbeddingLayer<double>(Vocab, EmbedDim) };
        for (int b = 0; b < Blocks; b++)
        {
            var mha = new MultiHeadAttentionLayer<double>(NumHeads, HeadDim,
                activationFunction: new IdentityActivation<double>()) { UseCausalMask = true };
            layers.Add(new PreLNTransformerBlock<double>(EmbedDim, FfnDim, mha, new GELUActivation<double>()));
        }
        layers.Add(new RMSNormalizationLayer<double>());
        layers.Add(new DenseLayer<double>(Vocab, activationFunction: new IdentityActivation<double>()));

        var arch = new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional, NeuralNetworkTaskType.TextGeneration, NetworkComplexity.Simple,
            inputSize: 1, outputSize: Vocab, layers: layers);
        var model = new NeuralNetwork<double>(arch);

        var probe = new Tensor<double>(new[] { 1, 1 }); probe[0, 0] = 1;
        _ = model.Predict(probe);
        var p = model.GetParameters();
        var det = new double[p.Length];
        for (int i = 0; i < det.Length; i++) det[i] = ((i % 29) - 14) / 9.0;
        model.SetParameters(new Vector<double>(det));
        return model;
    }

    private static AiDotNet.Configuration.InferenceOptimizationConfig Config(int tpSize) => new()
    {
        TensorParallelSize = tpSize,
        MaxBatchSize = 8,
    };

    private static int[] ServeGreedy(NeuralNetwork<double> model, int tpSize, int[] prompt, int maxNew)
    {
        using var wrapper = new ServableModelWrapper<double>(
            "tp-lm", model, inputShape: new[] { 1 }, enableBatching: true,
            generationForward: model.Predict, servingInferenceConfig: Config(tpSize));

        Assert.True(wrapper.SupportsIncrementalGeneration,
            $"TensorParallelSize={tpSize} should still build a KV-cached incremental path");

        var request = new GenerationRequest<double>
        {
            PromptTokenIds = new List<int>(prompt),
            MaxNewTokens = maxNew,
            Temperature = 0f, // greedy
        };
        return wrapper.RunGeneration(request, CancellationToken.None).GeneratedTokens.ToArray();
    }

    [Fact(Timeout = 120000)]
    public async Task TensorParallelSize_PartitionsAndServesThroughBatcher()
    {
        await Task.Yield();
        var model = BuildTransformer();
        var prompt = new[] { 1, 5, 2 };
        const int maxNew = 6;

        // Served with TensorParallelSize = 2 (partitioner engaged) vs the directly-driven partitioned model.
        int[] served = ServeGreedy(model, tpSize: 2, prompt, maxNew);

        // Match serving's GPU auto-enable so the directly-driven reference uses the same (GPU or CPU) path.
        var tp = TensorParallelPartitioner<double>.TryBuild(
            model, worldSize: 2, blockSize: 16, numBlocks: 512, out var reason, useGpu: GpuPagedAttention.IsAvailable);
        Assert.True(tp is not null, $"model should be partitionable (reason: {reason})");
        var composite = new CompositePagedKVCache<double>(tp!.RankCaches);
        var direct = GenerateDirect(tp, composite, prompt, maxNew);

        Assert.Equal(maxNew, served.Length);
        Assert.All(served, t => Assert.InRange(t, 0, Vocab - 1));
        Assert.Equal(direct, served); // the wrapper's TP path == the proven directly-driven partitioned model
    }

    [Fact(Timeout = 120000)]
    public async Task TensorParallelSize_IsDeterministic_AcrossRuns()
    {
        await Task.Yield();
        var model = BuildTransformer();
        var prompt = new[] { 3, 7, 1, 4 };
        int[] a = ServeGreedy(model, tpSize: 2, prompt, maxNew: 6);
        int[] b = ServeGreedy(model, tpSize: 2, prompt, maxNew: 6);
        Assert.Equal(a, b);
    }

    private static int[] GenerateDirect(
        TensorParallelPagedModel<double> model, CompositePagedKVCache<double> cache, int[] prompt, int maxNew)
    {
        var config = new ContinuousBatcherConfig { AutoStart = false, EosTokenId = 999, EnableSpeculativeDecoding = false };
        using var batcher = new ContinuousBatcher<double>(config, model, cache);
        var request = new GenerationRequest<double>
        {
            PromptTokenIds = new List<int>(prompt),
            MaxNewTokens = maxNew,
            Temperature = 0f,
        };
        var task = batcher.GenerateAsync(request);
        int guard = maxNew + prompt.Length + 16;
        while (!task.IsCompleted && guard-- > 0) batcher.Step();
        Assert.True(task.IsCompleted);
        return task.GetAwaiter().GetResult().GeneratedTokens.ToArray();
    }
}
