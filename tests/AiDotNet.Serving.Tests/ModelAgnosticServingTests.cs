using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.Configuration;
using AiDotNet.Interfaces;
using AiDotNet.Serving.Engine;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Demonstrates that the serving engine is model-agnostic: because it drives a model only through
/// <see cref="ICausalLmModel{T}"/>, models that internally quantize their weights, or shard their forward pass
/// across tensor-parallel / pipeline-parallel workers, serve unchanged. Also verifies KV-cache quantization
/// raises the derived block-pool capacity.
/// </summary>
public class ModelAgnosticServingTests
{
    private const int Vocab = 64;

    /// <summary>Simulates a tensor-parallel output projection: two "workers" each fill half of the vocab
    /// logits in parallel, then the halves are concatenated — mimicking column-parallel + gather.</summary>
    private sealed class TensorParallelCounterLm : ICausalLmModel<double>
    {
        public int VocabularySize => Vocab;
        public int? EosTokenId => null;
        public Tensor<double> ForwardLogits(Tensor<double> tokenIds)
        {
            int n = tokenIds.Shape[tokenIds.Shape.Length - 1];
            int last = (int)Math.Round(Convert.ToDouble(tokenIds[0, n - 1]));
            int target = (last + 1) % Vocab;
            var t = new Tensor<double>(new[] { 1, n, Vocab });

            int half = Vocab / 2;
            Parallel.Invoke(
                () => { for (int v = 0; v < half; v++) t[0, n - 1, v] = v == target ? 1.0 : 0.0; },
                () => { for (int v = half; v < Vocab; v++) t[0, n - 1, v] = v == target ? 1.0 : 0.0; });
            return t;
        }
    }

    /// <summary>Simulates pipeline-parallel execution: the forward runs through sequential "stages" before
    /// producing logits (each stage is a no-op transform here; the point is multi-stage forward serves).</summary>
    private sealed class PipelineParallelCounterLm : ICausalLmModel<double>
    {
        public int VocabularySize => Vocab;
        public int? EosTokenId => null;
        public Tensor<double> ForwardLogits(Tensor<double> tokenIds)
        {
            int n = tokenIds.Shape[tokenIds.Shape.Length - 1];
            int value = (int)Math.Round(Convert.ToDouble(tokenIds[0, n - 1]));
            // Three sequential stages, each transforming the activation and passing it to the next (pipeline).
            Func<int, int>[] stages = { x => x + 1, x => x - 1, x => x }; // net-identity across the pipeline
            foreach (var stage in stages) value = stage(value);
            int target = (value + 1) % Vocab;
            var t = new Tensor<double>(new[] { 1, n, Vocab });
            t[0, n - 1, target] = 1.0;
            return t;
        }
    }

    /// <summary>Simulates INT8 weight quantization: logits are rounded to a coarse grid (as a quantized model's
    /// would be), which must not change the argmax the sampler selects.</summary>
    private sealed class QuantizedCounterLm : ICausalLmModel<double>
    {
        public int VocabularySize => Vocab;
        public int? EosTokenId => null;
        public Tensor<double> ForwardLogits(Tensor<double> tokenIds)
        {
            int n = tokenIds.Shape[tokenIds.Shape.Length - 1];
            int last = (int)Math.Round(Convert.ToDouble(tokenIds[0, n - 1]));
            int target = (last + 1) % Vocab;
            var t = new Tensor<double>(new[] { 1, n, Vocab });
            for (int v = 0; v < Vocab; v++)
            {
                double logit = v == target ? 5.0 : 0.1 * (v % 3); // small noise on non-target logits
                t[0, n - 1, v] = Math.Round(logit * 16.0) / 16.0;  // quantize to a coarse grid
            }
            return t;
        }
    }

    private static int[] Run(ICausalLmModel<double> model, int prompt, int maxTokens)
    {
        using var engine = new ContinuousBatchingEngine<double>(new RecomputeModelRunner<double>(model));
        engine.AddRequest(new GenerationRequest("r", new[] { prompt },
            new SamplingParameters { Temperature = 0.0, MaxTokens = maxTokens }));
        RequestOutput? final = null;
        int steps = 0;
        while (engine.HasUnfinishedRequests)
        {
            if (++steps > 1000) throw new InvalidOperationException("no convergence");
            foreach (var o in engine.Step()) if (o.IsFinished) final = o;
        }
        return System.Linq.Enumerable.ToArray(final!.Outputs[0].TokenIds);
    }

    [Fact]
    public void TensorParallelModel_ServesCorrectly()
        => Assert.Equal(new[] { 6, 7, 8, 9 }, Run(new TensorParallelCounterLm(), 5, 4));

    [Fact]
    public void PipelineParallelModel_ServesCorrectly()
        => Assert.Equal(new[] { 6, 7, 8, 9 }, Run(new PipelineParallelCounterLm(), 5, 4));

    [Fact]
    public void QuantizedModel_ServesCorrectly()
        => Assert.Equal(new[] { 6, 7, 8, 9 }, Run(new QuantizedCounterLm(), 5, 4));

    [Fact]
    public void KvCacheQuantization_RaisesDerivedBlockCapacity()
    {
        var baseline = new InferenceOptimizationConfig { MaxBatchSize = 4, PagedKVCacheBlockSize = 16 };
        var quantized = new InferenceOptimizationConfig
        {
            MaxBatchSize = 4,
            PagedKVCacheBlockSize = 16,
            KVCacheQuantization = KVCacheQuantizationMode.Int8,
        };

        var baseOpts = ServingConfigMapper.ToEngineOptions(baseline, maxContextTokens: 2048);
        var quantOpts = ServingConfigMapper.ToEngineOptions(quantized, maxContextTokens: 2048);

        Assert.Equal(KVCacheQuantizationMode.None, baseOpts.KvCacheQuantization);
        Assert.Equal(KVCacheQuantizationMode.Int8, quantOpts.KvCacheQuantization);
        Assert.Equal(2 * baseOpts.NumKvBlocks, quantOpts.NumKvBlocks); // Int8 KV ⇒ ~2× capacity
    }
}
