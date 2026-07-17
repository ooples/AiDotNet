using System;
using System.Collections.Generic;
using AiDotNet.Configuration;
using AiDotNet.Interfaces;
using AiDotNet.Agentic.Models.Local;
using AiDotNet.Serving.Engine;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Tests for the config bridge (<see cref="ServingConfigMapper"/>) and tokenizer auto-resolution — the pieces
/// that let the serving engine be configured from the library-wide <see cref="InferenceOptimizationConfig"/>
/// with no hand-tuning, and let a model carry its own tokenizer.
/// </summary>
public class ServingConfigMapperTests
{
    private const int Vocab = 128;

    private sealed class CounterLm : ICausalLmModel<double>
    {
        public int VocabularySize => Vocab;
        public int? EosTokenId => null;
        public Tensor<double> ForwardLogits(Tensor<double> tokenIds)
        {
            int n = tokenIds.Shape[tokenIds.Shape.Length - 1];
            int last = (int)Math.Round(Convert.ToDouble(tokenIds[0, n - 1]));
            var t = new Tensor<double>(new[] { 1, n, Vocab });
            t[0, n - 1, (last + 1) % Vocab] = 1.0;
            return t;
        }
    }

    private sealed class CharTokenizer : IGenerationTokenizer
    {
        public int EosTokenId => -1;
        public IReadOnlyList<int> Encode(string text) => System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Select(text, c => (int)c));
        public string Decode(IReadOnlyList<int> tokenIds) => new string(System.Linq.Enumerable.ToArray(System.Linq.Enumerable.Select(tokenIds, id => (char)id)));
    }

    // A model that carries its own tokenizer.
    private sealed class SelfTokenizingLm : ICausalLmModel<double>, IProvidesGenerationTokenizer
    {
        private readonly CounterLm _inner = new();
        public int VocabularySize => _inner.VocabularySize;
        public int? EosTokenId => _inner.EosTokenId;
        public Tensor<double> ForwardLogits(Tensor<double> tokenIds) => _inner.ForwardLogits(tokenIds);
        public IGenerationTokenizer GetGenerationTokenizer() => new CharTokenizer();
    }

    [Fact]
    public void Default_ProducesUsableOptions()
    {
        var opts = ServingConfigMapper.ToEngineOptions(null);
        opts.Validate(); // must be valid
        Assert.Equal(16, opts.BlockSize); // InferenceOptimizationConfig.Default PagedKVCacheBlockSize
        Assert.True(opts.MaxNumSequences >= 1);
        Assert.True(opts.NumKvBlocks >= opts.MaxNumSequences);
    }

    [Fact]
    public void MapsBatchAndBlockSize_FromConfig()
    {
        var config = new InferenceOptimizationConfig
        {
            EnableBatching = true,
            MaxBatchSize = 8,
            PagedKVCacheBlockSize = 32,
        };
        var opts = ServingConfigMapper.ToEngineOptions(config, eosTokenId: 5, maxContextTokens: 2048);

        Assert.Equal(32, opts.BlockSize);
        Assert.Equal(8, opts.MaxNumSequences);
        Assert.Equal(5, opts.EosTokenId);
        // 8 sequences must each fit ceil(2048/32)+1 = 65 blocks -> pool >= 8*65.
        Assert.True(opts.NumKvBlocks >= 8 * 65);
        opts.Validate();
    }

    [Fact]
    public void BatchingDisabled_UsesSingleSequence()
    {
        var config = new InferenceOptimizationConfig { EnableBatching = false, MaxBatchSize = 64 };
        var opts = ServingConfigMapper.ToEngineOptions(config);
        Assert.Equal(1, opts.MaxNumSequences);
    }

    [Fact]
    public void PagedModelSizing_DerivesBlocksFromMemoryBudgetAndShape()
    {
        // 4 layers, 8 kv heads, headDim 64, block 16, fp16 -> 2*4*8*64*16*2 = 131,072 bytes = 128 KiB/block.
        // A 64 MiB budget therefore yields 64 MiB / 128 KiB = 512 blocks.
        var config = new InferenceOptimizationConfig { KVCacheMaxSizeMB = 64, MaxBatchSize = 8 };
        long perBlock = ServingConfigMapper.KvBytesPerBlock(4, 8, 64, 16, KVCacheQuantizationMode.None);
        Assert.Equal(131072L, perBlock);

        var opts = ServingConfigMapper.ToEngineOptionsForPagedModel(config, null, 4, 8, 64, 16);
        Assert.Equal(16, opts.BlockSize);
        Assert.Equal(512, opts.NumKvBlocks); // 64 MiB / 128 KiB
        opts.Validate();
    }

    [Fact]
    public void PagedModelSizing_Int8_FitsTwiceAsManyBlocks()
    {
        var fp16 = new InferenceOptimizationConfig { KVCacheMaxSizeMB = 64 };
        var int8 = new InferenceOptimizationConfig { KVCacheMaxSizeMB = 64, KVCacheQuantization = KVCacheQuantizationMode.Int8 };

        var a = ServingConfigMapper.ToEngineOptionsForPagedModel(fp16, null, 4, 8, 64, 16);
        var b = ServingConfigMapper.ToEngineOptionsForPagedModel(int8, null, 4, 8, 64, 16);
        Assert.Equal(2 * a.NumKvBlocks, b.NumKvBlocks);
    }

    [Fact]
    public void SpeculationHelpers_ReadConfig()
    {
        Assert.False(ServingConfigMapper.IsSpeculativeEnabled(null));
        var spec = new InferenceOptimizationConfig { EnableSpeculativeDecoding = true, SpeculationDepth = 6 };
        Assert.True(ServingConfigMapper.IsSpeculativeEnabled(spec));
        Assert.Equal(6, ServingConfigMapper.SpeculationDepth(spec));
    }

    [Fact]
    public void TextGenerator_AutoResolvesTokenizer_FromModel()
    {
        using var gen = new TextGenerator<double>(new SelfTokenizingLm());
        // No tokenizer passed; the model provides one. 'A'(65) -> 'B','C' via counter.
        var text = gen.Generate("A", new SamplingParameters { Temperature = 0.0, MaxTokens = 2 });
        Assert.Equal("BC", text);
    }

    [Fact]
    public void TextGenerator_ConfigDerivesEngineOptions()
    {
        var config = new InferenceOptimizationConfig { MaxBatchSize = 4, PagedKVCacheBlockSize = 16 };
        using var gen = new TextGenerator<double>(new CounterLm(), config: config);
        // Still generates correctly with config-derived options.
        var ids = gen.Generate(new[] { 1 }, new SamplingParameters { Temperature = 0.0, MaxTokens = 3 });
        Assert.Equal(new[] { 2, 3, 4 }, ids);
    }
}
