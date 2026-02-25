using System;
using System.Linq;
using AiDotNet.Inference.PagedAttention;
using AiDotNet.Inference.SpeculativeDecoding;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Inference;

/// <summary>
/// Integration tests for Inference PagedAttention BlockManager, SpeculativeDecoding
/// config/stats, and NGramDraftModel. Tests memory management, allocation/deallocation,
/// reference counting, and speculative decoding configuration.
/// </summary>
public class InferencePagedAttentionIntegrationTests
{
    #region BlockManager Basic Allocation Tests

    [Fact]
    public void BlockManager_InitialState_AllBlocksFree()
    {
        var config = new BlockManagerConfig { NumBlocks = 10, BlockSize = 16 };
        var manager = new BlockManager<double>(config);

        Assert.Equal(10, manager.FreeBlockCount);
        Assert.Equal(0, manager.AllocatedBlockCount);
        Assert.Equal(10, manager.TotalBlocks);
        Assert.Equal(0.0, manager.MemoryUtilization);
    }

    [Fact]
    public void BlockManager_AllocateSingle_ReducesFreeCount()
    {
        var config = new BlockManagerConfig { NumBlocks = 5 };
        var manager = new BlockManager<double>(config);

        int blockId = manager.AllocateBlock();

        Assert.True(blockId >= 0);
        Assert.Equal(4, manager.FreeBlockCount);
        Assert.Equal(1, manager.AllocatedBlockCount);
    }

    [Fact]
    public void BlockManager_AllocateAll_NoMoreFree()
    {
        var config = new BlockManagerConfig { NumBlocks = 3 };
        var manager = new BlockManager<double>(config);

        int b0 = manager.AllocateBlock();
        int b1 = manager.AllocateBlock();
        int b2 = manager.AllocateBlock();

        Assert.True(b0 >= 0 && b1 >= 0 && b2 >= 0);
        Assert.Equal(0, manager.FreeBlockCount);
        Assert.Equal(3, manager.AllocatedBlockCount);

        // Next allocation should return -1
        int overflow = manager.AllocateBlock();
        Assert.Equal(-1, overflow);
    }

    [Fact]
    public void BlockManager_AllocateMultiple_AllocatesCorrectCount()
    {
        var config = new BlockManagerConfig { NumBlocks = 10 };
        var manager = new BlockManager<double>(config);

        var blocks = manager.AllocateBlocks(4);

        Assert.NotNull(blocks);
        Assert.Equal(4, blocks.Length);
        Assert.Equal(6, manager.FreeBlockCount);
        Assert.Equal(4, manager.AllocatedBlockCount);

        // All block IDs should be unique
        Assert.Equal(blocks.Length, blocks.Distinct().Count());
    }

    [Fact]
    public void BlockManager_AllocateMultiple_InsufficientBlocks_ReturnsNull()
    {
        var config = new BlockManagerConfig { NumBlocks = 3 };
        var manager = new BlockManager<double>(config);

        var blocks = manager.AllocateBlocks(5); // Only 3 available

        Assert.Null(blocks);
        Assert.Equal(3, manager.FreeBlockCount); // No change
    }

    #endregion

    #region BlockManager Free Tests

    [Fact]
    public void BlockManager_FreeSingle_RestoresFreeCount()
    {
        var config = new BlockManagerConfig { NumBlocks = 5 };
        var manager = new BlockManager<double>(config);

        int blockId = manager.AllocateBlock();
        Assert.Equal(4, manager.FreeBlockCount);

        manager.FreeBlock(blockId);
        Assert.Equal(5, manager.FreeBlockCount);
        Assert.Equal(0, manager.AllocatedBlockCount);
    }

    [Fact]
    public void BlockManager_FreeMultiple_RestoresAll()
    {
        var config = new BlockManagerConfig { NumBlocks = 10 };
        var manager = new BlockManager<double>(config);

        var blocks = manager.AllocateBlocks(5);
        Assert.NotNull(blocks);
        Assert.Equal(5, manager.AllocatedBlockCount);

        manager.FreeBlocks(blocks);
        Assert.Equal(10, manager.FreeBlockCount);
        Assert.Equal(0, manager.AllocatedBlockCount);
    }

    [Fact]
    public void BlockManager_FreeUnallocated_NoEffect()
    {
        var config = new BlockManagerConfig { NumBlocks = 5 };
        var manager = new BlockManager<double>(config);

        // Free a block that was never allocated
        manager.FreeBlock(999);
        Assert.Equal(5, manager.FreeBlockCount); // No change
    }

    [Fact]
    public void BlockManager_AllocateFreeCycle_ReusesBlocks()
    {
        var config = new BlockManagerConfig { NumBlocks = 2 };
        var manager = new BlockManager<double>(config);

        // Allocate both
        int b0 = manager.AllocateBlock();
        int b1 = manager.AllocateBlock();
        Assert.Equal(0, manager.FreeBlockCount);

        // Free one
        manager.FreeBlock(b0);
        Assert.Equal(1, manager.FreeBlockCount);

        // Allocate again - should reuse freed block
        int b2 = manager.AllocateBlock();
        Assert.True(b2 >= 0);
        Assert.Equal(0, manager.FreeBlockCount);
    }

    #endregion

    #region BlockManager Memory Utilization Tests

    [Fact]
    public void BlockManager_MemoryUtilization_CalculatesCorrectly()
    {
        var config = new BlockManagerConfig { NumBlocks = 10 };
        var manager = new BlockManager<double>(config);

        Assert.Equal(0.0, manager.MemoryUtilization, 1e-6);

        manager.AllocateBlocks(5);
        Assert.Equal(0.5, manager.MemoryUtilization, 1e-6);

        manager.AllocateBlocks(5);
        Assert.Equal(1.0, manager.MemoryUtilization, 1e-6);
    }

    [Fact]
    public void BlockManager_Config_BytesPerBlock_GoldenReference()
    {
        var config = new BlockManagerConfig
        {
            BlockSize = 16,     // tokens per block
            NumLayers = 2,
            NumHeads = 4,
            HeadDimension = 8
        };

        // BytesPerBlock = BlockSize * NumLayers * NumHeads * HeadDim * sizeof(float) * 2
        // = 16 * 2 * 4 * 8 * 4 * 2 = 8192
        Assert.Equal(8192, config.BytesPerBlock);
    }

    [Fact]
    public void BlockManagerConfig_DefaultValues()
    {
        var config = new BlockManagerConfig();

        Assert.Equal(16, config.BlockSize);
        Assert.Equal(1024, config.NumBlocks);
        Assert.Equal(32, config.NumLayers);
        Assert.Equal(32, config.NumHeads);
        Assert.Equal(128, config.HeadDimension);
        Assert.False(config.UseGpuMemory);
    }

    #endregion

    #region SpeculativeDecodingConfig Tests

    [Fact]
    public void SpeculativeDecodingConfig_DefaultValues()
    {
        var config = new SpeculativeDecodingConfig<double>();

        Assert.Equal(5, config.NumDraftTokens);
        Assert.Null(config.Seed);
        Assert.False(config.UseTreeSpeculation);
        Assert.Equal(2, config.TreeBranchFactor);
        Assert.Equal(4, config.MaxTreeDepth);
        Assert.False(config.AdaptiveDraftLength);
    }

    [Fact]
    public void SpeculativeDecodingConfig_SetProperties()
    {
        var config = new SpeculativeDecodingConfig<double>
        {
            NumDraftTokens = 8,
            Seed = 42,
            UseTreeSpeculation = true,
            TreeBranchFactor = 3,
            MaxTreeDepth = 6,
            AdaptiveDraftLength = true
        };

        Assert.Equal(8, config.NumDraftTokens);
        Assert.Equal(42, config.Seed);
        Assert.True(config.UseTreeSpeculation);
        Assert.Equal(3, config.TreeBranchFactor);
        Assert.Equal(6, config.MaxTreeDepth);
        Assert.True(config.AdaptiveDraftLength);
    }

    [Fact]
    public void SpeculativeDecodingConfig_Float_DefaultValues()
    {
        var config = new SpeculativeDecodingConfig<float>();

        Assert.Equal(5, config.NumDraftTokens);
        Assert.False(config.UseTreeSpeculation);
    }

    #endregion

    #region SpeculativeDecodingStats Tests

    [Fact]
    public void SpeculativeDecodingStats_DefaultValues()
    {
        var stats = new SpeculativeDecodingStats();

        Assert.Equal(0L, stats.TotalTokensGenerated);
        Assert.Equal(0L, stats.TotalDraftTokens);
        Assert.Equal(0L, stats.AcceptedDraftTokens);
        Assert.Equal(0L, stats.TotalVerificationCalls);
        Assert.Equal(0.0, stats.AcceptanceRate);
        Assert.Equal(0.0, stats.TokensPerVerification);
        Assert.Equal(0.0, stats.SpeedupEstimate);
    }

    [Fact]
    public void SpeculativeDecodingStats_SetAndVerify()
    {
        var stats = new SpeculativeDecodingStats
        {
            TotalTokensGenerated = 100,
            TotalDraftTokens = 80,
            AcceptedDraftTokens = 60,
            TotalVerificationCalls = 20,
            AcceptanceRate = 0.75,
            TokensPerVerification = 5.0,
            SpeedupEstimate = 2.5
        };

        Assert.Equal(100L, stats.TotalTokensGenerated);
        Assert.Equal(80L, stats.TotalDraftTokens);
        Assert.Equal(60L, stats.AcceptedDraftTokens);
        Assert.Equal(20L, stats.TotalVerificationCalls);
        Assert.Equal(0.75, stats.AcceptanceRate);
        Assert.Equal(5.0, stats.TokensPerVerification);
        Assert.Equal(2.5, stats.SpeedupEstimate);
    }

    [Fact]
    public void SpeculativeDecodingStats_AcceptanceRate_Consistency()
    {
        var stats = new SpeculativeDecodingStats
        {
            TotalDraftTokens = 200,
            AcceptedDraftTokens = 150,
        };

        // AcceptanceRate should be set manually (not auto-calculated)
        // but we can verify the expected relationship
        double expectedRate = (double)stats.AcceptedDraftTokens / stats.TotalDraftTokens;
        Assert.Equal(0.75, expectedRate, 1e-6);
    }

    #endregion

    #region NGramDraftModel Tests

    [Fact]
    public void NGramDraftModel_Constructs_WithDefaults()
    {
        var model = new NGramDraftModel<double>();

        Assert.Equal(8, model.MaxDraftTokens);
        Assert.Equal(50000, model.VocabSize);
    }

    [Fact]
    public void NGramDraftModel_Constructs_WithCustomParams()
    {
        var model = new NGramDraftModel<double>(ngramSize: 4, vocabSize: 1000, seed: 42);

        Assert.Equal(8, model.MaxDraftTokens);
        Assert.Equal(1000, model.VocabSize);
    }

    [Fact]
    public void NGramDraftModel_Train_LearnsCounts()
    {
        var model = new NGramDraftModel<double>(ngramSize: 2, vocabSize: 10, seed: 42);

        // Train on simple repeating patterns
        var corpus = new[]
        {
            new Vector<int>(new[] { 1, 2, 3, 1, 2, 3, 1, 2, 3 }),
            new Vector<int>(new[] { 4, 5, 6, 4, 5, 6 }),
        };

        model.Train(corpus);

        // After training, model should be able to generate drafts
        var input = new Vector<int>(new[] { 1, 2 });
        var temperature = 1.0;
        var result = model.GenerateDraft(input, numDraftTokens: 3, temperature);

        Assert.NotNull(result);
        Assert.Equal(3, result.NumTokens);
        Assert.Equal(3, result.Tokens.Length);
    }

    [Fact]
    public void NGramDraftModel_GenerateDraft_UntainedModel_StillGenerates()
    {
        var model = new NGramDraftModel<double>(ngramSize: 2, vocabSize: 5, seed: 42);
        // Don't train - should still produce tokens (random fallback)

        var input = new Vector<int>(new[] { 0, 1 });
        var result = model.GenerateDraft(input, numDraftTokens: 3, 1.0);

        Assert.NotNull(result);
        Assert.Equal(3, result.NumTokens);
    }

    [Fact]
    public void NGramDraftModel_Deterministic_SameSeedSameOutput()
    {
        var corpus = new[]
        {
            new Vector<int>(new[] { 1, 2, 3, 4, 5, 1, 2, 3, 4, 5 }),
        };

        var model1 = new NGramDraftModel<double>(ngramSize: 2, vocabSize: 10, seed: 42);
        model1.Train(corpus);

        var model2 = new NGramDraftModel<double>(ngramSize: 2, vocabSize: 10, seed: 42);
        model2.Train(corpus);

        var input = new Vector<int>(new[] { 1, 2 });
        var result1 = model1.GenerateDraft(input, numDraftTokens: 5, 1.0);
        var result2 = model2.GenerateDraft(input, numDraftTokens: 5, 1.0);

        for (int i = 0; i < result1.NumTokens; i++)
        {
            Assert.Equal(result1.Tokens[i], result2.Tokens[i]);
        }
    }

    [Fact]
    public void NGramDraftModel_DraftResult_Properties()
    {
        var result = new DraftResult<double>();

        Assert.Equal(0, result.NumTokens);
        Assert.Equal(0, result.Tokens.Length);
    }

    #endregion

    #region End-to-End BlockManager Tests

    [Fact]
    public void BlockManager_SimulateMultipleSequences()
    {
        var config = new BlockManagerConfig { NumBlocks = 20, BlockSize = 16 };
        var manager = new BlockManager<double>(config);

        // Simulate 3 sequences requesting blocks
        var seq1Blocks = manager.AllocateBlocks(5);
        var seq2Blocks = manager.AllocateBlocks(3);
        var seq3Blocks = manager.AllocateBlocks(4);

        Assert.NotNull(seq1Blocks);
        Assert.NotNull(seq2Blocks);
        Assert.NotNull(seq3Blocks);
        Assert.Equal(8, manager.FreeBlockCount); // 20 - 12 = 8

        // Sequence 2 finishes
        manager.FreeBlocks(seq2Blocks);
        Assert.Equal(11, manager.FreeBlockCount); // 8 + 3 = 11

        // Sequence 4 starts - can use freed blocks
        var seq4Blocks = manager.AllocateBlocks(6);
        Assert.NotNull(seq4Blocks);
        Assert.Equal(5, manager.FreeBlockCount); // 11 - 6 = 5

        // Clean up remaining sequences
        manager.FreeBlocks(seq1Blocks);
        manager.FreeBlocks(seq3Blocks);
        manager.FreeBlocks(seq4Blocks);

        Assert.Equal(20, manager.FreeBlockCount);
        Assert.Equal(0, manager.AllocatedBlockCount);
    }

    [Fact]
    public void BlockManager_AllBlockIdsUnique()
    {
        var config = new BlockManagerConfig { NumBlocks = 100 };
        var manager = new BlockManager<double>(config);

        var allBlocks = new int[100];
        for (int i = 0; i < 100; i++)
        {
            allBlocks[i] = manager.AllocateBlock();
            Assert.True(allBlocks[i] >= 0);
        }

        // All IDs should be unique
        Assert.Equal(100, allBlocks.Distinct().Count());
    }

    #endregion
}
