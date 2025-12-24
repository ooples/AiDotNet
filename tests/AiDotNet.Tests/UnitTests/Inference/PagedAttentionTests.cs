using AiDotNet.Inference.PagedAttention;
using AiDotNet.Inference.Quantization;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Inference;

/// <summary>
/// Unit tests for PagedAttention components.
/// </summary>
public class BlockManagerTests
{
    [Fact]
    public void BlockManager_Creation_InitializesCorrectly()
    {
        // Arrange & Act
        var config = new BlockManagerConfig
        {
            BlockSize = 16,
            NumBlocks = 100,
            NumLayers = 32,
            NumHeads = 32,
            HeadDimension = 128
        };
        var manager = new BlockManager<float>(config);

        // Assert
        Assert.Equal(100, manager.TotalBlocks);
        Assert.Equal(100, manager.FreeBlockCount);
        Assert.Equal(0, manager.AllocatedBlockCount);
        Assert.Equal(0, manager.MemoryUtilization);
    }

    [Fact]
    public void BlockManager_AllocateBlock_DecrementsFreeCount()
    {
        // Arrange
        var config = new BlockManagerConfig { NumBlocks = 10 };
        var manager = new BlockManager<float>(config);

        // Act
        int blockId = manager.AllocateBlock();

        // Assert
        Assert.True(blockId >= 0);
        Assert.Equal(9, manager.FreeBlockCount);
        Assert.Equal(1, manager.AllocatedBlockCount);
    }

    [Fact]
    public void BlockManager_AllocateBlocks_AllocatesMultiple()
    {
        // Arrange
        var config = new BlockManagerConfig { NumBlocks = 20 };
        var manager = new BlockManager<float>(config);

        // Act
        var blocks = manager.AllocateBlocks(5);

        // Assert
        Assert.NotNull(blocks);
        Assert.Equal(5, blocks.Length);
        Assert.Equal(15, manager.FreeBlockCount);
        Assert.Equal(5, manager.AllocatedBlockCount);
    }

    [Fact]
    public void BlockManager_AllocateBlocks_ReturnsNullWhenNotEnough()
    {
        // Arrange
        var config = new BlockManagerConfig { NumBlocks = 5 };
        var manager = new BlockManager<float>(config);

        // Act
        var blocks = manager.AllocateBlocks(10);

        // Assert
        Assert.Null(blocks);
        Assert.Equal(5, manager.FreeBlockCount); // No change
    }

    [Fact]
    public void BlockManager_FreeBlock_ReturnsToPool()
    {
        // Arrange
        var config = new BlockManagerConfig { NumBlocks = 10 };
        var manager = new BlockManager<float>(config);
        int blockId = manager.AllocateBlock();

        // Act
        manager.FreeBlock(blockId);

        // Assert
        Assert.Equal(10, manager.FreeBlockCount);
        Assert.Equal(0, manager.AllocatedBlockCount);
    }

    [Fact]
    public void BlockManager_AddReference_IncreasesRefCount()
    {
        // Arrange
        var config = new BlockManagerConfig { NumBlocks = 10 };
        var manager = new BlockManager<float>(config);
        int blockId = manager.AllocateBlock();

        // Act
        manager.AddReference(blockId);

        // Assert
        Assert.Equal(2, manager.GetReferenceCount(blockId));
    }

    [Fact]
    public void BlockManager_FreeBlock_WithMultipleRefs_OnlyDecrementsRef()
    {
        // Arrange
        var config = new BlockManagerConfig { NumBlocks = 10 };
        var manager = new BlockManager<float>(config);
        int blockId = manager.AllocateBlock();
        manager.AddReference(blockId); // ref count = 2

        // Act
        manager.FreeBlock(blockId);

        // Assert
        Assert.Equal(1, manager.GetReferenceCount(blockId));
        Assert.Equal(9, manager.FreeBlockCount); // Still allocated
    }

    [Fact]
    public void BlockManager_CopyOnWrite_CreatesNewBlock()
    {
        // Arrange
        var config = new BlockManagerConfig { NumBlocks = 10 };
        var manager = new BlockManager<float>(config);
        int blockId = manager.AllocateBlock();
        manager.AddReference(blockId); // ref count = 2

        // Act
        int newBlockId = manager.CopyOnWrite(blockId);

        // Assert
        Assert.NotEqual(blockId, newBlockId);
        Assert.Equal(1, manager.GetReferenceCount(blockId));
        Assert.Equal(1, manager.GetReferenceCount(newBlockId));
    }

    [Fact]
    public void BlockManager_CopyOnWrite_NoopForSingleRef()
    {
        // Arrange
        var config = new BlockManagerConfig { NumBlocks = 10 };
        var manager = new BlockManager<float>(config);
        int blockId = manager.AllocateBlock();

        // Act
        int result = manager.CopyOnWrite(blockId);

        // Assert
        Assert.Equal(blockId, result); // Same block returned
    }

    [Fact]
    public void BlockManager_CanAllocate_ChecksAvailability()
    {
        // Arrange
        var config = new BlockManagerConfig { NumBlocks = 5 };
        var manager = new BlockManager<float>(config);
        manager.AllocateBlocks(3);

        // Assert
        Assert.True(manager.CanAllocate(2));
        Assert.False(manager.CanAllocate(3));
    }

    [Fact]
    public void BlockManager_BlocksForTokens_CalculatesCorrectly()
    {
        // Arrange
        var config = new BlockManagerConfig { BlockSize = 16, NumBlocks = 100 };
        var manager = new BlockManager<float>(config);

        // Assert
        Assert.Equal(1, manager.BlocksForTokens(1));
        Assert.Equal(1, manager.BlocksForTokens(16));
        Assert.Equal(2, manager.BlocksForTokens(17));
        Assert.Equal(7, manager.BlocksForTokens(100));
    }

    [Fact]
    public void BlockManager_GetStats_ReturnsCorrectStats()
    {
        // Arrange
        var config = new BlockManagerConfig { BlockSize = 16, NumBlocks = 100 };
        var manager = new BlockManager<float>(config);
        manager.AllocateBlocks(25);

        // Act
        var stats = manager.GetStats();

        // Assert
        Assert.Equal(100, stats.TotalBlocks);
        Assert.Equal(25, stats.AllocatedBlocks);
        Assert.Equal(75, stats.FreeBlocks);
        Assert.Equal(0.25, stats.MemoryUtilization, 0.001);
    }

    [Fact]
    public void BlockManager_Reset_FreesAllBlocks()
    {
        // Arrange
        var config = new BlockManagerConfig { NumBlocks = 50 };
        var manager = new BlockManager<float>(config);
        manager.AllocateBlocks(30);

        // Act
        manager.Reset();

        // Assert
        Assert.Equal(50, manager.FreeBlockCount);
        Assert.Equal(0, manager.AllocatedBlockCount);
    }
}

/// <summary>
/// Tests for BlockTable.
/// </summary>
public class BlockTableTests
{
    [Fact]
    public void BlockTable_Creation_InitializesEmpty()
    {
        // Act
        var table = new BlockTable(1, 16);

        // Assert
        Assert.Equal(1, table.SequenceId);
        Assert.Equal(16, table.BlockSize);
        Assert.Equal(0, table.NumLogicalBlocks);
        Assert.Equal(0, table.Capacity);
    }

    [Fact]
    public void BlockTable_AppendBlock_IncreasesCapacity()
    {
        // Arrange
        var table = new BlockTable(1, 16);

        // Act
        table.AppendBlock(5);
        table.AppendBlock(10);

        // Assert
        Assert.Equal(2, table.NumLogicalBlocks);
        Assert.Equal(32, table.Capacity);
    }

    [Fact]
    public void BlockTable_GetPhysicalBlock_ReturnsCorrectId()
    {
        // Arrange
        var table = new BlockTable(1, 16);
        table.AppendBlocks(new[] { 5, 10, 15 });

        // Assert
        Assert.Equal(5, table.GetPhysicalBlock(0));
        Assert.Equal(10, table.GetPhysicalBlock(1));
        Assert.Equal(15, table.GetPhysicalBlock(2));
    }

    [Fact]
    public void BlockTable_GetBlockAndOffset_CalculatesCorrectly()
    {
        // Arrange
        var table = new BlockTable(1, 16);
        table.AppendBlocks(new[] { 5, 10, 15 });

        // Assert
        Assert.Equal((5, 0), table.GetBlockAndOffset(0));
        Assert.Equal((5, 15), table.GetBlockAndOffset(15));
        Assert.Equal((10, 0), table.GetBlockAndOffset(16));
        Assert.Equal((10, 5), table.GetBlockAndOffset(21));
        Assert.Equal((15, 0), table.GetBlockAndOffset(32));
    }

    [Fact]
    public void BlockTable_ReplaceBlock_UpdatesMapping()
    {
        // Arrange
        var table = new BlockTable(1, 16);
        table.AppendBlocks(new[] { 5, 10, 15 });

        // Act
        int oldId = table.ReplaceBlock(1, 99);

        // Assert
        Assert.Equal(10, oldId);
        Assert.Equal(99, table.GetPhysicalBlock(1));
    }

    [Fact]
    public void BlockTable_RemoveLastBlock_DecreasesCapacity()
    {
        // Arrange
        var table = new BlockTable(1, 16);
        table.AppendBlocks(new[] { 5, 10, 15 });

        // Act
        int removed = table.RemoveLastBlock();

        // Assert
        Assert.Equal(15, removed);
        Assert.Equal(2, table.NumLogicalBlocks);
        Assert.Equal(32, table.Capacity);
    }

    [Fact]
    public void BlockTable_Copy_CreatesShallowCopy()
    {
        // Arrange
        var table = new BlockTable(1, 16);
        table.AppendBlocks(new[] { 5, 10, 15 });

        // Act
        var copy = table.Copy(2);

        // Assert
        Assert.Equal(2, copy.SequenceId);
        Assert.Equal(table.NumLogicalBlocks, copy.NumLogicalBlocks);
        Assert.Equal(table.GetPhysicalBlock(0), copy.GetPhysicalBlock(0));
    }

    [Fact]
    public void BlockTable_TruncateTo_RemovesExcessBlocks()
    {
        // Arrange
        var table = new BlockTable(1, 16);
        table.AppendBlocks(new[] { 5, 10, 15, 20 });

        // Act
        var removed = table.TruncateTo(2);

        // Assert
        Assert.Equal(2, removed.Count);
        Assert.Contains(20, removed);
        Assert.Contains(15, removed);
        Assert.Equal(2, table.NumLogicalBlocks);
    }

    [Fact]
    public void BlockTable_BlocksNeededFor_CalculatesCorrectly()
    {
        // Arrange
        var table = new BlockTable(1, 16);

        // Assert
        Assert.Equal(1, table.BlocksNeededFor(1));
        Assert.Equal(1, table.BlocksNeededFor(16));
        Assert.Equal(2, table.BlocksNeededFor(17));
        Assert.Equal(10, table.BlocksNeededFor(160));
    }
}

/// <summary>
/// Tests for BlockTableManager.
/// </summary>
public class BlockTableManagerTests
{
    [Fact]
    public void BlockTableManager_CreateBlockTable_AllocatesInitialBlocks()
    {
        // Arrange
        var blockManager = new BlockManager<float>(new BlockManagerConfig { NumBlocks = 100 });
        var tableManager = new BlockTableManager<float>(blockManager);

        // Act
        var table = tableManager.CreateBlockTable(1, 5);

        // Assert
        Assert.NotNull(table);
        Assert.Equal(5, table.NumLogicalBlocks);
        Assert.Equal(1, tableManager.ActiveTableCount);
    }

    [Fact]
    public void BlockTableManager_GetBlockTable_ReturnsExisting()
    {
        // Arrange
        var blockManager = new BlockManager<float>(new BlockManagerConfig { NumBlocks = 100 });
        var tableManager = new BlockTableManager<float>(blockManager);
        tableManager.CreateBlockTable(1, 2);

        // Act
        var table = tableManager.GetBlockTable(1);

        // Assert
        Assert.NotNull(table);
        Assert.Equal(1, table.SequenceId);
    }

    [Fact]
    public void BlockTableManager_FreeBlockTable_ReleasesBlocks()
    {
        // Arrange
        var blockManager = new BlockManager<float>(new BlockManagerConfig { NumBlocks = 100 });
        var tableManager = new BlockTableManager<float>(blockManager);
        tableManager.CreateBlockTable(1, 10);

        // Act
        tableManager.FreeBlockTable(1);

        // Assert
        Assert.Null(tableManager.GetBlockTable(1));
        Assert.Equal(100, blockManager.FreeBlockCount);
    }

    [Fact]
    public void BlockTableManager_ForkBlockTable_SharesBlocks()
    {
        // Arrange
        var blockManager = new BlockManager<float>(new BlockManagerConfig { NumBlocks = 100, BlockSize = 16 });
        var tableManager = new BlockTableManager<float>(blockManager);
        var sourceTable = tableManager.CreateBlockTable(1, 5);

        // Act
        var forkedTable = tableManager.ForkBlockTable(1, 2);

        // Assert
        Assert.NotNull(forkedTable);
        Assert.Equal(5, forkedTable!.NumLogicalBlocks);
        // Blocks should be shared (ref count increased)
        Assert.Equal(2, blockManager.GetReferenceCount(sourceTable!.GetPhysicalBlock(0)));
    }

    [Fact]
    public void BlockTableManager_EnsureCapacity_AllocatesMoreBlocks()
    {
        // Arrange
        var blockManager = new BlockManager<float>(new BlockManagerConfig { NumBlocks = 100, BlockSize = 16 });
        var tableManager = new BlockTableManager<float>(blockManager);
        tableManager.CreateBlockTable(1, 2); // 32 tokens capacity

        // Act
        bool success = tableManager.EnsureCapacity(1, 50); // Need 4 blocks total

        // Assert
        Assert.True(success);
        var table = tableManager.GetBlockTable(1);
        Assert.True(table!.NumLogicalBlocks >= 4);
    }
}

/// <summary>
/// Tests for PagedKVCache.
/// </summary>
public class PagedKVCacheTests
{
    [Fact]
    public void PagedKVCache_Creation_InitializesCorrectly()
    {
        // Arrange & Act
        var config = new PagedKVCacheConfig
        {
            BlockSize = 16,
            NumBlocks = 100,
            NumLayers = 12,
            NumHeads = 12,
            HeadDimension = 64
        };
        var cache = new PagedKVCache<float>(config);

        // Assert
        Assert.Equal(0, cache.ActiveSequenceCount);
        Assert.NotNull(cache.BlockManager);
        Assert.NotNull(cache.BlockTableManager);
    }

    [Fact]
    public void PagedKVCache_AllocateSequence_CreatesEntry()
    {
        // Arrange
        var config = new PagedKVCacheConfig
        {
            BlockSize = 16,
            NumBlocks = 100,
            NumLayers = 2,
            NumHeads = 4,
            HeadDimension = 8
        };
        var cache = new PagedKVCache<float>(config);

        // Act
        bool success = cache.AllocateSequence(1, 32); // 2 blocks needed

        // Assert
        Assert.True(success);
        Assert.Equal(1, cache.ActiveSequenceCount);
        Assert.Equal(32, cache.GetSequenceLength(1));
    }

    [Fact]
    public void PagedKVCache_ExtendSequence_IncreasesLength()
    {
        // Arrange
        var config = new PagedKVCacheConfig
        {
            BlockSize = 16,
            NumBlocks = 100,
            NumLayers = 2,
            NumHeads = 4,
            HeadDimension = 8
        };
        var cache = new PagedKVCache<float>(config);
        cache.AllocateSequence(1, 16);

        // Act
        bool success = cache.ExtendSequence(1, 20);

        // Assert
        Assert.True(success);
        Assert.Equal(36, cache.GetSequenceLength(1));
    }

    [Fact]
    public void PagedKVCache_FreeSequence_RemovesEntry()
    {
        // Arrange
        var config = new PagedKVCacheConfig
        {
            BlockSize = 16,
            NumBlocks = 100,
            NumLayers = 2,
            NumHeads = 4,
            HeadDimension = 8
        };
        var cache = new PagedKVCache<float>(config);
        cache.AllocateSequence(1, 32);

        // Act
        cache.FreeSequence(1);

        // Assert
        Assert.Equal(0, cache.ActiveSequenceCount);
        Assert.Equal(0, cache.GetSequenceLength(1));
    }

    [Fact]
    public void PagedKVCache_ForkSequence_CreatesSharedCopy()
    {
        // Arrange
        var config = new PagedKVCacheConfig
        {
            BlockSize = 16,
            NumBlocks = 100,
            NumLayers = 2,
            NumHeads = 4,
            HeadDimension = 8
        };
        var cache = new PagedKVCache<float>(config);
        cache.AllocateSequence(1, 32);

        // Act
        bool success = cache.ForkSequence(1, 2);

        // Assert
        Assert.True(success);
        Assert.Equal(2, cache.ActiveSequenceCount);
        Assert.Equal(32, cache.GetSequenceLength(2));
    }

    [Fact]
    public void PagedKVCache_WriteReadKey_RoundTrips()
    {
        // Arrange
        var config = new PagedKVCacheConfig
        {
            BlockSize = 16,
            NumBlocks = 100,
            NumLayers = 2,
            NumHeads = 4,
            HeadDimension = 8
        };
        var cache = new PagedKVCache<float>(config);
        cache.AllocateSequence(1, 16);

        var keyData = new float[4 * 8]; // num_heads * head_dim
        for (int i = 0; i < keyData.Length; i++) keyData[i] = i * 0.1f;

        // Act
        cache.WriteKey(1, 5, 0, keyData);
        var readKey = new float[4 * 8];
        cache.ReadKey(1, 5, 0, readKey);

        // Assert
        for (int i = 0; i < keyData.Length; i++)
        {
            Assert.Equal(keyData[i], readKey[i], 4);
        }
    }

    [Fact]
    public void PagedKVCache_WriteReadValue_RoundTrips()
    {
        // Arrange
        var config = new PagedKVCacheConfig
        {
            BlockSize = 16,
            NumBlocks = 100,
            NumLayers = 2,
            NumHeads = 4,
            HeadDimension = 8
        };
        var cache = new PagedKVCache<float>(config);
        cache.AllocateSequence(1, 16);

        var valueData = new float[4 * 8];
        for (int i = 0; i < valueData.Length; i++) valueData[i] = i * 0.2f;

        // Act
        cache.WriteValue(1, 10, 1, valueData);
        var readValue = new float[4 * 8];
        cache.ReadValue(1, 10, 1, readValue);

        // Assert
        for (int i = 0; i < valueData.Length; i++)
        {
            Assert.Equal(valueData[i], readValue[i], 4);
        }
    }

    [Fact]
    public void PagedKVCache_GetStats_ReturnsValidStats()
    {
        // Arrange
        var config = new PagedKVCacheConfig
        {
            BlockSize = 16,
            NumBlocks = 100,
            NumLayers = 2,
            NumHeads = 4,
            HeadDimension = 8
        };
        var cache = new PagedKVCache<float>(config);
        cache.AllocateSequence(1, 32);
        cache.AllocateSequence(2, 48);

        // Act
        var stats = cache.GetStats();

        // Assert
        Assert.Equal(2, stats.ActiveSequences);
        Assert.Equal(80, stats.TotalTokensCached);
        Assert.Equal(40, stats.AverageSequenceLength);
    }
}

/// <summary>
/// Tests for PagedAttentionKernel.
/// </summary>
public class PagedAttentionKernelTests
{
    private PagedKVCache<float> CreateTestCache()
    {
        return new PagedKVCache<float>(new PagedKVCacheConfig
        {
            BlockSize = 16,
            NumBlocks = 100,
            NumLayers = 2,
            NumHeads = 4,
            HeadDimension = 8
        });
    }

    [Fact]
    public void PagedAttentionKernel_ComputeAttention_ProducesOutput()
    {
        // Arrange
        using var cache = CreateTestCache();
        cache.AllocateSequence(1, 16);

        // Write some KV data
        var kvData = new float[4 * 8];
        for (int pos = 0; pos < 16; pos++)
        {
            for (int i = 0; i < kvData.Length; i++) kvData[i] = (pos + 1) * 0.1f;
            cache.WriteKey(1, pos, 0, kvData);
            cache.WriteValue(1, pos, 0, kvData);
        }

        var kernel = new PagedAttentionKernel<float>(cache);
        var query = new float[4 * 8];
        for (int i = 0; i < query.Length; i++) query[i] = 0.5f;

        var output = new float[4 * 8];

        // Act
        kernel.ComputeAttention(query, 1, 0, output, 1.0f / MathF.Sqrt(8));

        // Assert
        Assert.Contains(output, v => v != 0); // Output should not be all zeros
    }

    [Fact]
    public void PagedAttentionKernel_ComputeTiledAttention_ProducesOutput()
    {
        // Arrange
        using var cache = CreateTestCache();
        cache.AllocateSequence(1, 32); // 2 blocks

        // Write some KV data
        var kvData = new float[4 * 8];
        for (int pos = 0; pos < 32; pos++)
        {
            for (int i = 0; i < kvData.Length; i++) kvData[i] = MathF.Sin(pos + i);
            cache.WriteKey(1, pos, 0, kvData);
            cache.WriteValue(1, pos, 0, kvData);
        }

        var kernel = new PagedAttentionKernel<float>(cache);
        var query = new float[4 * 8];
        for (int i = 0; i < query.Length; i++) query[i] = MathF.Cos(i);

        var output = new float[4 * 8];

        // Act
        kernel.ComputeTiledPagedAttention(query, 1, 0, output, 1.0f / MathF.Sqrt(8));

        // Assert
        Assert.Contains(output, v => v != 0);
    }

    [Fact]
    public void PagedAttentionKernel_UpdateCache_ExtendsSequence()
    {
        // Arrange
        using var cache = CreateTestCache();
        cache.AllocateSequence(1, 8);

        var kernel = new PagedAttentionKernel<float>(cache);
        var key = new float[4 * 8];
        var value = new float[4 * 8];

        // Act
        kernel.UpdateCache(key, value, 1, 8, 0);

        // Assert
        Assert.True(cache.GetSequenceLength(1) >= 8);
    }

    [Fact]
    public void PagedAttentionKernel_ComputeBatchedAttention_ProcessesMultiple()
    {
        // Arrange
        using var cache = CreateTestCache();
        cache.AllocateSequence(1, 16);
        cache.AllocateSequence(2, 16);

        // Write KV data for both
        var kvData = new float[4 * 8];
        for (int seq = 1; seq <= 2; seq++)
        {
            for (int pos = 0; pos < 16; pos++)
            {
                for (int i = 0; i < kvData.Length; i++) kvData[i] = seq * pos * 0.1f;
                cache.WriteKey(seq, pos, 0, kvData);
                cache.WriteValue(seq, pos, 0, kvData);
            }
        }

        var kernel = new PagedAttentionKernel<float>(cache);
        var queries = new float[2 * 4 * 8]; // batch_size * num_heads * head_dim
        for (int i = 0; i < queries.Length; i++) queries[i] = 0.5f;

        var outputs = new float[2 * 4 * 8];

        // Act
        kernel.ComputeBatchedAttention(queries, new long[] { 1, 2 }, 0, outputs, 0.35f);

        // Assert
        Assert.Contains(outputs, v => v != 0);
    }

    [Fact]
    public void PagedAttentionKernel_ForwardQuantized_MatchesFloatWithinTolerance()
    {
        // Arrange
        using var cacheFloat = CreateTestCache();
        cacheFloat.AllocateSequence(1, 1);
        var kernelFloat = new PagedAttentionKernel<float>(cacheFloat);

        using var cacheQ = CreateTestCache();
        cacheQ.AllocateSequence(1, 1);
        var kernelQ = new PagedAttentionKernel<float>(cacheQ);

        int hiddenDim = kernelFloat.Config.NumHeads * kernelFloat.Config.HeadDimension;
        int projDim = hiddenDim;

        var rnd = RandomHelper.CreateSeededRandom(42);
        var hidden = new float[hiddenDim];
        for (int i = 0; i < hidden.Length; i++)
        {
            hidden[i] = (float)(rnd.NextDouble() * 0.2 - 0.1);
        }

        float[] MakeWeights(int rows, int cols)
        {
            var w = new float[rows * cols];
            for (int i = 0; i < w.Length; i++)
            {
                w[i] = (float)(rnd.NextDouble() * 0.02 - 0.01);
            }
            return w;
        }

        var wQ = MakeWeights(projDim, hiddenDim);
        var wK = MakeWeights(projDim, hiddenDim);
        var wV = MakeWeights(projDim, hiddenDim);
        var wO = MakeWeights(hiddenDim, projDim);

        var qWQ = Int8WeightOnlyQuantization.QuantizePerRow(wQ, projDim, hiddenDim);
        var qWK = Int8WeightOnlyQuantization.QuantizePerRow(wK, projDim, hiddenDim);
        var qWV = Int8WeightOnlyQuantization.QuantizePerRow(wV, projDim, hiddenDim);
        var qWO = Int8WeightOnlyQuantization.QuantizePerRow(wO, hiddenDim, projDim);

        var outFloat = new float[hiddenDim];
        var outQ = new float[hiddenDim];

        // Act
        kernelFloat.Forward(hidden, wQ, wK, wV, wO, sequenceId: 1, position: 0, layer: 0, output: outFloat);
        kernelQ.ForwardQuantized(hidden, qWQ, qWK, qWV, qWO, sequenceId: 1, position: 0, layer: 0, output: outQ);

        // Assert
        float maxAbsDiff = 0f;
        for (int i = 0; i < hiddenDim; i++)
        {
            float diff = MathF.Abs(outFloat[i] - outQ[i]);
            if (diff > maxAbsDiff) maxAbsDiff = diff;
        }

        Assert.True(maxAbsDiff <= 1e-2f, $"Max abs diff was {maxAbsDiff}");
    }
}

/// <summary>
/// Tests for PagedAttentionServer.
/// </summary>
public class PagedAttentionServerTests
{
    [Fact]
    public void PagedAttentionServer_RegisterSequence_Works()
    {
        // Arrange
        var config = new PagedKVCacheConfig
        {
            BlockSize = 16,
            NumBlocks = 100,
            NumLayers = 2,
            NumHeads = 4,
            HeadDimension = 8
        };
        using var server = new PagedAttentionServer<float>(config);

        // Act
        bool success = server.RegisterSequence(1, 32);

        // Assert
        Assert.True(success);
    }

    [Fact]
    public void PagedAttentionServer_UnregisterSequence_Frees()
    {
        // Arrange
        var config = new PagedKVCacheConfig
        {
            BlockSize = 16,
            NumBlocks = 100,
            NumLayers = 2,
            NumHeads = 4,
            HeadDimension = 8
        };
        using var server = new PagedAttentionServer<float>(config);
        server.RegisterSequence(1, 32);

        // Act
        server.UnregisterSequence(1);

        // Assert
        Assert.Equal(0, server.GetStats().ActiveSequences);
    }

    [Fact]
    public void PagedAttentionServer_ForkSequence_ForBeamSearch()
    {
        // Arrange
        var config = new PagedKVCacheConfig
        {
            BlockSize = 16,
            NumBlocks = 100,
            NumLayers = 2,
            NumHeads = 4,
            HeadDimension = 8
        };
        using var server = new PagedAttentionServer<float>(config);
        server.RegisterSequence(1, 32);

        // Act
        bool success = server.ForkSequence(1, new long[] { 2, 3, 4 });

        // Assert
        Assert.True(success);
        Assert.Equal(4, server.GetStats().ActiveSequences);
    }

    [Fact]
    [Trait("Category", "Integration")]  // Skip on net471 - 4GB allocation exceeds .NET Framework array size limits
    public void PagedAttentionServer_ForModel_CreatesValidServer()
    {
        // Act
        using var server = PagedAttentionServer<float>.ForModel("llama-7b", 4L * 1024 * 1024 * 1024);

        // Assert
        Assert.NotNull(server.KVCache);
        Assert.NotNull(server.Kernel);
    }
}

/// <summary>
/// Integration tests for PagedAttention.
/// </summary>
public class PagedAttentionIntegrationTests
{
    [Fact]
    public void PagedAttention_MultipleSequences_ManagesMemoryEfficiently()
    {
        // Arrange - Small config for testing
        var config = new PagedKVCacheConfig
        {
            BlockSize = 4,
            NumBlocks = 50,
            NumLayers = 2,
            NumHeads = 2,
            HeadDimension = 4
        };
        using var cache = new PagedKVCache<float>(config);

        // Act - Allocate multiple sequences with varying lengths
        cache.AllocateSequence(1, 10);
        cache.AllocateSequence(2, 5);
        cache.AllocateSequence(3, 15);
        cache.AllocateSequence(4, 8);

        // Assert
        Assert.Equal(4, cache.ActiveSequenceCount);

        var stats = cache.GetStats();
        Assert.Equal(38, stats.TotalTokensCached);

        // Free some and verify memory is reclaimed
        cache.FreeSequence(2);
        cache.FreeSequence(4);

        Assert.Equal(2, cache.ActiveSequenceCount);
        Assert.Equal(25, cache.GetStats().TotalTokensCached);
    }

    [Fact]
    public void PagedAttention_BeamSearchFork_SharesMemory()
    {
        // Arrange
        var config = new PagedKVCacheConfig
        {
            BlockSize = 4,
            NumBlocks = 100,
            NumLayers = 2,
            NumHeads = 2,
            HeadDimension = 4
        };
        using var cache = new PagedKVCache<float>(config);

        // Initial sequence
        cache.AllocateSequence(1, 16);

        var initialBlocks = cache.BlockManager.AllocatedBlockCount;

        // Fork for beam search (4 beams)
        for (int i = 2; i <= 5; i++)
        {
            cache.ForkSequence(1, i);
        }

        // Assert - forked sequences should share blocks (via copy-on-write)
        Assert.Equal(5, cache.ActiveSequenceCount);

        // Allocated blocks should NOT be 5x since they share via COW
        var afterForkBlocks = cache.BlockManager.AllocatedBlockCount;
        Assert.Equal(initialBlocks, afterForkBlocks); // Same blocks, just shared
    }

    [Fact]
    public void PagedAttention_SequenceExtension_WorksAcrossBlocks()
    {
        // Arrange
        var config = new PagedKVCacheConfig
        {
            BlockSize = 4,
            NumBlocks = 100,
            NumLayers = 1,
            NumHeads = 2,
            HeadDimension = 4
        };
        using var cache = new PagedKVCache<float>(config);
        cache.AllocateSequence(1, 3); // Starts in first block

        // Act - Extend to span multiple blocks
        cache.ExtendSequence(1, 5); // Now 8 tokens = 2 blocks
        cache.ExtendSequence(1, 6); // Now 14 tokens = 4 blocks

        // Assert
        Assert.Equal(14, cache.GetSequenceLength(1));

        var table = cache.GetBlockTable(1);
        Assert.NotNull(table);
        Assert.True(table.Length >= 4);
    }
}
