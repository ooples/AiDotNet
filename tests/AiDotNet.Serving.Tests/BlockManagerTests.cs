using System.Collections.Generic;
using System.Linq;
using AiDotNet.Serving.Engine;
using Xunit;

namespace AiDotNet.Serving.Tests;

/// <summary>
/// Invariant tests for <see cref="BlockManager"/> — the paged KV-cache allocator. These prove the correctness
/// backbone every downstream component relies on: a physical block is never handed out twice, free + used
/// blocks always sum to the pool size, copy-on-write duplicates exactly the block being written (and only when
/// shared), and fork/free reference counting reclaims memory precisely.
/// </summary>
public class BlockManagerTests
{
    // ---- Construction / basic accounting -------------------------------------------------

    [Fact]
    public void NewManager_AllBlocksFree()
    {
        var bm = new BlockManager(totalBlocks: 8, blockSize: 4);
        Assert.Equal(8, bm.TotalBlocks);
        Assert.Equal(8, bm.NumFreeBlocks);
        Assert.Equal(0, bm.NumUsedBlocks);
        Assert.Equal(4, bm.BlockSize);
        Assert.Equal(0, bm.NumSequences);
        Assert.Equal(0.0, bm.Usage);
    }

    [Theory]
    [InlineData(1, 4, 1)]
    [InlineData(4, 4, 1)]
    [InlineData(5, 4, 2)]
    [InlineData(16, 4, 4)]
    [InlineData(17, 4, 5)]
    public void BlocksForTokens_CeilingDivision(int tokens, int blockSize, int expected)
    {
        var bm = new BlockManager(totalBlocks: 32, blockSize: blockSize);
        Assert.Equal(expected, bm.BlocksForTokens(tokens));
    }

    // ---- Allocation ----------------------------------------------------------------------

    [Fact]
    public void Allocate_ConsumesCeilingBlocks_AndConserves()
    {
        var bm = new BlockManager(totalBlocks: 8, blockSize: 4);
        var table = bm.Allocate("s1", numTokens: 6); // ceil(6/4) = 2 blocks

        Assert.Equal(2, table.Count);
        Assert.Equal(2, bm.NumUsedBlocks);
        Assert.Equal(6, bm.NumFreeBlocks);
        Assert.Equal(8, bm.NumFreeBlocks + bm.NumUsedBlocks); // conservation
        Assert.True(bm.Contains("s1"));
        Assert.Equal(6, bm.GetLength("s1"));
        Assert.Equal(2, bm.FilledSlotsInLastBlock("s1")); // 6 % 4
    }

    [Fact]
    public void Allocate_DistinctSequences_NeverShareBlocks()
    {
        var bm = new BlockManager(totalBlocks: 16, blockSize: 4);
        var a = bm.Allocate("a", 8).ToArray();
        var b = bm.Allocate("b", 8).ToArray();

        Assert.Empty(a.Intersect(b)); // no double-allocation
        Assert.Equal(4, bm.NumUsedBlocks);
    }

    [Fact]
    public void Allocate_Twice_SameSequence_Throws()
    {
        var bm = new BlockManager(8, 4);
        bm.Allocate("s1", 4);
        Assert.Throws<System.InvalidOperationException>(() => bm.Allocate("s1", 4));
    }

    [Fact]
    public void CanAllocate_RespectsPoolSize()
    {
        var bm = new BlockManager(totalBlocks: 2, blockSize: 4);
        Assert.True(bm.CanAllocate(8));   // exactly 2 blocks
        Assert.False(bm.CanAllocate(9));  // needs 3
    }

    [Fact]
    public void Allocate_BeyondCapacity_Throws()
    {
        var bm = new BlockManager(totalBlocks: 2, blockSize: 4);
        Assert.Throws<System.InvalidOperationException>(() => bm.Allocate("s1", 12)); // needs 3, have 2
    }

    // ---- Append (decode growth) ----------------------------------------------------------

    [Fact]
    public void Append_WithinLastBlock_NoNewBlock_NoCopy()
    {
        var bm = new BlockManager(8, 4);
        bm.Allocate("s1", 2); // 1 block, 2/4 filled
        var copies = bm.Append("s1", 1); // -> 3/4 filled, still 1 block

        Assert.Empty(copies);
        Assert.Single(bm.GetBlockTable("s1"));
        Assert.Equal(3, bm.GetLength("s1"));
        Assert.Equal(1, bm.NumUsedBlocks);
    }

    [Fact]
    public void Append_CrossingBlockBoundary_AllocatesNewBlock()
    {
        var bm = new BlockManager(8, 4);
        bm.Allocate("s1", 4);            // exactly 1 full block
        var copies = bm.Append("s1", 1); // 5th token -> needs a 2nd block

        Assert.Empty(copies);
        Assert.Equal(2, bm.GetBlockTable("s1").Count);
        Assert.Equal(5, bm.GetLength("s1"));
        Assert.Equal(1, bm.FilledSlotsInLastBlock("s1"));
    }

    [Fact]
    public void CanAppend_FalseWhenPoolExhausted()
    {
        var bm = new BlockManager(totalBlocks: 1, blockSize: 4);
        bm.Allocate("s1", 4); // consumes the only block, now full
        Assert.False(bm.CanAppend("s1", 1)); // would need a 2nd block, none free
        Assert.Throws<System.InvalidOperationException>(() => bm.Append("s1", 1));
    }

    [Fact]
    public void Append_ManyTokens_MaintainsConservationEveryStep()
    {
        var bm = new BlockManager(totalBlocks: 64, blockSize: 4);
        bm.Allocate("s1", 1);
        for (int i = 0; i < 100; i++)
        {
            if (!bm.CanAppend("s1", 1)) break;
            bm.Append("s1", 1);
            Assert.Equal(bm.TotalBlocks, bm.NumFreeBlocks + bm.NumUsedBlocks);
        }
        Assert.Equal(101, bm.GetLength("s1"));
        Assert.Equal(bm.BlocksForTokens(101), bm.GetBlockTable("s1").Count);
    }

    // ---- Fork + copy-on-write ------------------------------------------------------------

    [Fact]
    public void Fork_SharesAllBlocks_NoNewAllocation()
    {
        var bm = new BlockManager(16, 4);
        var parent = bm.Allocate("p", 8).ToArray(); // 2 blocks
        int usedBefore = bm.NumUsedBlocks;

        var child = bm.Fork("p", "c").ToArray();

        Assert.Equal(parent, child);                 // identical physical blocks
        Assert.Equal(usedBefore, bm.NumUsedBlocks);  // fork allocates nothing
        Assert.Equal(8, bm.GetLength("c"));
    }

    [Fact]
    public void Append_IntoSharedPartialBlock_TriggersCopyOnWrite()
    {
        var bm = new BlockManager(16, 4);
        bm.Allocate("p", 2);        // 1 block, 2/4 filled (has room)
        bm.Fork("p", "c");          // block now shared (refcount 2)
        int freeBefore = bm.NumFreeBlocks;

        var pTable = bm.GetBlockTable("p").ToArray();
        var copies = bm.Append("c", 1); // c writes into the shared partial block -> COW

        Assert.Single(copies);
        var cTable = bm.GetBlockTable("c");
        Assert.Equal(pTable[0], copies[0].Source);       // copied from the shared block
        Assert.Equal(cTable[0], copies[0].Destination);  // into c's new block
        Assert.NotEqual(pTable[0], cTable[0]);           // c no longer aliases p
        Assert.Equal(freeBefore - 1, bm.NumFreeBlocks);  // exactly one new block consumed

        // Parent's block table is untouched by the child's write.
        Assert.Equal(pTable, bm.GetBlockTable("p").ToArray());
    }

    [Fact]
    public void Append_IntoSharedFullBlock_AllocatesNewBlock_NoCopy()
    {
        var bm = new BlockManager(16, 4);
        bm.Allocate("p", 4);   // exactly 1 FULL block
        bm.Fork("p", "c");     // shared, but full -> next write goes to a fresh block, no copy
        var copies = bm.Append("c", 1);

        Assert.Empty(copies);
        Assert.Equal(2, bm.GetBlockTable("c").Count);
        Assert.Single(bm.GetBlockTable("p")); // parent unchanged
    }

    [Fact]
    public void Append_Unshared_NeverCopies()
    {
        var bm = new BlockManager(16, 4);
        bm.Allocate("s1", 2);
        bm.Fork("s1", "s2");
        bm.Free("s2");                    // s1's block is unshared again (refcount back to 1)
        var copies = bm.Append("s1", 1);  // in-place, no COW

        Assert.Empty(copies);
    }

    // ---- Free / reclamation --------------------------------------------------------------

    [Fact]
    public void Free_ReturnsBlocksToPool()
    {
        var bm = new BlockManager(8, 4);
        bm.Allocate("s1", 8); // 2 blocks
        Assert.Equal(2, bm.NumUsedBlocks);

        bm.Free("s1");

        Assert.Equal(0, bm.NumUsedBlocks);
        Assert.Equal(8, bm.NumFreeBlocks);
        Assert.False(bm.Contains("s1"));
    }

    [Fact]
    public void Free_SharedBlocks_OnlyReclaimedAtLastReference()
    {
        var bm = new BlockManager(16, 4);
        bm.Allocate("p", 8); // 2 blocks
        bm.Fork("p", "c");   // both reference the same 2 blocks
        Assert.Equal(2, bm.NumUsedBlocks);

        bm.Free("p");
        Assert.Equal(2, bm.NumUsedBlocks); // c still holds them
        Assert.Equal(14, bm.NumFreeBlocks);

        bm.Free("c");
        Assert.Equal(0, bm.NumUsedBlocks); // now reclaimed
        Assert.Equal(16, bm.NumFreeBlocks);
    }

    [Fact]
    public void Free_UnknownSequence_IsNoOp()
    {
        var bm = new BlockManager(8, 4);
        bm.Free("nope"); // must not throw
        Assert.Equal(8, bm.NumFreeBlocks);
    }

    [Fact]
    public void AllocateForkAppendFree_FullCycle_ReturnsPoolToPristine()
    {
        var bm = new BlockManager(totalBlocks: 32, blockSize: 4);

        bm.Allocate("p", 10);       // 3 blocks
        bm.Fork("p", "c1");
        bm.Fork("p", "c2");
        for (int i = 0; i < 20; i++)
        {
            if (bm.CanAppend("c1")) bm.Append("c1");
            if (bm.CanAppend("c2")) bm.Append("c2");
            if (bm.CanAppend("p")) bm.Append("p");
            Assert.Equal(bm.TotalBlocks, bm.NumFreeBlocks + bm.NumUsedBlocks);
        }

        bm.Free("p");
        bm.Free("c1");
        bm.Free("c2");

        Assert.Equal(0, bm.NumUsedBlocks);
        Assert.Equal(32, bm.NumFreeBlocks); // every block reclaimed, no leak, no double-free
        Assert.Equal(0, bm.NumSequences);
    }

    // ---- Reallocation after free (no double-alloc across lifecycles) ---------------------

    [Fact]
    public void ReAllocateAfterFree_ReusesBlocksWithoutOverlapAcrossLiveSequences()
    {
        var bm = new BlockManager(totalBlocks: 4, blockSize: 4); // tiny pool forces reuse
        var first = bm.Allocate("s1", 16).ToArray(); // all 4 blocks
        Assert.Equal(0, bm.NumFreeBlocks);

        bm.Free("s1");
        var live = bm.Allocate("s2", 8).ToArray(); // 2 blocks, reused from pool
        var s3 = bm.Allocate("s3", 8).ToArray();   // other 2 blocks

        Assert.Empty(live.Intersect(s3)); // two live sequences never overlap
        Assert.Equal(4, bm.NumUsedBlocks);
    }
}
