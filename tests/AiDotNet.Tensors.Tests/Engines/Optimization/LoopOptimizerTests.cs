using System;
using System.Collections.Concurrent;
using AiDotNet.Tensors.Engines.Optimization;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

public class LoopOptimizerTests
{
    [Fact]
    public void Tile2D_VisitsAllTiles()
    {
        int rows = 10;
        int cols = 9;
        int tileSize = 4;
        int count = 0;

        LoopOptimizer.Tile2D(rows, cols, tileSize, (iStart, iEnd, jStart, jEnd) =>
        {
            Assert.InRange(iStart, 0, rows - 1);
            Assert.InRange(iEnd, 1, rows);
            Assert.InRange(jStart, 0, cols - 1);
            Assert.InRange(jEnd, 1, cols);
            count++;
        });

        int expectedTilesI = (rows + tileSize - 1) / tileSize;
        int expectedTilesJ = (cols + tileSize - 1) / tileSize;
        Assert.Equal(expectedTilesI * expectedTilesJ, count);
    }

    [Fact]
    public void UnrollBy4_InvokesActionForAllIndices()
    {
        const int length = 17;
        int seen = 0;

        LoopOptimizer.UnrollBy4(length, _ => seen++);

        Assert.Equal(length, seen);
    }

    [Fact]
    public void UnrollBy8_InvokesActionForAllIndices()
    {
        const int length = 17;
        int seen = 0;

        LoopOptimizer.UnrollBy8(length, _ => seen++);

        Assert.Equal(length, seen);
    }

    [Fact]
    public void StripMine_CoversFullRange()
    {
        const int total = 10;
        const int strip = 4;
        int covered = 0;

        LoopOptimizer.StripMine(total, strip, (start, end) => covered += (end - start));

        Assert.Equal(total, covered);
    }

    [Fact]
    public void Fuse_RunsAllActionsEachIteration()
    {
        const int length = 5;
        int a = 0;
        int b = 0;

        LoopOptimizer.Fuse(length, _ => a++, _ => b++);

        Assert.Equal(length, a);
        Assert.Equal(length, b);
    }

    [Fact]
    public void OptimalOrder2D_RowMajorAndColumnMajorVisitAll()
    {
        const int rows = 3;
        const int cols = 4;

        int rowMajor = 0;
        LoopOptimizer.OptimalOrder2D(rows, cols, rowMajorAccess: true, (_, _) => rowMajor++);
        Assert.Equal(rows * cols, rowMajor);

        int colMajor = 0;
        LoopOptimizer.OptimalOrder2D(rows, cols, rowMajorAccess: false, (_, _) => colMajor++);
        Assert.Equal(rows * cols, colMajor);
    }

    [Fact]
    public void ParallelTile2D_VisitsAllTiles()
    {
        int rows = 9;
        int cols = 9;
        int tileSize = 4;

        var tiles = new ConcurrentBag<(int, int, int, int)>();

        LoopOptimizer.ParallelTile2D(rows, cols, tileSize, (iStart, iEnd, jStart, jEnd) =>
        {
            tiles.Add((iStart, iEnd, jStart, jEnd));
        });

        int expectedTilesI = (rows + tileSize - 1) / tileSize;
        int expectedTilesJ = (cols + tileSize - 1) / tileSize;
        Assert.Equal(expectedTilesI * expectedTilesJ, tiles.Count);
    }

    [Fact]
    public void DetermineOptimalTileSize_ReturnsAtLeastOne_AndAtMostDimension()
    {
        int tileSize = LoopOptimizer.DetermineOptimalTileSize(dimension: 128);

        Assert.InRange(tileSize, 1, 128);
    }
}

