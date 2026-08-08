using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.Video.Denoising;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Video.Denoising;

/// <summary>
/// Verifies Shift-Net's grouped spatial-temporal shift module against the paper
/// (Li et al., CVPR 2023, arXiv:2206.10810).
/// </summary>
/// <remarks>
/// Shifting is parameter-free, so its correctness is a property of the operation rather than of trained
/// weights — which makes it fully checkable. Each test targets a way the scheme gets simplified: one
/// global shift instead of per-slice, temporal-only, clamped instead of zero-filled borders.
/// </remarks>
public class GroupedSpatialTemporalShiftTests
{
    private static Tensor<double> Features(int h, int w, int c, Func<int, int, int, double> value)
    {
        var t = new Tensor<double>(new[] { h, w, c });
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int k = 0; k < c; k++)
                    t[(((y * w) + x) * c) + k] = value(y, x, k);
        return t;
    }

    [Fact]
    public void TheDisplacementSetIsThePapersTwentyFiveOffsets()
    {
        // M = 25 from {-9,-5,0,5,9} squared. The formula |d| = k*(s-1)+1 with s = 5 yields magnitudes
        // 1, 5, 9 — but the implementation's stated set omits 1 and includes 0, and only that set gives
        // exactly 25 pairs.
        Assert.Equal(25, GroupedSpatialTemporalShift<double>.SliceCount);
        Assert.Equal(new[] { -9, -5, 0, 5, 9 }, GroupedSpatialTemporalShift<double>.Displacements);
        Assert.Equal(25, GroupedSpatialTemporalShift<double>.Offsets.Count);
        Assert.Equal(5, GroupedSpatialTemporalShift<double>.BaseShiftLength);

        // (0, 0) must be present: without an unshifted slice the module cannot represent "no motion".
        Assert.Contains((0, 0), GroupedSpatialTemporalShift<double>.Offsets);
    }

    [Fact]
    public void SplitGroupsHalvesTheChannelsAndKeepsTheHalvesDistinct()
    {
        // f_i^a, f_i^b in R^(h x w x c/2). Channel k < half goes to a, the rest to b.
        var features = Features(1, 1, 4, (y, x, k) => k);
        var (a, b) = new GroupedSpatialTemporalShift<double>().SplitGroups(features);

        Assert.Equal(2, a.Shape[2]);
        Assert.Equal(2, b.Shape[2]);
        Assert.Equal(new[] { 0.0, 1.0 }, new[] { a[0], a[1] });
        Assert.Equal(new[] { 2.0, 3.0 }, new[] { b[0], b[1] });
    }

    [Fact]
    public void SplitGroupsRejectsAnOddChannelCount()
    {
        // An off-by-one in the split shifts every downstream slice boundary, so it fails loudly.
        var odd = Features(1, 1, 3, (y, x, k) => 1.0);
        Assert.Throws<ArgumentException>(() => new GroupedSpatialTemporalShift<double>().SplitGroups(odd));
    }

    [Fact]
    public void ShiftMovesContentAndZeroFillsTheVacatedBorder()
    {
        // A single bright pixel at (0,0) shifted by (+1, 0) must land at (0,1), and (0,0) must become
        // ZERO. Clamping would leave the original value there and invent a correspondence; wrapping
        // would move it to the far edge.
        var features = Features(2, 2, 1, (y, x, k) => (y == 0 && x == 0) ? 1.0 : 0.0);
        var shifted = new GroupedSpatialTemporalShift<double>().Shift(features, dx: 1, dy: 0);

        Assert.Equal(1.0, shifted[(0 * 2) + 1], 10);   // arrived at (0,1)
        Assert.Equal(0.0, shifted[(0 * 2) + 0], 10);   // vacated, zero-filled
    }

    [Fact]
    public void ShiftIsAPureMoveWithNoBlending()
    {
        // The paper's argument is that shifting is FREE. Any interpolation or weighted blend would make
        // it a convolution and defeat the point, so shifted values must be EXACTLY the originals.
        var features = Features(3, 3, 1, (y, x, k) => (y * 3) + x + 1);
        var shifted = new GroupedSpatialTemporalShift<double>().Shift(features, dx: 1, dy: 1);

        // (2,2) should hold whatever was at (1,1).
        Assert.Equal(features[(1 * 3) + 1], shifted[(2 * 3) + 2], 12);
    }

    [Fact]
    public void GroupedSpatialShiftGivesEachSliceADifferentDisplacement()
    {
        // "Grouped" means per-slice displacements, not one global shift. With 25 channels there is one
        // channel per slice, so a single bright row lands in a different place for each channel.
        // A single global shift would move every channel identically.
        int m = GroupedSpatialTemporalShift<double>.SliceCount;
        var features = Features(21, 21, m, (y, x, k) => (y == 10 && x == 10) ? 1.0 : 0.0);

        var shifted = new GroupedSpatialTemporalShift<double>().GroupedSpatialShift(features);

        // Locate the bright pixel per channel and confirm at least two channels differ.
        var found = new (int Y, int X)[m];
        for (int k = 0; k < m; k++)
        {
            found[k] = (-1, -1);
            for (int y = 0; y < 21; y++)
                for (int x = 0; x < 21; x++)
                    if (shifted[(((y * 21) + x) * m) + k] > 0.5) found[k] = (y, x);
        }

        int distinct = 0;
        var seen = new System.Collections.Generic.HashSet<(int, int)>();
        foreach (var f in found) if (f.Y >= 0 && seen.Add(f)) distinct++;

        Assert.True(distinct > 1,
            $"Only {distinct} distinct destination(s); the slices are not being shifted independently.");
    }

    [Fact]
    public void GroupedSpatialShiftMatchesTheDeclaredOffsetForEachSlice()
    {
        // Stronger than "they differ": slice m must land exactly at Offsets[m].
        int m = GroupedSpatialTemporalShift<double>.SliceCount;
        var module = new GroupedSpatialTemporalShift<double>();
        var features = Features(21, 21, m, (y, x, k) => (y == 10 && x == 10) ? 1.0 : 0.0);
        var shifted = module.GroupedSpatialShift(features);

        for (int k = 0; k < m; k++)
        {
            var (dx, dy) = GroupedSpatialTemporalShift<double>.Offsets[k];
            int ey = 10 + dy, ex = 10 + dx;
            if (ey < 0 || ey >= 21 || ex < 0 || ex >= 21) continue;

            Assert.Equal(1.0, shifted[(((ey * 21) + ex) * m) + k], 10);
        }
    }

    [Fact]
    public void GroupedSpatialShiftRejectsTooFewChannels()
    {
        // Fewer than M channels cannot form M slices.
        var narrow = Features(4, 4, 4, (y, x, k) => 1.0);
        Assert.Throws<ArgumentException>(
            () => new GroupedSpatialTemporalShift<double>().GroupedSpatialShift(narrow));
    }

    [Fact]
    public void TemporalShiftKeepsOwnAGroupAndTakesTheNeighboursBGroup()
    {
        // The frame's own a group passes through untouched; the b half comes from the neighbour. Getting
        // this backwards would discard the frame's own information.
        int m = GroupedSpatialTemporalShift<double>.SliceCount;
        int c = m * 2;   // even, and enough channels for M slices in each half
        var current = Features(21, 21, c, (y, x, k) => k < c / 2 ? 7.0 : 8.0);
        var previous = Features(21, 21, c, (y, x, k) => k < c / 2 ? 1.0 : 2.0);

        var result = new GroupedSpatialTemporalShift<double>()
            .ForwardTemporalShift(current, previous);

        // a half preserved from `current` (7.0), never the neighbour's 1.0.
        Assert.Equal(7.0, result[(((10 * 21) + 10) * c) + 0], 10);
        // b half came from `previous` (2.0 shifted), never `current`'s 8.0.
        Assert.Equal(2.0, result[(((10 * 21) + 10) * c) + (c / 2)], 10);
    }

    [Fact]
    public void ForwardAndBackwardShiftsDrawFromOppositeNeighbours()
    {
        // FTS and BTS alternate so information flows both ways. With distinguishable neighbours the two
        // directions must produce different b halves.
        int m = GroupedSpatialTemporalShift<double>.SliceCount;
        int c = m * 2;
        var current = Features(21, 21, c, (y, x, k) => 0.0);
        var past = Features(21, 21, c, (y, x, k) => k < c / 2 ? 0.0 : 3.0);
        var future = Features(21, 21, c, (y, x, k) => k < c / 2 ? 0.0 : 9.0);
        var module = new GroupedSpatialTemporalShift<double>();

        var fromPast = module.ForwardTemporalShift(current, past);
        var fromFuture = module.BackwardTemporalShift(current, future);

        int bIndex = (((10 * 21) + 10) * c) + (c / 2);
        Assert.Equal(3.0, fromPast[bIndex], 10);
        Assert.Equal(9.0, fromFuture[bIndex], 10);
    }

    [Fact]
    public void TemporalShiftRejectsMismatchedFrameShapes()
    {
        int c = GroupedSpatialTemporalShift<double>.SliceCount * 2;
        var a = Features(8, 8, c, (y, x, k) => 1.0);
        var b = Features(4, 4, c, (y, x, k) => 1.0);

        Assert.Throws<ArgumentException>(
            () => new GroupedSpatialTemporalShift<double>().ForwardTemporalShift(a, b));
    }

    [Fact]
    public void LossIsPlainL1()
    {
        // L = (1/T) sum ||H_i - O_i||_1. No perceptual or adversarial term: the paper's claim is that a
        // simple baseline suffices, so extra losses would confound its own comparison.
        var predicted = new Vector<double>(new[] { 1.0, 3.0 });
        var target = new Vector<double>(new[] { 0.0, 0.0 });

        Assert.Equal(2.0, GroupedSpatialTemporalShift<double>.Loss(predicted, target), 10);
    }
}
