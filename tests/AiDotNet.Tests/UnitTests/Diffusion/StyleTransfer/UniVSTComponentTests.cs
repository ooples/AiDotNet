using System;
using System.Collections.Generic;
using AiDotNet.Diffusion.StyleTransfer;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Diffusion.StyleTransfer;

/// <summary>
/// Verifies UniVST's three components against the paper (Song, Lin, Zhan, Yan, Cao, Ji,
/// "UniVST: A Unified Framework for Training-free Localized Video Style Transfer",
/// arXiv:2410.20084, TPAMI 2025).
/// </summary>
/// <remarks>
/// Each test targets a way the method gets flattened in practice — a single global mask instead of
/// per-frame propagation, nearest-neighbour instead of a k-way majority, a constant blend instead of
/// the beta ramp, latent-only AdaIN, averaging without warping — rather than merely calling the code.
/// UniVST is TRAINING-FREE, so there is no loss to test: any loss here would be an invention.
/// </remarks>
public class UniVSTComponentTests
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

    private static Tensor<double> Mask(int h, int w, Func<int, int, double> value)
    {
        var t = new Tensor<double>(new[] { h, w });
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                t[(y * w) + x] = value(y, x);
        return t;
    }

    // ------------------------------------------------------------ mask propagation

    [Fact]
    public void AnchorsAreTheFirstFramePlusTheNinePrecedingFrames()
    {
        // "first frame + previous 9". Frame 0 stays pinned no matter how far we advance; dropping it
        // in favour of a purely local chain is what makes a propagated mask drift.
        var prop = new UniVSTMaskPropagation<double>(anchorHistory: 9, seed: 1);

        var at15 = prop.SelectAnchorFrames(15);
        Assert.Contains(0, at15);
        Assert.Equal(10, at15.Count);             // frame 0 plus frames 6..14
        Assert.Contains(6, at15);
        Assert.Contains(14, at15);
        Assert.DoesNotContain(5, at15);
        Assert.DoesNotContain(15, at15);          // never itself

        // Early frames simply have fewer anchors, with no duplicate of frame 0.
        Assert.Equal(new[] { 0 }, prop.SelectAnchorFrames(1));
        Assert.Empty(prop.SelectAnchorFrames(0));
    }

    [Fact]
    public void IdenticalFeaturesPropagateTheMaskUnchanged()
    {
        // The soundness floor: if every frame looks identical, correspondence is the identity and the
        // mask must survive verbatim. A propagator that erodes or grows the region fails here.
        //
        // The features put foreground and background on two clearly separated directions, which is
        // what a k-way MAJORITY vote requires and what real inversion features supply — points inside
        // a region resemble each other. Feature vectors that were near-identical everywhere would make
        // the k neighbours of a foreground point mostly background, and the vote would (correctly)
        // flip it; that would be testing the fixture, not the propagator.
        const int h = 6, w = 6, c = 3;
        bool IsFg(int y, int x) => y >= 2 && y <= 3 && x >= 2 && x <= 3;

        // Third channel carries a small unique per-point offset so points are still distinguishable
        // within a region, without moving them off their region's direction.
        var frame = Features(h, w, c, (y, x, k) => k switch
        {
            0 => IsFg(y, x) ? 1.0 : 0.0,
            1 => IsFg(y, x) ? 0.0 : 1.0,
            _ => 0.01 * ((y * w) + x),
        });

        var frames = new List<Tensor<double>> { frame, frame, frame };
        var mask = Mask(h, w, (y, x) => IsFg(y, x) ? 1.0 : 0.0);

        var masks = new UniVSTMaskPropagation<double>(neighbors: 3, downsampleRate: 1.0, seed: 7)
            .Propagate(frames, mask);

        Assert.Equal(3, masks.Count);
        for (int f = 0; f < masks.Count; f++)
            for (int p = 0; p < h * w; p++)
                Assert.Equal(mask[p], masks[f][p], 10);
    }

    [Fact]
    public void PropagationFollowsCorrespondenceWhenContentMoves()
    {
        // The whole point is tracking WITHOUT a tracking model. Frame 1 is frame 0 shifted one column
        // right, so the foreground must move with it. A propagator that just copies frame 0's mask
        // (a single global mask) leaves the region behind and fails.
        const int h = 5, w = 5, c = 6;

        // A distinctive per-column signature so a column matches only its own shifted copy.
        double Sig(int x, int k) => Math.Sin((x * 2.0) + 1.0 + (k * 0.5)) * (1.0 + x);

        var f0 = Features(h, w, c, (y, x, k) => Sig(x, k));
        var f1 = Features(h, w, c, (y, x, k) => Sig(x - 1, k));   // shifted right by one column
        var mask0 = Mask(h, w, (y, x) => x == 1 ? 1.0 : 0.0);      // foreground = column 1

        var masks = new UniVSTMaskPropagation<double>(neighbors: 3, downsampleRate: 1.0, seed: 11)
            .Propagate(new List<Tensor<double>> { f0, f1 }, mask0);

        int col2 = 0, col1 = 0;
        for (int y = 0; y < h; y++)
        {
            if (masks[1][(y * w) + 2] > 0.5) col2++;
            if (masks[1][(y * w) + 1] > 0.5) col1++;
        }

        Assert.True(col2 > col1,
            $"Foreground should have followed the shift into column 2; got col2={col2} col1={col1}.");
    }

    [Fact]
    public void PropagationRejectsMismatchedFrameShapes()
    {
        var a = Features(4, 4, 2, (y, x, k) => 1.0);
        var b = Features(3, 3, 2, (y, x, k) => 1.0);
        var mask = Mask(4, 4, (y, x) => 0.0);

        Assert.Throws<ArgumentException>(() =>
            new UniVSTMaskPropagation<double>(seed: 1).Propagate(new List<Tensor<double>> { a, b }, mask));
    }

    [Fact]
    public void PropagationRejectsAMaskThatDoesNotMatchTheFeatureResolution()
    {
        var a = Features(4, 4, 2, (y, x, k) => 1.0);
        var wrong = Mask(2, 2, (y, x) => 1.0);

        Assert.Throws<ArgumentException>(() =>
            new UniVSTMaskPropagation<double>(seed: 1).Propagate(new List<Tensor<double>> { a }, wrong));
    }

    [Fact]
    public void PropagationRejectsNonSpatialFeatureRank()
    {
        var flat = new Tensor<double>(new[] { 8 });
        var mask = Mask(2, 2, (y, x) => 0.0);
        Assert.Throws<ArgumentException>(() =>
            new UniVSTMaskPropagation<double>(seed: 1).Propagate(new List<Tensor<double>> { flat }, mask));
    }

    [Fact]
    public void PropagationRejectsInvalidHyperparameters()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new UniVSTMaskPropagation<double>(neighbors: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new UniVSTMaskPropagation<double>(anchorHistory: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new UniVSTMaskPropagation<double>(downsampleRate: 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new UniVSTMaskPropagation<double>(downsampleRate: 1.5));
    }

    // ------------------------------------------------------------ AdaIN stylization

    [Fact]
    public void AdaInGivesTheContentTheStylesPerChannelStatistics()
    {
        // AdaIN's definition: after the transform, each channel carries the STYLE's mean and standard
        // deviation. This is the assertion that pins the formula rather than a rescaling that merely
        // looks similar.
        const int c = 2, n = 8;
        var content = new Tensor<double>(new[] { c, n });
        var style = new Tensor<double>(new[] { c, n });
        var rng = new Random(5);
        for (int i = 0; i < c * n; i++) content[i] = rng.NextDouble() * 3.0;
        for (int i = 0; i < c * n; i++) style[i] = (rng.NextDouble() * 10.0) - 5.0;

        var result = new UniVSTAdaInStylization<double>().AdaIn(content, style);

        for (int ch = 0; ch < c; ch++)
        {
            (double m, double s) = MeanStd(result, ch * n, n);
            (double sm, double ss) = MeanStd(style, ch * n, n);
            Assert.Equal(sm, m, 6);
            Assert.Equal(ss, s, 6);
        }
    }

    [Fact]
    public void AdaInIsPerChannelNotGlobal()
    {
        // Two channels with deliberately different style statistics. A single global mean/variance
        // would land both channels on the same numbers and lose the colour relationships that carry
        // the style.
        const int n = 6;
        var content = new Tensor<double>(new[] { 2, n });
        var style = new Tensor<double>(new[] { 2, n });
        for (int i = 0; i < n; i++)
        {
            content[i] = i;
            content[n + i] = i * 2.0;
            style[i] = 100.0 + i;          // channel 0: mean ~102.5
            style[n + i] = -50.0 + (i * 0.1); // channel 1: mean ~ -49.75
        }

        var result = new UniVSTAdaInStylization<double>().AdaIn(content, style);

        (double m0, _) = MeanStd(result, 0, n);
        (double m1, _) = MeanStd(result, n, n);
        Assert.True(m0 > 50.0, $"Channel 0 should take its own style mean; got {m0}.");
        Assert.True(m1 < 0.0, $"Channel 1 should take its own style mean; got {m1}.");
    }

    [Fact]
    public void AdaInLeavesAConstantChannelFiniteRatherThanDividingByZero()
    {
        var content = new Tensor<double>(new[] { 1, 4 });
        for (int i = 0; i < 4; i++) content[i] = 2.0;      // zero variance
        var style = new Tensor<double>(new[] { 1, 4 });
        for (int i = 0; i < 4; i++) style[i] = i;

        var result = new UniVSTAdaInStylization<double>().AdaIn(content, style);
        for (int i = 0; i < 4; i++)
            Assert.False(double.IsNaN(result[i]) || double.IsInfinity(result[i]),
                $"Constant channel produced a non-finite value at {i}: {result[i]}.");
    }

    [Fact]
    public void AdaInRejectsMismatchedChannelCounts()
    {
        var a = new Tensor<double>(new[] { 2, 3 });
        var b = new Tensor<double>(new[] { 3, 3 });
        Assert.Throws<ArgumentException>(() => new UniVSTAdaInStylization<double>().AdaIn(a, b));
    }

    [Fact]
    public void BetaRampsLinearlyFromOneTenthToNineTenths()
    {
        // beta_tau2 = 0.1 at 0.4T rising to beta_tau3 = 0.9 at 1.0T, linearly. A constant blend is
        // the obvious simplification and it removes the schedule the paper relies on.
        var s = new UniVSTAdaInStylization<double>();

        Assert.Equal(0.1, s.BetaAt(0.4), 10);
        Assert.Equal(0.9, s.BetaAt(1.0), 10);
        Assert.Equal(0.5, s.BetaAt(0.7), 10);   // midpoint of [0.4, 1.0]

        // Clamped, not extrapolated: outside the ramp beta must not run past its endpoints.
        Assert.Equal(0.1, s.BetaAt(0.0), 10);
        Assert.Equal(0.1, s.BetaAt(0.2), 10);
        Assert.Equal(0.9, s.BetaAt(1.5), 10);

        // Monotone increasing across the ramp.
        double prev = -1.0;
        for (double f = 0.4; f <= 1.0001; f += 0.05)
        {
            double b = s.BetaAt(f);
            Assert.True(b >= prev, $"beta must not decrease across the ramp; {b} < {prev} at {f}.");
            prev = b;
        }
    }

    [Fact]
    public void LatentAdaInAppliesOnlyInItsNarrowLateWindow()
    {
        // [0.1T, 0.15T]. Applying latent AdaIN across the whole trajectory overwhelms content
        // structure, so the narrowness is the design.
        var s = new UniVSTAdaInStylization<double>();
        Assert.True(s.IsLatentAdaInActive(0.10));
        Assert.True(s.IsLatentAdaInActive(0.125));
        Assert.True(s.IsLatentAdaInActive(0.15));
        Assert.False(s.IsLatentAdaInActive(0.09));
        Assert.False(s.IsLatentAdaInActive(0.16));
        Assert.False(s.IsLatentAdaInActive(0.5));
    }

    [Fact]
    public void KeyValueWindowStartsAtFourTenthsAndRunsToT()
    {
        var s = new UniVSTAdaInStylization<double>();
        Assert.False(s.IsKeyValueAdaInActive(0.39));
        Assert.True(s.IsKeyValueAdaInActive(0.4));
        Assert.True(s.IsKeyValueAdaInActive(1.0));
    }

    [Fact]
    public void QueryBlendUsesGammaAndAppliesAtEveryTimestep()
    {
        // gamma = 0.35 on the EDIT query, so the content query keeps the larger share (0.65). Getting
        // the weights backwards would let the stylized branch stop attending to original structure.
        // Note there is no timestep argument at all: unlike the other two operations this one has no
        // window, and that asymmetry is deliberate.
        var edit = new Tensor<double>(new[] { 1, 2 });
        var content = new Tensor<double>(new[] { 1, 2 });
        edit[0] = 1.0; edit[1] = 1.0;
        content[0] = 0.0; content[1] = 0.0;

        var blended = new UniVSTAdaInStylization<double>().BlendQuery(edit, content);
        Assert.Equal(0.35, blended[0], 10);
        Assert.Equal(0.35, blended[1], 10);
    }

    [Fact]
    public void KeyValueBlendLeansOnRawStyleAtTheLowBetaEnd()
    {
        // K~ = beta * AdaIN(K_edit, K_style) + (1 - beta) * K_style. At beta -> 0 the result is the
        // RAW STYLE key/value, NOT the edit branch. Reading the complement as "the edit tensor" is an
        // easy misimplementation that inverts the early-timestep behaviour.
        var options = new UniVSTOptions { BetaAtRampStart = 0.0, BetaAtRampEnd = 0.0 };
        var s = new UniVSTAdaInStylization<double>(options);

        var edit = new Tensor<double>(new[] { 1, 4 });
        var style = new Tensor<double>(new[] { 1, 4 });
        for (int i = 0; i < 4; i++) { edit[i] = i * 10.0; style[i] = 1.0 + i; }

        var result = s.BlendKeyValue(edit, style, 0.4);
        for (int i = 0; i < 4; i++)
            Assert.Equal(style[i], result[i], 8);
    }

    [Fact]
    public void MaskGatingKeepsContentOutsideAndEditedInside()
    {
        // Z = M * edited + (1 - M) * content. This is what makes the stylization LOCALIZED; without
        // it UniVST would restyle the whole frame like the methods it improves on.
        const int c = 2, h = 2, w = 2;
        var edited = new Tensor<double>(new[] { c, h, w });
        var content = new Tensor<double>(new[] { c, h, w });
        for (int i = 0; i < c * h * w; i++) { edited[i] = 9.0; content[i] = 1.0; }

        // Foreground only at (0,0), and the mask must broadcast across BOTH channels.
        var mask = Mask(h, w, (y, x) => (y == 0 && x == 0) ? 1.0 : 0.0);

        var result = new UniVSTAdaInStylization<double>().ApplyMask(edited, content, mask);

        for (int ch = 0; ch < c; ch++)
        {
            int o = ch * h * w;
            Assert.Equal(9.0, result[o + 0], 10);   // inside the mask
            Assert.Equal(1.0, result[o + 1], 10);   // outside
            Assert.Equal(1.0, result[o + 2], 10);
            Assert.Equal(1.0, result[o + 3], 10);
        }
    }

    [Fact]
    public void MaskGatingRejectsAMaskThatDoesNotTileTheLatent()
    {
        var edited = new Tensor<double>(new[] { 2, 2, 2 });
        var content = new Tensor<double>(new[] { 2, 2, 2 });
        var mask = new Tensor<double>(new[] { 3 });
        Assert.Throws<ArgumentException>(() =>
            new UniVSTAdaInStylization<double>().ApplyMask(edited, content, mask));
    }

    // ------------------------------------------------------------ consistent smoothing

    [Fact]
    public void SmoothingWindowIsClampedAtTheSequenceEndsNotWrapped()
    {
        // Wrapping would average the last frame into the first and invent motion across a cut.
        var sm = new UniVSTConsistentSmoothing<double>(new UniVSTOptions { SmoothingHalfWindow = 2 });
        Assert.Equal(5, sm.WindowLength);

        Assert.Equal(new[] { 0, 1, 2 }, sm.WindowIndices(0, 8));           // no negative wrap
        Assert.Equal(new[] { 5, 6, 7 }, sm.WindowIndices(7, 8));           // no wrap to 0
        Assert.Equal(new[] { 2, 3, 4, 5, 6 }, sm.WindowIndices(4, 8));     // full 2m+1 in the middle
    }

    [Fact]
    public void SmoothingAveragesOverTheFramesActuallyContributed()
    {
        // At the ends the window is short. Dividing by the nominal 2m+1 there would pull those frames
        // toward zero and darken the start and end of every video.
        var sm = new UniVSTConsistentSmoothing<double>(new UniVSTOptions { SmoothingHalfWindow = 1 });

        var frames = new List<Tensor<double>>();
        for (int i = 0; i < 3; i++)
        {
            var f = new Tensor<double>(new[] { 1, 1, 1 });
            f[0] = 6.0;                       // identical frames
            frames.Add(f);
        }

        // Identity flow, so warping is a no-op and the mean of identical frames must be the value
        // itself — at every position including the two ends.
        var zeroFlow = new Tensor<double>(new[] { 2, 1, 1 });
        var result = sm.SmoothSequence(frames, (i, j) => zeroFlow);

        Assert.Equal(3, result.Count);
        for (int i = 0; i < 3; i++)
            Assert.Equal(6.0, result[i][0], 8);
    }

    [Fact]
    public void SmoothingSkipsPairsWithNoFlowRatherThanTreatingThemAsAligned()
    {
        // A null flow means no correspondence was established. Substituting a zero field would average
        // in an UNALIGNED neighbour and reintroduce the ghosting this step exists to remove, so the
        // frame must pass through untouched when it has no usable neighbours.
        var sm = new UniVSTConsistentSmoothing<double>(new UniVSTOptions { SmoothingHalfWindow = 1 });

        var frames = new List<Tensor<double>>();
        for (int i = 0; i < 3; i++)
        {
            var f = new Tensor<double>(new[] { 1, 1, 1 });
            f[0] = i == 1 ? 100.0 : 0.0;      // a spike in the middle frame only
            frames.Add(f);
        }

        var result = sm.SmoothSequence(frames, (i, j) => null);

        Assert.Equal(0.0, result[0][0], 8);
        Assert.Equal(100.0, result[1][0], 8);   // untouched, not pulled toward its neighbours
        Assert.Equal(0.0, result[2][0], 8);
    }

    [Fact]
    public void SmoothingAppliesOnlyInsideItsTimestepWindow()
    {
        var sm = new UniVSTConsistentSmoothing<double>();
        Assert.False(sm.IsActive(0.29));
        Assert.True(sm.IsActive(0.3));
        Assert.True(sm.IsActive(0.35));
        Assert.True(sm.IsActive(0.4));
        Assert.False(sm.IsActive(0.41));
    }

    [Fact]
    public void RefinedNoiseAndStepReproduceTheDdimUpdate()
    {
        // The correction re-enters through the NOISE, not by overwriting the latent, so the sampler's
        // own update rule still holds. Feeding back the exact x0 that generated z_t must therefore
        // return the standard DDIM step for z_{t-1}.
        const double alphaT = 0.36, alphaPrev = 0.64;
        var sm = new UniVSTConsistentSmoothing<double>();

        var x0 = new Tensor<double>(new[] { 1, 3 });
        var eps = new Tensor<double>(new[] { 1, 3 });
        for (int i = 0; i < 3; i++) { x0[i] = 0.5 * (i + 1); eps[i] = 0.25 - (0.1 * i); }

        // Build z_t exactly as the forward process would.
        var zt = new Tensor<double>(new[] { 1, 3 });
        for (int i = 0; i < 3; i++)
            zt[i] = (Math.Sqrt(alphaT) * x0[i]) + (Math.Sqrt(1.0 - alphaT) * eps[i]);

        var recovered = sm.RefineNoise(zt, x0, alphaT);
        for (int i = 0; i < 3; i++)
            Assert.Equal(eps[i], recovered[i], 8);   // epsilon solved back out

        var zPrev = sm.StepWithRefinedNoise(x0, recovered, alphaPrev);
        for (int i = 0; i < 3; i++)
        {
            double expected = (Math.Sqrt(alphaPrev) * x0[i]) + (Math.Sqrt(1.0 - alphaPrev) * eps[i]);
            Assert.Equal(expected, zPrev[i], 8);
        }
    }

    [Fact]
    public void RefineNoiseRejectsAnOutOfRangeAlpha()
    {
        var sm = new UniVSTConsistentSmoothing<double>();
        var a = new Tensor<double>(new[] { 1, 1 });
        Assert.Throws<ArgumentOutOfRangeException>(() => sm.RefineNoise(a, a, 0.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => sm.RefineNoise(a, a, 1.5));
    }

    [Fact]
    public void RefineNoiseIsZeroWhenNoNoiseRemains()
    {
        // alpha-bar == 1 is t = 0: there is no noise component left to solve for, so the residual is
        // zero rather than a division by zero.
        var sm = new UniVSTConsistentSmoothing<double>();
        var zt = new Tensor<double>(new[] { 1, 2 });
        var x0 = new Tensor<double>(new[] { 1, 2 });
        zt[0] = 3.0; zt[1] = 4.0; x0[0] = 3.0; x0[1] = 4.0;

        var eps = sm.RefineNoise(zt, x0, 1.0);
        Assert.Equal(0.0, eps[0], 10);
        Assert.Equal(0.0, eps[1], 10);
    }

    // ------------------------------------------------------------ options

    [Fact]
    public void OptionsDefaultsAreThePapersPublishedValues()
    {
        var o = new UniVSTOptions();
        Assert.Equal(50, o.DdimSteps);
        Assert.Equal(16, o.NumFrames);
        Assert.Equal(0.4, o.MaskFeatureTimestepFraction, 10);
        Assert.Equal(2, o.MaskFeatureUpBlockIndex);
        Assert.Equal(10, o.MaskMatchNeighbors);
        Assert.Equal(9, o.MaskAnchorHistory);
        Assert.Equal(0.5, o.MaskDownsampleRate, 10);
        Assert.Equal(0.10, o.LatentAdaInStartFraction, 10);
        Assert.Equal(0.15, o.LatentAdaInEndFraction, 10);
        Assert.Equal(0.35, o.QueryBlendGamma, 10);
        Assert.Equal(0.4, o.KeyValueAdaInStartFraction, 10);
        Assert.Equal(1.0, o.KeyValueAdaInEndFraction, 10);
        Assert.Equal(0.1, o.BetaAtRampStart, 10);
        Assert.Equal(0.9, o.BetaAtRampEnd, 10);
        Assert.Equal(0.3, o.SmoothingStartFraction, 10);
        Assert.Equal(0.4, o.SmoothingEndFraction, 10);
        o.Validate();
    }

    [Fact]
    public void OptionsRejectAnInvertedInterval()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new UniVSTOptions { LatentAdaInStartFraction = 0.5, LatentAdaInEndFraction = 0.2 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new UniVSTOptions { SmoothingStartFraction = 0.9, SmoothingEndFraction = 0.1 }.Validate());
    }

    [Fact]
    public void OptionsRejectFractionsOutsideTheUnitInterval()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new UniVSTOptions { MaskFeatureTimestepFraction = 1.4 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new UniVSTOptions { QueryBlendGamma = -0.1 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new UniVSTOptions { DdimSteps = 0 }.Validate());
    }

    private static (double Mean, double Std) MeanStd(Tensor<double> t, int offset, int count)
    {
        double mean = 0.0;
        for (int i = 0; i < count; i++) mean += t[offset + i];
        mean /= count;
        double var = 0.0;
        for (int i = 0; i < count; i++)
        {
            double d = t[offset + i] - mean;
            var += d * d;
        }
        return (mean, Math.Sqrt(var / count));
    }
}
