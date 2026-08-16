using System;
using AiDotNet.Finance.Trading.Factors;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance.Factors;

/// <summary>
/// Verifies the two components of <see cref="Stockformer{T}"/> whose specifics come from the
/// reference implementation rather than the paper text: the single-level sym2 band split and the
/// multi-task objective.
/// </summary>
/// <remarks>
/// Ma, Xue, Lu and Chen, arXiv:2401.06139 (Expert Systems with Applications 273:126803, 2025);
/// reference implementation github.com/Eric991005/Multitask-Stockformer.
/// </remarks>
public class StockformerComponentTests
{
    // ---------------------------------------------------------------- band split

    [Fact]
    public void SplitProducesTwoBandsOfHalfLength()
    {
        // level = 1 halves the series. With the paper's T1 = 20 window the encoder therefore spans
        // 10 timesteps, not 20 — the mistake this asserts against is sizing layers from the input
        // window instead of the band length.
        var bands = new StockformerBands<double>(waveletOrder: 2, levels: 1);
        Assert.Equal(10, bands.BandLength(20));

        var series = new Vector<double>(20);
        for (int i = 0; i < 20; i++) series[i] = Math.Sin(i * 0.3);

        var (low, high) = bands.Split(series);
        Assert.True(low.Length > 0, "Low band came back empty.");
        Assert.True(high.Length > 0, "High band came back empty.");
    }

    [Fact]
    public void LowBandTracksTrendAndHighBandCarriesTheJitter()
    {
        // The entire point of the split. A ramp plus alternating spikes: the trend belongs in the low
        // band and the spikes in the high band. If the two were swapped — or if the split were
        // decorative — the high band would carry the larger sustained magnitude.
        var bands = new StockformerBands<double>();
        var series = new Vector<double>(32);
        for (int i = 0; i < 32; i++) series[i] = i + (i % 2 == 0 ? 3.0 : -3.0);

        var (low, high) = bands.Split(series);

        double lowRange = Range(low);
        double highRange = Range(high);
        Assert.True(lowRange > highRange,
            $"Expected the ramp to dominate the low band, but low range {lowRange:F3} <= high range {highRange:F3}.");

        // And the high band must actually see the alternating component rather than being ~zero.
        Assert.True(highRange > 1e-6, "High band is flat; the oscillation was not separated out.");
    }

    [Fact]
    public void ConstructorRejectsUnusableConfiguration()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new StockformerBands<double>(2, levels: 0));
        // Symlet supports a fixed set of orders; an unsupported one must fail loudly rather than
        // silently substituting a different wavelet than the paper names.
        Assert.Throws<ArgumentException>(() => new StockformerBands<double>(waveletOrder: 3));
    }

    [Fact]
    public void SplitRejectsASeriesTooShortToDecompose()
    {
        var bands = new StockformerBands<double>();
        Assert.Throws<ArgumentException>(() => bands.Split(new Vector<double>(1)));
    }

    [Fact]
    public void SplitAllHandlesEveryStockIndependently()
    {
        var bands = new StockformerBands<double>();
        var perStock = new Matrix<double>(4, 16);
        for (int s = 0; s < 4; s++)
        {
            for (int t = 0; t < 16; t++) perStock[s, t] = (s + 1) * Math.Cos(t * 0.4);
        }

        var (low, high) = bands.SplitAll(perStock);
        Assert.Equal(4, low.Rows);
        Assert.Equal(4, high.Rows);
        Assert.Equal(bands.BandLength(16), low.Columns);
        Assert.Equal(bands.BandLength(16), high.Columns);
    }

    // ---------------------------------------------------------------- multi-task loss

    [Fact]
    public void MaskedMaeAveragesOverValidEntriesOnly()
    {
        // THE discriminating case for the mask renormalization. Labels [1, 0, 5] with sentinel 0
        // leave two valid entries with absolute errors 0 and 2, so the answer is 1.0.
        // An implementation that masks but forgets mask /= mean(mask) averages over all THREE
        // entries and returns 0.667 — quietly shrinking this task relative to the classification
        // term, which is exactly the kind of silent reweighting the paper's 1:1 sum forbids.
        var preds  = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var labels = new Vector<double>(new[] { 1.0, 0.0, 5.0 });

        Assert.Equal(1.0, StockformerMultiTaskLoss<double>.MaskedMae(preds, labels, 0.0), 10);
    }

    [Fact]
    public void MaskedMaeReturnsZeroWhenEverythingIsMasked()
    {
        var preds  = new Vector<double>(new[] { 4.0, 5.0 });
        var labels = new Vector<double>(new[] { 0.0, 0.0 });

        // 0/0 in the reference's mask/mean(mask). Finite beats NaN propagating into the total.
        Assert.Equal(0.0, StockformerMultiTaskLoss<double>.MaskedMae(preds, labels, 0.0), 10);
    }

    [Fact]
    public void CrossEntropyMatchesTheClosedFormOnUniformLogits()
    {
        // Two classes, equal logits -> -log(1/2) = log 2, whichever class is the target.
        var logits  = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });
        var targets = new Vector<double>(new[] { 0.0, 1.0 });

        Assert.Equal(Math.Log(2.0), StockformerMultiTaskLoss<double>.CrossEntropy(logits, targets, 2), 10);
    }

    [Fact]
    public void CrossEntropyIsComputedFromRawLogitsAndSurvivesLargeValues()
    {
        // torch.nn.CrossEntropyLoss applies softmax internally, so callers pass raw logits. The
        // log-sum-exp shift must keep a large logit from overflowing to infinity.
        var logits  = new Vector<double>(new[] { 1000.0, 0.0 });
        var targets = new Vector<double>(new[] { 0.0 });

        double loss = StockformerMultiTaskLoss<double>.CrossEntropy(logits, targets, 2);
        Assert.False(double.IsNaN(loss) || double.IsInfinity(loss), $"Overflowed: {loss}");
        Assert.Equal(0.0, loss, 6);   // confidently correct -> ~zero loss
    }

    [Fact]
    public void TotalIsTheUNWEIGHTEDSumOfBothTasksAcrossBothRepresentations()
    {
        // The reference sums the two tasks 1:1 and supervises each head on BOTH representations
        // (four terms total). A weighted form exists in the train script but is commented out, so
        // any coefficient here would be invented.
        var reg      = new Vector<double>(new[] { 1.0, 2.0 });
        var regLow   = new Vector<double>(new[] { 1.5, 2.5 });
        var target   = new Vector<double>(new[] { 2.0, 4.0 });
        var targetLo = new Vector<double>(new[] { 2.0, 4.0 });
        var cls      = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });
        var clsLow   = new Vector<double>(new[] { 0.0, 0.0, 0.0, 0.0 });
        var dir      = new Vector<double>(new[] { 0.0, 1.0 });

        var (regression, classification, total) = StockformerMultiTaskLoss<double>.Compute(
            reg, regLow, target, targetLo, cls, clsLow, dir, numClasses: 2);

        // Regression: MAE(|1-2|,|2-4|)=1.5 plus MAE(|1.5-2|,|2.5-4|)=1.0 -> 2.5
        Assert.Equal(2.5, regression, 10);
        // Classification: log2 + log2
        Assert.Equal(2.0 * Math.Log(2.0), classification, 10);
        Assert.Equal(regression + classification, total, 10);
    }

    [Fact]
    public void ComputeRejectsMismatchedShapes()
    {
        var two   = new Vector<double>(new[] { 1.0, 2.0 });
        var three = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

        Assert.Throws<ArgumentException>(() => StockformerMultiTaskLoss<double>.MaskedMae(two, three));
        // 3 logits cannot cover 2 samples x 2 classes.
        Assert.Throws<ArgumentException>(
            () => StockformerMultiTaskLoss<double>.CrossEntropy(three, two, 2));
    }

    private static double Range(Vector<double> v)
    {
        double min = double.MaxValue, max = double.MinValue;
        for (int i = 0; i < v.Length; i++)
        {
            if (v[i] < min) min = v[i];
            if (v[i] > max) max = v[i];
        }
        return max - min;
    }
}
