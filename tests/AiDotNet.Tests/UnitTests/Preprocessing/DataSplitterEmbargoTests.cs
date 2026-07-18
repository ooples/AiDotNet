using AiDotNet.Preprocessing.DataPreparation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Preprocessing;

/// <summary>
/// Tests for the purged/embargoed chronological split: on the time-ordered (no-shuffle) path a gap of
/// <c>embargo</c> rows is dropped at each partition boundary so a horizon-h label can never straddle it. On
/// the shuffled (i.i.d.) path the embargo is a no-op. Row value == original index makes leakage observable.
/// </summary>
public sealed class DataSplitterEmbargoTests
{
    private static (Matrix<double> x, Vector<double> y) Sequential(int n)
    {
        var x = new Matrix<double>(n, 1);
        var y = new Vector<double>(n);
        for (var i = 0; i < n; i++)
        {
            x[i, 0] = i;
            y[i] = i;
        }

        return (x, y);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Chronological_embargo_drops_a_gap_at_each_boundary()
    {
        const int n = 200;
        const int embargo = 5;
        var (x, y) = Sequential(n);

        var (_, yTr, _, yVal, _, yTe) = DataSplitter.Split<double, Matrix<double>, Vector<double>>(
            x, y, trainRatio: 0.7, validationRatio: 0.15, shuffle: false, embargo: embargo);

        double lastTrain = yTr[yTr.Length - 1];
        double firstVal = yVal[0];
        double lastVal = yVal[yVal.Length - 1];
        double firstTest = yTe[0];

        // A horizon-`embargo` label from the last training row lands strictly BEFORE the first validation row
        // (i.e. inside the dropped gap) — no future information crosses the boundary.
        Assert.True(lastTrain + embargo < firstVal, $"train→val leak: {lastTrain}+{embargo} !< {firstVal}");
        Assert.True(lastVal + embargo < firstTest, $"val→test leak: {lastVal}+{embargo} !< {firstTest}");

        // Chronological order is preserved (no shuffle).
        Assert.Equal(0.0, yTr[0]);
        // Exactly two embargo gaps' worth of rows are dropped.
        Assert.Equal(n - 2 * embargo, yTr.Length + yVal.Length + yTe.Length);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Shuffled_split_ignores_embargo_and_uses_every_row()
    {
        const int n = 200;
        var (x, y) = Sequential(n);

        var (_, yTr, _, yVal, _, yTe) = DataSplitter.Split<double, Matrix<double>, Vector<double>>(
            x, y, trainRatio: 0.7, validationRatio: 0.15, shuffle: true, embargo: 5);

        // i.i.d. data has no temporal boundary to protect, so the embargo is a no-op: all rows are used.
        Assert.Equal(n, yTr.Length + yVal.Length + yTe.Length);
    }

    [Fact]
    [Trait("category", "unit")]
    public void Zero_embargo_preserves_the_contiguous_split()
    {
        const int n = 200;
        var (x, y) = Sequential(n);

        var (_, yTr, _, yVal, _, yTe) = DataSplitter.Split<double, Matrix<double>, Vector<double>>(
            x, y, trainRatio: 0.7, validationRatio: 0.15, shuffle: false, embargo: 0);

        // Default embargo (0) is byte-for-byte the historical contiguous chronological split.
        Assert.Equal(n, yTr.Length + yVal.Length + yTe.Length);
        Assert.Equal(0.0, yTr[0]);
        Assert.Equal(yTr[yTr.Length - 1] + 1, yVal[0]); // val starts immediately after train (no gap)
    }
}
