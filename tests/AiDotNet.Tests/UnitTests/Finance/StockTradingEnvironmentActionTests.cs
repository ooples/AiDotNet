using AiDotNet.Finance.Trading.Environments;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Finance;

/// <summary>
/// The discrete action an action-score vector selects: the scan
/// <c>best = 0; if (Compare(v[i], v[best]) &gt; 0) best = i</c>, in which NaN ranks below every number.
/// </summary>
public sealed class StockTradingEnvironmentActionTests
{
    [Fact]
    public void FirstMaxIndex_Decimal_PicksTheFirstLargestScore()
    {
        // decimal has no infinity: substituting -infinity for NaN threw OverflowException for every decimal action.
        Assert.Equal(1, StockTradingEnvironment<decimal>.FirstMaxIndex(new Vector<decimal>(new[] { 1m, 3m, 3m })));
    }

    [Fact]
    public void FirstMaxIndex_RealNegativeInfinityBeatsAnEarlierNaN()
    {
        // CompareTo puts NaN below -infinity, so the scan moves to index 1. Replacing NaN with -infinity made the
        // two equal and returned 0.
        Assert.Equal(1, StockTradingEnvironment<double>.FirstMaxIndex(
            new Vector<double>(new[] { double.NaN, double.NegativeInfinity, double.NaN })));
    }

    [Fact]
    public void FirstMaxIndex_AllNaN_IsZero_AndTiesKeepTheFirst()
    {
        Assert.Equal(0, StockTradingEnvironment<double>.FirstMaxIndex(new Vector<double>(new[] { double.NaN, double.NaN })));
        Assert.Equal(2, StockTradingEnvironment<double>.FirstMaxIndex(
            new Vector<double>(new[] { 0.5, double.NaN, 2.0, 1.0, 2.0 })));
    }
}
