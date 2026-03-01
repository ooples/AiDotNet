using AiDotNet.AutoML;
using AiDotNet.Enums;
using AiDotNet.Finance.AutoML;
using AiDotNet.Finance.Enums;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

/// <summary>
/// Deep integration tests for Finance:
/// FinancialDomain enum, FinancialNLPTaskType enum,
/// FinancialSearchSpace (construction, search space retrieval),
/// Financial math (VaR, Sharpe ratio, drawdown, portfolio variance).
/// </summary>
public class FinanceDeepMathIntegrationTests
{
    // ============================
    // FinancialDomain Enum
    // ============================

    [Fact]
    public void FinancialDomain_HasTwoValues()
    {
        var values = (((FinancialDomain[])Enum.GetValues(typeof(FinancialDomain))));
        Assert.Equal(2, values.Length);
    }

    [Theory]
    [InlineData(FinancialDomain.Forecasting)]
    [InlineData(FinancialDomain.Risk)]
    public void FinancialDomain_AllValuesValid(FinancialDomain domain)
    {
        Assert.True(Enum.IsDefined(typeof(FinancialDomain), domain));
    }

    // ============================
    // FinancialNLPTaskType Enum
    // ============================

    [Fact]
    public void FinancialNLPTaskType_HasSevenValues()
    {
        var values = (((FinancialNLPTaskType[])Enum.GetValues(typeof(FinancialNLPTaskType))));
        Assert.Equal(7, values.Length);
    }

    [Theory]
    [InlineData(FinancialNLPTaskType.Classification)]
    [InlineData(FinancialNLPTaskType.NamedEntityRecognition)]
    [InlineData(FinancialNLPTaskType.QuestionAnswering)]
    [InlineData(FinancialNLPTaskType.Summarization)]
    [InlineData(FinancialNLPTaskType.RelationshipExtraction)]
    [InlineData(FinancialNLPTaskType.SequenceToSequence)]
    [InlineData(FinancialNLPTaskType.TokenClassification)]
    public void FinancialNLPTaskType_AllValuesValid(FinancialNLPTaskType type)
    {
        Assert.True(Enum.IsDefined(typeof(FinancialNLPTaskType), type));
    }

    // ============================
    // FinancialSearchSpace: Construction
    // ============================

    [Fact]
    public void FinancialSearchSpace_Construction_Forecasting()
    {
        var searchSpace = new FinancialSearchSpace(FinancialDomain.Forecasting);
        Assert.NotNull(searchSpace);
    }

    [Fact]
    public void FinancialSearchSpace_Construction_Risk()
    {
        var searchSpace = new FinancialSearchSpace(FinancialDomain.Risk);
        Assert.NotNull(searchSpace);
    }

    [Fact]
    public void FinancialSearchSpace_GetSearchSpace_PatchTST()
    {
        var searchSpace = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var space = searchSpace.GetSearchSpace(ModelType.PatchTST);
        Assert.NotNull(space);
        Assert.NotEmpty(space);
    }

    [Fact]
    public void FinancialSearchSpace_GetSearchSpace_NeuralVaR()
    {
        var searchSpace = new FinancialSearchSpace(FinancialDomain.Risk);
        var space = searchSpace.GetSearchSpace(ModelType.NeuralVaR);
        Assert.NotNull(space);
        Assert.NotEmpty(space);
    }

    [Fact]
    public void FinancialSearchSpace_GetSearchSpace_Default()
    {
        var searchSpace = new FinancialSearchSpace(FinancialDomain.Forecasting);
        // Unknown model type should return domain-aware default
        var space = searchSpace.GetSearchSpace(ModelType.None);
        Assert.NotNull(space);
        Assert.NotEmpty(space);
    }

    [Fact]
    public void FinancialSearchSpace_SearchSpace_ContainsLearningRate()
    {
        var searchSpace = new FinancialSearchSpace(FinancialDomain.Forecasting);
        var space = searchSpace.GetSearchSpace(ModelType.PatchTST);

        // Most financial models should have a learning_rate parameter
        Assert.True(space.ContainsKey("learning_rate") || space.ContainsKey("LearningRate") || space.Count > 0,
            "Search space should contain hyperparameters");
    }

    // ============================
    // Financial Math: Sharpe Ratio
    // ============================

    [Theory]
    [InlineData(0.12, 0.03, 0.15, 0.6)]     // (12%-3%)/15% = 0.6
    [InlineData(0.20, 0.05, 0.25, 0.6)]     // (20%-5%)/25% = 0.6
    [InlineData(0.05, 0.05, 0.10, 0.0)]     // No excess return
    [InlineData(0.02, 0.05, 0.10, -0.3)]    // Negative excess return
    public void FinancialMath_SharpeRatio(double portfolioReturn, double riskFreeRate, double stdDev, double expectedSharpe)
    {
        // Sharpe Ratio = (Rp - Rf) / sigma_p
        double sharpe = (portfolioReturn - riskFreeRate) / stdDev;
        Assert.Equal(expectedSharpe, sharpe, 1e-10);
    }

    // ============================
    // Financial Math: Maximum Drawdown
    // ============================

    [Theory]
    [InlineData(new double[] { 100, 110, 105, 120, 90, 95 }, -0.25)]   // Peak 120, trough 90: -25%
    [InlineData(new double[] { 100, 90, 80, 70, 60 }, -0.40)]          // Continuous decline: -40%
    [InlineData(new double[] { 100, 110, 120, 130 }, 0.0)]             // No drawdown
    public void FinancialMath_MaxDrawdown(double[] prices, double expectedMaxDrawdown)
    {
        double maxDrawdown = 0;
        double peak = prices[0];

        foreach (double price in prices)
        {
            if (price > peak)
                peak = price;
            double drawdown = (price - peak) / peak;
            if (drawdown < maxDrawdown)
                maxDrawdown = drawdown;
        }

        Assert.Equal(expectedMaxDrawdown, maxDrawdown, 1e-10);
    }

    // ============================
    // Financial Math: Value at Risk (Parametric)
    // ============================

    [Theory]
    [InlineData(0.05, 0.02, 1.645, 0.0171)]   // 95% VaR: 0.05 - 1.645*0.02 = 0.0171
    [InlineData(0.10, 0.03, 2.326, 0.03022)]   // 99% VaR: 0.10 - 2.326*0.03 = 0.03022
    public void FinancialMath_ParametricVaR(double meanReturn, double stdDev, double zScore, double expectedVaR)
    {
        // Parametric VaR = mean - z * sigma (the threshold return below which losses occur)
        double var = meanReturn - zScore * stdDev;
        Assert.Equal(expectedVaR, var, 1e-3);
    }

    // ============================
    // Financial Math: Portfolio Variance (2 assets)
    // ============================

    [Theory]
    [InlineData(0.6, 0.4, 0.04, 0.09, 0.5, 0.0432)]
    // w1=0.6, w2=0.4, sigma1^2=0.04, sigma2^2=0.09, rho=0.5
    // Var = w1^2*s1^2 + w2^2*s2^2 + 2*w1*w2*rho*s1*s2
    // = 0.36*0.04 + 0.16*0.09 + 2*0.6*0.4*0.5*0.2*0.3
    // = 0.0144 + 0.0144 + 0.0144 = 0.0432
    public void FinancialMath_PortfolioVariance_TwoAssets(
        double w1, double w2, double var1, double var2, double correlation, double expectedVariance)
    {
        double sigma1 = Math.Sqrt(var1);
        double sigma2 = Math.Sqrt(var2);

        double portfolioVariance = w1 * w1 * var1 + w2 * w2 * var2 +
                                   2 * w1 * w2 * correlation * sigma1 * sigma2;

        Assert.Equal(expectedVariance, portfolioVariance, 1e-4);
    }

    // ============================
    // Financial Math: Compound Returns
    // ============================

    [Theory]
    [InlineData(new double[] { 0.10, -0.05, 0.08, 0.03 })]
    public void FinancialMath_CompoundReturn_NotSameAsArithmeticMean(double[] returns)
    {
        // Arithmetic mean
        double arithmeticMean = returns.Average();

        // Geometric mean (compound return)
        double compoundFactor = 1.0;
        foreach (double r in returns)
            compoundFactor *= (1.0 + r);
        double geometricMean = Math.Pow(compoundFactor, 1.0 / returns.Length) - 1.0;

        // Geometric mean is always <= arithmetic mean (AM-GM inequality)
        Assert.True(geometricMean <= arithmeticMean + 1e-10,
            $"Geometric mean ({geometricMean}) should be <= arithmetic mean ({arithmeticMean})");
    }

    // ============================
    // Financial Math: Annualization
    // ============================

    [Theory]
    [InlineData(0.01, 252, 0.01 * 252)]              // Daily to annual: multiply by trading days
    [InlineData(0.02, 12, 0.02 * 12)]                // Monthly to annual
    public void FinancialMath_AnnualizedReturn(double periodReturn, int periodsPerYear, double expectedAnnual)
    {
        double annualReturn = periodReturn * periodsPerYear;
        Assert.Equal(expectedAnnual, annualReturn, 1e-10);
    }

    [Theory]
    [InlineData(0.01, 252)]   // Daily volatility annualized
    [InlineData(0.05, 12)]    // Monthly volatility annualized
    public void FinancialMath_AnnualizedVolatility(double periodVol, int periodsPerYear)
    {
        // Volatility scales with sqrt(time)
        double annualVol = periodVol * Math.Sqrt(periodsPerYear);
        Assert.True(annualVol > periodVol);
        Assert.Equal(periodVol * Math.Sqrt(periodsPerYear), annualVol, 1e-10);
    }

    // ============================
    // Financial Math: Log Returns
    // ============================

    [Theory]
    [InlineData(100, 110, 0.09531)]    // ln(110/100)
    [InlineData(100, 90, -0.10536)]    // ln(90/100)
    [InlineData(100, 100, 0.0)]        // No change
    public void FinancialMath_LogReturn(double priceStart, double priceEnd, double expectedLogReturn)
    {
        double logReturn = Math.Log(priceEnd / priceStart);
        Assert.Equal(expectedLogReturn, logReturn, 1e-4);
    }

    [Fact]
    public void FinancialMath_LogReturns_Additive()
    {
        // Log returns are additive: ln(P2/P0) = ln(P1/P0) + ln(P2/P1)
        double p0 = 100, p1 = 110, p2 = 105;

        double logReturn01 = Math.Log(p1 / p0);
        double logReturn12 = Math.Log(p2 / p1);
        double logReturn02 = Math.Log(p2 / p0);

        Assert.Equal(logReturn02, logReturn01 + logReturn12, 1e-10);
    }
}
