using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for financial neural network models (forecasting, risk, portfolio, NLP).
/// Inherits all NN invariant tests and adds finance-specific invariants:
/// finite predictions, different inputs produce different outputs, output bounded,
/// and zero input stability.
/// </summary>
public abstract class FinancialModelTestBase : NeuralNetworkModelTestBase
{
    // =====================================================
    // FINANCIAL INVARIANT: Predictions Should Be Finite
    // Financial model predictions (prices, returns, risk scores)
    // must always be finite. NaN/Inf in financial output is catastrophic.
    // =====================================================

    [Fact]
    public void FinancialPredictions_ShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        Assert.True(output.Length > 0, "Financial model produced empty output.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Financial prediction[{i}] is NaN — catastrophic for trading/risk systems.");
            Assert.False(double.IsInfinity(output[i]),
                $"Financial prediction[{i}] is Infinity — would cause unbounded positions.");
        }
    }

    // =====================================================
    // FINANCIAL INVARIANT: Different Market Data → Different Predictions
    // A financial model that ignores its input data is useless.
    // =====================================================

    [Fact]
    public void DifferentMarketData_DifferentPredictions()
    {
        var network = CreateNetwork();

        var bullish = CreateConstantTensor(InputShape, 0.8);  // simulating upward data
        var bearish = CreateConstantTensor(InputShape, 0.2);  // simulating downward data

        var predBull = network.Predict(bullish);
        var predBear = network.Predict(bearish);

        bool anyDifferent = false;
        int minLen = Math.Min(predBull.Length, predBear.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(predBull[i] - predBear[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Financial model produces identical predictions for different market data — model is ignoring input.");
    }

    // =====================================================
    // FINANCIAL INVARIANT: Output Bounded
    // Financial predictions should be in a reasonable range.
    // Extreme values indicate numerical instability that would
    // cause catastrophic trading decisions.
    // =====================================================

    [Fact]
    public void Output_ShouldBeBounded()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        for (int i = 0; i < output.Length; i++)
        {
            Assert.True(Math.Abs(output[i]) < 1e8,
                $"Financial output[{i}] = {output[i]:E4} is unbounded. " +
                "Would cause extreme positions in a trading system.");
        }
    }

    // =====================================================
    // FINANCIAL INVARIANT: Zero Input Should Not Crash
    // A market with no data/activity is a valid edge case.
    // =====================================================

    [Fact]
    public void ZeroInput_ShouldNotCrash()
    {
        var network = CreateNetwork();
        var zeroInput = CreateConstantTensor(InputShape, 0.0);

        var output = network.Predict(zeroInput);
        Assert.True(output.Length > 0, "Financial model produced empty output for zero input.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Financial output[{i}] is NaN for zero market data.");
        }
    }
}
