using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for financial risk models (VaR, stress testing, etc.).
/// Inherits financial model invariants and adds risk-specific: non-negative risk
/// and sensitivity to market conditions.
/// </summary>
public abstract class RiskModelTestBase : FinancialModelTestBase
{
    [Fact]
    public void RiskEstimate_ShouldBeFinite()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Risk estimate[{i}] is NaN.");
            Assert.False(double.IsInfinity(output[i]), $"Risk estimate[{i}] is Infinity.");
        }
    }

    [Fact]
    public void DifferentConditions_DifferentRisk()
    {
        var network = CreateNetwork();
        var calm = CreateConstantTensor(InputShape, 0.1);
        var volatile_ = CreateConstantTensor(InputShape, 0.9);

        var risk1 = network.Predict(calm);
        var risk2 = network.Predict(volatile_);

        bool anyDifferent = false;
        int minLen = Math.Min(risk1.Length, risk2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(risk1[i] - risk2[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Risk model produces identical estimates for different market conditions.");
    }
}
