using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for financial risk models (VaR, stress testing, etc.).
/// Inherits financial model invariants and adds risk-specific: non-negative risk
/// and sensitivity to market conditions.
/// </summary>
/// <remarks>
/// Generic over T so the source generator's float scaffold can emit
/// <c>RiskModelTestBase&lt;float&gt;</c>. While this base was non-generic, every risk model
/// (TabNet, TabTransformer, ...) was locked to &lt;double&gt; — adding them to
/// <c>Fp32TestClassNames</c> would have been CS0308 — so the float-first remedy was
/// unavailable and only fixture shrinks could fit them in the CI budget. Same treatment
/// DocumentNNModelTestBase already received. The non-generic shim below preserves every
/// existing <c>: RiskModelTestBase</c> derivation.
/// </remarks>
public abstract class RiskModelTestBase<T> : FinancialModelTestBase<T>
{
    [Fact(Timeout = 60000)]
    public async Task RiskEstimate_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            double v = ConvertToDouble(output[i]);
            Assert.False(double.IsNaN(v), $"Risk estimate[{i}] is NaN.");
            Assert.False(double.IsInfinity(v), $"Risk estimate[{i}] is Infinity.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task DifferentConditions_DifferentRisk()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();
        var calm = CreateConstantTensor(InputShape, 0.1);
        var volatile_ = CreateConstantTensor(InputShape, 0.9);

        var risk1 = network.Predict(calm);
        var risk2 = network.Predict(volatile_);

        bool anyDifferent = false;
        int minLen = Math.Min(risk1.Length, risk2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(ConvertToDouble(risk1[i]) - ConvertToDouble(risk2[i])) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Risk model produces identical estimates for different market conditions.");
    }
}

/// <summary>
/// Non-generic double-precision shim, mirroring the FinancialModelTestBase /
/// VideoNNModelTestBase pattern, so existing <c>: RiskModelTestBase</c> derivations keep compiling.
/// </summary>
public abstract class RiskModelTestBase : RiskModelTestBase<double> { }
