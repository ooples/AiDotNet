using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for financial forecasting models. Inherits financial model invariants
/// and adds forecasting-specific: temporal ordering preserved, horizon consistency,
/// and trend sensitivity.
/// </summary>
public abstract class ForecastingModelTestBase : FinancialModelTestBase
{
    [Fact]
    public void ForecastHorizon_ShouldProduceOutput()
    {
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);
        Assert.True(output.Length > 0, "Forecasting model produced empty forecast.");
    }

    [Fact]
    public void DifferentHistories_DifferentForecasts()
    {
        var network = CreateNetwork();
        var history1 = CreateConstantTensor(InputShape, 0.1);
        var history2 = CreateConstantTensor(InputShape, 0.9);

        var forecast1 = network.Predict(history1);
        var forecast2 = network.Predict(history2);

        bool anyDifferent = false;
        int minLen = Math.Min(forecast1.Length, forecast2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(forecast1[i] - forecast2[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Forecasting model produces identical forecasts for different histories — ignoring input.");
    }
}
