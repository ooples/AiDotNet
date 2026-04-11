using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for portfolio optimization models. Inherits financial model invariants
/// and adds portfolio-specific: finite allocations and non-empty output.
/// </summary>
public abstract class PortfolioOptimizerTestBase : FinancialModelTestBase
{
    [Fact(Timeout = 60000)]
    public async Task Allocations_ShouldBeFinite()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), $"Allocation[{i}] is NaN.");
            Assert.True(Math.Abs(output[i]) < 1e6,
                $"Allocation[{i}] = {output[i]:E4} is unreasonable.");
        }
    }

    [Fact(Timeout = 60000)]
    public async Task Portfolio_ShouldBeNonEmpty()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var output = network.Predict(input);
        Assert.True(output.Length > 0, "Portfolio optimizer produced empty allocation.");
    }
}
