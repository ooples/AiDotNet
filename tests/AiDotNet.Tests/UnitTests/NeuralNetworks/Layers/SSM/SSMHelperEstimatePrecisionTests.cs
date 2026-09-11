using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers.SSM;
using Moq;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Regression test for the int32 overflow in <c>EstimateMemorySavings</c>: the reduced-precision
/// byte estimate computed <c>paramCount * targetBitWidth</c> in int arithmetic, which overflows
/// once paramCount * bits exceeds int.MaxValue (e.g. 300M parameters at 8 bits).
/// </summary>
public class SSMHelperEstimatePrecisionTests
{
    [Fact(Timeout = 120000)]
    public async Task EstimateMemorySavings_LargeLayer_DoesNotOverflow()
    {
        const long paramCount = 300_000_000;
        var layer = new Mock<ILayer<float>>();
        layer.Setup(l => l.ParameterCount).Returns(paramCount);

        var (originalBytes, reducedBytes, ratio) =
            SSMQuantizationHelper<float>.EstimateMemorySavings(layer.Object, 8);

        // 300M params * 8 bits / 8 = 300,000,000 bytes, plus (300M / 128) groups * 8 bytes = 18,750,000.
        Assert.Equal(paramCount * 4, originalBytes);
        Assert.Equal(318_750_000L, reducedBytes);
        Assert.True(ratio > 3.0 && ratio < 4.0, $"Ratio {ratio} should be just under 4x for 8-bit.");
    }
}
