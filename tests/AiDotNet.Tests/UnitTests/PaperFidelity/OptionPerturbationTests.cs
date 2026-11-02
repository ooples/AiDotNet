using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

public class OptionPerturbationTests
{
    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    public void FloatingPointBoundaryHasAnInteriorProbabilityCandidate(double boundary)
    {
        foreach (var type in new[] { typeof(double), typeof(float), typeof(decimal), typeof(double?) })
        {
            var candidates = OptionSensitivityHarness.Perturbations(type, boundary);
            Assert.Contains(candidates, value => Convert.ToDouble(value) > 0 && Convert.ToDouble(value) < 1);
        }
    }
}
