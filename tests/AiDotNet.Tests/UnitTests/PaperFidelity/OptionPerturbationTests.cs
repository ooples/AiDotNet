using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

public class OptionPerturbationTests
{
    static OptionPerturbationTests() => CloneRegistry.Register(new ClonePlan(
        typeof(ThresholdModel), Array.Empty<ClonePlanEntry>(), new[] { "_options" }));

    public sealed class ThresholdOptions : ModelOptions
    {
        public double Threshold { get; set; } = 1;
    }

    public sealed class ThresholdModel
    {
        private readonly ThresholdOptions _options;
        public ThresholdModel(ThresholdOptions options) => _options = options;
        public bool Trained { get; set; }
        public long Output => _options.Threshold < .75 ? 1 : 0;
    }

    private sealed class InferenceProbe { }
    private sealed class TrainingProbe { }
    private sealed class IgnoredProbe { }

    [Fact]
    public void LaterValidCandidateCanCrossAnInferenceThreshold()
    {
        var probe = new OptionSensitivityProbe<ThresholdModel>(
            () => new(new()), model => model.Output, train: null);
        var result = OptionSensitivityHarness.Measure(typeof(InferenceProbe), "Threshold", probe);
        Assert.Equal(OptionSensitivityOutcome.Sensitive, result.Outcome);
    }

    [Fact]
    public void LaterValidCandidateCanCrossATrainingThreshold()
    {
        var probe = new OptionSensitivityProbe<ThresholdModel>(
            () => new(new()), model => model.Trained ? model.Output : 0,
            model => model.Trained = true);
        var result = OptionSensitivityHarness.Measure(typeof(TrainingProbe), "Threshold", probe);
        Assert.Equal(OptionSensitivityOutcome.Sensitive, result.Outcome);
    }

    [Fact]
    public void IgnoredOptionStillFailsAfterAllCandidates()
    {
        var probe = new OptionSensitivityProbe<ThresholdModel>(
            () => new(new()), _ => 0, train: null);
        var result = OptionSensitivityHarness.Measure(typeof(IgnoredProbe), "Threshold", probe);
        Assert.Equal(OptionSensitivityOutcome.Insensitive, result.Outcome);
    }

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
