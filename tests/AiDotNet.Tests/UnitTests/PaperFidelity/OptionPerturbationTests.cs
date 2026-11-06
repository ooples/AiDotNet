using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

public class OptionPerturbationTests
{
    static OptionPerturbationTests()
    {
        CloneRegistry.Register(new ClonePlan(
            typeof(ThresholdModel), Array.Empty<ClonePlanEntry>(), new[] { "_options" }));
        CloneRegistry.Register(new ClonePlan(
            typeof(FixedArrayModel), Array.Empty<ClonePlanEntry>(), new[] { "_options" }));
        CloneRegistry.Register(new ClonePlan(
            typeof(ResettingModel), Array.Empty<ClonePlanEntry>(), new[] { "_options" }));
        CloneRegistry.Register(new ClonePlan(
            typeof(ClampingModel), Array.Empty<ClonePlanEntry>(), new[] { "_options" }));
        CloneRegistry.Register(new ClonePlan(
            typeof(OverwritingModel), Array.Empty<ClonePlanEntry>(), new[] { "_options" }));
        CloneRegistry.Register(new ClonePlan(
            typeof(LateRandomModel), Array.Empty<ClonePlanEntry>(), new[] { "_options" }));
        CloneRegistry.Register(new ClonePlan(
            typeof(DeepFixtureModel), Array.Empty<ClonePlanEntry>(), new[] { "_options" }));
    }

    public sealed class FixedArrayOptions : ModelOptions
    {
        public int[] Widths { get; set; } = new[] { 2, 4 };
    }

    public sealed class FixedArrayModel
    {
        private readonly FixedArrayOptions _options;
        public FixedArrayModel(FixedArrayOptions options)
        {
            if (options.Widths.Length != 2 || options.Widths.Any(width => width <= 0))
                throw new ArgumentException("Exactly two positive stage widths are required.");
            _options = options;
        }
        public long Output => _options.Widths.Sum();
    }

    [Fact]
    public void FixedLengthArrayCanChangeWithoutChangingItsLength()
    {
        var probe = new OptionSensitivityProbe<FixedArrayModel>(
            () => new(new()), model => model.Output, train: null);
        var result = OptionSensitivityHarness.Measure(typeof(FixedArrayModel), "Widths", probe);
        Assert.Equal(OptionSensitivityOutcome.Sensitive, result.Outcome);
    }

    [Fact]
    public void ArrayCandidatesChangeEachElementWithoutMutatingTheOriginal()
    {
        var original = new[] { 2, 4 };
        var candidates = OptionSensitivityHarness.Perturbations(typeof(int[]), original).Cast<int[]>().ToArray();
        Assert.Equal(new[] { 2, 4 }, original);
        Assert.Contains(candidates, values => values.Length == 2 && values[0] != 2 && values[1] == 4);
        Assert.Contains(candidates, values => values.Length == 2 && values[0] == 2 && values[1] != 4);
    }

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

    public sealed class DepthOptions : ModelOptions
    {
        public int Depth { get; set; } = 5;
    }

    public sealed class DeepFixtureModel
    {
        private readonly DepthOptions _options;
        public DeepFixtureModel(DepthOptions options)
        {
            if (options.Depth > 6) throw new OutOfMemoryException("Doubling a deep fixture exceeds its allocation budget.");
            _options = options;
        }
        public long Output => _options.Depth;
    }

    [Fact]
    public void StructuralOptionCanBeMeasuredBeforeAnExponentialExpansion()
    {
        var probe = new OptionSensitivityProbe<DeepFixtureModel>(() => new(new()), model => model.Output, train: null);
        var result = OptionSensitivityHarness.Measure(typeof(DeepFixtureModel), "Depth", probe);
        Assert.Equal(OptionSensitivityOutcome.Sensitive, result.Outcome);
    }

    public sealed class LateRandomModel
    {
        private readonly ThresholdOptions _options;
        private Random? _random;
        public LateRandomModel(ThresholdOptions options) => _options = options;
        public long Output { get; set; }
        public long Read() => (_random ??= AiDotNet.NeuralNetworks.Layers.LayerInitializationSeedScope
            .CreateComponentRandom(null)).Next() + (long)(_options.Threshold * 100);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RandomnessFirstUsedAfterConstructionStillGetsTheFixtureSeed(bool duringTraining)
    {
        var probe = new OptionSensitivityProbe<LateRandomModel>(
            () => new(new()), model => duringTraining ? model.Output : model.Read(),
            duringTraining ? model => model.Output = model.Read() : null);
        var result = OptionSensitivityHarness.Measure(typeof(LateRandomModel), "Threshold", probe);
        Assert.Equal(OptionSensitivityOutcome.Sensitive, result.Outcome);
    }

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

    public sealed class LevelOptions : ModelOptions
    {
        public double Level { get; set; } = 0.95;
    }

    /// <summary>Resets an out-of-range level to its default, as many models do with a probability.</summary>
    public sealed class ResettingModel
    {
        private readonly LevelOptions _options;
        public ResettingModel(LevelOptions options)
        {
            if (options.Level >= 1.0) options.Level = 0.95;
            _options = options;
        }
        public long Output => _options.Level < 0.9 ? 1 : 0;
    }

    public sealed class ClampedOptions : ModelOptions
    {
        private double _level = 0.75;
        public double Level { get => _level; set => _level = Math.Min(1.0, Math.Max(0.0, value)); }
    }

    public sealed class ClampingModel
    {
        private readonly ClampedOptions _options;
        public ClampingModel(ClampedOptions options) => _options = options;
        public long Output => _options.Level > 0.9 ? 1 : 0;
    }

    /// <summary>Ignores the configured level entirely by overwriting it.</summary>
    public sealed class OverwritingModel
    {
        private readonly LevelOptions _options;
        public OverwritingModel(LevelOptions options)
        {
            options.Level = 0.95;
            _options = options;
        }
        public long Output => _options.Level < 0.9 ? 1 : 0;
    }

    private sealed class ResettingProbe { }
    private sealed class ClampingProbe { }
    private sealed class OverwritingProbe { }

    /// <summary>
    /// The first candidate (0.95 * 2 = 1.9) is out of range and the model resets it to 0.95; the next one (0.475) is
    /// held and changes the output. Fails if the first unheld candidate is reported as OptionsNotRouted.
    /// </summary>
    [Fact]
    public void ModelResettingAnInvalidCandidateIsStillRouted()
    {
        var probe = new OptionSensitivityProbe<ResettingModel>(
            () => new(new()), model => model.Output, train: null);
        var result = OptionSensitivityHarness.Measure(typeof(ResettingProbe), "Level", probe);
        Assert.Equal(OptionSensitivityOutcome.Sensitive, result.Outcome);
    }

    /// <summary>
    /// The options setter clamps 0.75 * 2 = 1.5 to 1.0, which the model holds and acts on. Fails if the held value is
    /// compared with the raw candidate instead of what the options read back.
    /// </summary>
    [Fact]
    public void ClampingSetterIsComparedWithItsReadBackValue()
    {
        var probe = new OptionSensitivityProbe<ClampingModel>(
            () => new(new()), model => model.Output, train: null);
        var result = OptionSensitivityHarness.Measure(typeof(ClampingProbe), "Level", probe);
        Assert.Equal(OptionSensitivityOutcome.Sensitive, result.Outcome);
    }

    /// <summary>A model that overwrites the option for every candidate is still reported as not routed.</summary>
    [Fact]
    public void ModelOverwritingEveryCandidateIsNotRouted()
    {
        var probe = new OptionSensitivityProbe<OverwritingModel>(
            () => new(new()), model => model.Output, train: null);
        var result = OptionSensitivityHarness.Measure(typeof(OverwritingProbe), "Level", probe);
        Assert.Equal(OptionSensitivityOutcome.OptionsNotRouted, result.Outcome);
    }

    /// <summary>An unset (null) list option still gets a candidate: a one-element list.</summary>
    [Fact]
    public void NullListHasAOneElementCandidate()
    {
        var candidates = OptionSensitivityHarness.Perturbations(typeof(int[]), null).ToArray();
        Assert.Contains(candidates, value => value is int[] items && items.Length == 1);
    }

    public enum ThreeWay { A, B, C }

    [Fact]
    public void SmallEnumIsCoveredCompletely()
    {
        Assert.Equal(new object?[] { ThreeWay.C, ThreeWay.A }, OptionSensitivityHarness.Perturbations(typeof(ThreeWay), ThreeWay.B).ToArray());
        Assert.Equal(new object?[] { ThreeWay.A, ThreeWay.B, ThreeWay.C }, OptionSensitivityHarness.Perturbations(typeof(ThreeWay?), null).ToArray());
    }

    /// <summary>
    /// Every enum candidate is a rebuild plus training, and an option that changes nothing tries them all:
    /// TimeSeriesModelType's ~70 values ran Timer's ModelType check past its 300 s budget.
    /// </summary>
    [Fact]
    public void LargeEnumIsSampledAcrossItsRange()
    {
        var values = Enum.GetValues(typeof(AiDotNet.Enums.TimeSeriesModelType)).Cast<object>().ToList();
        Assert.True(values.Count > OptionSensitivityHarness.MaxEnumCandidates);
        var current = values[3];

        var candidates = OptionSensitivityHarness.Perturbations(typeof(AiDotNet.Enums.TimeSeriesModelType), current).ToList();

        Assert.Equal(OptionSensitivityHarness.MaxEnumCandidates, candidates.Count);
        Assert.Equal(candidates.Count, candidates.Distinct().Count());
        Assert.DoesNotContain(current, candidates);
        Assert.Equal(values[4], candidates[0]);
        // Spread over the range rather than the next few neighbours.
        var positions = candidates.Select(candidate => (values.IndexOf(candidate ?? current) - 3 + values.Count) % values.Count).ToList();
        Assert.True(positions.Max() - positions.Min() >= values.Count / 2, $"Candidates sit at offsets {string.Join(", ", positions)}.");
    }
}
