using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.PaperFidelity;

public class OptionPerturbationTests
{
    [Theory]
    [InlineData("Temperature")]
    [InlineData("TextToSemanticTemperature")]
    [InlineData("SemanticToAcousticTemperature")]
    public void TemperaturesIncludeZeroAndExcludeNegativeValues(string option)
    {
        var candidates = OptionSensitivityHarness.Perturbations(typeof(double), 0.9, option).Cast<double>().ToArray();
        Assert.Contains(0, candidates);
        Assert.All(candidates, value => Assert.True(value >= 0));
        Assert.DoesNotContain(0.9, candidates);
    }

    [Theory]
    [InlineData("HiftUpsampleRates")]
    [InlineData("VocoderUpsampleRates")]
    public void EqualUpsamplingRatesCanChangeWhilePreservingTheWaveformHop(string option)
    {
        var original = new[] { 8, 8 };
        var candidates = OptionSensitivityHarness.Perturbations(typeof(int[]), original, option).Cast<int[]>().ToArray();
        Assert.Contains(candidates, rates => !rates.SequenceEqual(original)
            && rates.Length == original.Length && rates.All(rate => rate > 0)
            && rates.Aggregate(1, (product, rate) => product * rate) == 64);
        Assert.Equal(new[] { 8, 8 }, original);
    }

    [Theory]
    [InlineData(8)]
    [InlineData(16)]
    public void RotaryHeadWidthsPreferAnEvenStructuralChange(int current)
    {
        var candidates = OptionSensitivityHarness.Perturbations(typeof(int), current, "EmotionHeadDim").Cast<int>().ToArray();
        Assert.Equal(2 * current, candidates[0]);
        Assert.Contains(current + 1, candidates);
        Assert.DoesNotContain(current, candidates);
    }

    [Theory]
    [InlineData("Beta1", 0.9, false, false)]
    [InlineData("Beta2", 0.999, false, false)]
    [InlineData("TeacherMomentumStart", 0.5, false, true)]
    [InlineData("TeacherMomentumEnd", 1.0, false, true)]
    [InlineData("TimeMaskProbability", 0.025, false, true)]
    [InlineData("TopP", 0.95, true, true)]
    [InlineData("GuidanceRescale", 0.75, false, true)]
    public void ProbabilityAndMomentCandidatesStayInTheirMathematicalDomain(string option, double current, bool strictLower, bool inclusiveUpper)
    {
        var candidates = OptionSensitivityHarness.Perturbations(typeof(double), current, option).Cast<double>().ToArray();
        Assert.NotEmpty(candidates);
        Assert.All(candidates, value =>
        {
            Assert.True(strictLower ? value > 0 : value >= 0);
            Assert.True(inclusiveUpper ? value <= 1 : value < 1);
            Assert.NotEqual(current, value);
        });
    }

    [Theory]
    [InlineData(80, 128)]
    [InlineData(128, 80)]
    public void LogMelCandidatesIncludeTheOtherSupportedFrontendWidth(int current, int alternative)
    {
        var candidates = OptionSensitivityHarness.Perturbations(typeof(int), current, "NumMels").Cast<int>().ToArray();
        Assert.Contains(alternative, candidates);
        Assert.DoesNotContain(current, candidates);
    }

    [Theory]
    [InlineData("FilterThreshold", 0.98, 1.0, false)]
    [InlineData("RepetitionPenalty", 1.2, 2.0, true)]
    public void SamplerCandidatesStayWithinSupportedBounds(string option, double current, double upper, bool strictLower)
    {
        var candidates = OptionSensitivityHarness.Perturbations(typeof(double), current, option).Cast<double>().ToArray();
        Assert.NotEmpty(candidates);
        Assert.All(candidates, value =>
        {
            Assert.True(strictLower ? value > 0 : value >= 0);
            Assert.True(value < upper);
            Assert.NotEqual(current, value);
        });
    }

    [Fact]
    public void SampleRatesPreferSimpleRatiosWithoutDroppingFineGrainedCandidates()
    {
        var candidates = OptionSensitivityHarness.Perturbations(typeof(int), 22050, "SampleRate").Cast<int>().ToArray();
        Assert.Equal(44100, candidates[0]);
        Assert.Equal(11025, candidates[1]);
        Assert.Contains(22051, candidates);
        Assert.Equal(candidates.Length, candidates.Distinct().Count());
    }

    [Fact]
    public void CandidatesStartFromThePreparedControlValue()
    {
        var probe = new OptionSensitivityProbe<ThresholdModel>(
            () => new(new ThresholdOptions()), model => model.Output, train: null,
            prepareOptions: options => ((ThresholdOptions)options).Threshold = 0.5);
        var result = OptionSensitivityHarness.Measure(typeof(PreparedValueProbe), "Threshold", probe);
        Assert.Equal(OptionSensitivityOutcome.Sensitive, result.Outcome);
        Assert.Contains("Threshold 0.5 ->", result.Detail);
    }

    private sealed class PreparedValueProbe { }

    [Theory]
    [InlineData(3)]
    [InlineData(5)]
    public void IntegerCandidatesIncludeDifferentPositiveOddKernel(int kernel)
    {
        var candidates = OptionSensitivityHarness.Perturbations(typeof(int), kernel).Cast<int>().ToArray();
        Assert.Contains(candidates, value => value > 0 && value % 2 == 1 && value != kernel);
    }

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
        CloneRegistry.Register(new ClonePlan(
            typeof(RejectingOptionsModel), Array.Empty<ClonePlanEntry>(), new[] { "_options" }));
        CloneRegistry.Register(new ClonePlan(
            typeof(CheckpointShapeModel), Array.Empty<ClonePlanEntry>(), new[] { "_options" }));
        CloneRegistry.Register(new ClonePlan(typeof(OptionPreservingFallbackModel), Array.Empty<ClonePlanEntry>(),
            constructorCandidates: new IReadOnlyList<string>[] { new[] { "_items", "_options" }, new[] { "_options" } }));
        CloneRegistry.Register(new ClonePlan(
            typeof(ProductConstrainedArrayModel), Array.Empty<ClonePlanEntry>(), new[] { "_options" }));
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

    public sealed class ProductConstrainedArrayModel
    {
        private readonly FixedArrayOptions _options;
        public ProductConstrainedArrayModel(FixedArrayOptions options)
        {
            if (options.Widths.Length != 2 || options.Widths.Any(width => width <= 0)
                || (long)options.Widths[0] * options.Widths[1] != 8)
                throw new ArgumentException("The two stage rates must have product eight.", nameof(options));
            _options = options;
        }
        public long Output => _options.Widths[0];
        public int[] Shape => _options.Widths;
    }

    private sealed class ParameterShapeProbe { }
    private sealed class FixedParameterShapeProbe { }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void EqualParameterCountsStillDetectAChangedTensorShape(bool observeShapes)
    {
        var probe = new OptionSensitivityProbe<ProductConstrainedArrayModel>(
            () => new(new FixedArrayOptions()), _ => 0, train: null,
            prepareModel: model =>
            {
                if (!model.Shape.SequenceEqual(new[] { 2, 4 })) throw new ArgumentException("The checkpoint is 2 by 4.");
            },
            parameterCount: _ => 8,
            parameterShapes: model => new[] { observeShapes ? model.Shape : new[] { 2, 4 } });
        var result = OptionSensitivityHarness.Measure(
            observeShapes ? typeof(ParameterShapeProbe) : typeof(FixedParameterShapeProbe), "Widths", probe);
        Assert.Equal(observeShapes ? OptionSensitivityOutcome.Sensitive : OptionSensitivityOutcome.NoValidPerturbation,
            result.Outcome);
        if (observeShapes) Assert.Contains("ordered parameter tensor shapes", result.Detail);
    }

    [Fact]
    public void StageRatesCanChangeWithoutChangingTheirProduct()
    {
        var probe = new OptionSensitivityProbe<ProductConstrainedArrayModel>(
            () => new(new FixedArrayOptions()), model => model.Output, train: null);
        var result = OptionSensitivityHarness.Measure(typeof(ProductConstrainedArrayModel), "Widths", probe);
        Assert.Equal(OptionSensitivityOutcome.Sensitive, result.Outcome);
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
    private sealed class SpeechSpanProbe { }
    private sealed class UnexpectedOperationProbe { }
    private sealed class PreparedControlContextProbe { }

    [Fact]
    public void PreparedConfigurationContextsHaveIndependentCachedControls()
    {
        OptionSensitivityProbe<ThresholdModel> Probe(double threshold, string group) => new(
            () => new(new ThresholdOptions()), model => model.Output, train: null,
            prepareOptions: options => ((ThresholdOptions)options).Threshold = threshold,
            controlGroup: group);
        var first = OptionSensitivityHarness.Measure(typeof(PreparedControlContextProbe), "Threshold", Probe(1, "high"));
        var second = OptionSensitivityHarness.Measure(typeof(PreparedControlContextProbe), "Threshold", Probe(0.5, "low"));
        Assert.Equal(OptionSensitivityOutcome.Sensitive, first.Outcome);
        Assert.Equal(OptionSensitivityOutcome.Sensitive, second.Outcome);
        Assert.Contains("-> 0.5 changes", first.Detail);
        Assert.Contains("0.5 -> 1 changes", second.Detail);
    }

    [Fact]
    public void ChangedGenerationFormatIsAnObservableContractChange()
    {
        var probe = new OptionSensitivityProbe<ThresholdModel>(() => new(new ThresholdOptions()),
            model => model.Output == 1
                ? throw new AiDotNet.TextToSpeech.Latest.KaniSpeechSpanException("No complete speech frame.") : 0,
            train: null);
        var result = OptionSensitivityHarness.Measure(typeof(SpeechSpanProbe), "Threshold", probe);
        Assert.Equal(OptionSensitivityOutcome.Sensitive, result.Outcome);
        Assert.Contains("generated speech contract", result.Detail);
    }

    [Fact]
    public void UnexpectedOperationFailuresStillPropagate()
    {
        var probe = new OptionSensitivityProbe<ThresholdModel>(() => new(new ThresholdOptions()),
            model => model.Output == 1 ? throw new InvalidOperationException("Unexpected model failure.") : 0,
            train: null);
        Assert.Throws<InvalidOperationException>(() =>
            OptionSensitivityHarness.Measure(typeof(UnexpectedOperationProbe), "Threshold", probe));
    }

    public sealed class DepthOptions : ModelOptions
    {
        public int Depth { get; set; } = 5;
    }

    public sealed class CheckpointShapeModel : IDisposable
    {
        private readonly DepthOptions _options;
        public CheckpointShapeModel(DepthOptions options) => _options = options;
        public int ParameterCount => _options.Depth;
        public bool Disposed { get; private set; }
        public void Dispose() => Disposed = true;
    }

    private sealed class CheckpointShapeProbe { }
    private sealed class FixedCheckpointShapeProbe { }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void CheckpointLoadingDoesNotHideAParameterCountChange(bool observeCount)
    {
        var rebuilt = new List<CheckpointShapeModel>();
        var probe = new OptionSensitivityProbe<CheckpointShapeModel>(
            () => new(new DepthOptions()), _ => 0, train: null,
            prepareModel: model =>
            {
                rebuilt.Add(model);
                if (model.ParameterCount != 5) throw new ArgumentException("The fixture checkpoint has five parameters.");
            },
            parameterCount: model => observeCount ? model.ParameterCount : 5);
        var result = OptionSensitivityHarness.Measure(
            observeCount ? typeof(CheckpointShapeProbe) : typeof(FixedCheckpointShapeProbe), "Depth", probe);
        Assert.Equal(observeCount ? OptionSensitivityOutcome.Sensitive : OptionSensitivityOutcome.NoValidPerturbation,
            result.Outcome);
        Assert.All(rebuilt, model => Assert.True(model.Disposed));
    }

    public sealed class RejectingOptionsModel
    {
        private readonly DepthOptions _options;
        public RejectingOptionsModel() : this(new DepthOptions()) { }
        public RejectingOptionsModel(DepthOptions options)
        {
            if (options.Depth > 6) throw new ArgumentOutOfRangeException(nameof(options));
            _options = options;
        }
    }

    public sealed class OptionPreservingFallbackModel
    {
        private readonly int[] _items = Array.Empty<int>();
        private readonly DepthOptions _options;
        public OptionPreservingFallbackModel() : this(new DepthOptions()) { }
        public OptionPreservingFallbackModel(DepthOptions options) => _options = options;
        public OptionPreservingFallbackModel(int[] items, DepthOptions options)
        {
            if (items.Length == 0) throw new ArgumentException("At least one item is required.", nameof(items));
            _items = items;
            _options = options;
        }
        public int Depth => _options.Depth;
    }

    [Fact]
    public void RejectedAuxiliaryStateCanUseANarrowerConstructorThatPreservesOptions()
    {
        var model = new OptionPreservingFallbackModel(new DepthOptions { Depth = 9 });
        var copy = Assert.IsType<OptionPreservingFallbackModel>(CloneEngine.CopyConfiguration(model));
        Assert.Equal(9, copy.Depth);
    }

    [Fact]
    public void RejectedOptionsDoNotFallBackToTheParameterlessConstructor()
    {
        var options = new DepthOptions();
        var model = new RejectingOptionsModel(options);
        options.Depth = 7;
        var error = Assert.Throws<System.Reflection.TargetInvocationException>(() => CloneEngine.CopyConfiguration(model));
        Assert.IsType<ArgumentOutOfRangeException>(error.InnerException);
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
