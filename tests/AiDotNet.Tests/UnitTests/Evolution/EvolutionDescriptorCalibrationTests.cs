using AiDotNet;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution;

/// <summary>
/// Covers deriving a whole archive grid from what a seed population measured. Bounds are the one piece of setup that
/// cannot be guessed from the code, and both ways of getting them wrong fail silently: too wide and every candidate
/// shares one cell, too narrow and nothing ever competes.
/// </summary>
public sealed class EvolutionDescriptorCalibrationTests
{
    [Fact]
    public void AnAxisCoversTheObservedSpanPlusItsMargin()
    {
        EvolutionDescriptorDefinition axis = Assert.Single(
            EvolutionDescriptorCalibration.FromObservations(
                Observations(("x", 2.0), ("x", 6.0)),
                options: new EvolutionDescriptorCalibrationOptions { Padding = 0.25, BinCount = 8 }));

        // Span 4, so a quarter is 1 at each end.
        Assert.Equal("x", axis.Name);
        Assert.Equal(1.0, axis.Minimum, 10);
        Assert.Equal(7.0, axis.Maximum, 10);
        Assert.Equal(8, axis.BinCount);
        Assert.Equal(EvolutionOutOfRangePolicy.Grow, axis.OutOfRangePolicy);
    }

    [Fact]
    public void TheSeededExtremesLandInsideTheGridRatherThanOnItsEdge()
    {
        // The point of the margin: the best and worst seeds have somewhere to be improved on before the grid has to
        // grow, and neither sits in a bin that a rounding step could push out of.
        EvolutionDescriptorDefinition axis = Assert.Single(
            EvolutionDescriptorCalibration.FromObservations(Observations(("x", 2.0), ("x", 6.0))));

        Assert.True(axis.TryGetBin(2.0, out int low));
        Assert.True(axis.TryGetBin(6.0, out int high));
        Assert.True(low > 0);
        Assert.True(high < axis.BinCount - 1);
    }

    [Fact]
    public void TheSameObservationsInAnyOrderGiveTheSameGrid()
    {
        // The grid enters the archive's definition hash, so an order-dependent answer would make two runs from the
        // same seeds refuse each other's checkpoints.
        IReadOnlyList<EvolutionDescriptorDefinition> forwards = EvolutionDescriptorCalibration.FromObservations(
            Observations(("x", 2.0), ("x", 6.0), ("x", 4.0)));
        IReadOnlyList<EvolutionDescriptorDefinition> backwards = EvolutionDescriptorCalibration.FromObservations(
            Observations(("x", 4.0), ("x", 6.0), ("x", 2.0)));

        Assert.Equal(forwards[0].ToCanonicalString(), backwards[0].ToCanonicalString());
    }

    [Fact]
    public void EveryReportedDescriptorBecomesAnAxisInAStableOrder()
    {
        var observations = new List<IReadOnlyDictionary<string, double>>
        {
            Values(("recall", 0.2), ("accuracy", 0.4)),
            Values(("latency", 30.0), ("accuracy", 0.8))
        };

        IReadOnlyList<EvolutionDescriptorDefinition> axes =
            EvolutionDescriptorCalibration.FromObservations(observations);

        Assert.Equal(new[] { "accuracy", "latency", "recall" }, axes.Select(axis => axis.Name));
    }

    [Fact]
    public void NamingTheAxesFixesTheirOrderAndLeavesTheRestOut()
    {
        var observations = new List<IReadOnlyDictionary<string, double>>
        {
            Values(("accuracy", 0.2), ("latency", 30.0), ("ignored", 1.0))
        };

        IReadOnlyList<EvolutionDescriptorDefinition> axes = EvolutionDescriptorCalibration.FromObservations(
            observations, new[] { "latency", "accuracy" });

        Assert.Equal(new[] { "latency", "accuracy" }, axes.Select(axis => axis.Name));
    }

    [Fact]
    public void ADescriptorEverySeedAgreedOnGetsAUsableWindowRatherThanAHairlineOne()
    {
        // Freezing a single-valued axis on its own nudges the bounds apart by one representable double. A bin that
        // narrow needs astronomically many growth steps to reach the first value that differs, so the axis is dead
        // in practice. Calibration gives it a real window to grow from.
        EvolutionDescriptorDefinition axis = Assert.Single(
            EvolutionDescriptorCalibration.FromObservations(
                Observations(("flag", 0.0), ("flag", 0.0)),
                options: new EvolutionDescriptorCalibrationOptions { DegenerateSpan = 2.0 }));

        Assert.Equal(-1.0, axis.Minimum, 10);
        Assert.Equal(1.0, axis.Maximum, 10);
        Assert.True(axis.BinWidth > 0.01);
    }

    [Fact]
    public void AValueOutsideTheSeededSpanWidensTheGridRatherThanBeingDiscarded()
    {
        // Calibrating from seeds fixes a sensible bin width; the growth policy is what makes a narrow seed
        // population cost a little widening instead of a wrong grid.
        EvolutionDescriptorDefinition axis = Assert.Single(
            EvolutionDescriptorCalibration.FromObservations(Observations(("x", 0.0), ("x", 10.0))));

        Assert.False(axis.TryGetBin(500.0, out _));

        EvolutionDescriptorDefinition widened = axis.Widen(500.0);
        Assert.True(widened.TryGetBin(500.0, out _));
        Assert.Equal(axis.BinWidth, widened.BinWidth, 10);
    }

    [Fact]
    public void AMeasurementThatFailedDoesNotDecideAnAxis()
    {
        // A seed that could not be measured contributes nothing rather than dragging a bound to zero.
        var observations = new List<IReadOnlyDictionary<string, double>>
        {
            Values(("x", 5.0)),
            Values(("x", double.NaN)),
            Values(("x", 7.0))
        };

        EvolutionDescriptorDefinition axis = Assert.Single(
            EvolutionDescriptorCalibration.FromObservations(observations,
                options: new EvolutionDescriptorCalibrationOptions { Padding = 0 }));

        Assert.Equal(5.0, axis.Minimum, 10);
        Assert.Equal(7.0, axis.Maximum, 10);
    }

    [Fact]
    public void ADescriptorNobodyReportedIsAnErrorRatherThanASkippedAxis()
    {
        // Quietly dropping an axis the caller asked for would build a different archive than the one requested, and
        // the run would look like it worked.
        ArgumentException failure = Assert.Throws<ArgumentException>(() =>
            EvolutionDescriptorCalibration.FromObservations(Observations(("x", 1.0)), new[] { "missing" }));

        Assert.Contains("missing", failure.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NothingMeasurableAtAllIsAnErrorThatSaysWhatToDo()
    {
        ArgumentException failure = Assert.Throws<ArgumentException>(() =>
            EvolutionDescriptorCalibration.FromObservations(
                new List<IReadOnlyDictionary<string, double>> { Values() }));

        Assert.Contains("nothing to calibrate", failure.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void BlankOrRepeatedAxisNamesAreRefused()
    {
        Assert.Throws<ArgumentException>(() =>
            EvolutionDescriptorCalibration.FromObservations(Observations(("x", 1.0)), new[] { " " }));
        Assert.Throws<ArgumentException>(() =>
            EvolutionDescriptorCalibration.FromObservations(Observations(("x", 1.0)), new[] { "x", "x" }));
        Assert.Throws<ArgumentException>(() =>
            EvolutionDescriptorCalibration.FromObservations(Observations(("x", 1.0)), Array.Empty<string>()));
    }

    [Fact]
    public void AnInvalidCalibrationSettingIsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new EvolutionDescriptorCalibrationOptions { BinCount = 0 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new EvolutionDescriptorCalibrationOptions { Padding = -1 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new EvolutionDescriptorCalibrationOptions { DegenerateSpan = 0 }.Validate());
    }

    [Fact]
    public async Task CalibratingFromATaskMeasuresEverySeedOnce()
    {
        var task = new CountingCalibrationTask();
        TestGenome[] seeds = { new(10), new(30), new(20) };

        IReadOnlyList<EvolutionDescriptorDefinition> axes = await EvolutionDescriptorCalibration.CalibrateAsync(
            task, seeds, options: new EvolutionDescriptorCalibrationOptions { Padding = 0 });

        Assert.Equal(3, task.Evaluations);
        EvolutionDescriptorDefinition axis = Assert.Single(axes);
        Assert.Equal(10, axis.Minimum, 10);
        Assert.Equal(30, axis.Maximum, 10);
    }

    [Fact]
    public async Task ASeedThatFailedToEvaluateIsSkippedRatherThanFailingTheCalibration()
    {
        // A seed population where some members do not run is ordinary; the survivors still describe the space.
        var task = new CountingCalibrationTask(failOnValue: 30);
        TestGenome[] seeds = { new(10), new(30), new(20) };

        IReadOnlyList<EvolutionDescriptorDefinition> axes = await EvolutionDescriptorCalibration.CalibrateAsync(
            task, seeds, options: new EvolutionDescriptorCalibrationOptions { Padding = 0 });

        EvolutionDescriptorDefinition axis = Assert.Single(axes);
        Assert.Equal(10, axis.Minimum, 10);
        Assert.Equal(20, axis.Maximum, 10);
    }

    [Fact]
    public async Task CalibratingWithNoSeedsIsRefused()
    {
        await Assert.ThrowsAsync<ArgumentException>(() => EvolutionDescriptorCalibration.CalibrateAsync(
            new CountingCalibrationTask(), Array.Empty<TestGenome>()));
    }

    [Fact]
    public void AConfiguredProgramDescriptorGetsItsOwnRangeRatherThanTheLengthRange()
    {
        // Regression: every configured descriptor was mapped onto the program-length axis, so a descriptor
        // reporting a ratio between zero and one was given a range spanning thousands of characters and every
        // candidate landed in the first bin. The archive then kept one overall winner and the search lost the
        // diversity pressure it exists for, with no error anywhere.
        var options = new ProgramEvolutionOptions { Language = ProgramLanguage.Python };
        options.SeedPrograms.Add("def solve(x):\n    return x\n");
        options.SeedPrograms.Add("def solve(x):\n    return x + 1\n");
        options.Descriptors.Add(new RatioDescriptor(0.2));

        EvolutionDescriptorDefinition axis = Assert.Single(
            AiModelBuilder<double, Matrix<double>, Vector<double>>.CalibrateProgramDescriptors(options));

        Assert.Equal("ratio", axis.Name);
        Assert.True(axis.Maximum < 1.0, "the axis should cover the descriptor's own scale, not the program length");
        Assert.True(axis.TryGetBin(0.2, out _));
    }

    [Fact]
    public void EveryConfiguredProgramDescriptorBecomesAnAxis()
    {
        // Only the first was mapped before, so a caller who named three behaviours got a one-dimensional archive.
        var options = new ProgramEvolutionOptions { Language = ProgramLanguage.Python };
        options.SeedPrograms.Add("def solve(x):\n    return x\n");
        options.Descriptors.Add(new RatioDescriptor(0.2, "first"));
        options.Descriptors.Add(new RatioDescriptor(0.6, "second"));

        IReadOnlyList<EvolutionDescriptorDefinition> axes =
            AiModelBuilder<double, Matrix<double>, Vector<double>>.CalibrateProgramDescriptors(options);

        Assert.Equal(new[] { "first", "second" }, axes.Select(axis => axis.Name));
    }

    private static List<IReadOnlyDictionary<string, double>> Observations(params (string Name, double Value)[] values)
    {
        var observations = new List<IReadOnlyDictionary<string, double>>(values.Length);
        foreach ((string name, double value) in values) observations.Add(Values((name, value)));
        return observations;
    }

    private static Dictionary<string, double> Values(params (string Name, double Value)[] values)
    {
        var observation = new Dictionary<string, double>(StringComparer.Ordinal);
        foreach ((string name, double value) in values) observation[name] = value;
        return observation;
    }

    /// <summary>A descriptor reporting a fixed value on a scale nothing like program length.</summary>
    private sealed class RatioDescriptor : IProgramDescriptor
    {
        private readonly double _value;

        public RatioDescriptor(double value, string name = "ratio")
        {
            _value = value;
            Name = name;
        }

        public string Name { get; }

        public double Compute(ProgramGenome genome) => _value + (genome.Source.Length % 3) * 0.01;
    }

    /// <summary>A task that reports the genome's own value as its descriptor and counts evaluations.</summary>
    private sealed class CountingCalibrationTask : IEvolutionTask<TestGenome>
    {
        private readonly int? _failOnValue;
        private int _evaluations;

        public CountingCalibrationTask(int? failOnValue = null) => _failOnValue = failOnValue;

        public int Evaluations => _evaluations;

        public string Id => "calibration-task";

        public string VersionHash => "calibration-task-v1";

        public string EvaluatorVersionHash => "calibration-evaluator-v1";

        public ValueTask<EvolutionCanonicalGenome<TestGenome>> CanonicalizeAsync(TestGenome genome,
            CancellationToken cancellationToken = default) =>
            new(new EvolutionCanonicalGenome<TestGenome>(genome,
                genome.Value.ToString(System.Globalization.CultureInfo.InvariantCulture)));

        public ValueTask<EvolutionTaskResult> EvaluateAsync(EvolutionCandidate<TestGenome> candidate,
            EvolutionEvaluationContext context, CancellationToken cancellationToken = default)
        {
            _evaluations++;
            int value = candidate.CanonicalGenome.Genome.Value;
            if (_failOnValue == value)
                return new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Failed("nope", "synthetic failure"));

            return new ValueTask<EvolutionTaskResult>(EvolutionTaskResult.Completed(
                value, new Dictionary<string, double>(StringComparer.Ordinal) { ["x"] = value }));
        }
    }
}
