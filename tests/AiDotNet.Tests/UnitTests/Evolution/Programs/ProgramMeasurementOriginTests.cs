using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramMeasurementOriginTests
{
    private static readonly ProgramGenome Genome = new("return 1;", ProgramLanguage.CSharp);
    private static readonly EvolutionEvaluationContext Context = new(0, 1, 2, 1);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Descriptor_merge_preserves_original_measurements_and_current_cost(bool reused)
    {
        var measurement = Result(1, reused ? 0 : 5, reused);
        var task = new ProgramEvolutionTask(Evaluator(measurement), new ProgramDescriptorSet(new ProgramLengthDescriptor()));
        var candidate = new EvolutionCandidate<ProgramGenome>(0, await task.CanonicalizeAsync(Genome),
            new EvolutionLineage(null, null, "seed", null, 0, 0, 1));
        var result = await task.EvaluateAsync(candidate, Context);
        Assert.Same(measurement.MeasurementOrigin, result.MeasurementOrigin);
        Assert.Equal(measurement.CostUnits, result.CostUnits);
        Assert.Equal(Genome.Source.Length, result.Descriptors["length"]);
        Assert.Equal(12, result.MeasurementOrigin!.OriginalCostUnits);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Correctness_gate_preserves_fitness_origin_and_charges_current_correctness(bool reused)
    {
        int correctnessCalls = 0;
        var checks = new DelegateProgramFitnessEvaluator((_, _, _) =>
        {
            correctnessCalls++;
            return new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 1, costUnits: 3));
        });
        var measurement = Result(0.75, reused ? 0 : 5, reused);
        var gate = new CorrectnessGatedProgramFitnessEvaluator(checks, Evaluator(measurement));
        var result = await gate.EvaluateAsync(Genome, Context);
        Assert.Equal(1, correctnessCalls);
        Assert.Equal(reused ? 3 : 8, result.CostUnits);
        Assert.Same(measurement.MeasurementOrigin, result.MeasurementOrigin);
        Assert.Equal(0.75, result.Quality);
    }

    [Fact]
    public async Task Rejected_correctness_preserves_its_origin_and_never_calls_fitness()
    {
        var check = Result(0.5, 3, false);
        var fitness = new DelegateProgramFitnessEvaluator((_, _, _) => throw new InvalidOperationException("Fitness must not run."));
        var result = await new CorrectnessGatedProgramFitnessEvaluator(Evaluator(check), fitness).EvaluateAsync(Genome, Context);
        Assert.Equal(EvolutionEvaluationStatus.Rejected, result.Status);
        Assert.Same(check.MeasurementOrigin, result.MeasurementOrigin); Assert.Equal(3, result.CostUnits);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Judge_does_not_apply_original_uncertainty_to_a_blended_score_or_spend_on_unsupported_origin(bool enabled)
    {
        var client = new FakeChatClient("{\"correctness\":1,\"efficiency\":1,\"readability\":1}");
        var measurement = Result(0.5, 5, false);
        var judge = new LlmJudgeProgramFitnessEvaluator<double>(client, Evaluator(measurement),
            options: new LlmFeedbackOptions { Enabled = enabled });
        var result = await judge.EvaluateAsync(Genome, Context);
        Assert.Equal(0, client.Calls); Assert.Same(measurement.MeasurementOrigin, result.MeasurementOrigin);
        Assert.Equal(5, result.CostUnits);
        if (enabled)
        {
            Assert.Equal(EvolutionEvaluationStatus.Failed, result.Status);
            Assert.Contains(result.Diagnostics, diagnostic => diagnostic.Code == "program_judge_measurement_origin_unsupported");
        }
        else Assert.Same(measurement, result);
    }

    private static IProgramFitnessEvaluator Evaluator(EvolutionTaskResult result) =>
        new DelegateProgramFitnessEvaluator((_, _, _) => new ValueTask<EvolutionTaskResult>(result));

    private static EvolutionTaskResult Result(double quality, double cost, bool reused)
    {
        var origin = new EvolutionMeasurementOrigin(EvolutionHash.Compute("fitness-scope"), "original-run", "original-evaluation",
            new[] { "original-sample" }, new DateTimeOffset(2026, 9, 11, 0, 0, 0, TimeSpan.Zero), 12, "fitness-units-v1", "stats-v1",
            standardError: 0.1);
        if (reused) origin = origin.AsReused(EvolutionMeasurementOriginKind.PersistentReuse);
        return new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, quality, costUnits: cost).WithMeasurementOrigin(origin);
    }
}
