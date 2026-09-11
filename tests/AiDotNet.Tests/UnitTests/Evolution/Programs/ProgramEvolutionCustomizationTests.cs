using AiDotNet;
using AiDotNet.Configuration;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramEvolutionCustomizationTests
{
    private const string Unit = "synthetic-test-work-v1";

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public async Task Missing_evaluator_receipts_cannot_be_converted_into_known_zero_cost(int missingStage)
    {
        var ledger = new EvolutionResourceLedger("missing-receipt", EvolutionResources.Of("cost_units", 10));
        var accounting = new ProgramEvolutionResourceOptions(ledger, 7, Unit);
        IProgramFitnessEvaluator absent = new NullResultProgramFitnessEvaluator();
        IProgramFitnessEvaluator good = new DelegateProgramFitnessEvaluator((_, _, _) => new ValueTask<EvolutionTaskResult>(
            new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 1, costUnits: 0.5)));
        IProgramFitnessEvaluator evaluator = missingStage == 0 ? absent : new CorrectnessGatedProgramFitnessEvaluator(
            missingStage == 1 ? absent : good, missingStage == 2 ? absent : good);
        var task = new ProgramEvolutionTask(evaluator, options: new ProgramEvolutionOptions { ResourceAccounting = accounting });
        var metered = new ResourceMeteredEvolutionTask<ProgramGenome>(task, ledger, new[] { 7m });
        var genome = new ProgramGenome("return 1", ProgramLanguage.Python);
        var candidate = new EvolutionCandidate<ProgramGenome>(0, new EvolutionCanonicalGenome<ProgramGenome>(genome, genome.Id),
            new EvolutionLineage(null, null, "seed", null, 0, 0, 0UL));
        var result = await metered.EvaluateAsync(candidate, new EvolutionEvaluationContext(0, 1, 1, 1));
        Assert.Equal(EvolutionEvaluationStatus.Failed, result.Status);
        Assert.Equal(1, ledger.Snapshot().Unknown);
        Assert.Equal(7m, ledger.Snapshot().Spent["cost_units"]);
        Assert.Equal(7, result.CostUnits);
    }

    [Fact]
    public void Malformed_cost_unit_identity_cannot_alias_another_unit_through_replacement_encoding()
    {
        var ledger = new EvolutionResourceLedger("unicode-unit", EvolutionResources.Of("cost_units", 10));
        Assert.ThrowsAny<ArgumentException>(() => new ProgramEvolutionResourceOptions(ledger, 1, "unit" + '\ud800'));
    }

    private static ProgramEvolutionOptions Options(IProgramVariationOperator variation, ProgramEvolutionResourceOptions? accounting = null)
    {
        var options = new ProgramEvolutionOptions { Language = ProgramLanguage.Python, CustomVariation = variation, ResourceAccounting = accounting };
        options.SeedPrograms.Add("return 1");
        options.TestCases.Add(new ProgramInputOutputExample { Input = "", ExpectedOutput = "2" });
        options.Engine.MaxEvaluationAttempts = 2;
        options.Engine.MaxProposals = 2; // The engine counts the initial seed as a proposal.
        options.Engine.MaxGenerations = 1;
        options.Engine.ProposalBatchSize = 1;
        return options;
    }

    private static IAiModelBuilder<double, Matrix<double>, Vector<double>> Builder(ProgramEvolutionOptions options, FakeProgramExecutionEngine execution) =>
        new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureProgramExecutionEngine(execution).ConfigureProgramEvolution(options);

    private static FakeProgramExecutionEngine Execution() => new((source, _) => FakeExecutionOutcome.Success(source.EndsWith("2", StringComparison.Ordinal) ? "2" : "1"));

    [Fact]
    public async Task Custom_loop_needs_no_facade_chat_client_and_preserves_reported_usage_and_fitness()
    {
        var variation = new CustomVariation();
        var execution = Execution();
        var options = Options(variation);
        Assert.Same(variation, options.Clone().CustomVariation);
        var result = await Builder(options, execution).BuildAsync();
        Assert.Equal("return 2", result.ProgramEvolution?.BestProgram?.Source);
        Assert.Equal(1, result.ProgramEvolution?.BestQuality);
        Assert.Equal(1, variation.Calls);
        Assert.Equal(2, execution.Calls);
        Assert.Equal(variation.GetUsage().ChatCalls, result.ProgramEvolution?.LlmUsage.ChatCalls);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(0)]
    public async Task Custom_proposals_still_pass_the_configured_correctness_gate(double pass)
    {
        var variation = new CustomVariation();
        var execution = Execution();
        var result = await Builder(Options(variation), execution)
            .ConfigureProgramCorrectness(new DelegateProgramFitnessEvaluator(_ => pass)).BuildAsync();
        Assert.Equal(pass == 1, result.ProgramEvolution?.HasBestProgram);
        Assert.Equal(pass == 1 ? 2 : 0, execution.Calls);
        Assert.Equal(pass == 1 ? 1 : 0, variation.Calls);
    }

    [Fact]
    public async Task Custom_loop_output_does_not_create_an_unused_legacy_provenance_sink()
    {
        string directory = Path.Combine(Path.GetTempPath(), "aidotnet-custom-output-" + Guid.NewGuid().ToString("N"));
        try
        {
            var options = Options(new CustomVariation());
            options.Engine.OutputDirectory = directory;
            var result = await Builder(options, Execution()).BuildAsync();
            Assert.Equal("return 2", result.ProgramEvolution?.BestProgram?.Source);
            Assert.False(Directory.Exists(Path.Combine(directory, "provenance")));
        }
        finally
        {
            if (Directory.Exists(directory)) Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task Proposal_correctness_and_fitness_share_one_ledger_without_double_charges()
    {
        var ledger = new EvolutionResourceLedger("shared-program", EvolutionResources.Of("cost_units", 3.25m));
        var variation = new CostedVariation(ledger);
        var accounting = new ProgramEvolutionResourceOptions(ledger, 1.5m, Unit);
        var options = Options(variation, accounting);
        Assert.Same(accounting, options.Clone().ResourceAccounting);
        var gate = new DelegateProgramFitnessEvaluator((_, _, _) => new ValueTask<EvolutionTaskResult>(
            new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 1, costUnits: 0.5)));
        var result = await Builder(options, Execution()).ConfigureProgramCorrectness(gate).BuildAsync();
        Assert.Equal("return 2", result.ProgramEvolution?.BestProgram?.Source);
        EvolutionResourceSnapshot snapshot = ledger.Snapshot();
        Assert.Equal(3.25m, snapshot.Spent["cost_units"]);
        Assert.Equal(3, snapshot.Settled);
        Assert.Equal(2, snapshot.Receipts.Count(receipt => receipt.Stage == EvolutionResourceStage.Evaluation));
        Assert.Single(snapshot.Receipts.Where(receipt => receipt.Stage == EvolutionResourceStage.Proposal));
        Assert.Equal(0, snapshot.Unknown);
        Assert.False(snapshot.MaximumViolated);
    }

    [Fact]
    public async Task Denied_proposal_cannot_invoke_the_backend_or_spend_its_reservation()
    {
        var ledger = new EvolutionResourceLedger("denied-program", EvolutionResources.Of("cost_units", 1.1m));
        var variation = new CostedVariation(ledger);
        var result = await Builder(Options(variation, new ProgramEvolutionResourceOptions(ledger, 1, Unit)), Execution()).BuildAsync();
        Assert.Equal("return 1", result.ProgramEvolution?.BestProgram?.Source);
        Assert.Equal(0, variation.Calls);
        Assert.Equal(1m, ledger.Snapshot().Spent["cost_units"]);
        Assert.True(ledger.Snapshot().Denied > 0);
    }

    [Fact]
    public async Task Evaluator_overrun_is_charged_and_cannot_create_a_winner()
    {
        var ledger = new EvolutionResourceLedger("overrun-program", EvolutionResources.Of("cost_units", 10));
        var variation = new CustomVariation();
        var result = await Builder(Options(variation, new ProgramEvolutionResourceOptions(ledger, 0.5m, Unit)), Execution()).BuildAsync();
        Assert.False(result.ProgramEvolution?.HasBestProgram);
        Assert.Equal(0, variation.Calls);
        Assert.Equal(1m, ledger.Snapshot().Spent["cost_units"]);
        Assert.True(ledger.Snapshot().MaximumViolated);
    }

    [Fact]
    public async Task Automatic_checkpoints_are_refused_before_any_accounted_work()
    {
        var ledger = new EvolutionResourceLedger("checkpoint-program", EvolutionResources.Of("cost_units", 10));
        var variation = new CustomVariation();
        var options = Options(variation, new ProgramEvolutionResourceOptions(ledger, 1, Unit));
        options.Engine.CheckpointInterval = 1;
        var execution = Execution();
        await Assert.ThrowsAsync<NotSupportedException>(() => Builder(options, execution).BuildAsync());
        Assert.Equal(0, execution.Calls);
        Assert.Equal(0, variation.Calls);
        Assert.Equal(0, ledger.Snapshot().Admitted);
    }

    [Fact]
    public void Custom_provenance_and_mismatched_cost_units_fail_during_configuration()
    {
        var ledger = new EvolutionResourceLedger("configuration-program", EvolutionResources.Of("cost_units", 10));
        var options = Options(new CustomVariation());
        options.Provenance.Enabled = true;
        Assert.Throws<ArgumentException>(options.Validate);
        options = Options(new CostedVariation(ledger), new ProgramEvolutionResourceOptions(ledger, 1, "different"));
        Assert.Throws<ArgumentException>(options.Validate);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Nonpositive_evaluation_maximum_is_rejected(int maximum)
    {
        var ledger = new EvolutionResourceLedger("bounds", EvolutionResources.Of("cost_units", 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ProgramEvolutionResourceOptions(ledger, maximum, Unit));
    }

    [Fact]
    public void Accounting_requires_a_real_ledger_bounded_maximum_and_printable_unit_identity()
    {
        var ledger = new EvolutionResourceLedger("bounds", EvolutionResources.Of("cost_units", 10));
#pragma warning disable CS8625 // Deliberately exercise the public null guard.
        Assert.Throws<ArgumentNullException>(() => new ProgramEvolutionResourceOptions(null, 1, Unit));
#pragma warning restore CS8625
        Assert.Throws<ArgumentOutOfRangeException>(() => new ProgramEvolutionResourceOptions(ledger, EvolutionResources.MaximumAmount + 1, Unit));
        foreach (string invalid in new[] { "", " ", "unit\n", new string('a', 257) })
            Assert.Throws<ArgumentException>(() => new ProgramEvolutionResourceOptions(ledger, 1, invalid));
        var missing = new EvolutionResourceLedger("no-cost", EvolutionResources.Of("calls", 10));
        Assert.Throws<ArgumentException>(() => new ProgramEvolutionResourceOptions(missing, 1, Unit));
    }

    [Fact]
    public void Task_fingerprint_distinguishes_marker_pairs_and_declared_cost_units()
    {
        var evaluator = new DelegateProgramFitnessEvaluator(_ => 1);
        var first = new ProgramEvolutionOptions { EvolveBlockStartMarker = "a|b", EvolveBlockEndMarker = "c" };
        var second = new ProgramEvolutionOptions { EvolveBlockStartMarker = "a", EvolveBlockEndMarker = "b|c" };
        Assert.NotEqual(new ProgramEvolutionTask(evaluator, options: first).VersionHash, new ProgramEvolutionTask(evaluator, options: second).VersionHash);
        var ledger = new EvolutionResourceLedger("fingerprint", EvolutionResources.Of("cost_units", 10));
        first.ResourceAccounting = new ProgramEvolutionResourceOptions(ledger, 1, Unit);
        second = first.Clone();
        second.ResourceAccounting = new ProgramEvolutionResourceOptions(ledger, 1, "changed-units");
        Assert.NotEqual(new ProgramEvolutionTask(evaluator, options: first).VersionHash, new ProgramEvolutionTask(evaluator, options: second).VersionHash);
    }

    private sealed class CustomVariation : IProgramVariationOperator
    {
        public string Id => "custom-test";
        public string VersionHash => "custom-test-v1";
        public int Calls { get; private set; }
        public ValueTask<ProgramGenome> ProposeAsync(EvolutionVariationContext<ProgramGenome> context, CancellationToken cancellationToken = default)
        {
            Calls++;
            return new(new ProgramGenome("return 2", ProgramLanguage.Python));
        }
        public void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult) { }
        // Deliberately distinctive reported test usage verifies forwarding, not real model execution.
        public ProgramEvolutionLlmUsage GetUsage() => new(proposals: Calls, chatCalls: Calls * 2, inputTokens: Calls * 3);
    }

    private sealed class CostedVariation : IProgramVariationOperator, ICheckpointableVariationOperator<ProgramGenome>, IEvolutionProposalCostProvider
    {
        private readonly Source _source = new();
        private readonly ResourceMeteredVariationOperator<ProgramGenome> _metered;
        public CostedVariation(EvolutionResourceLedger ledger) => _metered = new(_source, ledger, EvolutionResources.Of("cost_units", 0.25m), Unit);
        public int Calls => _source.Calls;
        public string Id => _metered.Id;
        public string VersionHash => _metered.VersionHash;
        public string CostUnitVersionHash => Unit;
        public ValueTask<ProgramGenome> ProposeAsync(EvolutionVariationContext<ProgramGenome> context, CancellationToken cancellationToken = default) => _metered.ProposeAsync(context, cancellationToken);
        public void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult) => _metered.Observe(evaluation, insertionResult);
        public string CaptureState() => _metered.CaptureState();
        public void RestoreState(string state) => _metered.RestoreState(state);
        public EvolutionProposalCost GetProposalCost(long generation) => _metered.GetProposalCost(generation);
        public ProgramEvolutionLlmUsage GetUsage() => new(proposals: Calls);
    }

    private sealed class Source : ICostedEvolutionProposalSource<ProgramGenome>
    {
        public string Id => "costed-test";
        public string VersionHash => "costed-test-v1";
        public int Calls { get; private set; }
        public ValueTask<EvolutionResourceResult<ProgramGenome>> ProposeAsync(EvolutionVariationContext<ProgramGenome> context, CancellationToken cancellationToken = default)
        {
            Calls++;
            return new(new EvolutionResourceResult<ProgramGenome>(new ProgramGenome("return 2", ProgramLanguage.Python), EvolutionResources.Of("cost_units", 0.25m)));
        }
        public void Observe(EvolutionEvaluation evaluation, EvolutionArchiveInsertionResult? insertionResult) { }
        public string CaptureState() => "stateless-test-v1";
        public void RestoreState(string state)
        {
            if (state != CaptureState()) throw new ArgumentException("Unexpected test state.", nameof(state));
        }
    }
}
