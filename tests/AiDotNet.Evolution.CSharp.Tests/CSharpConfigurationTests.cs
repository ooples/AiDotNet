using AiDotNet.Configuration;
using AiDotNet.Evolution.Programs;
using AiDotNet.Interfaces;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Moq;
using System.Text.Json;
using Xunit;
using static AiDotNet.Evolution.CSharp.Tests.CompilerTestSupport;

namespace AiDotNet.Evolution.CSharp.Tests;

public sealed class CSharpConfigurationTests
{
    [Fact]
    public void Reading_unconfigured_provenance_must_not_enable_a_conflicting_sink()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        var options = Options();
        var ledger = Ledger();
        builder.ConfigureCSharpProgramEvolution(new ScriptedClient(), new ProgramEvolutionOptions(), options,
            new ProgramEvolutionResourceOptions(ledger, 1, options.CostUnitVersionHash));
        Assert.Equal(0.1m, ledger.Snapshot().Spent["cost_units"]);
    }

    [Fact]
    public async Task Installed_operator_consumes_failed_proposal_cost_bookkeeping_after_engine_commit()
    {
        ProgramEvolutionOptions? installed = null;
        var facade = new Mock<IAiModelBuilder<double, Matrix<double>, Vector<double>>>();
        facade.Setup(builder => builder.ConfigureProgramEvolution(It.IsAny<ProgramEvolutionOptions>()))
            .Callback<ProgramEvolutionOptions>(options => installed = options).Returns(facade.Object);
        var program = Program();
        program.SeedPrograms.Add(Source);
        program.TestCases.Add(new ProgramInputOutputExample { Input = "", ExpectedOutput = "1" });
        program.Engine.MaxProposals = 2;
        program.Engine.MaxEvaluationAttempts = 2;
        program.Engine.MaxGenerations = 1;
        var options = Options();
        var ledger = Ledger();
        var client = new ScriptedClient { Handler = (_, messages) => ScriptedClient.Response(Reply(messages, "INVALID")) };
        facade.Object.ConfigureCSharpProgramEvolution(client, program, options,
            new ProgramEvolutionResourceOptions(ledger, 1, options.CostUnitVersionHash));
        Assert.NotNull(installed);
        await new AiModelBuilder<double, Matrix<double>, Vector<double>>().ConfigureProgramExecutionEngine(new ScriptedExecution())
            .ConfigureProgramEvolution(installed).BuildAsync();
        var checkpointable = Assert.IsAssignableFrom<ICheckpointableVariationOperator<ProgramGenome>>(installed.CustomVariation);
        using JsonDocument state = JsonDocument.Parse(checkpointable.CaptureState());
        Assert.Empty(state.RootElement.GetProperty("Pending").EnumerateObject());
        Assert.IsAssignableFrom<IOutcomeAwareVariationOperator<ProgramGenome>>(installed.CustomVariation);
        Assert.Equal(2, client.Conversations.Count);
        Assert.Equal(3.654m, ledger.Snapshot().Spent["cost_units"]);
    }

    [Theory]
    [InlineData("MaxSourceChars", 255)]
    [InlineData("MaxSourceChars", 65537)]
    [InlineData("MaxResponseChars", 255)]
    [InlineData("MaxResponseChars", 262145)]
    [InlineData("MaxRepairs", -1)]
    [InlineData("MaxRepairs", 8)]
    [InlineData("MaxEdits", 0)]
    [InlineData("MaxEdits", 17)]
    [InlineData("MaxCatalogNodes", 0)]
    [InlineData("MaxCatalogNodes", 65)]
    [InlineData("MaxInputTokens", 0)]
    [InlineData("MaxInputTokens", 1048577)]
    [InlineData("MaxOutputTokens", 0)]
    [InlineData("MaxOutputTokens", 65537)]
    [InlineData("CompilationTimeoutSeconds", 0)]
    [InlineData("CompilationTimeoutSeconds", 31)]
    public void Unsupported_work_bounds_are_rejected(string property, int value)
    {
        var options = Options();
        typeof(CSharpProgramEvolutionOptions).GetProperty(property)!.SetValue(options, value);
        Assert.Throws<ArgumentOutOfRangeException>(() => options.Snapshot());
    }

    [Fact]
    public void Identities_prices_paths_and_every_semantic_option_are_validated_and_fingerprinted()
    {
        var options = Options();
        string baseline = options.ConfigurationHash;
        foreach (var property in typeof(CSharpProgramEvolutionOptions).GetProperties())
        {
            var changed = options.Snapshot();
            if (property.PropertyType == typeof(int)) property.SetValue(changed, (int)property.GetValue(changed)! + 1);
            else if (property.PropertyType == typeof(decimal)) property.SetValue(changed, (decimal)property.GetValue(changed)! + 0.001m);
            else if (property.PropertyType == typeof(string) && property.Name != "AuditDirectory") property.SetValue(changed, (string)property.GetValue(changed)! + "changed");
            else continue; // Audit location is operational, reference content is fingerprinted by the compiler.
            Assert.NotEqual(baseline, changed.ConfigurationHash);
        }
        foreach (string name in new[] { "Id", "TargetIdentity", "ModelVersionIdentity", "CostUnitVersionHash" })
            foreach (string invalid in new[] { "", " ", "bad\n", new string('x', 257), "bad" + '\ud800' })
            {
                var changed = Options();
                typeof(CSharpProgramEvolutionOptions).GetProperty(name)!.SetValue(changed, invalid);
                Assert.ThrowsAny<ArgumentException>(() => changed.Snapshot());
            }
        foreach (string name in new[] { "SetupCostUnits", "ModelCallCostUnits", "CompilationCostUnits", "ParseCostUnits", "AuditCostUnits", "InputTokenCostUnits", "OutputTokenCostUnits" })
        {
            var changed = Options();
            var property = typeof(CSharpProgramEvolutionOptions).GetProperty(name)!;
            property.SetValue(changed, -1m);
            Assert.Throws<ArgumentOutOfRangeException>(() => changed.Snapshot());
            property.SetValue(changed, 1_000_000_001m);
            Assert.Throws<ArgumentOutOfRangeException>(() => changed.Snapshot());
        }
        options.Id = new string('x', 65);
        Assert.Throws<ArgumentException>(() => options.Snapshot());
        options = Options(); options.AuditDirectory = " ";
        Assert.Throws<ArgumentException>(() => options.Snapshot());
        options = Options(); options.ReferencePaths = Array.Empty<string>();
        Assert.Throws<ArgumentException>(() => options.Snapshot());
        options.ReferencePaths = Enumerable.Repeat("path", 65).ToArray();
        Assert.Throws<ArgumentException>(() => options.Snapshot());
        options.ReferencePaths = new[] { " " };
        Assert.Throws<ArgumentException>(() => options.Snapshot());
    }

    [Fact]
    public async Task Facade_runs_compiler_proposals_through_independent_correctness_and_fitness_with_shared_costs()
    {
        var program = Program();
        program.Language = ProgramLanguage.Generic; // The C# extension resolves an unspecified language.
        program.SeedPrograms.Add(Source);
        program.TestCases.Add(new ProgramInputOutputExample { Input = "", ExpectedOutput = "2" });
        program.Engine.MaxEvaluationAttempts = 2;
        program.Engine.MaxProposals = 2;
        program.Engine.MaxGenerations = 1;
        program.Engine.ProposalBatchSize = 1;
        var options = Options();
        var ledger = Ledger();
        var client = new ScriptedClient();
        var execution = new ScriptedExecution();
        var gate = new DelegateProgramFitnessEvaluator((_, _, _) => new ValueTask<EvolutionTaskResult>(
            new EvolutionTaskResult(EvolutionEvaluationStatus.Completed, 1, costUnits: 0.5)));
        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureProgramExecutionEngine(execution).ConfigureProgramCorrectness(gate)
            .ConfigureCSharpProgramEvolution(client, program, options, new ProgramEvolutionResourceOptions(ledger, 1.5m, options.CostUnitVersionHash))
            .BuildAsync();
        Assert.Equal(1, result.ProgramEvolution?.LlmUsage.ChatCalls);
        Assert.Equal(2, execution.Calls);
        Assert.Equal(4.382m, ledger.Snapshot().Spent["cost_units"]);
        Assert.Equal(Source.Replace("return 1", "return 2", StringComparison.Ordinal), result.ProgramEvolution?.BestProgram?.Source);
        Assert.Equal(2, ledger.Snapshot().Receipts.Count(receipt => receipt.Stage == EvolutionResourceStage.Evaluation));
        Assert.Equal(4, ledger.Snapshot().Settled);
        Assert.Null(program.CustomVariation); // Configuration never mutates the caller's options.
    }

    [Fact]
    public void Conflicting_language_provenance_cost_units_and_checkpoint_modes_fail_before_setup()
    {
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>();
        var options = Options();
        var ledger = Ledger();
        var resources = new ProgramEvolutionResourceOptions(ledger, 1, options.CostUnitVersionHash);
        var program = Program();
        program.Language = ProgramLanguage.Python;
        Assert.Throws<ArgumentException>(() => builder.ConfigureCSharpProgramEvolution(new ScriptedClient(), program, options, resources));
        program = Program(); program.Provenance.Enabled = true;
        Assert.Throws<ArgumentException>(() => builder.ConfigureCSharpProgramEvolution(new ScriptedClient(), program, options, resources));
        program = Program(); program.Engine.CheckpointInterval = 1;
        Assert.Throws<NotSupportedException>(() => builder.ConfigureCSharpProgramEvolution(new ScriptedClient(), program, options, resources));
        Assert.Throws<ArgumentException>(() => builder.ConfigureCSharpProgramEvolution(new ScriptedClient(), Program(), options, new ProgramEvolutionResourceOptions(ledger, 1, "wrong")));
        Assert.Equal(0, ledger.Snapshot().Admitted);
        Assert.False(Directory.Exists(options.AuditDirectory));
        Assert.Throws<ArgumentException>(() => CSharpProposalSource<double>.Create(new ScriptedClient { ModelId = " " }, options, Program(), ledger));
        program = Program(); program.TaskDescription = new string('x', 4097);
        Assert.Throws<ArgumentException>(() => CSharpProposalSource<double>.Create(new ScriptedClient(), options, program, ledger));
    }

    [Fact]
    public void Optional_package_exposes_only_facade_configuration_not_compiler_plumbing()
    {
        Assert.Equal(new[] { typeof(CSharpProgramEvolutionExtensions), typeof(CSharpProgramEvolutionOptions) }.OrderBy(type => type.Name),
            typeof(CSharpProgramEvolutionOptions).Assembly.GetExportedTypes().OrderBy(type => type.Name));
        Assert.DoesNotContain(typeof(AiModelBuilder<,,>).Assembly.GetReferencedAssemblies(), assembly => assembly.Name?.StartsWith("Microsoft.CodeAnalysis", StringComparison.Ordinal) == true);
    }

    private sealed class ScriptedExecution : IProgramExecutionEngine
    {
        internal int Calls { get; private set; }
        public bool TryExecute(ProgramLanguage language, string sourceCode, string input, out string output, out string? errorMessage, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Calls++;
            output = sourceCode.Contains("return 2", StringComparison.Ordinal) ? "2" : "1";
            errorMessage = null;
            return true;
        }
        public Task<ProgramExecuteResponse> ExecuteAsync(ProgramExecuteRequest request, CancellationToken cancellationToken = default)
        {
            bool success = TryExecute(request.Language, request.SourceCode, request.StdIn ?? string.Empty, out string output, out string? error, cancellationToken);
            return Task.FromResult(new ProgramExecuteResponse
            {
                Success = success,
                Language = request.Language,
                ExitCode = success ? 0 : 1,
                StdOut = output,
                StdErr = error ?? string.Empty,
                Error = error
            });
        }
    }
}
