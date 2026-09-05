using AiDotNet;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Models.Results;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNetTests.UnitTests.Evolution;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Proves the evolution configuration surface is a real configuration-file surface: a YAML document drives a whole
/// run through the builder, the settings survive a write and a read unchanged, secrets and machine-specific values
/// can be referenced instead of embedded, and a malformed file is refused where the mistake is rather than after the
/// first evaluation.
/// </summary>
public sealed class EvolutionYamlConfigTests
{
    private const string RunConfiguration = @"
evolution:
  runId: ${EVO_TEST_RUN_ID:-yaml-evolution}
  seed: 4242
  maxEvaluationAttempts: 12
  maxProposals: 60
  maxGenerations: 60
  proposalBatchSize: 4
  islandCount: 2
  migrationInterval: 0
  migrantsPerIsland: 1
  migrationTopology: FullyConnected
  migrationRate: 0.25
  preventRepeatedMigration: true
  dispatch: Continuous
  maxInFlight: 3
  maxInFlightPerIsland: 2
  inspirationCount: 2
  archiveDirection: Maximize
  descriptors:
    - name: x
      minimum: 0
      maximum: 100
      binCount: 10
      outOfRangePolicy: Clamp
";

    [Fact(Timeout = 120000)]
    public async Task AYamlFileConfiguresAWholeEvolutionRunThroughTheBuilder()
    {
        YamlModelConfig config = YamlConfigLoader.LoadFromString(RunConfiguration);
        EvolutionOptions options = Assert.IsType<EvolutionOptions>(config.Evolution);

        AiModelResult<double, Matrix<double>, Vector<double>> result =
            await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
                .ConfigureEvolution(options)
                .ConfigureEvolution(new SyntheticEvolutionTask(), new IncrementVariation())
                .ConfigureEvolutionSeeds(new[] { new TestGenome(30) })
                .BuildAsync();

        EvolutionRunSummary summary = Assert.IsType<EvolutionRunSummary>(result.EvolutionSummary);
        Assert.Equal("yaml-evolution", summary.RunId);
        Assert.Equal(EvolutionStopReason.EvaluationBudgetReached, summary.StopReason);
        Assert.Equal(12, summary.EvaluationAttempts);
        Assert.Equal(2, summary.IslandCount);
        Assert.All(summary.Islands, island => Assert.Equal(10, island.TotalCells));
        Assert.NotEmpty(summary.Elites);
        Assert.All(summary.Elites, elite => Assert.Contains("x", elite.Descriptors.Keys));
    }

    [Fact]
    public void EverySettingSurvivesBeingWrittenOutAndLoadedBackIn()
    {
        YamlModelConfig loaded = YamlConfigLoader.LoadFromString(RunConfiguration);
        string rewritten = YamlConfigLoader.SaveToString(loaded);
        YamlModelConfig again = YamlConfigLoader.LoadFromString(rewritten);

        EvolutionOptions first = Assert.IsType<EvolutionOptions>(loaded.Evolution);
        EvolutionOptions second = Assert.IsType<EvolutionOptions>(again.Evolution);

        Assert.Equal(first.RunId, second.RunId);
        Assert.Equal(first.Seed, second.Seed);
        Assert.Equal(first.MaxEvaluationAttempts, second.MaxEvaluationAttempts);
        Assert.Equal(first.ProposalBatchSize, second.ProposalBatchSize);
        Assert.Equal(first.IslandCount, second.IslandCount);
        Assert.Equal(first.MigrationTopology, second.MigrationTopology);
        Assert.Equal(first.MigrationRate, second.MigrationRate);
        Assert.Equal(first.PreventRepeatedMigration, second.PreventRepeatedMigration);
        Assert.Equal(first.Dispatch, second.Dispatch);
        Assert.Equal(first.MaxInFlight, second.MaxInFlight);
        Assert.Equal(first.MaxInFlightPerIsland, second.MaxInFlightPerIsland);
        Assert.Equal(first.ArchiveDirection, second.ArchiveDirection);

        // The archive axes are the part a general object mapper cannot express, so they get an explicit check.
        Assert.Equal(
            first.Descriptors.Select(Describe),
            second.Descriptors.Select(Describe));

        // Two runs configured from the two documents would be the same run, which is the property that makes a
        // configuration file worth committing.
        Assert.Equal(
            first.ToEngineOptions().ToSemanticCanonicalString(),
            second.ToEngineOptions().ToSemanticCanonicalString());
    }

    [Fact]
    public void SettingsTheFacadeExposesReachTheEngineRatherThanBeingDropped()
    {
        EvolutionOptions options = Assert.IsType<EvolutionOptions>(
            YamlConfigLoader.LoadFromString(RunConfiguration).Evolution).SnapshotAndValidate();

        Assert.Equal(EvolutionMigrationTopology.FullyConnected, options.MigrationTopology);
        Assert.Equal(0.25, options.MigrationRate);
        Assert.True(options.PreventRepeatedMigration);
        Assert.Equal(EvolutionDispatchMode.Continuous, options.Dispatch);
        Assert.Equal(3, options.MaxInFlight);
        Assert.Equal(2, options.MaxInFlightPerIsland);
    }

    [Fact]
    public void AReferenceToAnEnvironmentVariableIsResolvedAndAMissingOneIsNamed()
    {
        Assert.Equal("run-a", YamlVariableResolver.Resolve("runId: ${NAME}", _ => "run-a").Split(' ')[1]);
        Assert.Equal("runId: fallback", YamlVariableResolver.Resolve("runId: ${NAME:-fallback}", _ => null));
        Assert.Equal("runId: fallback", YamlVariableResolver.Resolve("runId: ${NAME:-fallback}", _ => string.Empty));
        Assert.Equal("runId: ${NAME}", YamlVariableResolver.Resolve("runId: $${NAME}", _ => "ignored"));

        // A template or regular expression that happens to use braces is left alone rather than refused.
        Assert.Equal("pattern: ${0,3}", YamlVariableResolver.Resolve("pattern: ${0,3}", _ => "ignored"));

        ArgumentException missing = Assert.Throws<ArgumentException>(
            () => YamlVariableResolver.Resolve("apiKey: ${OPENAI_API_KEY}", _ => null));
        Assert.Contains("OPENAI_API_KEY", missing.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AMalformedDescriptorIsRefusedWithTheReasonAndNotSilentlyIgnored()
    {
        ArgumentException unknownKey = Assert.Throws<ArgumentException>(() => YamlConfigLoader.LoadFromString(
            "evolution:\n  descriptors:\n    - name: x\n      minimum: 0\n      maximum: 1\n      binCount: 4\n      binWidth: 2\n"));
        Assert.Contains("binWidth", Flatten(unknownKey), StringComparison.Ordinal);

        ArgumentException incomplete = Assert.Throws<ArgumentException>(() => YamlConfigLoader.LoadFromString(
            "evolution:\n  descriptors:\n    - name: x\n      minimum: 0\n"));
        Assert.Contains("binCount", Flatten(incomplete), StringComparison.Ordinal);

        ArgumentException inverted = Assert.Throws<ArgumentException>(() => YamlConfigLoader.LoadFromString(
            "evolution:\n  descriptors:\n    - name: x\n      minimum: 5\n      maximum: 1\n      binCount: 4\n"));
        Assert.Contains("maximum", Flatten(inverted), StringComparison.OrdinalIgnoreCase);

        ArgumentException policy = Assert.Throws<ArgumentException>(() => YamlConfigLoader.LoadFromString(
            "evolution:\n  descriptors:\n    - name: x\n      minimum: 0\n      maximum: 1\n      binCount: 4\n      outOfRangePolicy: Squash\n"));
        Assert.Contains("Squash", Flatten(policy), StringComparison.Ordinal);
    }

    [Fact]
    public void TheEvolutionSectionsAppearInTheGeneratedSchemaAndReference()
    {
        // An editor validates a configuration file against the generated schema, and a reader learns the sections
        // exist from the generated reference. A section missing from either is a section nobody can discover.
        string schema = YamlJsonSchema.Generate();
        Assert.Contains("\"evolution\"", schema, StringComparison.Ordinal);
        Assert.Contains("\"evolutionSeeds\"", schema, StringComparison.Ordinal);
        Assert.Contains("\"programEvolution\"", schema, StringComparison.Ordinal);

        string reference = YamlDocsGenerator.Generate();
        Assert.Contains("evolution", reference, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("programEvolution", reference, StringComparison.OrdinalIgnoreCase);
    }

    private static string Describe(EvolutionDescriptorDefinition descriptor) => descriptor.ToCanonicalString();

    /// <summary>Joins an exception and its causes, because YAML reports the reason on the inner exception.</summary>
    private static string Flatten(Exception exception)
    {
        var text = new System.Text.StringBuilder();
        for (Exception? current = exception; current is not null; current = current.InnerException)
            text.Append(current.Message).Append(' ');
        return text.ToString();
    }
}
