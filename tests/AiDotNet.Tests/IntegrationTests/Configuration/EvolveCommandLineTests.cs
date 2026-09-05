using AiDotNet.Evolve.Cli;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Covers the command-line tool: every command's exit code and the text a person reads from it. The commands are
/// called directly rather than through a spawned process, so the assertions are about behaviour rather than about
/// build layout, and a failing configuration file is proven to fail loudly rather than to run something unintended.
/// </summary>
public sealed class EvolveCommandLineTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "aidotnet-evolve-cli-" + Guid.NewGuid().ToString("N"));

    private const string ValidConfiguration = @"
evolution:
  runId: ${EVOLVE_CLI_TEST_RUN:-cli-run}
  seed: 99
  maxEvaluationAttempts: 8
  islandCount: 2
  migrationInterval: 0
  dispatch: Continuous
  descriptors:
    - name: length
      minimum: 0
      maximum: 400
      binCount: 8
      outOfRangePolicy: Clamp
";

    public EvolveCommandLineTests() => Directory.CreateDirectory(_directory);

    [Fact]
    public async Task NoArgumentsPrintsUsageAndReportsThatNothingWasAskedFor()
    {
        (int code, string output, string error) = await Execute();

        Assert.Equal(EvolveCommandLine.ExitUsage, code);
        Assert.Contains("aidotnet-evolve", output, StringComparison.Ordinal);
        Assert.Contains("--config", output, StringComparison.Ordinal);
        Assert.Empty(error);
    }

    [Fact]
    public async Task AskingForHelpSucceeds()
    {
        (int code, string output, _) = await Execute("--help");

        Assert.Equal(EvolveCommandLine.ExitSuccess, code);
        Assert.Contains("${NAME}", output, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ValidateResolvesTheFileAndPrintsWhatTheRunWouldUse()
    {
        string path = WriteConfig(ValidConfiguration);

        (int code, string output, string error) = await Execute("validate", "--config", path);

        Assert.Equal(EvolveCommandLine.ExitSuccess, code);
        Assert.Empty(error);
        Assert.Contains("is valid.", output, StringComparison.Ordinal);
        Assert.Contains("cli-run", output, StringComparison.Ordinal);
        Assert.Contains("Continuous", output, StringComparison.Ordinal);
        Assert.Contains("length [0, 400] in 8 bins, Clamp", output, StringComparison.Ordinal);
        Assert.Contains("program section   absent", output, StringComparison.Ordinal);
    }

    [Fact]
    public async Task CommandLineOverridesWinOverTheFileWithoutEditingIt()
    {
        string path = WriteConfig(ValidConfiguration);

        (int code, string output, _) = await Execute(
            "validate", "--config", path, "--run-id", "override", "--seed", "7", "--max-evaluations", "3");

        Assert.Equal(EvolveCommandLine.ExitSuccess, code);
        Assert.Contains("run id            override", output, StringComparison.Ordinal);
        Assert.Contains("seed              7", output, StringComparison.Ordinal);
        Assert.Contains("evaluation budget 3", output, StringComparison.Ordinal);
        Assert.Equal(ValidConfiguration, File.ReadAllText(path));
    }

    [Fact]
    public async Task AConfigurationMistakeIsReportedWithItsReasonAndANonZeroExitCode()
    {
        string missingSection = WriteConfig("optimizer:\n  type: Adam\n", "no-evolution.yaml");
        (int missingCode, _, string missingError) = await Execute("validate", "--config", missingSection);
        Assert.Equal(EvolveCommandLine.ExitUsage, missingCode);
        Assert.Contains("no 'evolution:' section", missingError, StringComparison.Ordinal);

        string badDescriptor = WriteConfig(
            "evolution:\n  descriptors:\n    - name: x\n      minimum: 5\n      maximum: 1\n      binCount: 4\n",
            "bad-descriptor.yaml");
        (int badCode, _, string badError) = await Execute("validate", "--config", badDescriptor);
        Assert.Equal(EvolveCommandLine.ExitUsage, badCode);
        Assert.Contains("maximum", badError, StringComparison.OrdinalIgnoreCase);

        (int missingOption, _, string missingOptionError) = await Execute("validate");
        Assert.Equal(EvolveCommandLine.ExitUsage, missingOption);
        Assert.Contains("--config is required", missingOptionError, StringComparison.Ordinal);

        (int unknown, _, string unknownError) = await Execute("evolve-everything");
        Assert.Equal(EvolveCommandLine.ExitUsage, unknown);
        Assert.Contains("Unknown command", unknownError, StringComparison.Ordinal);

        (int stray, _, string strayError) = await Execute("validate", "config.yaml");
        Assert.Equal(EvolveCommandLine.ExitUsage, stray);
        Assert.Contains("Unexpected argument", strayError, StringComparison.Ordinal);
    }

    [Fact]
    public async Task RunRefusesAFileThatCannotDescribeAWholeRunRatherThanStartingOne()
    {
        string path = WriteConfig(ValidConfiguration);

        (int code, string output, string error) = await Execute("run", "--config", path);

        Assert.Equal(EvolveCommandLine.ExitUsage, code);
        Assert.Empty(output);
        Assert.Contains("no 'programEvolution:' section", error, StringComparison.Ordinal);
        Assert.Contains("Use the library directly", error, StringComparison.Ordinal);
    }

    [Fact]
    public async Task SchemaAndDocsWriteFilesAnEditorAndAReaderCanUse()
    {
        string schemaPath = Path.Combine(_directory, "schema", "aidotnet.schema.json");
        (int schemaCode, string schemaOutput, _) = await Execute("schema", "--out", schemaPath);
        Assert.Equal(EvolveCommandLine.ExitSuccess, schemaCode);
        Assert.Contains("Wrote", schemaOutput, StringComparison.Ordinal);
        Assert.Contains("\"evolution\"", File.ReadAllText(schemaPath), StringComparison.Ordinal);

        string docsPath = Path.Combine(_directory, "reference.md");
        (int docsCode, _, _) = await Execute("docs", "--out", docsPath);
        Assert.Equal(EvolveCommandLine.ExitSuccess, docsCode);
        Assert.Contains("YAML Configuration Reference", File.ReadAllText(docsPath), StringComparison.Ordinal);

        // Without --out the generated text goes to standard output, so it can be piped.
        (int piped, string pipedOutput, _) = await Execute("schema");
        Assert.Equal(EvolveCommandLine.ExitSuccess, piped);
        Assert.Contains("\"evolution\"", pipedOutput, StringComparison.Ordinal);
    }

    private string WriteConfig(string content, string name = "evolution.yaml")
    {
        string path = Path.Combine(_directory, name);
        File.WriteAllText(path, content);
        return path;
    }

    private static async Task<(int Code, string Output, string Error)> Execute(params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int code = await EvolveCommandLine.ExecuteAsync(args, output, error);
        return (code, output.ToString(), error.ToString());
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
        }
        catch (IOException)
        {
            // A leftover temporary directory is not worth failing a test over.
        }
    }
}
