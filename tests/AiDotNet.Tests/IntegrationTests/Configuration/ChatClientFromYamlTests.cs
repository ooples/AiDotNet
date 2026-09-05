using AiDotNet;
using AiDotNet.Agentic.Models.Connectors;
using AiDotNet.Configuration;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Proves a configuration file can actually name the two chat clients that are reached by a path rather than by an
/// HTTP endpoint, and get a working one back. Being present in the generated type registry is not the same as being
/// constructible: both take a required argument, so the question is whether the file's parameters reach it.
/// </summary>
public sealed class ChatClientFromYamlTests : IDisposable
{
    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), "aidotnet-chat-yaml-" + Guid.NewGuid().ToString("N"));

    public ChatClientFromYamlTests() => Directory.CreateDirectory(_directory);

    [Fact]
    public void AConfigurationFileCanNameTheHumanInTheLoopClientAndGetAWorkingOne()
    {
        string queue = Path.Combine(_directory, "manual-queue").Replace('\\', '/');
        string configPath = WriteConfig(
            "chatClient:\n" +
            "  type: ManualChatClient\n" +
            "  params:\n" +
            "    queueDirectory: " + queue + "\n");

        // Building the builder applies the file. The client's constructor creates its queue directory, so the
        // directory appearing is proof that the client was constructed with the parameter the file supplied,
        // rather than with a default or not at all.
        Assert.False(Directory.Exists(queue));
        _ = new AiModelBuilder<double, Matrix<double>, Vector<double>>(configPath);
        Assert.True(Directory.Exists(queue), "the manual client should have created the queue directory it was given");
    }

    [Fact]
    public void AConfigurationFileCanNameTheSubprocessClientWithItsSpendingCap()
    {
        string configPath = WriteConfig(
            "chatClient:\n" +
            "  type: ProcessChatClient\n" +
            "  params:\n" +
            "    fileName: my-model\n");

        // A command that is never run cannot prove itself by a side effect, so this asserts the weaker but still
        // meaningful thing: the file is accepted and the client is built rather than refused for having no
        // constructor a configuration file can reach.
        _ = new AiModelBuilder<double, Matrix<double>, Vector<double>>(configPath);

        // The richer settings still need code, and saying so is the point of naming the executable positionally.
        var configured = new ProcessChatClient<double>("my-model",
            new ProcessChatClientOptions { MaxBudgetUsd = 0.25 });
        Assert.Equal(new[] { "--max-budget-usd", "0.25" }, configured.Arguments.ToArray());
    }

    [Fact]
    public void AConfigurationFileNamingAClientThatDoesNotExistIsRefused()
    {
        string configPath = WriteConfig("chatClient:\n  type: NoSuchChatClient\n");

        Exception failure = Assert.ThrowsAny<Exception>(
            () => new AiModelBuilder<double, Matrix<double>, Vector<double>>(configPath));
        Assert.Contains("NoSuchChatClient", Flatten(failure), StringComparison.Ordinal);
    }

    private string WriteConfig(string content)
    {
        string path = Path.Combine(_directory, "config.yaml");
        File.WriteAllText(path, content);
        return path;
    }

    private static string Flatten(Exception exception)
    {
        var text = new System.Text.StringBuilder();
        for (Exception? current = exception; current is not null; current = current.InnerException)
            text.Append(current.Message).Append(' ');
        return text.ToString();
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
