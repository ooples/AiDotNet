using System.Text;
using Xunit;

namespace AiDotNet.Evolve.Cli.Tests;

public sealed class RunEvidenceBundleTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "evidence-test-" + Guid.NewGuid().ToString("N"));
    private static Dictionary<string, byte[]> Files() => new()
    {
        ["configuration.json"] = Encoding.UTF8.GetBytes("{\"Seed\":7}"),
        ["environment.json"] = Encoding.UTF8.GetBytes("{}"),
        ["validation.json"] = Encoding.UTF8.GetBytes("{\"Passed\":true}"),
        ["result.json"] = Encoding.UTF8.GetBytes("{}"),
        ["program.txt"] = Encoding.UTF8.GetBytes("print(7)")
    };

    [Fact]
    public void RoundTripIsByteExactAndNeverOverwritesAnExistingDestination()
    {
        var files = Files();
        RunEvidenceBundle.Create(_root, files, Array.Empty<string>());
        var read = RunEvidenceBundle.ReadAndVerify(_root);
        Assert.Equal(files.Count, read.Count);
        foreach (var pair in files) Assert.Equal(pair.Value, read[pair.Key]);
        Assert.Throws<IOException>(() => RunEvidenceBundle.Create(_root, files, Array.Empty<string>()));
        Assert.Equal("print(7)", File.ReadAllText(Path.Combine(_root, "program.txt")));
    }

    [Theory]
    [InlineData("program.txt", "private-token")]
    [InlineData("configuration.json", "{\"Value\":\"private-\\u0074oken\"}")]
    [InlineData("validation.json", "{\"private-token\":true}")]
    public void KnownSecretsAreRefusedBeforePublishingEvenWhenJsonEscapesThem(string file, string content)
    {
        var files = Files();
        files[file] = Encoding.UTF8.GetBytes(content);
        Assert.Throws<InvalidDataException>(() => RunEvidenceBundle.Create(_root, files, new[] { "private-token" }));
        Assert.False(Directory.Exists(_root));
    }

    [Theory]
    [InlineData("../program.txt")]
    [InlineData("PROGRAM.TXT")]
    [InlineData("extra.json")]
    public void OnlyTheFixedEvidenceFileSetIsAllowed(string name)
    {
        var files = Files();
        files.Remove("program.txt");
        files.Add(name, Encoding.UTF8.GetBytes("{}"));
        Assert.Throws<InvalidDataException>(() => RunEvidenceBundle.Create(_root, files, Array.Empty<string>()));
        Assert.False(Directory.Exists(_root));
    }

    [Fact]
    public void TamperingMissingDocumentsAndDuplicateJsonPropertiesAreRefused()
    {
        var files = Files();
        files["result.json"] = Encoding.UTF8.GetBytes("{\"Passed\":true,\"Passed\":false}");
        Assert.Throws<InvalidDataException>(() => RunEvidenceBundle.Create(_root, files, Array.Empty<string>()));
        files = Files();
        files.Remove("validation.json");
        Assert.Throws<InvalidDataException>(() => RunEvidenceBundle.Create(_root, files, Array.Empty<string>()));
        RunEvidenceBundle.Create(_root, Files(), Array.Empty<string>());
        File.AppendAllText(Path.Combine(_root, "program.txt"), " ");
        Assert.Throws<InvalidDataException>(() => RunEvidenceBundle.ReadAndVerify(_root));
    }

    [Fact]
    public void SourceAndMetadataSizesAreBoundedBeforePublication()
    {
        foreach (var (name, length) in new[] { ("program.txt", 4 * 1024 * 1024 + 1), ("result.json", 128 * 1024 + 1) })
        {
            var files = Files();
            files[name] = new byte[length];
            Assert.Throws<InvalidDataException>(() => RunEvidenceBundle.Create(_root, files, Array.Empty<string>()));
        }
        Assert.False(Directory.Exists(_root));
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); // This instance's GUID directory only.
    }
}
