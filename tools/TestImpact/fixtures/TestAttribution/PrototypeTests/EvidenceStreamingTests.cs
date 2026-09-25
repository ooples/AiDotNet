using AiDotNet.TestImpact;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "RunnerProtocol")]
public sealed class EvidenceStreamingTests
{
    private static string PathForEvidence() => Path.Combine(Path.GetTempPath(), "attribution-document-" + Guid.NewGuid().ToString("N") + ".json");
    private sealed record Payload(string Name, int Count);

    [Fact]
    public void CompactStreamingPreservesTheDocument()
    {
        string path = PathForEvidence();
        RunnerBinding.WriteNew(path, new Payload("source", 3), indented: false);
        Assert.Equal(new("source", 3), ExecutionEvidence.ReadDocumentFile<Payload>(path));
        Assert.DoesNotContain('\n', File.ReadAllText(path));
    }

    [Fact]
    public void DuplicateUnknownAndMissingPropertiesAreRejected()
    {
        foreach (string json in new[] { "{\"Name\":\"a\",\"Name\":\"b\",\"Count\":1}",
            "{\"Name\":\"a\",\"Count\":1,\"Extra\":true}", "{\"Name\":\"a\"}" })
        {
            string path = PathForEvidence();
            File.WriteAllText(path, json);
            Assert.Equal(EvidenceFailure.Format, Assert.Throws<EvidenceException>(() => ExecutionEvidence.ReadDocumentFile<Payload>(path)).Reason);
        }
    }

    [Fact]
    public void OversizedFilesAreRejectedBeforeDeserialization()
    {
        string path = PathForEvidence();
        RunnerBinding.WriteNew(path, new Payload("source", 3));
        Assert.Equal(EvidenceFailure.Format, Assert.Throws<EvidenceException>(() => ExecutionEvidence.ReadDocumentFile<Payload>(path, 2)).Reason);
    }
}
