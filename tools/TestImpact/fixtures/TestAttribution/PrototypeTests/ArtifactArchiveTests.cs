using System.IO.Compression;
using AiDotNet.TestImpact;
using Xunit;

namespace PrototypeTests;

[Trait("Scenario", "WorkflowProtocol")]
public sealed class ArtifactArchiveTests
{
    private static (string Zip, string Output) Archive(params (string Name, int Attributes)[] entries)
    {
        string root = Path.Combine(Path.GetTempPath(), "attribution-archive-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        string zip = Path.Combine(root, "input.zip");
        using (ZipArchive archive = ZipFile.Open(zip, ZipArchiveMode.Create))
            foreach (var item in entries)
            {
                ZipArchiveEntry entry = archive.CreateEntry(item.Name);
                entry.ExternalAttributes = item.Attributes;
                using var writer = new StreamWriter(entry.Open());
                writer.Write("evidence");
            }
        return (zip, Path.Combine(root, "output"));
    }

    [Fact]
    public void NormalArchiveExtractsWithoutOverwriting()
    {
        var artifact = Archive(("proof/report.json", 0));
        ArtifactArchive.Extract(artifact.Zip, artifact.Output);
        Assert.Equal("evidence", File.ReadAllText(Path.Combine(artifact.Output, "proof/report.json")));
        Assert.Throws<IOException>(() => ArtifactArchive.Extract(artifact.Zip, artifact.Output));
    }

    [Fact]
    public void TraversalAndAbsolutePathsAreRejectedBeforeExtraction()
    {
        foreach (string path in new[] { "../outside", "/absolute", "C:/outside", "good/../../outside", "good\\outside" })
        {
            var artifact = Archive(("first.json", 0), (path, 0));
            Assert.Throws<EvidenceException>(() => ArtifactArchive.Extract(artifact.Zip, artifact.Output));
            Assert.False(Directory.Exists(artifact.Output));
        }
    }

    [Fact]
    public void AlternateStreamsAndReservedNamesAreRejected()
    {
        foreach (string path in new[] { "report.json:secret", "CON", "aux.json", "COM1.txt", "NUL/x", "proof./x", "proof /x" })
            Assert.Throws<EvidenceException>(() => ArtifactArchive.ResolveContained(Path.GetTempPath(), path));
    }

    [Fact]
    public void DuplicateAndCaseCollidingEntriesAreRejected()
    {
        var artifact = Archive(("report.json", 0), ("REPORT.json", 0));
        Assert.Throws<InvalidDataException>(() => ArtifactArchive.Extract(artifact.Zip, artifact.Output));
        Assert.False(Directory.Exists(artifact.Output));
    }

    [Fact]
    public void SymbolicLinksAreRejectedBeforeExtraction()
    {
        var artifact = Archive(("link", unchecked((int)0xa0000000)));
        Assert.Throws<InvalidDataException>(() => ArtifactArchive.Extract(artifact.Zip, artifact.Output));
        Assert.False(Directory.Exists(artifact.Output));
    }
}
