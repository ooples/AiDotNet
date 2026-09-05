using AiDotNet.Enums;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class EvolveBlockTests
{
    private const string PythonSource =
        "import math\n" +
        "# EVOLVE-BLOCK-START\n" +
        "def solve(x):\n" +
        "    return x\n" +
        "# EVOLVE-BLOCK-END\n" +
        "print(solve(1))\n";

    [Fact]
    public void ExtractSplitsSourceIntoPrefixBodyAndSuffix()
    {
        EvolveBlockExtractionResult result = EvolveBlock.Extract(PythonSource);

        Assert.Equal(EvolveBlockStatus.Complete, result.Status);
        Assert.True(result.IsWellFormed);
        Assert.True(result.TryGetPrimaryRegion(out EvolveBlockRegion region));
        Assert.Equal("import math\n# EVOLVE-BLOCK-START\n", region.Prefix);
        Assert.Equal("def solve(x):\n    return x\n", region.Body);
        Assert.Equal("# EVOLVE-BLOCK-END\nprint(solve(1))\n", region.Suffix);
        Assert.Equal(PythonSource, region.ToSource());
        Assert.Equal(1, region.StartLineIndex);
        Assert.Equal(4, region.EndLineIndex);
        Assert.Equal(2, region.BodyLineCount);
    }

    [Fact]
    public void RewritePreservesEverythingOutsideTheBlock()
    {
        Assert.True(EvolveBlock.TryReplaceFirst(PythonSource, "def solve(x):\n    return x * 2", default, out string rewritten));

        Assert.StartsWith("import math\n# EVOLVE-BLOCK-START\n", rewritten, StringComparison.Ordinal);
        Assert.EndsWith("# EVOLVE-BLOCK-END\nprint(solve(1))\n", rewritten, StringComparison.Ordinal);
        Assert.Contains("return x * 2", rewritten, StringComparison.Ordinal);
        Assert.DoesNotContain("    return x\n", rewritten, StringComparison.Ordinal);
    }

    [Fact]
    public void RewriteAppendsMissingTerminatorSoTheEndMarkerKeepsItsLine()
    {
        Assert.True(EvolveBlock.TryReplaceFirst(PythonSource, "pass", default, out string rewritten));
        Assert.Contains("# EVOLVE-BLOCK-START\npass\n# EVOLVE-BLOCK-END", rewritten, StringComparison.Ordinal);
    }

    [Fact]
    public void RewriteWithEmptyBodyLeavesAdjacentMarkers()
    {
        Assert.True(EvolveBlock.TryReplaceFirst(PythonSource, string.Empty, default, out string rewritten));
        Assert.Contains("# EVOLVE-BLOCK-START\n# EVOLVE-BLOCK-END", rewritten, StringComparison.Ordinal);
    }

    [Fact]
    public void CarriageReturnLineEndingsArePreservedOnRewrite()
    {
        string source = PythonSource.Replace("\n", "\r\n");
        EvolveBlockExtractionResult result = EvolveBlock.Extract(source);

        Assert.Equal(EvolveBlockStatus.Complete, result.Status);
        Assert.True(result.TryGetPrimaryRegion(out EvolveBlockRegion region));
        Assert.Equal(source, region.ToSource());
        Assert.Equal("\r\n", region.NewLine);

        string rewritten = region.Rewrite("pass\npass");
        Assert.Contains("# EVOLVE-BLOCK-START\r\npass\r\npass\r\n# EVOLVE-BLOCK-END", rewritten, StringComparison.Ordinal);
        Assert.DoesNotContain("pass\npass", rewritten, StringComparison.Ordinal);
    }

    [Fact]
    public void SourceWithoutTrailingNewLineRoundTrips()
    {
        string source = "a\n# EVOLVE-BLOCK-START\nb\n# EVOLVE-BLOCK-END\nc";
        EvolveBlockExtractionResult result = EvolveBlock.Extract(source);
        Assert.True(result.TryGetPrimaryRegion(out EvolveBlockRegion region));
        Assert.Equal(source, region.ToSource());
    }

    [Theory]
    [InlineData("    # EVOLVE-BLOCK-START")]
    [InlineData("\t# EVOLVE-BLOCK-START")]
    [InlineData("# EVOLVE-BLOCK-START   ")]
    [InlineData("# EVOLVE-BLOCK-START  # keep this tidy")]
    public void MarkersToleratedWithSurroundingText(string startLine)
    {
        string source = startLine + "\nbody\n# EVOLVE-BLOCK-END\n";
        EvolveBlockExtractionResult result = EvolveBlock.Extract(source);

        Assert.Equal(EvolveBlockStatus.Complete, result.Status);
        Assert.True(result.TryGetPrimaryRegion(out EvolveBlockRegion region));
        Assert.Equal("body\n", region.Body);
    }

    [Fact]
    public void MultipleBlocksAreAllRecovered()
    {
        string source =
            "# EVOLVE-BLOCK-START\none\n# EVOLVE-BLOCK-END\n" +
            "middle\n" +
            "# EVOLVE-BLOCK-START\ntwo\n# EVOLVE-BLOCK-END\n";

        EvolveBlockExtractionResult result = EvolveBlock.Extract(source);
        Assert.Equal(EvolveBlockStatus.Complete, result.Status);
        Assert.Equal(2, result.Regions.Count);
        Assert.Equal("one\n", result.Regions[0].Body);
        Assert.Equal("two\n", result.Regions[1].Body);

        Assert.True(EvolveBlock.TryReplaceAll(source, new[] { "ONE", "TWO" }, default, out string rewritten));
        Assert.Contains("# EVOLVE-BLOCK-START\nONE\n# EVOLVE-BLOCK-END", rewritten, StringComparison.Ordinal);
        Assert.Contains("# EVOLVE-BLOCK-START\nTWO\n# EVOLVE-BLOCK-END", rewritten, StringComparison.Ordinal);
        Assert.Contains("middle", rewritten, StringComparison.Ordinal);
    }

    [Fact]
    public void ReplaceAllRefusesAMismatchedReplacementCount()
    {
        Assert.False(EvolveBlock.TryReplaceAll(PythonSource, new[] { "a", "b" }, default, out string rewritten));
        Assert.Equal(PythonSource, rewritten);
    }

    [Fact]
    public void MissingMarkersReportNotPresent()
    {
        EvolveBlockExtractionResult result = EvolveBlock.Extract("print(1)\n");
        Assert.Equal(EvolveBlockStatus.NotPresent, result.Status);
        Assert.True(result.IsWellFormed);
        Assert.False(result.HasRegions);
        Assert.False(result.TryGetPrimaryRegion(out _));
        Assert.False(EvolveBlock.TryReplaceFirst("print(1)\n", "x", default, out string rewritten));
        Assert.Equal("print(1)\n", rewritten);
    }

    [Fact]
    public void UnterminatedBlockIsReportedAndDiscarded()
    {
        EvolveBlockExtractionResult result = EvolveBlock.Extract("# EVOLVE-BLOCK-START\nbody\n");
        Assert.Equal(EvolveBlockStatus.Unterminated, result.Status);
        Assert.False(result.IsWellFormed);
        Assert.False(result.HasRegions);
        Assert.Single(result.Diagnostics);
        Assert.Contains("never closed", result.Diagnostics[0], StringComparison.Ordinal);
    }

    [Fact]
    public void RestartedBlockIsReported()
    {
        EvolveBlockExtractionResult result = EvolveBlock.Extract(
            "# EVOLVE-BLOCK-START\nlost\n# EVOLVE-BLOCK-START\nkept\n# EVOLVE-BLOCK-END\n");

        Assert.Equal(EvolveBlockStatus.RestartedBlock, result.Status);
        Assert.Single(result.Regions);
        Assert.Equal("kept\n", result.Regions[0].Body);
        Assert.Single(result.Diagnostics);
    }

    [Fact]
    public void StrayEndMarkerIsReported()
    {
        EvolveBlockExtractionResult result = EvolveBlock.Extract("# EVOLVE-BLOCK-END\nprint(1)\n");
        Assert.Equal(EvolveBlockStatus.UnmatchedEnd, result.Status);
        Assert.False(result.HasRegions);
        Assert.Single(result.Diagnostics);
    }

    [Fact]
    public void DiagnosticsAreBounded()
    {
        var builder = new System.Text.StringBuilder();
        for (int index = 0; index < EvolveBlockExtractionResult.MaxDiagnostics + 20; index++)
        {
            builder.Append("# EVOLVE-BLOCK-END\n");
        }

        EvolveBlockExtractionResult result = EvolveBlock.Extract(builder.ToString());
        Assert.Equal(EvolveBlockExtractionResult.MaxDiagnostics, result.Diagnostics.Count);
    }

    [Fact]
    public void LanguageMarkersUseValidCommentSyntax()
    {
        Assert.Equal(EvolveBlockMarkers.Slash, EvolveBlockMarkers.ForLanguage(ProgramLanguage.CSharp));
        Assert.Equal(EvolveBlockMarkers.Slash, EvolveBlockMarkers.ForLanguage(ProgramLanguage.Rust));
        Assert.Equal(EvolveBlockMarkers.DoubleDash, EvolveBlockMarkers.ForLanguage(ProgramLanguage.SQL));
        Assert.Equal(EvolveBlockMarkers.Hash, EvolveBlockMarkers.ForLanguage(ProgramLanguage.Python));
        Assert.Equal(EvolveBlockMarkers.Hash, EvolveBlockMarkers.ForLanguage(ProgramLanguage.Generic));

        string csharp = "// EVOLVE-BLOCK-START\nint x = 1;\n// EVOLVE-BLOCK-END\n";
        EvolveBlockExtractionResult result = EvolveBlock.Extract(csharp, ProgramLanguage.CSharp);
        Assert.Equal(EvolveBlockStatus.Complete, result.Status);
        Assert.Equal("int x = 1;\n", result.Regions[0].Body);

        Assert.Equal(EvolveBlockStatus.NotPresent, EvolveBlock.Extract(csharp).Status);
    }

    [Fact]
    public void DefaultMarkersMatchTheReferenceImplementation()
    {
        Assert.Equal("# EVOLVE-BLOCK-START", EvolveBlock.DefaultStartMarker);
        Assert.Equal("# EVOLVE-BLOCK-END", EvolveBlock.DefaultEndMarker);
        var markers = default(EvolveBlockMarkers);
        Assert.Equal(EvolveBlock.DefaultStartMarker, markers.Start);
        Assert.Equal(EvolveBlock.DefaultEndMarker, markers.End);
    }

    [Fact]
    public void WrapAddsMarkersOnlyWhenAbsent()
    {
        string wrapped = EvolveBlock.Wrap("print(1)\n");
        Assert.Equal("# EVOLVE-BLOCK-START\nprint(1)\n# EVOLVE-BLOCK-END\n", wrapped);
        Assert.Equal(wrapped, EvolveBlock.Wrap(wrapped));

        string crlf = EvolveBlock.Wrap("a\r\nb\r\n");
        Assert.Equal("# EVOLVE-BLOCK-START\r\na\r\nb\r\n# EVOLVE-BLOCK-END\r\n", crlf);

        string csharp = EvolveBlock.Wrap("int x = 1;", EvolveBlockMarkers.Slash);
        Assert.Equal("// EVOLVE-BLOCK-START\nint x = 1;\n// EVOLVE-BLOCK-END\n", csharp);
    }

    [Fact]
    public void MarkerPairValidationRejectsBlankAndDuplicateMarkers()
    {
        Assert.Throws<ArgumentException>(() => new EvolveBlockMarkers(" ", "end"));
        Assert.Throws<ArgumentException>(() => new EvolveBlockMarkers("start", string.Empty));
        Assert.Throws<ArgumentException>(() => new EvolveBlockMarkers("same", "same"));
#pragma warning disable CS8600, CS8625
        Assert.Throws<ArgumentNullException>(() => new EvolveBlockMarkers(null, "end"));
#pragma warning restore CS8600, CS8625
    }

    [Fact]
    public void ContainsStartMarkerDetectsThePairInUse()
    {
        Assert.True(EvolveBlock.ContainsStartMarker(PythonSource));
        Assert.False(EvolveBlock.ContainsStartMarker(PythonSource, EvolveBlockMarkers.Slash));
    }
}
