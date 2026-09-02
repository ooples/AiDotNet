using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class ProgramDiffTests
{
    private const string Source = "def solve(x):\n    return x\n\nprint(solve(1))\n";

    private const string Response =
        "Here is the change.\n" +
        "<<<<<<< SEARCH\n" +
        "    return x\n" +
        "=======\n" +
        "    return x * 2\n" +
        ">>>>>>> REPLACE\n";

    [Fact]
    public void ParsesTheReferenceBlockFormat()
    {
        ProgramDiffParseResult parsed = ProgramDiff.Parse(Response);

        Assert.True(parsed.IsSuccess);
        Assert.Single(parsed.Blocks);
        Assert.Empty(parsed.Failures);
        Assert.Equal("    return x", parsed.Blocks[0].SearchText);
        Assert.Equal("    return x * 2", parsed.Blocks[0].ReplaceText);
        Assert.Equal(0, parsed.Blocks[0].Ordinal);
        Assert.Equal(2, parsed.Blocks[0].ResponseLine);
    }

    [Fact]
    public void ParsesCarriageReturnResponsesTheReferenceRegexWouldMiss()
    {
        ProgramDiffParseResult parsed = ProgramDiff.Parse(Response.Replace("\n", "\r\n"));

        Assert.True(parsed.IsSuccess);
        Assert.Single(parsed.Blocks);
        Assert.Equal("    return x", parsed.Blocks[0].SearchText);
        Assert.DoesNotContain("\r", parsed.Blocks[0].SearchText, StringComparison.Ordinal);
    }

    [Fact]
    public void ParsesMarkersWithTrailingSpacesAndIndentation()
    {
        string response =
            "  <<<<<<< SEARCH   \n" +
            "    return x\n" +
            "  =======  \n" +
            "    return 0\n" +
            "  >>>>>>> REPLACE \n";

        ProgramDiffParseResult parsed = ProgramDiff.Parse(response);
        Assert.True(parsed.IsSuccess);
        Assert.Equal("    return 0", parsed.Blocks[0].ReplaceText);
    }

    [Fact]
    public void ParsesSeveralBlocksInOneResponse()
    {
        string response = Response +
            "<<<<<<< SEARCH\nprint(solve(1))\n=======\nprint(solve(2))\n>>>>>>> REPLACE\n";

        ProgramDiffParseResult parsed = ProgramDiff.Parse(response);
        Assert.Equal(2, parsed.Blocks.Count);
        Assert.Equal(1, parsed.Blocks[1].Ordinal);
    }

    [Fact]
    public void StrictModeRejectsCarriageReturns()
    {
        var options = new ProgramDiffOptions { AllowCarriageReturns = false };
        ProgramDiffParseResult parsed = ProgramDiff.Parse(Response.Replace("\n", "\r\n"), options);

        Assert.False(parsed.IsSuccess);
        Assert.Empty(parsed.Blocks);
        Assert.Equal(ProgramDiffFailureReason.CarriageReturnRejected, parsed.Failures[0].Reason);
    }

    [Fact]
    public void ResponseWithoutBlocksIsReportedAsNoBlocksFound()
    {
        ProgramDiffParseResult parsed = ProgramDiff.Parse("I could not improve this program.");
        Assert.False(parsed.HasBlocks);
        Assert.Single(parsed.Failures);
        Assert.Equal(ProgramDiffFailureReason.NoBlocksFound, parsed.Failures[0].Reason);
    }

    [Theory]
    [InlineData("<<<<<<< SEARCH\nabc\n=======\nxyz\n", "never closed")]
    [InlineData("<<<<<<< SEARCH\nabc\n>>>>>>> REPLACE\n", "without a divider")]
    [InlineData("=======\nxyz\n>>>>>>> REPLACE\n", "no matching SEARCH")]
    public void MalformedBlocksAreReportedNotSilentlyDropped(string response, string expectedFragment)
    {
        ProgramDiffParseResult parsed = ProgramDiff.Parse(response);

        Assert.False(parsed.IsSuccess);
        Assert.Empty(parsed.Blocks);
        Assert.Contains(parsed.Failures,
            failure => failure.Reason == ProgramDiffFailureReason.MalformedBlock
                && failure.Message.IndexOf(expectedFragment, StringComparison.Ordinal) >= 0);
    }

    [Fact]
    public void InterruptedBlockIsReportedAndTheSecondBlockStillParses()
    {
        string response =
            "<<<<<<< SEARCH\nlost\n" +
            "<<<<<<< SEARCH\n    return x\n=======\n    return 3\n>>>>>>> REPLACE\n";

        ProgramDiffParseResult parsed = ProgramDiff.Parse(response);
        Assert.Single(parsed.Blocks);
        Assert.Single(parsed.Failures);
        Assert.Equal(ProgramDiffFailureReason.MalformedBlock, parsed.Failures[0].Reason);
        Assert.Equal("    return 3", parsed.Blocks[0].ReplaceText);
    }

    [Fact]
    public void EmptySearchSectionIsRejected()
    {
        ProgramDiffParseResult parsed = ProgramDiff.Parse("<<<<<<< SEARCH\n\n=======\nnew\n>>>>>>> REPLACE\n");
        Assert.Empty(parsed.Blocks);
        Assert.Equal(ProgramDiffFailureReason.EmptySearchText, parsed.Failures[0].Reason);
    }

    [Fact]
    public void BlockLimitIsEnforced()
    {
        var builder = new System.Text.StringBuilder();
        for (int index = 0; index < 5; index++)
        {
            builder.Append("<<<<<<< SEARCH\nline")
                .Append(index)
                .Append("\n=======\nnew")
                .Append(index)
                .Append("\n>>>>>>> REPLACE\n");
        }

        ProgramDiffParseResult parsed = ProgramDiff.Parse(builder.ToString(), new ProgramDiffOptions { MaxBlocks = 2 });
        Assert.Equal(2, parsed.Blocks.Count);
        Assert.Equal(3, parsed.Failures.Count);
        Assert.All(parsed.Failures, failure => Assert.Equal(ProgramDiffFailureReason.BlockLimitExceeded, failure.Reason));
    }

    [Fact]
    public void ApplyingABlockEditsOnlyTheMatchedWindow()
    {
        ProgramDiffApplyResult result = ProgramDiff.ApplyResponse(Source, Response);

        Assert.True(result.IsSuccess);
        Assert.Equal(1, result.AppliedCount);
        Assert.Empty(result.Failures);
        Assert.Equal("def solve(x):\n    return x * 2\n\nprint(solve(1))\n", result.ModifiedSource);
        Assert.False(result.IsUnchanged);
    }

    [Fact]
    public void ApplyPreservesCarriageReturnLineEndings()
    {
        string crlfSource = Source.Replace("\n", "\r\n");
        ProgramDiffApplyResult result = ProgramDiff.ApplyResponse(crlfSource, Response);

        Assert.True(result.IsSuccess);
        Assert.Equal("def solve(x):\r\n    return x * 2\r\n\r\nprint(solve(1))\r\n", result.ModifiedSource);
    }

    [Fact]
    public void UnmatchedSearchTextIsReportedInsteadOfSilentlySkipped()
    {
        string response = "<<<<<<< SEARCH\n    return y\n=======\n    return 0\n>>>>>>> REPLACE\n";
        ProgramDiffApplyResult result = ProgramDiff.ApplyResponse(Source, response);

        Assert.False(result.IsSuccess);
        Assert.Equal(0, result.AppliedCount);
        Assert.True(result.IsUnchanged);
        Assert.Contains(result.Failures, failure => failure.Reason == ProgramDiffFailureReason.SearchTextNotFound);
        Assert.Contains(result.Failures, failure => failure.SearchExcerpt.IndexOf("return y", StringComparison.Ordinal) >= 0);
    }

    [Fact]
    public void ChildIdenticalToParentIsReportedAsUnchanged()
    {
        string response = "<<<<<<< SEARCH\n    return x\n=======\n    return x\n>>>>>>> REPLACE\n";
        ProgramDiffApplyResult result = ProgramDiff.ApplyResponse(Source, response);

        Assert.False(result.IsSuccess);
        Assert.Equal(1, result.AppliedCount);
        Assert.True(result.IsUnchanged);
        Assert.Contains(result.Failures, failure => failure.Reason == ProgramDiffFailureReason.ResultUnchanged);
    }

    [Fact]
    public void UnchangedResultCanBeAcceptedWhenRejectionIsDisabled()
    {
        var options = new ProgramEvolutionOptions();
        options.Diff.RejectWhenNoBlockApplied = false;
        string response = "<<<<<<< SEARCH\n    return x\n=======\n    return x\n>>>>>>> REPLACE\n";

        ProgramDiffApplyResult result = ProgramDiff.ApplyResponse(Source, response, options);
        Assert.True(result.IsSuccess);
        Assert.True(result.IsUnchanged);
    }

    [Fact]
    public void EmptyReplacementDeletesTheMatchedLinesInsteadOfLeavingABlankLine()
    {
        string response = "<<<<<<< SEARCH\nprint(solve(1))\n=======\n>>>>>>> REPLACE\n";
        ProgramDiffApplyResult result = ProgramDiff.ApplyResponse(Source, response);

        Assert.True(result.IsSuccess);
        Assert.Equal("def solve(x):\n    return x\n\n", result.ModifiedSource);
    }

    [Fact]
    public void FuzzyWhitespaceMatchingIsOptional()
    {
        string response = "<<<<<<< SEARCH\n    return   x\n=======\n    return 7\n>>>>>>> REPLACE\n";

        ProgramDiffApplyResult strict = ProgramDiff.ApplyResponse(Source, response);
        Assert.False(strict.IsSuccess);
        Assert.Contains(strict.Failures, failure => failure.Reason == ProgramDiffFailureReason.SearchTextNotFound);

        var options = new ProgramEvolutionOptions();
        options.Diff.FuzzyWhitespace = true;
        ProgramDiffApplyResult fuzzy = ProgramDiff.ApplyResponse(Source, response, options);
        Assert.True(fuzzy.IsSuccess);
        Assert.Contains("    return 7", fuzzy.ModifiedSource, StringComparison.Ordinal);
    }

    [Fact]
    public void FuzzyMatchingStillRequiresMatchingIndentation()
    {
        string response = "<<<<<<< SEARCH\nreturn x\n=======\nreturn 7\n>>>>>>> REPLACE\n";
        var options = new ProgramEvolutionOptions();
        options.Diff.FuzzyWhitespace = true;

        ProgramDiffApplyResult result = ProgramDiff.ApplyResponse(Source, response, options);
        Assert.False(result.IsSuccess);
    }

    [Fact]
    public void OutOfBlockEditsAreRejectedWhenEvolveBlocksAreEnforced()
    {
        string fenced =
            "import math\n" +
            "# EVOLVE-BLOCK-START\n" +
            "def solve(x):\n" +
            "    return x\n" +
            "# EVOLVE-BLOCK-END\n" +
            "print(solve(1))\n";

        var options = new ProgramEvolutionOptions
        {
            Language = ProgramLanguage.Python,
            EnforceEvolveBlocks = true
        };

        string outside = "<<<<<<< SEARCH\nimport math\n=======\nimport cmath\n>>>>>>> REPLACE\n";
        ProgramDiffApplyResult rejected = ProgramDiff.ApplyResponse(fenced, outside, options);
        Assert.False(rejected.IsSuccess);
        Assert.Equal(0, rejected.AppliedCount);
        Assert.Contains(rejected.Failures, failure => failure.Reason == ProgramDiffFailureReason.OutsideEvolveBlock);
        Assert.Equal(fenced, rejected.ModifiedSource);

        string inside = "<<<<<<< SEARCH\n    return x\n=======\n    return x * 3\n>>>>>>> REPLACE\n";
        ProgramDiffApplyResult accepted = ProgramDiff.ApplyResponse(fenced, inside, options);
        Assert.True(accepted.IsSuccess);
        Assert.Contains("return x * 3", accepted.ModifiedSource, StringComparison.Ordinal);
    }

    [Fact]
    public void FailuresConvertToBoundedRedactedDiagnostics()
    {
        string response = "<<<<<<< SEARCH\n" + new string('z', 5000) + "\n=======\nnew\n>>>>>>> REPLACE\n";
        ProgramDiffApplyResult result = ProgramDiff.ApplyResponse(Source, response);

        ProgramDiffFailure failure = Assert.Single(
            result.Failures, item => item.Reason == ProgramDiffFailureReason.SearchTextNotFound);
        Assert.True(failure.SearchExcerpt.Length <= 240);
        EvolutionDiagnostic diagnostic = failure.ToDiagnostic();
        Assert.Equal("program_diff_searchtextnotfound", diagnostic.Code);
        Assert.True(diagnostic.IsRedacted);
        Assert.True(diagnostic.Message.Length <= 4096);
    }

    [Fact]
    public void ControlCharactersAreSanitizedOutOfExcerpts()
    {
        string response = "<<<<<<< SEARCH\nmissing\u0007value\n=======\nnew\n>>>>>>> REPLACE\n";
        ProgramDiffApplyResult result = ProgramDiff.ApplyResponse(Source, response);
        ProgramDiffFailure failure = result.Failures[0];
        Assert.DoesNotContain("\u0007", failure.SearchExcerpt, StringComparison.Ordinal);
        Assert.Contains("missing\u00B7value", failure.SearchExcerpt, StringComparison.Ordinal);
    }

    [Fact]
    public void SummaryMatchesTheReferenceLayout()
    {
        ProgramDiffParseResult parsed = ProgramDiff.Parse(Response);
        Assert.Equal("Change 1: 'return x' to 'return x * 2'", ProgramDiff.FormatSummary(parsed.Blocks));

        ProgramDiffParseResult multiline = ProgramDiff.Parse(
            "<<<<<<< SEARCH\na\nb\n=======\nc\nd\n>>>>>>> REPLACE\n");
        Assert.Equal("Change 1: Replace:\n  a\n  b\nwith:\n  c\n  d", ProgramDiff.FormatSummary(multiline.Blocks));
    }

    [Fact]
    public void UnifiedDiffRendersTheChangedHunk()
    {
        ProgramDiffApplyResult result = ProgramDiff.ApplyResponse(Source, Response);
        string diff = result.UnifiedDiff;

        Assert.StartsWith("@@ -1,4 +1,4 @@\n", diff, StringComparison.Ordinal);
        Assert.Contains("-    return x\n", diff, StringComparison.Ordinal);
        Assert.Contains("+    return x * 2\n", diff, StringComparison.Ordinal);
        Assert.Contains(" def solve(x):\n", diff, StringComparison.Ordinal);
    }

    [Fact]
    public void UnifiedDiffIsEmptyWhenNothingChanged() =>
        Assert.Equal(string.Empty, ProgramDiff.CreateUnifiedDiff(Source, Source));

    [Fact]
    public void UnifiedDiffHandlesLargeRewritesWithoutQuadraticMemory()
    {
        var left = new System.Text.StringBuilder();
        var right = new System.Text.StringBuilder();
        for (int index = 0; index < 1200; index++)
        {
            left.Append("old ").Append(index).Append('\n');
            right.Append("new ").Append(index).Append('\n');
        }

        string diff = ProgramDiff.CreateUnifiedDiff(left.ToString(), right.ToString());
        Assert.Contains("@@ -1,1200 +1,1200 @@", diff, StringComparison.Ordinal);
        Assert.Contains("-old 0\n", diff, StringComparison.Ordinal);
        Assert.Contains("+new 1199\n", diff, StringComparison.Ordinal);
    }

    [Fact]
    public void ApplyRejectsAnEmptyBlockList()
    {
        ProgramDiffApplyResult result = ProgramDiff.Apply(Source, Array.Empty<ProgramDiffBlock>());
        Assert.False(result.IsSuccess);
        Assert.Equal(ProgramDiffFailureReason.NoBlocksFound, result.Failures[0].Reason);
    }

    [Fact]
    public void HandMadeBlockWithEmptySearchIsRejectedAtApplyTime()
    {
        ProgramDiffApplyResult result = ProgramDiff.Apply(Source, new[] { new ProgramDiffBlock(string.Empty, "x") });
        Assert.False(result.IsSuccess);
        Assert.Equal(ProgramDiffFailureReason.EmptySearchText, result.Failures[0].Reason);
    }

    [Fact]
    public void DiffOptionsValidateMarkers()
    {
        Assert.Throws<ArgumentException>(() => new ProgramDiffOptions { SearchMarker = " " }.Validate());
        Assert.Throws<ArgumentException>(() => new ProgramDiffOptions { DividerMarker = "<<<<<<< SEARCH" }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new ProgramDiffOptions { MaxBlocks = 0 }.Validate());

        var clone = new ProgramDiffOptions { FuzzyWhitespace = true, MaxBlocks = 7 }.Clone();
        Assert.True(clone.FuzzyWhitespace);
        Assert.Equal(7, clone.MaxBlocks);
    }

    [Fact]
    public void EvolutionOptionsResolveMarkersFromLanguage()
    {
        var options = new ProgramEvolutionOptions { Language = ProgramLanguage.CSharp };
        Assert.Equal(EvolveBlockMarkers.Slash, options.ResolveEvolveBlockMarkers());

        options.EvolveBlockStartMarker = "/* START */";
        Assert.Throws<ArgumentException>(() => options.ResolveEvolveBlockMarkers());

        options.EvolveBlockEndMarker = "/* END */";
        Assert.Equal(new EvolveBlockMarkers("/* START */", "/* END */"), options.ResolveEvolveBlockMarkers());

        Assert.Throws<ArgumentOutOfRangeException>(() => new ProgramEvolutionOptions { MaxProgramChars = 0 }.Validate());
    }
}
