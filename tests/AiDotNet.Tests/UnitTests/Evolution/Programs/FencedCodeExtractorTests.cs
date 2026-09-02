using AiDotNet.Enums;
using AiDotNet.Evolution.Programs;
using AiDotNet.ProgramSynthesis.Enums;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class FencedCodeExtractorTests
{
    [Fact]
    public void LabelledFenceIsPreferredOverProse()
    {
        string response = "Here you go:\n```python\nprint(1)\n```\nHope that helps.";
        FencedCodeExtractionResult result = FencedCodeExtractor.Extract(response, ProgramLanguage.Python);

        Assert.Equal(FencedCodeSelectionSource.LanguageLabeledFence, result.SelectionSource);
        Assert.True(result.IsConfident);
        Assert.Equal("print(1)", result.Code);
        Assert.Single(result.Blocks);
        Assert.Equal(ProgramLanguage.Python, result.Blocks[0].Language);
    }

    [Fact]
    public void LabelAliasesResolveToTheRequestedLanguage()
    {
        Assert.Equal("print(1)", FencedCodeExtractor.Extract("```py\nprint(1)\n```", ProgramLanguage.Python).Code);
        Assert.Equal("var x = 1;", FencedCodeExtractor.Extract("```c#\nvar x = 1;\n```", ProgramLanguage.CSharp).Code);
        Assert.Equal("fn main() {}", FencedCodeExtractor.Extract("```rs\nfn main() {}\n```", ProgramLanguage.Rust).Code);
    }

    [Fact]
    public void LongestMatchingFenceWinsRatherThanTheFirst()
    {
        string response =
            "For example:\n```python\nx = 1\n```\n" +
            "And the real answer:\n```python\ndef solve():\n    return 42\n```\n";

        FencedCodeExtractionResult result = FencedCodeExtractor.Extract(response, ProgramLanguage.Python);
        Assert.Equal(FencedCodeSelectionSource.LanguageLabeledFence, result.SelectionSource);
        Assert.Contains("def solve", result.Code, StringComparison.Ordinal);
        Assert.Equal(2, result.Blocks.Count);
    }

    [Fact]
    public void UnlabelledFenceIsTheSecondRungAndIsReported()
    {
        FencedCodeExtractionResult result = FencedCodeExtractor.Extract("```\nprint(1)\n```", ProgramLanguage.Python);

        Assert.Equal(FencedCodeSelectionSource.UnlabeledFence, result.SelectionSource);
        Assert.False(result.IsConfident);
        Assert.Equal("print(1)", result.Code);
        Assert.Contains(result.Diagnostics, note => note.IndexOf("python", StringComparison.Ordinal) >= 0);
    }

    [Fact]
    public void MismatchedLabelIsTheThirdRungAndIsReported()
    {
        FencedCodeExtractionResult result = FencedCodeExtractor.Extract("```java\nint x = 1;\n```", ProgramLanguage.Python);

        Assert.Equal(FencedCodeSelectionSource.OtherLabeledFence, result.SelectionSource);
        Assert.Equal("int x = 1;", result.Code);
        Assert.Contains(result.Diagnostics, note => note.IndexOf("java", StringComparison.Ordinal) >= 0);
    }

    [Fact]
    public void RawFallbackIsReportedAndCanBeDisabled()
    {
        FencedCodeExtractionResult allowed = FencedCodeExtractor.Extract("print(1)", ProgramLanguage.Python);
        Assert.Equal(FencedCodeSelectionSource.RawResponse, allowed.SelectionSource);
        Assert.Equal("print(1)", allowed.Code);
        Assert.NotEmpty(allowed.Diagnostics);

        FencedCodeExtractionResult refused = FencedCodeExtractor.Extract("print(1)", ProgramLanguage.Python, allowRawFallback: false);
        Assert.Equal(FencedCodeSelectionSource.None, refused.SelectionSource);
        Assert.False(refused.HasCode);
    }

    [Fact]
    public void NestedFencesSurviveALongerOuterFence()
    {
        string response =
            "Explanation\n" +
            "````markdown\n" +
            "```python\n" +
            "print(1)\n" +
            "```\n" +
            "````\n";

        FencedCodeExtractionResult result = FencedCodeExtractor.Extract(response, ProgramLanguage.Python);

        Assert.Single(result.Blocks);
        Assert.Equal(4, result.Blocks[0].FenceLength);
        Assert.Equal("```python\nprint(1)\n```", result.Blocks[0].Content);
        Assert.Equal(FencedCodeSelectionSource.OtherLabeledFence, result.SelectionSource);
    }

    [Fact]
    public void UnterminatedFenceIsReportedAndItsContentIsKept()
    {
        FencedCodeExtractionResult result = FencedCodeExtractor.Extract("```python\nprint(1)\n", ProgramLanguage.Python);

        Assert.Equal("print(1)", result.Code);
        Assert.Contains(result.Diagnostics, note => note.IndexOf("never closed", StringComparison.Ordinal) >= 0);
    }

    [Fact]
    public void TildeFencesAreSupported()
    {
        FencedCodeExtractionResult result = FencedCodeExtractor.Extract("~~~python\nprint(1)\n~~~", ProgramLanguage.Python);
        Assert.Equal(FencedCodeSelectionSource.LanguageLabeledFence, result.SelectionSource);
        Assert.Equal("print(1)", result.Code);
    }

    [Fact]
    public void IndentedFenceStripsOnlyTheFenceIndentation()
    {
        string response = "- item\n  ```python\n  if x:\n      return 1\n  ```\n";
        FencedCodeExtractionResult result = FencedCodeExtractor.Extract(response, ProgramLanguage.Python);
        Assert.Equal("if x:\n    return 1", result.Code);
    }

    [Fact]
    public void LeadingBlankLinesAreTrimmedButIndentationIsKept()
    {
        string response = "```python\n\n\n    if x:\n        return 1\n\n```";
        FencedCodeExtractionResult result = FencedCodeExtractor.Extract(response, ProgramLanguage.Python);
        Assert.Equal("    if x:\n        return 1", result.Code);
    }

    [Fact]
    public void CarriageReturnResponsesNormalizeToLineFeeds()
    {
        FencedCodeExtractionResult result = FencedCodeExtractor.Extract(
            "```python\r\nprint(1)\r\nprint(2)\r\n```\r\n", ProgramLanguage.Python);

        Assert.Equal("print(1)\nprint(2)", result.Code);
    }

    [Fact]
    public void EmptyResponseYieldsNoCode()
    {
        FencedCodeExtractionResult result = FencedCodeExtractor.Extract("   \n\n", ProgramLanguage.Python);
        Assert.Equal(FencedCodeSelectionSource.None, result.SelectionSource);
        Assert.False(result.HasCode);
        Assert.Empty(result.Blocks);
    }

    [Fact]
    public void EmptyFencesAreIgnored()
    {
        FencedCodeExtractionResult result = FencedCodeExtractor.Extract("```python\n```\n", ProgramLanguage.Python, allowRawFallback: false);
        Assert.Empty(result.Blocks);
        Assert.False(result.HasCode);
    }

    [Fact]
    public void ExtractBlocksReturnsEveryFence()
    {
        IReadOnlyList<FencedCodeBlock> blocks = FencedCodeExtractor.ExtractBlocks(
            "```python\na\n```\ntext\n```\nb\n```\n");

        Assert.Equal(2, blocks.Count);
        Assert.Equal("python", blocks[0].Label);
        Assert.True(blocks[1].IsUnlabeled);
        Assert.Equal(1, blocks[0].StartLine);
        Assert.Equal(5, blocks[1].StartLine);
    }

    [Fact]
    public void GenericLanguagePrefersUnlabelledFences()
    {
        FencedCodeExtractionResult result = FencedCodeExtractor.Extract(
            "```java\nint x;\n```\n```\nplain text block\n```\n", ProgramLanguage.Generic);

        Assert.Equal(FencedCodeSelectionSource.UnlabeledFence, result.SelectionSource);
        Assert.Equal("plain text block", result.Code);
    }

    [Fact]
    public void NullResponseIsRejected()
    {
#pragma warning disable CS8600, CS8625
        Assert.Throws<ArgumentNullException>(() => FencedCodeExtractor.Extract(null, ProgramLanguage.Python));
        Assert.Throws<ArgumentNullException>(() => FencedCodeExtractor.ExtractBlocks(null));
#pragma warning restore CS8600, CS8625
    }
}
