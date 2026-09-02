using System;
using System.Text;
using AiDotNet.Evolution.Prompts;
using Xunit;

namespace AiDotNetTests.UnitTests.Evolution.Programs;

public sealed class PromptTextRedactorTests
{
    private const char Escape = (char)27;
    private const char Bell = (char)7;
    private const char Tab = (char)9;
    private const char LineFeed = (char)10;
    private const char CarriageReturn = (char)13;

    [Fact]
    public void AnsiEscapeSequencesAreRemoved()
    {
        string text = Escape + "[31mfailed" + Escape + "[0m at line 4";
        Assert.Equal("failed at line 4", PromptTextRedactor.Redact(text));
    }

    [Fact]
    public void ControlCharactersOtherThanTabAndNewlineAreRemoved()
    {
        // A bell is dropped; a tab and a newline survive, and a carriage return is
        // normalized away so recorded text never depends on line endings.
        string text = "a" + Bell + "bc" + Tab + "d" + LineFeed + "e" + CarriageReturn + LineFeed;
        string expected = "abc" + Tab + "d" + LineFeed + "e" + LineFeed;
        Assert.Equal(expected, PromptTextRedactor.Redact(text));
    }

    [Theory]
    [InlineData("key is sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD")]
    [InlineData("Authorization: Bearer abcdefghijklmnopqrstuvwxyz012345")]
    [InlineData("password=hunter2")]
    [InlineData("api_key: abcd-1234-efgh")]
    public void CredentialShapedValuesAreRedacted(string text)
    {
        string redacted = PromptTextRedactor.Redact(text);
        Assert.Contains(PromptTextRedactor.RedactionMarker, redacted, StringComparison.Ordinal);
    }

    [Fact]
    public void LongOpaqueTokensAreRedactedButOrdinaryWordsAreNot()
    {
        string redacted = PromptTextRedactor.Redact("token " + new string('a', 48) + " end of a normal sentence");
        Assert.Contains(PromptTextRedactor.RedactionMarker, redacted, StringComparison.Ordinal);
        Assert.Contains("end of a normal sentence", redacted, StringComparison.Ordinal);
    }

    [Fact]
    public void OrdinaryProgramOutputSurvivesUntouched()
    {
        const string Output = "Traceback (most recent call last):\n  File \"solve.py\", line 12\nValueError: n must be positive";
        Assert.Equal(Output, PromptTextRedactor.Redact(Output));
    }

    [Fact]
    public void EmptyTextRedactsToEmpty()
    {
        Assert.Equal(string.Empty, PromptTextRedactor.Redact(string.Empty));
    }

    [Fact]
    public void RedactingAnAlreadyRedactedStringChangesNothingFurther()
    {
        // Artifacts pass through redaction on every prompt that quotes them, so the
        // operation has to be idempotent or the text drifts across iterations.
        const string Source = "password=hunter2 then ordinary output";
        string once = PromptTextRedactor.Redact(Source);
        Assert.Equal(once, PromptTextRedactor.Redact(once));
    }

    [Fact]
    public void ByteBoundingCutsToTheBudgetAndReportsTruncation()
    {
        string bounded = PromptTextRedactor.BoundToUtf8Bytes("abcdefghij", 4, out bool truncated);
        Assert.True(truncated);
        Assert.Equal("abcd", bounded);
    }

    [Fact]
    public void ByteBoundingLeavesTextThatFitsAlone()
    {
        string bounded = PromptTextRedactor.BoundToUtf8Bytes("abcd", 4, out bool truncated);
        Assert.False(truncated);
        Assert.Equal("abcd", bounded);
    }

    [Fact]
    public void ByteBoundingCountsBytesNotCharacters()
    {
        // Each of these is two UTF-8 bytes, so a six-byte budget holds three.
        const string Text = "ééééé";
        string bounded = PromptTextRedactor.BoundToUtf8Bytes(Text, 6, out bool truncated);
        Assert.True(truncated);
        Assert.Equal("ééé", bounded);
        Assert.Equal(6, new UTF8Encoding(false).GetByteCount(bounded));
    }

    [Fact]
    public void ByteBoundingNeverSplitsASurrogatePair()
    {
        // One astral character encodes to four UTF-8 bytes; a three-byte budget
        // must drop it whole rather than emit half a surrogate pair.
        const string Text = "\U0001F600tail";
        string bounded = PromptTextRedactor.BoundToUtf8Bytes(Text, 3, out bool truncated);
        Assert.True(truncated);
        Assert.Equal(string.Empty, bounded);

        string kept = PromptTextRedactor.BoundToUtf8Bytes(Text, 5, out _);
        Assert.Equal("\U0001F600t", kept);
    }

    [Fact]
    public void ZeroBudgetYieldsEmptyTextAndReportsTruncation()
    {
        string bounded = PromptTextRedactor.BoundToUtf8Bytes("abc", 0, out bool truncated);
        Assert.True(truncated);
        Assert.Equal(string.Empty, bounded);
    }

    [Fact]
    public void NegativeBudgetIsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => PromptTextRedactor.BoundToUtf8Bytes("abc", -1, out _));
    }

    [Fact]
    public void RedactAndBoundAppendsTheMarkerOnlyWhenItCut()
    {
        string cut = PromptTextRedactor.RedactAndBound("abcdefghij", 4, "[cut]", out bool truncated);
        Assert.True(truncated);
        Assert.Equal("abcd" + LineFeed + "[cut]", cut);

        string intact = PromptTextRedactor.RedactAndBound("abcd", 16, "[cut]", out bool wasCut);
        Assert.False(wasCut);
        Assert.Equal("abcd", intact);
    }

    [Fact]
    public void RedactAndBoundRedactsBeforeItMeasures()
    {
        const string Source = "password=hunter2 and then some more output text that follows";
        string result = PromptTextRedactor.RedactAndBound(Source, 4096, "[cut]", out _);
        Assert.DoesNotContain("hunter2", result, StringComparison.Ordinal);
    }
}
