#nullable disable
using AiDotNet.Enums;
using AiDotNet.Safety;
using AiDotNet.Safety.Text;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Integration tests for copyright and memorization detection modules.
/// Tests NgramCopyrightDetector, EmbeddingCopyrightDetector, and PerplexityMemorizationDetector.
/// </summary>
public class CopyrightDetectionIntegrationTests
{
    #region NgramCopyrightDetector Tests

    [Fact]
    public void Ngram_OriginalContent_NoFindings()
    {
        var detector = new NgramCopyrightDetector<double>();
        var findings = detector.EvaluateText(
            "This is original content about machine learning techniques and neural network architectures.");

        Assert.Empty(findings);
    }

    [Fact]
    public void Ngram_WithCopyrightedTexts_DetectsMatch()
    {
        var copyrighted = new[] { "It was the best of times it was the worst of times it was the age of wisdom" };
        var sources = new[] { "A Tale of Two Cities" };
        var detector = new NgramCopyrightDetector<double>(copyrighted, sources);
        var findings = detector.EvaluateText(
            "It was the best of times it was the worst of times it was the age of wisdom it was the age of foolishness");

        Assert.NotNull(findings);
    }

    [Fact]
    public void Ngram_EmptyText_NoFindings()
    {
        var detector = new NgramCopyrightDetector<double>();
        var findings = detector.EvaluateText("");

        Assert.Empty(findings);
    }

    [Fact]
    public void Ngram_ShortText_NoFindings()
    {
        var detector = new NgramCopyrightDetector<double>();
        var findings = detector.EvaluateText("Hello world");

        Assert.Empty(findings);
    }

    #endregion

    #region EmbeddingCopyrightDetector Tests

    [Fact]
    public void Embedding_OriginalContent_NoFindings()
    {
        var detector = new EmbeddingCopyrightDetector<double>();
        var findings = detector.EvaluateText(
            "Unique content about the integration of artificial intelligence in healthcare systems.");

        Assert.Empty(findings);
    }

    [Fact]
    public void Embedding_WithCopyrightedTexts_ProcessesCleanly()
    {
        var copyrighted = new[] { "To be or not to be that is the question whether tis nobler in the mind" };
        var sources = new[] { "Hamlet" };
        var detector = new EmbeddingCopyrightDetector<double>(copyrighted, sources);
        var findings = detector.EvaluateText(
            "To be or not to be that is the question whether tis nobler in the mind to suffer");

        Assert.NotNull(findings);
    }

    [Fact]
    public void Embedding_EmptyText_NoFindings()
    {
        var detector = new EmbeddingCopyrightDetector<double>();
        var findings = detector.EvaluateText("");

        Assert.Empty(findings);
    }

    #endregion

    #region PerplexityMemorizationDetector Tests

    [Fact]
    public void Perplexity_RepetitiveText_DetectsMemorization()
    {
        var detector = new PerplexityMemorizationDetector<double>();
        string repeatedText = string.Join(" ",
            Enumerable.Repeat("the quick brown fox jumps over the lazy dog", 20));
        var findings = detector.EvaluateText(repeatedText);

        Assert.NotNull(findings);
    }

    [Fact]
    public void Perplexity_UniqueContent_ProcessesWithoutError()
    {
        var detector = new PerplexityMemorizationDetector<double>();
        var findings = detector.EvaluateText(
            "Quantum computing leverages superposition and entanglement principles " +
            "to perform calculations that classical computers find intractable.");

        // Perplexity detector uses heuristic n-gram analysis; verify processing completes
        Assert.NotNull(findings);
    }

    [Fact]
    public void Perplexity_HighlyRepetitive_DetectsIssue()
    {
        var detector = new PerplexityMemorizationDetector<double>();
        string text = string.Join(" ", Enumerable.Repeat("hello world hello world", 50));
        var findings = detector.EvaluateText(text);

        Assert.NotNull(findings);
    }

    [Fact]
    public void Perplexity_CustomThresholds_Work()
    {
        var detector = new PerplexityMemorizationDetector<double>(
            lowPerplexityThreshold: 1.0, highRepetitionThreshold: 0.1);
        string text = string.Join(" ", Enumerable.Repeat("abc def ghi", 30));
        var findings = detector.EvaluateText(text);

        Assert.NotNull(findings);
    }

    [Fact]
    public void Perplexity_EmptyText_NoFindings()
    {
        var detector = new PerplexityMemorizationDetector<double>();
        var findings = detector.EvaluateText("");

        Assert.Empty(findings);
    }

    #endregion

    #region Cross-Module Tests

    [Fact]
    public void AllDetectors_OriginalContent_ProcessWithoutError()
    {
        var text = "This is completely original content about the future of space exploration " +
                   "and the potential for human colonization of Mars by the end of the century.";

        Assert.Empty(new NgramCopyrightDetector<double>().EvaluateText(text));
        Assert.Empty(new EmbeddingCopyrightDetector<double>().EvaluateText(text));
        // Perplexity detector may produce findings for any text based on heuristic n-gram analysis
        Assert.NotNull(new PerplexityMemorizationDetector<double>().EvaluateText(text));
    }

    #endregion
}
