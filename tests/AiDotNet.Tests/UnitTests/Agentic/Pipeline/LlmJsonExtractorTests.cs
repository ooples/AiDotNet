using System;
using AiDotNet.Agentic.Pipeline;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNetTests.UnitTests.Agentic.Pipeline;

public sealed class LlmJsonExtractorTests
{
    [Fact]
    public void APlainJsonReplyIsParsed()
    {
        Assert.True(LlmJsonExtractor.TryExtract("{\"readability\": 0.8}", out JObject json));
        Assert.Equal(0.8, json["readability"]?.Value<double>());
    }

    [Fact]
    public void AJsonLabelledFenceIsPreferred()
    {
        const string Reply = "Here is my judgement.\n```json\n{\"score\": 0.5}\n```\nHope that helps.";
        Assert.True(LlmJsonExtractor.TryExtract(Reply, out JObject json));
        Assert.Equal(0.5, json["score"]?.Value<double>());
    }

    [Fact]
    public void AnUnlabelledFenceIsAccepted()
    {
        const string Reply = "Result:\n```\n{\"score\": 0.25}\n```";
        Assert.True(LlmJsonExtractor.TryExtract(Reply, out JObject json));
        Assert.Equal(0.25, json["score"]?.Value<double>());
    }

    [Fact]
    public void ProseAroundABareObjectIsIgnored()
    {
        const string Reply = "I judge it as follows: {\"score\": 0.75} and that is my answer.";
        Assert.True(LlmJsonExtractor.TryExtract(Reply, out JObject json));
        Assert.Equal(0.75, json["score"]?.Value<double>());
    }

    [Fact]
    public void BracesInsideStringValuesDoNotEndTheObject()
    {
        // A regular expression that stops at the first '}' truncates this, and one
        // that runs to the last '}' swallows the trailing prose.
        const string Reply = "Answer: {\"reason\": \"use {n} buckets\", \"score\": 0.9} done.";
        Assert.True(LlmJsonExtractor.TryExtract(Reply, out JObject json));
        Assert.Equal("use {n} buckets", json["reason"]?.Value<string>());
        Assert.Equal(0.9, json["score"]?.Value<double>());
    }

    [Fact]
    public void EscapedQuotesInsideValuesAreHandled()
    {
        const string Reply = "{\"reason\": \"it said \\\"no\\\" twice\", \"score\": 0.1}";
        Assert.True(LlmJsonExtractor.TryExtract(Reply, out JObject json));
        Assert.Equal("it said \"no\" twice", json["reason"]?.Value<string>());
    }

    [Fact]
    public void NestedObjectsAreKeptWhole()
    {
        const string Reply = "prefix {\"outer\": {\"inner\": 1}, \"score\": 0.4} suffix";
        Assert.True(LlmJsonExtractor.TryExtract(Reply, out JObject json));
        Assert.Equal(1, json["outer"]?["inner"]?.Value<int>());
    }

    [Fact]
    public void AReplyWithNoJsonReportsAMissRatherThanThrowing()
    {
        Assert.False(LlmJsonExtractor.TryExtract("I would rather not answer.", out JObject json));
        Assert.Empty(json);
        Assert.Null(LlmJsonExtractor.Extract("I would rather not answer."));
    }

    [Fact]
    public void AJsonArrayIsNotMistakenForAnObject()
    {
        Assert.False(LlmJsonExtractor.TryExtract("[1, 2, 3]", out _));
    }

    [Fact]
    public void MalformedJsonInsideAFenceFallsThroughToTheBareObject()
    {
        const string Reply = "```json\n{\"score\": }\n```\nActually: {\"score\": 0.6}";
        Assert.True(LlmJsonExtractor.TryExtract(Reply, out JObject json));
        Assert.Equal(0.6, json["score"]?.Value<double>());
    }

    [Fact]
    public void NumbersMayArriveQuoted()
    {
        Assert.True(LlmJsonExtractor.TryExtract("{\"score\": \"0.42\"}", out JObject json));
        Assert.True(LlmJsonExtractor.TryReadNumber(json, "score", out double value));
        Assert.Equal(0.42, value);
    }

    [Fact]
    public void NonNumericAndNonFiniteFieldsAreReportedAsUnreadable()
    {
        Assert.True(LlmJsonExtractor.TryExtract("{\"score\": \"high\", \"other\": true}", out JObject json));
        Assert.False(LlmJsonExtractor.TryReadNumber(json, "score", out _));
        Assert.False(LlmJsonExtractor.TryReadNumber(json, "other", out _));
        Assert.False(LlmJsonExtractor.TryReadNumber(json, "absent", out _));
    }

    [Fact]
    public void AnEmptyReplyReportsAMiss()
    {
        Assert.False(LlmJsonExtractor.TryExtract(string.Empty, out _));
    }
}
