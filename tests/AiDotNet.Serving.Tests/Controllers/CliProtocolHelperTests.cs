using AiDotNet.Serving.Controllers;
using AiDotNet.Serving.Models.OpenAi;
using AiDotNet.Serving.Services;
using Newtonsoft.Json.Linq;
using Xunit;

namespace AiDotNet.Serving.Tests.Controllers;

/// <summary>
/// Exact-output tests for the pure helpers behind the OpenAI and Anthropic CLI routes: content flattening,
/// role labelling, and stop-sequence matching (including the streaming hold-back).
/// </summary>
public sealed class CliProtocolHelperTests
{
    [Fact]
    public void ContentText_ParallelToolResults_AreLabelledAndSeparable()
    {
        var content = JArray.Parse("""
            [
              { "type": "tool_result", "tool_use_id": "toolu_1", "content": "file a" },
              { "type": "tool_result", "tool_use_id": "toolu_2", "content": [ { "type": "text", "text": "file b" } ] }
            ]
            """);

        Assert.Equal("[tool_result toolu_1: file a][tool_result toolu_2: file b]", OpenAiJson.ContentText(content));
    }

    [Fact]
    public void ContentText_ToolUse_RendersNameAndCompactInput()
    {
        var content = JArray.Parse("""
            [ { "type": "text", "text": "Reading. " }, { "type": "tool_use", "id": "toolu_1", "name": "Read", "input": { "path": "a.cs" } } ]
            """);

        Assert.Equal("Reading. [tool_use: Read({\"path\":\"a.cs\"})]", OpenAiJson.ContentText(content));
    }

    [Fact]
    public void ContentText_ToolResultWithoutId_UsesUnknown()
    {
        var content = JArray.Parse("""[ { "type": "tool_result", "content": "ok" } ]""");

        Assert.Equal("[tool_result unknown: ok]", OpenAiJson.ContentText(content));
    }

    [Fact]
    public void ChatTemplate_DeveloperRole_IsLabelledSystem()
    {
        string prompt = ChatTemplate.Render(new[] { ("developer", "Be concise."), ("user", "Hi") });

        Assert.Equal("System: Be concise.\nUser: Hi\nAssistant:", prompt);
    }

    [Theory]
    [InlineData("abcEND", 3, "END")]
    [InlineData("abSTOPcdEND", 2, "STOP")]
    [InlineData("abc", -1, null)]
    public void FindStop_ReturnsEarliestMatchAndItsStop(string text, int expectedIndex, string? expectedStop)
    {
        int index = StopSequenceMatcher.FindStop(text, new[] { "END", "STOP", "" }, out string? matched);

        Assert.Equal(expectedIndex, index);
        Assert.Equal(expectedStop, matched);
    }

    [Theory]
    [InlineData("helloEN", 2)]   // "EN" could still become "END"
    [InlineData("helloE", 1)]
    [InlineData("hello", 0)]
    [InlineData("helloEND", 0)]  // a completed stop is FindStop's job, not a pending prefix
    public void PendingPrefixLength_HoldsBackOnlyAPossibleStopStart(string text, int expected)
    {
        Assert.Equal(expected, StopSequenceMatcher.PendingPrefixLength(text, new[] { "END" }));
    }

    [Theory]
    [InlineData("claude-sonnet-4-5", true)]
    [InlineData("GPT-4o", true)]
    [InlineData("default", true)]
    [InlineData("o3-mini", true)]
    [InlineData("my-custom-model", false)]
    public void IsStandardCliAlias_RecognisesCliDefaults(string name, bool expected)
    {
        Assert.Equal(expected, CliModelAliasResolver.IsStandardCliAlias(name));
    }
}
