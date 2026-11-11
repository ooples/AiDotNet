using System.Collections.Generic;
using AiDotNet.Serving.Models.OpenAi;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Serving.Models.Anthropic;

/// <summary>Anthropic Messages API (<c>POST /v1/messages</c>) and token-counting (<c>POST /v1/messages/count_tokens</c>) request payload.</summary>
public sealed class AnthropicMessageRequest
{
    [JsonProperty("model")] public string Model { get; set; } = string.Empty;
    [JsonProperty("messages")] public List<AnthropicMessage> Messages { get; set; } = new();
    [JsonProperty("system")] public JToken? System { get; set; }
    [JsonProperty("max_tokens")] public int? MaxTokens { get; set; }
    [JsonProperty("temperature")] public double? Temperature { get; set; }
    [JsonProperty("top_p")] public double? TopP { get; set; }
    [JsonProperty("top_k")] public int? TopK { get; set; }
    [JsonProperty("stream")] public bool Stream { get; set; }
    [JsonProperty("stop_sequences")] public List<string>? StopSequences { get; set; }

    public int ResolveMaxTokens(int fallback) => MaxTokens ?? fallback;
    public string SystemText() => OpenAiJson.ContentText(System);
    public IReadOnlyList<string> ResolveStop() => StopSequences ?? (IReadOnlyList<string>)global::System.Array.Empty<string>();
}

/// <summary>A single turn in an Anthropic Messages API request.</summary>
public sealed class AnthropicMessage
{
    [JsonProperty("role")] public string Role { get; set; } = "user";
    [JsonProperty("content")] public JToken? Content { get; set; }

    public string TextContent() => OpenAiJson.ContentText(Content);
}

/// <summary>Content block in an Anthropic message response.</summary>
public sealed class AnthropicContentBlock
{
    [JsonProperty("type")] public string Type { get; set; } = "text";
    [JsonProperty("text")] public string Text { get; set; } = string.Empty;
}

/// <summary>Token usage accounting for Anthropic Messages API.</summary>
public sealed class AnthropicUsage
{
    [JsonProperty("input_tokens")] public int InputTokens { get; set; }
    [JsonProperty("output_tokens")] public int OutputTokens { get; set; }
}

/// <summary>Non-streaming response for <c>POST /v1/messages</c>.</summary>
public sealed class AnthropicMessageResponse
{
    [JsonProperty("id")] public string Id { get; set; } = string.Empty;
    [JsonProperty("type")] public string Type { get; set; } = "message";
    [JsonProperty("role")] public string Role { get; set; } = "assistant";
    [JsonProperty("model")] public string Model { get; set; } = string.Empty;
    [JsonProperty("content")] public List<AnthropicContentBlock> Content { get; set; } = new();
    [JsonProperty("stop_reason")] public string? StopReason { get; set; }
    [JsonProperty("stop_sequence")] public string? StopSequence { get; set; }
    [JsonProperty("usage")] public AnthropicUsage Usage { get; set; } = new();
}

/// <summary>Response for <c>POST /v1/messages/count_tokens</c>.</summary>
public sealed class AnthropicCountTokensResponse
{
    [JsonProperty("input_tokens")] public int InputTokens { get; set; }
}
