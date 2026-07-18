using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Serving.Models.OpenAi;

// ============================ Requests ============================

/// <summary>OpenAI <c>/v1/chat/completions</c> request.</summary>
public sealed class ChatCompletionRequest
{
    [JsonProperty("model")] public string Model { get; set; } = string.Empty;
    [JsonProperty("messages")] public List<ChatMessage> Messages { get; set; } = new();
    [JsonProperty("max_tokens")] public int? MaxTokens { get; set; }
    [JsonProperty("max_completion_tokens")] public int? MaxCompletionTokens { get; set; }
    [JsonProperty("temperature")] public double? Temperature { get; set; }
    [JsonProperty("top_p")] public double? TopP { get; set; }
    [JsonProperty("top_k")] public int? TopK { get; set; }
    [JsonProperty("min_p")] public double? MinP { get; set; }
    [JsonProperty("stream")] public bool Stream { get; set; }
    [JsonProperty("stop")] public JToken? Stop { get; set; }
    [JsonProperty("n")] public int? N { get; set; }
    [JsonProperty("seed")] public int? Seed { get; set; }
    [JsonProperty("response_format")] public JToken? ResponseFormat { get; set; }
    [JsonProperty("logit_bias")] public JObject? LogitBias { get; set; }

    /// <summary>Resolves the effective max-new-tokens (max_tokens or max_completion_tokens, else fallback).</summary>
    public int ResolveMaxTokens(int fallback) => MaxTokens ?? MaxCompletionTokens ?? fallback;

    /// <summary>Normalized stop strings.</summary>
    public IReadOnlyList<string> ResolveStop() => OpenAiJson.StopList(Stop);
}

/// <summary>A chat message. <see cref="Content"/> may be a string or an array of content parts.</summary>
public sealed class ChatMessage
{
    [JsonProperty("role")] public string Role { get; set; } = "user";
    [JsonProperty("content")] public JToken? Content { get; set; }

    /// <summary>Flattens content (string or content-part array) to text.</summary>
    public string TextContent() => OpenAiJson.ContentText(Content);
}

/// <summary>OpenAI <c>/v1/completions</c> request.</summary>
public sealed class CompletionRequest
{
    [JsonProperty("model")] public string Model { get; set; } = string.Empty;
    [JsonProperty("prompt")] public JToken? Prompt { get; set; }
    [JsonProperty("max_tokens")] public int? MaxTokens { get; set; }
    [JsonProperty("temperature")] public double? Temperature { get; set; }
    [JsonProperty("top_p")] public double? TopP { get; set; }
    [JsonProperty("top_k")] public int? TopK { get; set; }
    [JsonProperty("min_p")] public double? MinP { get; set; }
    [JsonProperty("stream")] public bool Stream { get; set; }
    [JsonProperty("stop")] public JToken? Stop { get; set; }
    [JsonProperty("seed")] public int? Seed { get; set; }
    [JsonProperty("response_format")] public JToken? ResponseFormat { get; set; }
    [JsonProperty("logit_bias")] public JObject? LogitBias { get; set; }

    public int ResolveMaxTokens(int fallback) => MaxTokens ?? fallback;
    public string PromptText() => OpenAiJson.ContentText(Prompt);
    public IReadOnlyList<string> ResolveStop() => OpenAiJson.StopList(Stop);
}

// ============================ Responses ============================

/// <summary>Token accounting.</summary>
public sealed class Usage
{
    [JsonProperty("prompt_tokens")] public int PromptTokens { get; set; }
    [JsonProperty("completion_tokens")] public int CompletionTokens { get; set; }
    [JsonProperty("total_tokens")] public int TotalTokens { get; set; }
}

/// <summary>A role/content message in a response (non-streaming) or a streaming delta.</summary>
public sealed class ChatMessageOut
{
    [JsonProperty("role", NullValueHandling = NullValueHandling.Ignore)] public string? Role { get; set; }
    [JsonProperty("content")] public string? Content { get; set; }
}

/// <summary>A chat choice; carries <see cref="Message"/> (non-streaming) or <see cref="Delta"/> (streaming).</summary>
public sealed class ChatChoice
{
    [JsonProperty("index")] public int Index { get; set; }
    [JsonProperty("message", NullValueHandling = NullValueHandling.Ignore)] public ChatMessageOut? Message { get; set; }
    [JsonProperty("delta", NullValueHandling = NullValueHandling.Ignore)] public ChatMessageOut? Delta { get; set; }
    [JsonProperty("finish_reason")] public string? FinishReason { get; set; }
}

/// <summary>Non-streaming chat completion response.</summary>
public sealed class ChatCompletionResponse
{
    [JsonProperty("id")] public string Id { get; set; } = string.Empty;
    [JsonProperty("object")] public string Object { get; set; } = "chat.completion";
    [JsonProperty("created")] public long Created { get; set; }
    [JsonProperty("model")] public string Model { get; set; } = string.Empty;
    [JsonProperty("choices")] public List<ChatChoice> Choices { get; set; } = new();
    [JsonProperty("usage", NullValueHandling = NullValueHandling.Ignore)] public Usage? Usage { get; set; }
}

/// <summary>Streaming chat completion chunk (<c>chat.completion.chunk</c>).</summary>
public sealed class ChatCompletionChunk
{
    [JsonProperty("id")] public string Id { get; set; } = string.Empty;
    [JsonProperty("object")] public string Object { get; set; } = "chat.completion.chunk";
    [JsonProperty("created")] public long Created { get; set; }
    [JsonProperty("model")] public string Model { get; set; } = string.Empty;
    [JsonProperty("choices")] public List<ChatChoice> Choices { get; set; } = new();
}

/// <summary>A text-completion choice.</summary>
public sealed class CompletionChoice
{
    [JsonProperty("index")] public int Index { get; set; }
    [JsonProperty("text")] public string Text { get; set; } = string.Empty;
    [JsonProperty("finish_reason")] public string? FinishReason { get; set; }
}

/// <summary>Text completion response / streaming chunk (both use <c>text_completion</c>).</summary>
public sealed class CompletionResponse
{
    [JsonProperty("id")] public string Id { get; set; } = string.Empty;
    [JsonProperty("object")] public string Object { get; set; } = "text_completion";
    [JsonProperty("created")] public long Created { get; set; }
    [JsonProperty("model")] public string Model { get; set; } = string.Empty;
    [JsonProperty("choices")] public List<CompletionChoice> Choices { get; set; } = new();
    [JsonProperty("usage", NullValueHandling = NullValueHandling.Ignore)] public Usage? Usage { get; set; }
}

/// <summary>Response for <c>/v1/models</c>.</summary>
public sealed class ModelList
{
    [JsonProperty("object")] public string Object { get; set; } = "list";
    [JsonProperty("data")] public List<ModelCard> Data { get; set; } = new();
}

/// <summary>A single model entry.</summary>
public sealed class ModelCard
{
    [JsonProperty("id")] public string Id { get; set; } = string.Empty;
    [JsonProperty("object")] public string Object { get; set; } = "model";
    [JsonProperty("created")] public long Created { get; set; }
    [JsonProperty("owned_by")] public string OwnedBy { get; set; } = "aidotnet";
}

// ============================ Helpers ============================

internal static class OpenAiJson
{
    /// <summary>Flattens a string or a content-part array (<c>[{type:"text",text:"..."}]</c>) to text.</summary>
    public static string ContentText(JToken? t)
    {
        if (t == null) return string.Empty;
        if (t.Type == JTokenType.String) return t.Value<string>() ?? string.Empty;
        if (t.Type == JTokenType.Array)
        {
            var sb = new StringBuilder();
            foreach (var part in t)
            {
                var txt = part.Type == JTokenType.String ? part.Value<string>() : part["text"]?.Value<string>();
                if (!string.IsNullOrEmpty(txt)) sb.Append(txt);
            }
            return sb.ToString();
        }
        return t.ToString();
    }

    /// <summary>Normalizes an OpenAI <c>stop</c> field (string or array) to a list.</summary>
    public static IReadOnlyList<string> StopList(JToken? t)
    {
        var list = new List<string>();
        if (t == null) return list;
        if (t.Type == JTokenType.String)
        {
            var s = t.Value<string>();
            if (!string.IsNullOrEmpty(s)) list.Add(s!);
        }
        else if (t.Type == JTokenType.Array)
        {
            foreach (var e in t)
            {
                var s = e.Value<string>();
                if (!string.IsNullOrEmpty(s)) list.Add(s!);
            }
        }
        return list;
    }
}
