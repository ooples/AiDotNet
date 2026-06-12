using System.Net.Http;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using AiDotNet.Agentic.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

// Disambiguate from the legacy AiDotNet.PromptEngineering.Templates.ChatMessage (global using).
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// An <see cref="IChatClient{T}"/> for Cohere's Chat API, whose bespoke wire format splits a turn into the
/// latest <c>message</c>, a <c>chat_history</c> of prior turns, and a <c>preamble</c> for system instructions.
/// This connector maps the unified agentic types to and from that shape, including tool declarations and
/// tool-call parsing.
/// </summary>
/// <typeparam name="T">The numeric type shared across the agent stack.</typeparam>
/// <remarks>
/// <para>
/// Streaming is a single-shot fallback (complete answer as one update). Tool calling round-trips in Cohere's
/// native format: assistant tool-call turns carry <c>tool_calls</c> in the history, and tool results are sent
/// as <c>tool_results</c> (promoted to the live request leg when they are the latest turn).
/// </para>
/// <para><b>For Beginners:</b> Same agent code, pointed at Cohere. The class rearranges the conversation into
/// the layout Cohere expects (most recent message separate from the history) and translates the reply back.
/// </para>
/// </remarks>
public sealed class CohereChatClient<T> : ChatClientBase<T>
{
    private readonly string _apiKey;
    private readonly string _endpoint;

    /// <summary>
    /// Initializes a new Cohere client.
    /// </summary>
    /// <param name="apiKey">The Cohere API key.</param>
    /// <param name="modelName">The model id (default <c>command-r-plus</c>).</param>
    /// <param name="endpoint">The chat endpoint. <c>null</c> uses the public Cohere Chat API.</param>
    /// <param name="httpClient">Optional HTTP client (for testing or custom handlers).</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="apiKey"/> or <paramref name="modelName"/> is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="apiKey"/> or <paramref name="modelName"/> is empty/whitespace.</exception>
    public CohereChatClient(string apiKey, string modelName = "command-r-plus", string? endpoint = null, HttpClient? httpClient = null)
        : base(httpClient)
    {
        ValidateApiKey(apiKey);
        Guard.NotNullOrWhiteSpace(modelName);
        _apiKey = apiKey;
        _endpoint = endpoint ?? "https://api.cohere.com/v1/chat";
        ModelId = modelName;
    }

    /// <inheritdoc/>
    protected override async Task<ChatResponse> GetResponseCoreAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions options,
        CancellationToken cancellationToken)
    {
        var payload = BuildRequest(messages, options);
        using var request = new HttpRequestMessage(HttpMethod.Post, _endpoint)
        {
            Content = new StringContent(payload.ToString(Formatting.None), Encoding.UTF8, "application/json"),
        };
        request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);

        using var response = await HttpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        var body = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        var root = JObject.Parse(body);

        var contents = new List<AiContent>();
        var text = (string?)root["text"];
        if (text is not null && text.Length > 0)
        {
            contents.Add(new TextContent(text));
        }

        if (root["tool_calls"] is JArray toolCalls)
        {
            var index = 0;
            foreach (var call in toolCalls)
            {
                var name = (string?)call["name"];
                if (name is not null && name.Trim().Length > 0)
                {
                    var parameters = call["parameters"] as JObject ?? new JObject();
                    contents.Add(new ToolCallContent($"cohere-call-{index}", name, parameters.ToString(Formatting.None)));
                    index++;
                }
            }
        }

        if (contents.Count == 0)
        {
            contents.Add(new TextContent(string.Empty));
        }

        var finishReason = MapFinishReason((string?)root["finish_reason"], contents);
        var usage = ParseUsage(root["meta"]?["tokens"] as JObject);
        return new ChatResponse(ChatMessage.Assistant(contents), finishReason, usage, ModelId);
    }

    /// <inheritdoc/>
    protected override async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseCoreAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions options,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var response = await GetResponseCoreAsync(messages, options, cancellationToken).ConfigureAwait(false);
        yield return new ChatResponseUpdate(role: ChatRole.Assistant);
        if (response.Message.Text.Length > 0)
        {
            yield return ChatResponseUpdate.ForText(response.Message.Text);
        }

        yield return ChatResponseUpdate.ForFinish(response.FinishReason, response.Usage);
    }

    private JObject BuildRequest(IReadOnlyList<ChatMessage> messages, ChatOptions options)
    {
        var preamble = new StringBuilder();
        var history = new JArray();
        string message = string.Empty;
        JArray? liveToolResults = null;

        // Cohere's tool_results entries are keyed by the originating call
        // (name + parameters); the unified model keys results by call id, so
        // record each assistant tool call as we walk the conversation.
        var callIdToCall = new Dictionary<string, ToolCallContent>(StringComparer.Ordinal);

        // The "live" leg of the request is the last non-system turn: a user
        // turn becomes `message`; a tool-result turn becomes top-level
        // `tool_results` (the second leg of Cohere tool calling).
        var lastLiveIndex = LastNonSystemIndex(messages);

        for (var i = 0; i < messages.Count; i++)
        {
            var current = messages[i];
            if (current.Role == ChatRole.System)
            {
                if (preamble.Length > 0)
                {
                    preamble.Append('\n');
                }

                preamble.Append(current.Text);
                continue;
            }

            if (current.Role == ChatRole.Assistant)
            {
                foreach (var call in current.ToolCalls)
                {
                    callIdToCall[call.CallId] = call;
                }
            }

            if (i == lastLiveIndex)
            {
                if (current.Role == ChatRole.Tool)
                {
                    liveToolResults = BuildToolResults(current, callIdToCall);
                    continue;
                }

                if (current.Role == ChatRole.User)
                {
                    message = current.Text;
                    continue;
                }
            }

            history.Add(BuildHistoryEntry(current, callIdToCall));
        }

        if (message.Length == 0 && liveToolResults is null && messages.Count > 0)
        {
            message = messages[messages.Count - 1].Text;
        }

        var payload = new JObject
        {
            ["model"] = ModelId,
            ["message"] = message,
        };

        if (liveToolResults is { Count: > 0 })
        {
            payload["tool_results"] = liveToolResults;
        }

        if (history.Count > 0)
        {
            payload["chat_history"] = history;
        }

        if (preamble.Length > 0)
        {
            payload["preamble"] = preamble.ToString();
        }

        if (options.Temperature is { } temperature)
        {
            payload["temperature"] = temperature;
        }

        if (options.MaxOutputTokens is { } maxTokens)
        {
            payload["max_tokens"] = maxTokens;
        }

        if (options.TopP is { } topP)
        {
            payload["p"] = topP;
        }

        if (options.TopK is { } topK)
        {
            payload["k"] = topK;
        }

        if (options.Tools is { Count: > 0 } tools && options.ToolChoice != ToolChoiceMode.None)
        {
            payload["tools"] = BuildTools(tools);
        }

        return payload;
    }

    private static int LastNonSystemIndex(IReadOnlyList<ChatMessage> messages)
    {
        for (var i = messages.Count - 1; i >= 0; i--)
        {
            if (messages[i].Role != ChatRole.System)
            {
                return i;
            }
        }

        return -1;
    }

    // History entry preserving native structure: assistant tool-call turns
    // keep their tool_calls, and tool-result turns are TOOL entries — never
    // flattened to plain USER text.
    private static JObject BuildHistoryEntry(ChatMessage message, Dictionary<string, ToolCallContent> callIdToCall)
    {
        if (message.Role == ChatRole.Tool)
        {
            return new JObject
            {
                ["role"] = "TOOL",
                ["tool_results"] = BuildToolResults(message, callIdToCall),
            };
        }

        var entry = new JObject
        {
            ["role"] = message.Role == ChatRole.Assistant ? "CHATBOT" : "USER",
            ["message"] = message.Text,
        };

        if (message.Role == ChatRole.Assistant)
        {
            var toolCalls = message.ToolCalls;
            if (toolCalls.Count > 0)
            {
                var calls = new JArray();
                foreach (var call in toolCalls)
                {
                    calls.Add(new JObject
                    {
                        ["name"] = call.ToolName,
                        ["parameters"] = ParseParametersObject(call.ArgumentsJson),
                    });
                }

                entry["tool_calls"] = calls;
            }
        }

        return entry;
    }

    private static JArray BuildToolResults(ChatMessage message, Dictionary<string, ToolCallContent> callIdToCall)
    {
        var results = new JArray();
        foreach (var content in message.Contents)
        {
            if (content is not ToolResultContent result)
            {
                continue;
            }

            var call = callIdToCall.TryGetValue(result.CallId, out var origin) ? origin : null;
            results.Add(new JObject
            {
                ["call"] = new JObject
                {
                    ["name"] = call?.ToolName ?? result.CallId,
                    ["parameters"] = call is null ? new JObject() : ParseParametersObject(call.ArgumentsJson),
                },
                ["outputs"] = new JArray { BuildOutputObject(result) },
            });
        }

        return results;
    }

    private static JObject BuildOutputObject(ToolResultContent result)
    {
        JObject output;
        try
        {
            output = JToken.Parse(result.Result) is JObject resultObject
                ? new JObject(resultObject)
                : new JObject { ["text"] = result.Result };
        }
        catch (JsonException)
        {
            // Plain-text tool output — Cohere outputs entries must be objects.
            output = new JObject { ["text"] = result.Result };
        }

        if (result.IsError)
        {
            output["error"] = true;
        }

        return output;
    }

    private static JObject ParseParametersObject(string argumentsJson)
    {
        if (argumentsJson.Trim().Length == 0)
        {
            return new JObject();
        }

        try
        {
            return JObject.Parse(argumentsJson);
        }
        catch (JsonException)
        {
            // Preserve a non-object payload verbatim under a single key.
            return new JObject { ["value"] = argumentsJson };
        }
    }

    private static JArray BuildTools(IReadOnlyList<AiToolDefinition> tools)
    {
        var array = new JArray();
        foreach (var tool in tools)
        {
            array.Add(new JObject
            {
                ["name"] = tool.Name,
                ["description"] = tool.Description,
                ["parameter_definitions"] = BuildParameterDefinitions(tool.ParametersSchema),
            });
        }

        return array;
    }

    private static JObject BuildParameterDefinitions(JObject schema)
    {
        var definitions = new JObject();
        if (schema["properties"] is not JObject properties)
        {
            return definitions;
        }

        var required = (schema["required"] as JArray)?.Select(t => (string?)t).ToList() ?? new List<string?>();
        foreach (var property in properties)
        {
            if (property.Value is not JObject propSchema)
            {
                continue;
            }

            definitions[property.Key] = new JObject
            {
                ["description"] = (string?)propSchema["description"] ?? string.Empty,
                ["type"] = MapType((string?)propSchema["type"]),
                ["required"] = required.Contains(property.Key),
            };
        }

        return definitions;
    }

    private static string MapType(string? jsonSchemaType) => jsonSchemaType switch
    {
        "string" => "str",
        "integer" => "int",
        "number" => "float",
        "boolean" => "bool",
        "array" => "list",
        "object" => "dict",
        _ => "str",
    };

    private static ChatFinishReason MapFinishReason(string? reason, List<AiContent> contents)
    {
        if (contents.OfType<ToolCallContent>().Any())
        {
            return ChatFinishReason.ToolCalls;
        }

        return reason switch
        {
            "COMPLETE" => ChatFinishReason.Stop,
            "MAX_TOKENS" => ChatFinishReason.Length,
            null => ChatFinishReason.Stop,
            _ => ChatFinishReason.Unknown,
        };
    }

    private static ChatUsage? ParseUsage(JObject? tokens)
    {
        if (tokens is null)
        {
            return null;
        }

        var input = (int?)tokens["input_tokens"] ?? 0;
        var output = (int?)tokens["output_tokens"] ?? 0;
        return new ChatUsage(input, output);
    }
}
