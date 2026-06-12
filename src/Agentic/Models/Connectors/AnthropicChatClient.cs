using System.IO;
using System.Net.Http;
using System.Runtime.CompilerServices;
using AiDotNet.Agentic.Models;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// An <see cref="IChatClient{T}"/> for Anthropic's Claude Messages API with native tool use, streaming,
/// and multimodal (image) input.
/// </summary>
/// <typeparam name="T">The numeric type used across the AiDotNet ecosystem.</typeparam>
/// <remarks>
/// <para>
/// Anthropic's wire format differs from OpenAI's: system instructions are a top-level field (not a
/// message), message content is an array of typed blocks (<c>text</c>, <c>image</c>, <c>tool_use</c>,
/// <c>tool_result</c>), tool results are carried on user-role messages, and <c>max_tokens</c> is required.
/// This connector maps the provider-neutral model onto that shape and back, translating Claude's
/// <c>stop_reason</c> onto <see cref="ChatFinishReason"/>.
/// </para>
/// <para><b>For Beginners:</b> This adapter lets the library talk to Claude models. It hides Anthropic's
/// specific request/response format so the rest of the code uses the same messages and options as for any
/// other provider.
/// </para>
/// </remarks>
public sealed class AnthropicChatClient<T> : ChatClientBase<T>
{
    private const int DefaultMaxTokens = 4096;
    private static readonly JsonSerializerSettings JsonSettings = new() { NullValueHandling = NullValueHandling.Ignore };

    private readonly string _apiKey;
    private readonly string _endpoint;
    private readonly string _anthropicVersion;

    /// <summary>
    /// Initializes a new Anthropic chat client.
    /// </summary>
    /// <param name="apiKey">The Anthropic API key.</param>
    /// <param name="modelName">The model id (default <c>claude-3-5-sonnet-20241022</c>).</param>
    /// <param name="endpoint">Optional custom endpoint (defaults to the public Messages API URL).</param>
    /// <param name="anthropicVersion">The Anthropic API version header (default <c>2023-06-01</c>).</param>
    /// <param name="httpClient">Optional HTTP client.</param>
    /// <exception cref="ArgumentNullException">Thrown when a required argument is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">Thrown when a required argument is empty/whitespace.</exception>
    public AnthropicChatClient(
        string apiKey,
        string modelName = "claude-3-5-sonnet-20241022",
        string? endpoint = null,
        string anthropicVersion = "2023-06-01",
        HttpClient? httpClient = null)
        : base(httpClient)
    {
        ValidateApiKey(apiKey);
        Guard.NotNullOrWhiteSpace(modelName);
        Guard.NotNullOrWhiteSpace(anthropicVersion);
        if (endpoint is not null)
        {
            // Fail fast on a malformed custom endpoint instead of waiting for
            // the first request to blow up with an opaque HttpRequestException.
            Guard.NotNullOrWhiteSpace(endpoint);
            // Restrict to http(s): UriKind.Absolute alone still accepts file:/ftp:/etc.,
            // which would survive construction and fail opaquely on the first request.
            if (!Uri.TryCreate(endpoint, UriKind.Absolute, out var parsed)
                || (parsed.Scheme != Uri.UriSchemeHttp && parsed.Scheme != Uri.UriSchemeHttps))
            {
                throw new ArgumentException(
                    "Endpoint must be an absolute http(s) URI.", nameof(endpoint));
            }
            endpoint = parsed.ToString();
        }
        _apiKey = apiKey;
        _endpoint = endpoint ?? "https://api.anthropic.com/v1/messages";
        _anthropicVersion = anthropicVersion;
        ModelId = modelName;
    }

    /// <inheritdoc/>
    protected override async Task<ChatResponse> GetResponseCoreAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions options,
        CancellationToken cancellationToken)
    {
        var payload = BuildRequest(messages, options, stream: false);
        using var request = CreateHttpRequest(payload);
        using var response = await HttpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
        await EnsureSuccessAsync(response).ConfigureAwait(false);

        var body = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        var root = JObject.Parse(body);

        var contents = new List<AiContent>();
        if (root["content"] is JArray blocks)
        {
            foreach (var block in blocks)
            {
                var type = (string?)block["type"];
                if (type == "text")
                {
                    var text = (string?)block["text"] ?? string.Empty;
                    if (text.Length > 0)
                    {
                        contents.Add(new TextContent(text));
                    }
                }
                else if (type == "tool_use")
                {
                    var id = (string?)block["id"] ?? string.Empty;
                    var name = (string?)block["name"] ?? string.Empty;
                    var input = block["input"] as JObject ?? new JObject();
                    if (!string.IsNullOrWhiteSpace(id) && !string.IsNullOrWhiteSpace(name))
                    {
                        contents.Add(new ToolCallContent(id, name, input.ToString(Formatting.None)));
                    }
                }
            }
        }

        if (contents.Count == 0)
        {
            contents.Add(new TextContent(string.Empty));
        }

        var finishReason = ParseStopReason((string?)root["stop_reason"]);
        var usage = ParseUsage(root["usage"] as JObject);
        var modelId = (string?)root["model"] ?? ModelId;

        return new ChatResponse(ChatMessage.Assistant(contents), finishReason, usage, modelId);
    }

    /// <inheritdoc/>
    protected override async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseCoreAsync(
        IReadOnlyList<ChatMessage> messages,
        ChatOptions options,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var payload = BuildRequest(messages, options, stream: true);
        using var request = CreateHttpRequest(payload);
        using var response = await HttpClient
            .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
            .ConfigureAwait(false);
        await EnsureSuccessAsync(response).ConfigureAwait(false);

#if NET5_0_OR_GREATER
        using var stream = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
#else
        using var stream = await response.Content.ReadAsStreamAsync().ConfigureAwait(false);
#endif
        using var reader = new StreamReader(stream);

        ChatFinishReason? finishReason = null;
        int inputTokens = 0;
        int outputTokens = 0;
        bool roleEmitted = false;

        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var line = await reader.ReadLineAsync().ConfigureAwait(false);
            if (line is null)
            {
                break;
            }

            if (line.Length == 0 || !line.StartsWith("data:", StringComparison.Ordinal))
            {
                continue;
            }

            var data = line.Substring("data:".Length).Trim();
            if (data.Length == 0)
            {
                continue;
            }

            var evt = JObject.Parse(data);
            switch ((string?)evt["type"])
            {
                case "message_start":
                    inputTokens = (int?)evt["message"]?["usage"]?["input_tokens"] ?? inputTokens;
                    if (!roleEmitted)
                    {
                        roleEmitted = true;
                        yield return new ChatResponseUpdate(role: ChatRole.Assistant);
                    }
                    break;

                case "content_block_start":
                    if ((string?)evt["content_block"]?["type"] == "tool_use")
                    {
                        var index = (int?)evt["index"] ?? 0;
                        var id = (string?)evt["content_block"]?["id"];
                        var name = (string?)evt["content_block"]?["name"];
                        yield return ChatResponseUpdate.ForToolCall(new StreamingToolCallUpdate(index, id, name));
                    }
                    break;

                case "content_block_delta":
                    var deltaType = (string?)evt["delta"]?["type"];
                    if (deltaType == "text_delta")
                    {
                        var text = (string?)evt["delta"]?["text"] ?? string.Empty;
                        if (text.Length > 0)
                        {
                            yield return ChatResponseUpdate.ForText(text);
                        }
                    }
                    else if (deltaType == "input_json_delta")
                    {
                        var index = (int?)evt["index"] ?? 0;
                        var partial = (string?)evt["delta"]?["partial_json"];
                        yield return ChatResponseUpdate.ForToolCall(
                            new StreamingToolCallUpdate(index, argumentsJsonFragment: partial));
                    }
                    break;

                case "message_delta":
                    var reason = (string?)evt["delta"]?["stop_reason"];
                    if (!string.IsNullOrEmpty(reason))
                    {
                        finishReason = ParseStopReason(reason);
                    }
                    outputTokens = (int?)evt["usage"]?["output_tokens"] ?? outputTokens;
                    break;

                case "message_stop":
                    break;
            }
        }

        yield return ChatResponseUpdate.ForFinish(
            finishReason ?? ChatFinishReason.Stop,
            new ChatUsage(inputTokens, outputTokens));
    }

    private HttpRequestMessage CreateHttpRequest(JObject payload)
    {
        var json = JsonConvert.SerializeObject(payload, JsonSettings);
        var request = new HttpRequestMessage(HttpMethod.Post, _endpoint)
        {
            Content = new StringContent(json, System.Text.Encoding.UTF8, "application/json")
        };
        request.Headers.Add("x-api-key", _apiKey);
        request.Headers.Add("anthropic-version", _anthropicVersion);
        return request;
    }

    private JObject BuildRequest(IReadOnlyList<ChatMessage> messages, ChatOptions options, bool stream)
    {
        var system = new System.Text.StringBuilder();
        var messageArray = new JArray();

        foreach (var message in messages)
        {
            if (message.Role == ChatRole.System)
            {
                if (system.Length > 0) system.Append('\n');
                system.Append(message.Text);
            }
            else if (message.Role == ChatRole.Tool)
            {
                messageArray.Add(new JObject { ["role"] = "user", ["content"] = BuildToolResultBlocks(message) });
            }
            else
            {
                messageArray.Add(new JObject
                {
                    ["role"] = message.Role == ChatRole.Assistant ? "assistant" : "user",
                    ["content"] = BuildContentBlocks(message)
                });
            }
        }

        var payload = new JObject
        {
            ["model"] = ModelId,
            ["max_tokens"] = options.MaxOutputTokens ?? DefaultMaxTokens,
            ["messages"] = messageArray
        };

        if (system.Length > 0) payload["system"] = system.ToString();
        if (stream) payload["stream"] = true;
        if (options.Temperature is { } temperature) payload["temperature"] = temperature;
        if (options.TopP is { } topP) payload["top_p"] = topP;
        if (options.TopK is { } topK && topK > 0) payload["top_k"] = topK;

        if (options.StopSequences is { Count: > 0 } stops)
        {
            payload["stop_sequences"] = new JArray(stops.Cast<object>().ToArray());
        }

        if (options.Tools is { Count: > 0 } tools)
        {
            payload["tools"] = BuildTools(tools);
            payload["tool_choice"] = BuildToolChoice(options);
        }

        return payload;
    }

    private static JArray BuildContentBlocks(ChatMessage message)
    {
        var blocks = new JArray();
        foreach (var part in message.Contents)
        {
            switch (part)
            {
                case TextContent text:
                    blocks.Add(new JObject { ["type"] = "text", ["text"] = text.Text });
                    break;
                case ImageContent image:
                    blocks.Add(BuildImageBlock(image));
                    break;
                case ToolCallContent call:
                    blocks.Add(new JObject
                    {
                        ["type"] = "tool_use",
                        ["id"] = call.CallId,
                        ["name"] = call.ToolName,
                        ["input"] = ParseArguments(call.ArgumentsJson)
                    });
                    break;
            }
        }

        if (blocks.Count == 0)
        {
            blocks.Add(new JObject { ["type"] = "text", ["text"] = string.Empty });
        }

        return blocks;
    }

    private static JArray BuildToolResultBlocks(ChatMessage message)
    {
        var blocks = new JArray();
        foreach (var result in message.Contents.OfType<ToolResultContent>())
        {
            var block = new JObject
            {
                ["type"] = "tool_result",
                ["tool_use_id"] = result.CallId,
                ["content"] = result.Result
            };
            if (result.IsError) block["is_error"] = true;
            blocks.Add(block);
        }

        return blocks;
    }

    private static JObject BuildImageBlock(ImageContent image)
    {
        var data = image.Data;
        if (data is not null)
        {
            var mime = (image.MediaType ?? ImageMediaType.Png).ToMimeType();
            return new JObject
            {
                ["type"] = "image",
                ["source"] = new JObject
                {
                    ["type"] = "base64",
                    ["media_type"] = mime,
                    ["data"] = Convert.ToBase64String(data)
                }
            };
        }

        return new JObject
        {
            ["type"] = "image",
            ["source"] = new JObject { ["type"] = "url", ["url"] = image.Uri ?? string.Empty }
        };
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
                ["input_schema"] = tool.ParametersSchema
            });
        }

        return array;
    }

    private static JObject BuildToolChoice(ChatOptions options)
    {
        switch (options.ToolChoice ?? ToolChoiceMode.Auto)
        {
            case ToolChoiceMode.None:
                return new JObject { ["type"] = "none" };
            case ToolChoiceMode.Required when !string.IsNullOrWhiteSpace(options.RequiredToolName):
                return new JObject { ["type"] = "tool", ["name"] = options.RequiredToolName };
            case ToolChoiceMode.Required:
                return new JObject { ["type"] = "any" };
            default:
                return new JObject { ["type"] = "auto" };
        }
    }

    private static JObject ParseArguments(string argumentsJson)
    {
        // Silently substituting {} on parse failure would let us execute a
        // tool with wrong/default arguments and hide a provider-side defect.
        // Surface the failure with a typed exception that names the bad input.
        Guard.NotNullOrWhiteSpace(argumentsJson);
        try
        {
            var token = JToken.Parse(argumentsJson);
            if (token is JObject obj)
            {
                return obj;
            }

            throw new JsonException("Tool arguments must be a JSON object.");
        }
        catch (JsonException ex)
        {
            throw new ArgumentException(
                "Invalid tool arguments JSON.", nameof(argumentsJson), ex);
        }
    }

    private static ChatFinishReason ParseStopReason(string? reason) => reason switch
    {
        "end_turn" => ChatFinishReason.Stop,
        "stop_sequence" => ChatFinishReason.Stop,
        "max_tokens" => ChatFinishReason.Length,
        "tool_use" => ChatFinishReason.ToolCalls,
        _ => ChatFinishReason.Unknown
    };

    private static ChatUsage? ParseUsage(JObject? usage)
    {
        if (usage is null)
        {
            return null;
        }

        var input = (int?)usage["input_tokens"] ?? 0;
        var output = (int?)usage["output_tokens"] ?? 0;
        return new ChatUsage(input, output);
    }

    private static async Task EnsureSuccessAsync(HttpResponseMessage response)
    {
        if (response.IsSuccessStatusCode)
        {
            return;
        }

        var error = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
        throw new HttpResponseException(
            response.StatusCode,
            $"Anthropic request failed with status {(int)response.StatusCode}: {error}");
    }
}
