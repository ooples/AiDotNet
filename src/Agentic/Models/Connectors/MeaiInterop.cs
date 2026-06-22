using System.Text.Json;
using AiDotNet.Agentic.Models;
using Newtonsoft.Json.Linq;
using Meai = Microsoft.Extensions.AI;
using ChatMessage = AiDotNet.Agentic.Models.ChatMessage;

namespace AiDotNet.Agentic.Models.Connectors;

/// <summary>
/// Shared conversions between AiDotNet's agentic chat types and Microsoft.Extensions.AI (MEAI), used by both
/// the inbound <see cref="MeaiChatClient{T}"/> (MEAI model → AiDotNet) and the outbound
/// <see cref="AiDotNetMeaiChatClient{T}"/> (AiDotNet model → MEAI). Keeping the mapping in one place means the
/// two adapters round-trip identically (text, tool declarations, tool calls, tool results, usage).
/// </summary>
internal static class MeaiInterop
{
    // ---- JSON schema / arguments interchange (Newtonsoft <-> System.Text.Json) ----

    /// <summary>Converts a Newtonsoft schema object to a standalone System.Text.Json element.</summary>
    public static JsonElement ToJsonElement(JObject schema)
    {
        using var doc = JsonDocument.Parse(schema.ToString(Newtonsoft.Json.Formatting.None));
        // Clone so the element survives disposal of the document.
        return doc.RootElement.Clone();
    }

    /// <summary>Converts a System.Text.Json schema element to a Newtonsoft object (empty object if absent).</summary>
    public static JObject ToJObject(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.Undefined || element.ValueKind == JsonValueKind.Null)
        {
            return new JObject { ["type"] = "object", ["properties"] = new JObject() };
        }

        return JObject.Parse(element.GetRawText());
    }

    /// <summary>Serializes a MEAI tool-call argument dictionary to the JSON string AiDotNet carries.</summary>
    public static string ArgumentsToJson(IDictionary<string, object?>? arguments)
    {
        if (arguments is null || arguments.Count == 0)
        {
            return "{}";
        }

        return System.Text.Json.JsonSerializer.Serialize(arguments);
    }

    /// <summary>Parses an AiDotNet tool-call arguments JSON string into a MEAI argument dictionary.</summary>
    public static IDictionary<string, object?>? JsonToArguments(string? argumentsJson)
    {
        if (argumentsJson is null || argumentsJson.Trim().Length == 0)
        {
            return null;
        }

        try
        {
            return System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object?>>(argumentsJson);
        }
        catch (System.Text.Json.JsonException)
        {
            // Malformed arguments: surface them under a single "_raw" key rather than throwing, so the call
            // still reaches the consumer.
            return new Dictionary<string, object?> { ["_raw"] = argumentsJson };
        }
    }

    // ---- Tool definitions ----

    /// <summary>Wraps AiDotNet tool definitions as MEAI schema-only function declarations.</summary>
    public static IList<Meai.AITool> ToMeaiTools(IReadOnlyList<AiToolDefinition> tools)
    {
        var result = new List<Meai.AITool>(tools.Count);
        foreach (var tool in tools)
        {
            result.Add(new DeclarationFunction(tool.Name, tool.Description, ToJsonElement(tool.ParametersSchema)));
        }

        return result;
    }

    /// <summary>Reads MEAI function tools back into AiDotNet tool definitions.</summary>
    public static List<AiToolDefinition> FromMeaiTools(IEnumerable<Meai.AITool>? tools)
    {
        var result = new List<AiToolDefinition>();
        if (tools is null)
        {
            return result;
        }

        foreach (var tool in tools)
        {
            if (tool is Meai.AIFunction function)
            {
                result.Add(new AiToolDefinition(function.Name, function.Description, ToJObject(function.JsonSchema)));
            }
        }

        return result;
    }

    // ---- Role mapping ----

    public static Meai.ChatRole ToMeaiRole(ChatRole role) => role switch
    {
        ChatRole.System => Meai.ChatRole.System,
        ChatRole.User => Meai.ChatRole.User,
        ChatRole.Assistant => Meai.ChatRole.Assistant,
        ChatRole.Tool => Meai.ChatRole.Tool,
        _ => Meai.ChatRole.User
    };

    public static ChatRole FromMeaiRole(Meai.ChatRole role)
    {
        if (role == Meai.ChatRole.System) return ChatRole.System;
        if (role == Meai.ChatRole.Assistant) return ChatRole.Assistant;
        if (role == Meai.ChatRole.Tool) return ChatRole.Tool;
        return ChatRole.User;
    }

    // ---- Message mapping ----

    /// <summary>Maps an AiDotNet message (text, tool calls, tool results) to a MEAI message.</summary>
    public static Meai.ChatMessage ToMeaiMessage(ChatMessage message)
    {
        var contents = new List<Meai.AIContent>();
        foreach (var content in message.Contents)
        {
            switch (content)
            {
                case TextContent text:
                    contents.Add(new Meai.TextContent(text.Text));
                    break;
                case ToolCallContent call:
                    contents.Add(new Meai.FunctionCallContent(call.CallId, call.ToolName, JsonToArguments(call.ArgumentsJson)));
                    break;
                case ToolResultContent result:
                    contents.Add(new Meai.FunctionResultContent(result.CallId, result.Result));
                    break;
                // Image/multimodal content is not bridged through MEAI here; text and tool content are.
            }
        }

        return new Meai.ChatMessage(ToMeaiRole(message.Role), contents);
    }

    /// <summary>Maps a MEAI message (text, tool calls, tool results) to an AiDotNet message.</summary>
    public static ChatMessage FromMeaiMessage(Meai.ChatMessage message)
    {
        var contents = new List<AiContent>();
        foreach (var content in message.Contents)
        {
            switch (content)
            {
                case Meai.TextContent text:
                    contents.Add(new TextContent(text.Text ?? string.Empty));
                    break;
                case Meai.FunctionCallContent call:
                    contents.Add(new ToolCallContent(call.CallId, call.Name, ArgumentsToJson(call.Arguments)));
                    break;
                case Meai.FunctionResultContent result:
                    contents.Add(new ToolResultContent(result.CallId, result.Result?.ToString() ?? string.Empty));
                    break;
            }
        }

        if (contents.Count == 0)
        {
            contents.Add(new TextContent(message.Text ?? string.Empty));
        }

        return new ChatMessage(FromMeaiRole(message.Role), contents);
    }

    // ---- Finish reason / usage ----

    public static ChatFinishReason FromMeaiFinishReason(Meai.ChatFinishReason? reason)
    {
        if (reason is null)
        {
            return ChatFinishReason.Stop;
        }

        var value = reason.Value;
        if (value == Meai.ChatFinishReason.Stop) return ChatFinishReason.Stop;
        if (value == Meai.ChatFinishReason.Length) return ChatFinishReason.Length;
        if (value == Meai.ChatFinishReason.ToolCalls) return ChatFinishReason.ToolCalls;
        if (value == Meai.ChatFinishReason.ContentFilter) return ChatFinishReason.ContentFilter;
        return ChatFinishReason.Unknown;
    }

    public static Meai.ChatFinishReason ToMeaiFinishReason(ChatFinishReason reason) => reason switch
    {
        ChatFinishReason.Stop => Meai.ChatFinishReason.Stop,
        ChatFinishReason.Length => Meai.ChatFinishReason.Length,
        ChatFinishReason.ToolCalls => Meai.ChatFinishReason.ToolCalls,
        ChatFinishReason.ContentFilter => Meai.ChatFinishReason.ContentFilter,
        _ => Meai.ChatFinishReason.Stop
    };

    public static ChatUsage? FromMeaiUsage(Meai.UsageDetails? usage)
    {
        if (usage is null)
        {
            return null;
        }

        // Token counts are long? in MEAI; clamp (don't overflow-cast) into ChatUsage's int range.
        return new ChatUsage(ClampToInt(usage.InputTokenCount ?? 0), ClampToInt(usage.OutputTokenCount ?? 0));
    }

    private static int ClampToInt(long value) =>
        value < 0 ? 0 : (value > int.MaxValue ? int.MaxValue : (int)value);

    public static Meai.UsageDetails? ToMeaiUsage(ChatUsage? usage)
    {
        if (usage is null)
        {
            return null;
        }

        return new Meai.UsageDetails
        {
            InputTokenCount = usage.InputTokens,
            OutputTokenCount = usage.OutputTokens,
        };
    }

    // A schema-only MEAI function: it advertises name/description/parameters so a MEAI model can decide to
    // call it, but it is never executed through MEAI — the AiDotNet agent loop runs the real tool after the
    // call is surfaced back. Invoking it directly is therefore a programming error.
    private sealed class DeclarationFunction : Meai.AIFunction
    {
        private readonly JsonElement _schema;

        public DeclarationFunction(string name, string description, JsonElement schema)
        {
            Name = name;
            Description = description;
            _schema = schema;
        }

        public override string Name { get; }

        public override string Description { get; }

        public override JsonElement JsonSchema => _schema;

        protected override ValueTask<object?> InvokeCoreAsync(
            Meai.AIFunctionArguments arguments, CancellationToken cancellationToken) =>
            throw new NotSupportedException(
                $"'{Name}' is a schema-only declaration bridged from AiDotNet; it is executed by the AiDotNet " +
                "agent loop, not invoked through Microsoft.Extensions.AI.");
    }
}
