using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Agentic.Models;
using Newtonsoft.Json.Linq;

namespace AiDotNet.Agentic.Tools;

/// <summary>
/// Wraps any <see cref="IAgentTool"/> with schema guardrails: before the inner tool runs, the model-supplied
/// arguments are validated against the tool's JSON Schema (required properties present, declared types match).
/// A validation failure is returned as a descriptive <see cref="ToolInvocationResult.Error"/> the model can read
/// and correct on its next turn, rather than reaching the tool as malformed input.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is a safety wrapper. It checks that the assistant filled in a tool's inputs
/// correctly before the tool actually runs, and if not, hands back a clear "you got this wrong, here's why"
/// message so the assistant can fix its call — instead of the tool crashing or doing the wrong thing.</para>
/// </remarks>
public sealed class GuardedAgentTool : IAgentTool
{
    private readonly IAgentTool _inner;

    /// <summary>Wraps a tool with argument-schema validation.</summary>
    /// <param name="inner">The tool to guard.</param>
    public GuardedAgentTool(IAgentTool inner)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
    }

    /// <inheritdoc/>
    public string Name => _inner.Name;

    /// <inheritdoc/>
    public string Description => _inner.Description;

    /// <inheritdoc/>
    public JObject ParametersSchema => _inner.ParametersSchema;

    /// <inheritdoc/>
    public AiToolDefinition ToDefinition() => _inner.ToDefinition();

    /// <inheritdoc/>
    public Task<ToolInvocationResult> InvokeAsync(JObject arguments, CancellationToken cancellationToken = default)
    {
        var errors = Validate(arguments, _inner.ParametersSchema);
        if (errors.Count > 0)
        {
            return Task.FromResult(ToolInvocationResult.Error(
                $"Invalid arguments for tool '{Name}': {string.Join("; ", errors)}. Fix these and call the tool again."));
        }

        return _inner.InvokeAsync(arguments, cancellationToken);
    }

    /// <summary>Validates arguments against a (subset of) JSON Schema: required keys and declared property types.</summary>
    private static List<string> Validate(JObject? arguments, JObject schema)
    {
        var errors = new List<string>();
        arguments ??= new JObject();

        if (schema["required"] is JArray required)
        {
            foreach (var req in required)
            {
                if (req.Value<string>() is { Length: > 0 } key && arguments[key] is null)
                {
                    errors.Add($"missing required property '{key}'");
                }
            }
        }

        if (schema["properties"] is JObject properties)
        {
            foreach (var prop in properties.Properties())
            {
                var value = arguments[prop.Name];
                if (value is null) continue; // absence is handled by the required check above.
                if (prop.Value is JObject propSchema && propSchema["type"]?.Value<string>() is { } expected)
                {
                    if (!TypeMatches(expected, value.Type))
                    {
                        errors.Add($"property '{prop.Name}' should be {expected} but was {value.Type.ToString().ToLowerInvariant()}");
                    }
                }
            }
        }

        return errors;
    }

    private static bool TypeMatches(string schemaType, JTokenType actual) => schemaType switch
    {
        "number" => actual is JTokenType.Integer or JTokenType.Float,
        "integer" => actual is JTokenType.Integer,
        "string" => actual is JTokenType.String,
        "boolean" => actual is JTokenType.Boolean,
        "array" => actual is JTokenType.Array,
        "object" => actual is JTokenType.Object,
        "null" => actual is JTokenType.Null,
        _ => true, // unknown/unspecified schema type: do not block.
    };
}
