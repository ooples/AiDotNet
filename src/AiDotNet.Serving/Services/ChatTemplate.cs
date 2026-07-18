using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Serving.Services;

/// <summary>
/// Renders OpenAI chat messages into a single prompt string.
/// </summary>
/// <remarks>
/// This is a neutral, tokenizer-agnostic default (<c>Role: content</c> turns) that does not depend
/// on model-specific control tokens, so it works with any tokenizer. Model-specific chat templates
/// (Llama-3, ChatML, Mistral, …) can be layered on later via the model's own template metadata.
/// </remarks>
public static class ChatTemplate
{
    /// <summary>Renders messages into a plain prompt ending with an open assistant turn.</summary>
    public static string Render(IEnumerable<(string Role, string Content)> messages)
    {
        var sb = new StringBuilder();
        foreach (var (role, content) in messages)
        {
            sb.Append(Label(role)).Append(": ").Append(content).Append('\n');
        }
        sb.Append("Assistant:");
        return sb.ToString();
    }

    private static string Label(string role) => role?.ToLowerInvariant() switch
    {
        "system" => "System",
        "assistant" => "Assistant",
        "tool" => "Tool",
        _ => "User",
    };
}
