using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Serving.Engine.Http;

/// <summary>
/// Renders a chat conversation (a list of role-tagged messages) into the single prompt string the model
/// actually generates from. Different model families use different chat formats; this is the seam that lets a
/// server speak the model's format while exposing the standard OpenAI chat API.
/// </summary>
public interface IChatTemplate
{
    /// <summary>Renders the conversation into a prompt, ending where the assistant's reply should begin.</summary>
    string Render(IReadOnlyList<ChatMessage> messages);
}

/// <summary>
/// A generic, model-agnostic chat template: each message is emitted as a role-tagged block and the prompt ends
/// with an open assistant turn. Works out of the box; swap in a model-specific template via
/// <see cref="ServeOptions"/> when a model was trained with a particular format.
/// </summary>
public sealed class DefaultChatTemplate : IChatTemplate
{
    /// <inheritdoc/>
    public string Render(IReadOnlyList<ChatMessage> messages)
    {
        var sb = new StringBuilder();
        if (messages is not null)
        {
            foreach (var message in messages)
            {
                string role = string.IsNullOrWhiteSpace(message.Role) ? "user" : message.Role;
                sb.Append("<|").Append(role).Append("|>\n").Append(message.Content ?? string.Empty).Append('\n');
            }
        }
        sb.Append("<|assistant|>\n");
        return sb.ToString();
    }
}
