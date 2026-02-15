using System.Text;
using AiDotNet.Validation;

namespace AiDotNet.PromptEngineering.Templates;

/// <summary>
/// Prompt template for chat-based interactions with role-based messages.
/// </summary>
/// <remarks>
/// <para>
/// This template structures prompts as conversations with different roles (system, user, assistant).
/// It's designed for chat-based language models that understand conversational context.
/// </para>
/// <para><b>For Beginners:</b> A template for building conversations with different roles.
///
/// Example:
/// ```csharp
/// var template = new ChatPromptTemplate();
///
/// // Set system message (instructions for the AI)
/// template.AddSystemMessage("You are a helpful math tutor for elementary students.");
///
/// // Add conversation history if needed
/// template.AddUserMessage("What is 5 + 3?");
/// template.AddAssistantMessage("5 + 3 equals 8!");
///
/// // Add current query
/// template.AddUserMessage("What about 12 + 7?");
///
/// var prompt = template.Format(new Dictionary<string, string>());
///
/// // Result:
/// // System: You are a helpful math tutor for elementary students.
/// // User: What is 5 + 3?
/// // Assistant: 5 + 3 equals 8!
/// // User: What about 12 + 7?
/// ```
/// </para>
/// </remarks>
public class ChatPromptTemplate : PromptTemplateBase
{
    private readonly List<ChatMessage> _messages;
    private readonly string _messageFormat;

    /// <summary>
    /// Initializes a new instance of the ChatPromptTemplate class.
    /// </summary>
    /// <param name="messageFormat">Format for each message (default: "{role}: {content}").</param>
    public ChatPromptTemplate(string? messageFormat = null)
        : base("{placeholder}") // Base class requires a non-whitespace template; we rebuild it immediately.
    {
        _messages = new List<ChatMessage>();
        _messageFormat = messageFormat ?? "{role}: {content}";
        UpdateTemplate();
    }

    /// <summary>
    /// Adds a system message to the conversation.
    /// </summary>
    /// <param name="content">The system message content.</param>
    /// <returns>This template instance for method chaining.</returns>
    public ChatPromptTemplate AddSystemMessage(string content)
    {
        _messages.Add(new ChatMessage("system", content));
        UpdateTemplate();
        return this;
    }

    /// <summary>
    /// Adds a user message to the conversation.
    /// </summary>
    /// <param name="content">The user message content.</param>
    /// <returns>This template instance for method chaining.</returns>
    public ChatPromptTemplate AddUserMessage(string content)
    {
        _messages.Add(new ChatMessage("user", content));
        UpdateTemplate();
        return this;
    }

    /// <summary>
    /// Adds an assistant message to the conversation.
    /// </summary>
    /// <param name="content">The assistant message content.</param>
    /// <returns>This template instance for method chaining.</returns>
    public ChatPromptTemplate AddAssistantMessage(string content)
    {
        _messages.Add(new ChatMessage("assistant", content));
        UpdateTemplate();
        return this;
    }

    /// <summary>
    /// Adds a message with a custom role.
    /// </summary>
    /// <param name="role">The message role.</param>
    /// <param name="content">The message content.</param>
    /// <returns>This template instance for method chaining.</returns>
    public ChatPromptTemplate AddMessage(string role, string content)
    {
        _messages.Add(new ChatMessage(role, content));
        UpdateTemplate();
        return this;
    }

    /// <summary>
    /// Gets all messages in the conversation.
    /// </summary>
    public IReadOnlyList<ChatMessage> Messages => _messages.AsReadOnly();

    /// <summary>
    /// Updates the internal template based on current messages.
    /// </summary>
    private void UpdateTemplate()
    {
        var formattedMessages = _messages.Select(message =>
            _messageFormat
                .Replace("{role}", CapitalizeFirst(message.Role))
                .Replace("{content}", message.Content));

        Template = string.Join(Environment.NewLine, formattedMessages);
        InputVariables = new List<string>(); // Chat templates don't use variable substitution
    }

    /// <summary>
    /// Capitalizes the first letter of a string.
    /// </summary>
    private static string CapitalizeFirst(string str)
    {
        if (string.IsNullOrEmpty(str))
        {
            return str;
        }

        return char.ToUpper(str[0]) + str.Substring(1);
    }

    /// <summary>
    /// Formats the template (chat templates return the conversation as-is).
    /// </summary>
    protected override string FormatCore(Dictionary<string, string> variables)
    {
        // For chat templates, we don't do variable substitution
        // The template is built from messages
        return Template;
    }
}

/// <summary>
/// Represents a single message in a chat conversation.
/// </summary>
public class ChatMessage
{
    /// <summary>
    /// Gets or sets the role of the message sender.
    /// </summary>
    public string Role { get; set; }

    /// <summary>
    /// Gets or sets the content of the message.
    /// </summary>
    public string Content { get; set; }

    /// <summary>
    /// Initializes a new instance of the ChatMessage class.
    /// </summary>
    /// <param name="role">The message role.</param>
    /// <param name="content">The message content.</param>
    public ChatMessage(string role, string content)
    {
        Guard.NotNull(role);
        Role = role;
        Guard.NotNull(content);
        Content = content;
    }
}
