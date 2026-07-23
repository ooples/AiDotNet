namespace AiDotNet.Agentic.Models;

/// <summary>
/// Controls whether and how a chat model is allowed to call tools on a given request.
/// </summary>
/// <remarks>
/// <para>
/// When tools are supplied via <c>ChatOptions.Tools</c>, this mode tells the model how aggressively
/// it may use them. To force a <em>specific</em> tool, set <c>ChatOptions.RequiredToolName</c> in
/// addition to selecting <see cref="Required"/>.
/// </para>
/// <para><b>For Beginners:</b> Imagine giving an assistant a toolbox. This setting is your instruction
/// about the toolbox:
/// - <b>Auto</b>: "Use a tool if you think it helps." (the normal default)
/// - <b>None</b>: "Don't use any tools — just answer with text."
/// - <b>Required</b>: "You must call a tool before answering."
/// </para>
/// </remarks>
public enum ToolChoiceMode
{
    /// <summary>
    /// The model decides on its own whether to call a tool or respond directly. This is the default.
    /// </summary>
    Auto,

    /// <summary>
    /// The model is not allowed to call tools; it must respond with content only.
    /// </summary>
    None,

    /// <summary>
    /// The model must call at least one tool. Combine with a specific tool name to force that tool.
    /// </summary>
    Required
}
