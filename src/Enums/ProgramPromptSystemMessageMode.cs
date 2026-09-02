namespace AiDotNet.Enums;

/// <summary>Says how a configured system message is interpreted: as a template name or as literal text.</summary>
/// <remarks>
/// <para>
/// The reference OpenEvolve implementation guesses: it treats the configured system message as a template key if
/// a template with that name happens to exist, and otherwise as literal prompt text. A user whose literal
/// instruction collides with a template name therefore silently gets the template instead, and a user who
/// mistypes a template name silently gets that typo sent to the model as the system prompt. Making the choice
/// explicit removes both surprises.
/// </para>
/// <para><b>For Beginners:</b> You can set the standing instruction that is sent to the model in two ways: by
/// naming one of the built-in texts, or by writing the instruction out yourself. This setting says which of the
/// two you meant, so there is never any guessing. Use <see cref="TemplateKey"/> when you want a named built-in
/// text (optionally one you replaced), and <see cref="Literal"/> when the value you supplied is the actual
/// wording you want the model to see.</para>
/// </remarks>
public enum ProgramPromptSystemMessageMode
{
    /// <summary>The configured value names a <see cref="ProgramPromptTemplateKey"/> whose text is used.</summary>
    TemplateKey = 0,

    /// <summary>The configured value is the system message text itself.</summary>
    Literal = 1
}
