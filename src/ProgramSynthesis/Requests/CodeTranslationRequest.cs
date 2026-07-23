using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Requests;

/// <summary>
/// Request for translating code from one language to another.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Like translating a book, but for code.</para>
/// </remarks>
public sealed class CodeTranslationRequest : CodeTaskRequestBase
{
    public override CodeTask Task => CodeTask.Translation;

    public string Code { get; set; } = string.Empty;

    public ProgramLanguage SourceLanguage { get; set; } = ProgramLanguage.Generic;

    public ProgramLanguage TargetLanguage { get; set; } = ProgramLanguage.Generic;
}
