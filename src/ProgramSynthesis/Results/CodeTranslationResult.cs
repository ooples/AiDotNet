using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Results;

public sealed class CodeTranslationResult : CodeTaskResultBase
{
    public override CodeTask Task => CodeTask.Translation;

    public ProgramLanguage SourceLanguage { get; set; } = ProgramLanguage.Generic;

    public ProgramLanguage TargetLanguage { get; set; } = ProgramLanguage.Generic;

    public string TranslatedCode { get; set; } = string.Empty;
}
