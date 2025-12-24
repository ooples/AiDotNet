using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;

namespace AiDotNet.ProgramSynthesis.Execution;

public sealed class ProgramEvaluateIoRequest
{
    public ProgramLanguage Language { get; set; } = ProgramLanguage.Generic;

    public List<ProgramLanguage> AllowedLanguages { get; set; } = new();

    public ProgramLanguage? PreferredLanguage { get; set; }

    public bool AllowUndetectedLanguageFallback { get; set; }

    public string SourceCode { get; set; } = string.Empty;

    public List<ProgramInputOutputExample> TestCases { get; set; } = new();
}

