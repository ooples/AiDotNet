using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Execution;

public sealed class ProgramExecuteRequest
{
    public ProgramLanguage Language { get; set; } = ProgramLanguage.Generic;

    public List<ProgramLanguage> AllowedLanguages { get; set; } = new();

    public ProgramLanguage? PreferredLanguage { get; set; }

    public bool AllowUndetectedLanguageFallback { get; set; }

    public string SourceCode { get; set; } = string.Empty;

    public string? StdIn { get; set; }

    /// <summary>
    /// When true, the server should compile (or parse) the program but skip running it.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is primarily intended for typed-language verification workflows (e.g., C# compilation checks) where
    /// you want to validate code generation output without executing untrusted code.
    /// </para>
    /// </remarks>
    public bool CompileOnly { get; set; }
}

