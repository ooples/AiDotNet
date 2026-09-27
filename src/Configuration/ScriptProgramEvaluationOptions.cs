using AiDotNet.Enums;
using AiDotNet.Evolution;
using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.Configuration;

/// <summary>Configures how a caller-supplied evaluator script scores candidate programs inside the sandbox.</summary>
/// <remarks>
/// <para>
/// The evaluator script is itself untrusted code — it is usually written by the same model that writes the
/// candidates — so it runs through exactly the same execution engine, under exactly the same time, memory, output,
/// and concurrency limits. The reference OpenEvolve implementation writes its evaluator to
/// <c>evaluator_&lt;hash&gt;.py</c> and executes it unsandboxed inside its worker processes; running an evaluator
/// with fewer protections than the thing it evaluates is a hole this option set closes by construction, because
/// there is nowhere here to specify a second, laxer sandbox.
/// </para>
/// <para>
/// <see cref="EvaluatorScript"/> and <see cref="EvaluatorScriptLanguage"/> hold the script and the language it is
/// written in. <see cref="RequireEntryPoint"/> checks for <see cref="EntryPointMarker"/> when the evaluator is
/// constructed rather than when the first candidate is scored, so a script missing its entry point is a
/// configuration error you see immediately instead of a run that fails thousands of evaluations later.
/// </para>
/// <para><b>For Beginners:</b> Sometimes "is this program any good?" cannot be answered by comparing printed
/// output — you want to score style, structure, or a domain-specific rule. This lets you write that judgement as a
/// small script. The script receives the candidate's source code on its standard input and prints one JSON object
/// such as <c>{"quality": 0.8}</c> back. These settings say what that script is, what language it is in, and how
/// much of its extra commentary to keep.</para>
/// </remarks>
public sealed class ScriptProgramEvaluationOptions
{
    /// <summary>The largest artifact text retained per artifact, in characters.</summary>
    public const int MaxArtifactLengthCeiling = 4_000;

    /// <summary>The largest number of artifacts retained from one evaluation.</summary>
    public const int MaxArtifactCountCeiling = 16;

    /// <summary>Gets or sets the evaluator script source, or <c>null</c> when it is supplied directly to the evaluator.</summary>
    public string? EvaluatorScript { get; set; }

    /// <summary>Gets or sets the language the evaluator script is written in.</summary>
    /// <remarks>The sandbox must have an interpreter configured for this language, or every evaluation is refused.</remarks>
    public ProgramLanguage EvaluatorScriptLanguage { get; set; } = ProgramLanguage.Python;

    /// <summary>Gets or sets the text that must appear in the script for it to be accepted. Defaults to <c>"evaluate"</c>.</summary>
    public string EntryPointMarker { get; set; } = "evaluate";

    /// <summary>Gets or sets whether <see cref="EntryPointMarker"/> is enforced when the evaluator is constructed.</summary>
    public bool RequireEntryPoint { get; set; } = true;

    /// <summary>Gets or sets whether a larger reported quality is better.</summary>
    public EvolutionOptimizationDirection Direction { get; set; } = EvolutionOptimizationDirection.Maximize;

    /// <summary>Gets or sets how many artifacts from one evaluation become diagnostics. Defaults to 4.</summary>
    public int MaxArtifactCount { get; set; } = 4;

    /// <summary>Gets or sets how many characters of each artifact are retained. Defaults to 500.</summary>
    public int MaxArtifactLength { get; set; } = 500;

    /// <summary>Creates an independent copy so a running evaluator is unaffected by later mutation.</summary>
    /// <returns>A new instance carrying the same values.</returns>
    public ScriptProgramEvaluationOptions Clone() => new()
    {
        EvaluatorScript = EvaluatorScript,
        EvaluatorScriptLanguage = EvaluatorScriptLanguage,
        EntryPointMarker = EntryPointMarker,
        RequireEntryPoint = RequireEntryPoint,
        Direction = Direction,
        MaxArtifactCount = MaxArtifactCount,
        MaxArtifactLength = MaxArtifactLength
    };

    /// <summary>Rejects a configuration the evaluator could not honour.</summary>
    /// <exception cref="ArgumentException"><see cref="EntryPointMarker"/> is empty or white space while it is required.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <see cref="EvaluatorScriptLanguage"/> or <see cref="Direction"/> is not a defined value, or an artifact bound
    /// is negative or exceeds its ceiling.
    /// </exception>
    public void Validate()
    {
        if (!Enum.IsDefined(typeof(ProgramLanguage), EvaluatorScriptLanguage))
            throw new ArgumentOutOfRangeException(nameof(EvaluatorScriptLanguage), EvaluatorScriptLanguage,
                "Value must be a defined language.");
        if (!Enum.IsDefined(typeof(EvolutionOptimizationDirection), Direction))
            throw new ArgumentOutOfRangeException(nameof(Direction), Direction, "Value must be a defined direction.");
        if (RequireEntryPoint && string.IsNullOrWhiteSpace(EntryPointMarker))
            throw new ArgumentException(
                "EntryPointMarker cannot be empty while RequireEntryPoint is set.", nameof(EntryPointMarker));
        if (MaxArtifactCount < 0 || MaxArtifactCount > MaxArtifactCountCeiling)
            throw new ArgumentOutOfRangeException(nameof(MaxArtifactCount), MaxArtifactCount,
                $"Value must be between 0 and {MaxArtifactCountCeiling}.");
        if (MaxArtifactLength < 0 || MaxArtifactLength > MaxArtifactLengthCeiling)
            throw new ArgumentOutOfRangeException(nameof(MaxArtifactLength), MaxArtifactLength,
                $"Value must be between 0 and {MaxArtifactLengthCeiling}.");
    }
}
