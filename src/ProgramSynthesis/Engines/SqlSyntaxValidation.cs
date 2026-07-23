using AiDotNet.ProgramSynthesis.Interfaces;

namespace AiDotNet.ProgramSynthesis.Engines;

/// <summary>
/// Global registration point for the optional precise SQL syntax validator used by
/// <see cref="NeuralProgramSynthesizer{T}"/>.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> the program synthesizer validates candidate SQL programs. By
/// default it uses a lightweight generic structural check. If you want precise SQL
/// parsing, reference the opt-in <c>AiDotNet.Storage.Sqlite</c> package and set
/// <see cref="Validator"/> to a <c>SqliteSqlSyntaxValidator</c> instance — the
/// synthesizer will then use real SQLite parsing. This keeps the native SQLite
/// dependency out of the core package (audit-2026-05 finding #14).
/// </remarks>
public static class SqlSyntaxValidation
{
    /// <summary>
    /// The active SQL syntax validator, or <see langword="null"/> to use generic
    /// structural validation. Not set by default; assign a provider (e.g. from the
    /// <c>AiDotNet.Storage.Sqlite</c> package) to enable precise SQL validation.
    /// </summary>
    public static ISqlSyntaxValidator? Validator { get; set; }
}
