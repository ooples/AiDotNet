using System.Threading;
using AiDotNet.ProgramSynthesis.Enums;

namespace AiDotNet.ProgramSynthesis.Interfaces;

/// <summary>
/// Defines an execution boundary for running synthesized programs against inputs.
/// </summary>
/// <remarks>
/// <para>
/// Implementations should execute code in a sandboxed, resource-limited environment appropriate
/// for the target language (timeouts, memory limits, restricted I/O, etc.).
/// </para>
/// <para><b>For Beginners:</b> This is the "runner" that actually executes the generated code.
///
/// Program synthesis can generate code as text, but to verify it works we need to run it safely.
/// This interface lets you plug in a safe execution environment (for example, a container,
/// an isolated process, or a remote service) without embedding unsafe execution inside the library.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("ProgramExecutionEngine")]
public interface IProgramExecutionEngine
{
    /// <summary>
    /// Tries to execute the given program source against the provided input.
    /// </summary>
    /// <param name="language">The programming language the source is written in.</param>
    /// <param name="sourceCode">The program source code.</param>
    /// <param name="input">The input to execute the program with.</param>
    /// <param name="output">The captured output produced by the program (if successful).</param>
    /// <param name="errorMessage">An optional error message if execution failed.</param>
    /// <param name="cancellationToken">Optional cancellation token for the execution attempt.</param>
    /// <returns>True if execution succeeded and output is available; otherwise, false.</returns>
    bool TryExecute(
        ProgramLanguage language,
        string sourceCode,
        string input,
        out string output,
        out string? errorMessage,
        CancellationToken cancellationToken = default);
}

