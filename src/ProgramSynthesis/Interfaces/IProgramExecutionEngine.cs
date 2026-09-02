using System.Threading;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;

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

    /// <summary>
    /// Executes a program request and reports the full outcome, including exit code, captured streams, truncation
    /// flags, compilation diagnostics, and a structured error code.
    /// </summary>
    /// <param name="request">The program, its language, its standard input, and whether to compile without running.</param>
    /// <param name="cancellationToken">A token that cancels the execution.</param>
    /// <returns>The outcome of the attempt; never <c>null</c>.</returns>
    /// <remarks>
    /// <para>
    /// This is the richer of the two members and the one an evolution run should call. <see cref="TryExecute"/>
    /// collapses every failure into a single boolean and a message, which loses the distinction between a program
    /// that timed out, one that failed to compile, and one that ran and returned a non-zero exit code — precisely
    /// the distinctions a fitness function wants to score differently. It also cannot report that output was
    /// truncated, so a candidate whose answer was cut off at the output cap is indistinguishable from one that
    /// printed the wrong thing.
    /// </para>
    /// <para>
    /// Implementations must not throw for ordinary failure. A program that crashes, hangs, prints too much, or
    /// names an unsupported language is an everyday occurrence in program synthesis and must come back as a
    /// <see cref="ProgramExecuteResponse"/> whose <see cref="ProgramExecuteResponse.ErrorCode"/> says which it was.
    /// Reserve exceptions for programmer error, such as a <c>null</c> request or a disposed engine.
    /// </para>
    /// <para><b>For Beginners:</b> Use this instead of <see cref="TryExecute"/> whenever you care <em>why</em> a
    /// program failed. It hands back a small report: did it run, what did it print, what did it print on the error
    /// stream, was any of that cut short because it printed too much, what exit code did it finish with, and if it
    /// went wrong, was that a timeout, a compile error, or a crash.</para>
    /// </remarks>
    Task<ProgramExecuteResponse> ExecuteAsync(
        ProgramExecuteRequest request,
        CancellationToken cancellationToken = default);
}

