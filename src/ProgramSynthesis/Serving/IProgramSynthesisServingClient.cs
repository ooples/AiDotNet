using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;

namespace AiDotNet.ProgramSynthesis.Serving;

/// <summary>
/// Sends program-synthesis work to an AiDotNet.Serving host instead of running it in-process: code tasks, sandboxed
/// program execution and evaluation, and sandboxed SQL.
/// </summary>
/// <remarks>
/// <see cref="ProgramSynthesisServingClient"/> is the HTTP implementation. The host enforces its tier limits and
/// sandbox, so untrusted code never runs in the calling process.
/// </remarks>
public interface IProgramSynthesisServingClient
{
    /// <summary>Runs a code task (completion, generation, translation, review, ...) on the host.</summary>
    /// <param name="request">The task; its concrete request type selects the task.</param>
    /// <param name="cancellationToken">Cancels the call.</param>
    /// <returns>The task's result, of the result type that matches the request type.</returns>
    Task<CodeTaskResultBase> ExecuteCodeTaskAsync(CodeTaskRequestBase request, CancellationToken cancellationToken);

    /// <summary>Runs a program in the host's sandbox and returns what it printed.</summary>
    /// <param name="request">The source code, language and standard input.</param>
    /// <param name="cancellationToken">Cancels the call.</param>
    /// <returns>The compilation outcome, exit code, standard output and standard error of the run.</returns>
    Task<ProgramExecuteResponse> ExecuteProgramAsync(ProgramExecuteRequest request, CancellationToken cancellationToken);

    /// <summary>Runs a program in the host's sandbox against input/output test cases.</summary>
    /// <param name="request">The source code, language and the test cases.</param>
    /// <param name="cancellationToken">Cancels the call.</param>
    /// <returns>How many test cases passed, the pass rate, and each test case's result.</returns>
    Task<ProgramEvaluateIoResponse> EvaluateProgramIoAsync(ProgramEvaluateIoRequest request, CancellationToken cancellationToken);

    /// <summary>Runs a SQL query in the host's SQL sandbox.</summary>
    /// <param name="request">The query and the dialect or dataset to run it against.</param>
    /// <param name="cancellationToken">Cancels the call.</param>
    /// <returns>The columns and rows the query returned, or the error it raised.</returns>
    Task<SqlExecuteResponse> ExecuteSqlAsync(SqlExecuteRequest request, CancellationToken cancellationToken);
}

