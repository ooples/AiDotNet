using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;

namespace AiDotNet.ProgramSynthesis.Serving;

public interface IProgramSynthesisServingClient
{
    Task<CodeTaskResultBase> ExecuteCodeTaskAsync(CodeTaskRequestBase request, CancellationToken cancellationToken);

    Task<ProgramExecuteResponse> ExecuteProgramAsync(ProgramExecuteRequest request, CancellationToken cancellationToken);

    Task<ProgramEvaluateIoResponse> EvaluateProgramIoAsync(ProgramEvaluateIoRequest request, CancellationToken cancellationToken);

    Task<SqlExecuteResponse> ExecuteSqlAsync(SqlExecuteRequest request, CancellationToken cancellationToken);
}

