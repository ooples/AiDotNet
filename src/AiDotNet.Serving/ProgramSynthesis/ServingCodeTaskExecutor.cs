using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.Serving.Security;

namespace AiDotNet.Serving.ProgramSynthesis;

public sealed class ServingCodeTaskExecutor : IServingCodeTaskExecutor
{
    private readonly ServingHeuristicCodeModel _model;
    private readonly ILogger<ServingCodeTaskExecutor> _logger;

    public ServingCodeTaskExecutor(ILogger<ServingCodeTaskExecutor> logger)
    {
        _model = ServingHeuristicCodeModel.CreateDefault();
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    public Task<CodeTaskResultBase> ExecuteAsync(
        CodeTaskRequestBase request,
        ServingRequestContext requestContext,
        CancellationToken cancellationToken)
    {
        if (request is null) throw new ArgumentNullException(nameof(request));
        if (requestContext is null) throw new ArgumentNullException(nameof(requestContext));

        // The current task implementations are synchronous and deterministic; keep the Serving surface async for evolution.
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            var result = _model.PerformTask(request);
            return Task.FromResult(result);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Code task execution failed (Task={Task}, Language={Language}).", request.Task, request.Language);

            return Task.FromResult(CreateFailureResult(
                request.Task,
                request.Language,
                request.RequestId,
                "Code task execution failed."));
        }
    }

    private static CodeTaskResultBase CreateFailureResult(
        CodeTask task,
        ProgramLanguage language,
        string? requestId,
        string error)
    {
        CodeTaskResultBase result = task switch
        {
            CodeTask.Completion => new CodeCompletionResult(),
            CodeTask.Generation => new CodeGenerationResult(),
            CodeTask.Translation => new CodeTranslationResult(),
            CodeTask.Summarization => new CodeSummarizationResult(),
            CodeTask.BugDetection => new CodeBugDetectionResult(),
            CodeTask.BugFixing => new CodeBugFixingResult(),
            CodeTask.Refactoring => new CodeRefactoringResult(),
            CodeTask.Understanding => new CodeUnderstandingResult(),
            CodeTask.TestGeneration => new CodeTestGenerationResult(),
            CodeTask.Documentation => new CodeDocumentationResult(),
            CodeTask.Search => new CodeSearchResult(),
            CodeTask.CloneDetection => new CodeCloneDetectionResult(),
            CodeTask.CodeReview => new CodeReviewResult(),
            _ => new CodeSummarizationResult()
        };

        result.Language = language;
        result.RequestId = requestId;
        result.Success = false;
        result.Error = error;
        return result;
    }
}
