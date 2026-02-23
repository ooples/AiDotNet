using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using Newtonsoft.Json;
using AiDotNet.Validation;

namespace AiDotNet.ProgramSynthesis.Serving;

public sealed class ProgramSynthesisServingClient : IProgramSynthesisServingClient
{
    private static readonly JsonSerializerSettings DefaultJsonSettings = new()
    {
        NullValueHandling = NullValueHandling.Ignore
    };

    private readonly ProgramSynthesisServingClientOptions _options;
    private readonly HttpClient _httpClient;

    public ProgramSynthesisServingClient(ProgramSynthesisServingClientOptions options)
    {
        Guard.NotNull(options);
        _options = options;
        if (_options.BaseAddress is null)
        {
            throw new ArgumentException("BaseAddress is required.", nameof(options));
        }

        _httpClient = _options.HttpClient ?? new HttpClient();
        _httpClient.Timeout = TimeSpan.FromMilliseconds(_options.TimeoutMs > 0 ? _options.TimeoutMs : 100_000);

        if (_httpClient.BaseAddress is null)
        {
            _httpClient.BaseAddress = _options.BaseAddress;
        }
    }

    public async Task<CodeTaskResultBase> ExecuteCodeTaskAsync(CodeTaskRequestBase request, CancellationToken cancellationToken)
    {
        if (request is null) throw new ArgumentNullException(nameof(request));

        return request switch
        {
            CodeCompletionRequest completion =>
                await PostAsync<CodeCompletionResult>(GetCodeTaskRoute(CodeTask.Completion), completion, cancellationToken).ConfigureAwait(false),
            CodeGenerationRequest generation =>
                await PostAsync<CodeGenerationResult>(GetCodeTaskRoute(CodeTask.Generation), generation, cancellationToken).ConfigureAwait(false),
            CodeTranslationRequest translation =>
                await PostAsync<CodeTranslationResult>(GetCodeTaskRoute(CodeTask.Translation), translation, cancellationToken).ConfigureAwait(false),
            CodeSummarizationRequest summarization =>
                await PostAsync<CodeSummarizationResult>(GetCodeTaskRoute(CodeTask.Summarization), summarization, cancellationToken).ConfigureAwait(false),
            CodeBugDetectionRequest bugDetection =>
                await PostAsync<CodeBugDetectionResult>(GetCodeTaskRoute(CodeTask.BugDetection), bugDetection, cancellationToken).ConfigureAwait(false),
            CodeBugFixingRequest bugFixing =>
                await PostAsync<CodeBugFixingResult>(GetCodeTaskRoute(CodeTask.BugFixing), bugFixing, cancellationToken).ConfigureAwait(false),
            CodeRefactoringRequest refactoring =>
                await PostAsync<CodeRefactoringResult>(GetCodeTaskRoute(CodeTask.Refactoring), refactoring, cancellationToken).ConfigureAwait(false),
            CodeUnderstandingRequest understanding =>
                await PostAsync<CodeUnderstandingResult>(GetCodeTaskRoute(CodeTask.Understanding), understanding, cancellationToken).ConfigureAwait(false),
            CodeTestGenerationRequest testGeneration =>
                await PostAsync<CodeTestGenerationResult>(GetCodeTaskRoute(CodeTask.TestGeneration), testGeneration, cancellationToken).ConfigureAwait(false),
            CodeDocumentationRequest documentation =>
                await PostAsync<CodeDocumentationResult>(GetCodeTaskRoute(CodeTask.Documentation), documentation, cancellationToken).ConfigureAwait(false),
            CodeSearchRequest search =>
                await PostAsync<CodeSearchResult>(GetCodeTaskRoute(CodeTask.Search), search, cancellationToken).ConfigureAwait(false),
            CodeCloneDetectionRequest cloneDetection =>
                await PostAsync<CodeCloneDetectionResult>(GetCodeTaskRoute(CodeTask.CloneDetection), cloneDetection, cancellationToken).ConfigureAwait(false),
            CodeReviewRequest review =>
                await PostAsync<CodeReviewResult>(GetCodeTaskRoute(CodeTask.CodeReview), review, cancellationToken).ConfigureAwait(false),
            _ => throw new InvalidOperationException($"Unsupported code task request type: {request.GetType().Name}.")
        };
    }

    public Task<ProgramExecuteResponse> ExecuteProgramAsync(ProgramExecuteRequest request, CancellationToken cancellationToken)
    {
        if (request is null) throw new ArgumentNullException(nameof(request));
        return PostAsync<ProgramExecuteResponse>("api/program-synthesis/program/execute", request, cancellationToken);
    }

    public Task<ProgramEvaluateIoResponse> EvaluateProgramIoAsync(ProgramEvaluateIoRequest request, CancellationToken cancellationToken)
    {
        if (request is null) throw new ArgumentNullException(nameof(request));
        return PostAsync<ProgramEvaluateIoResponse>("api/program-synthesis/program/evaluate-io", request, cancellationToken);
    }

    public Task<SqlExecuteResponse> ExecuteSqlAsync(SqlExecuteRequest request, CancellationToken cancellationToken)
    {
        if (request is null) throw new ArgumentNullException(nameof(request));
        return PostAsync<SqlExecuteResponse>("api/program-synthesis/sql/execute", request, cancellationToken);
    }

    private static string GetCodeTaskRoute(CodeTask task) =>
        task switch
        {
            CodeTask.Completion => "api/program-synthesis/tasks/completion",
            CodeTask.Generation => "api/program-synthesis/tasks/generation",
            CodeTask.Translation => "api/program-synthesis/tasks/translation",
            CodeTask.Summarization => "api/program-synthesis/tasks/summarization",
            CodeTask.BugDetection => "api/program-synthesis/tasks/bug-detection",
            CodeTask.BugFixing => "api/program-synthesis/tasks/bug-fixing",
            CodeTask.Refactoring => "api/program-synthesis/tasks/refactoring",
            CodeTask.Understanding => "api/program-synthesis/tasks/understanding",
            CodeTask.TestGeneration => "api/program-synthesis/tasks/test-generation",
            CodeTask.Documentation => "api/program-synthesis/tasks/documentation",
            CodeTask.Search => "api/program-synthesis/tasks/search",
            CodeTask.CloneDetection => "api/program-synthesis/tasks/clone-detection",
            CodeTask.CodeReview => "api/program-synthesis/tasks/code-review",
            _ => throw new InvalidOperationException($"Unsupported code task: {task}.")
        };

    private async Task<TResponse> PostAsync<TResponse>(string relativeUrl, object body, CancellationToken cancellationToken)
        where TResponse : class
    {
        if (string.IsNullOrWhiteSpace(relativeUrl))
        {
            throw new ArgumentException("Relative URL is required.", nameof(relativeUrl));
        }

        var payload = JsonConvert.SerializeObject(body, DefaultJsonSettings);
        using var request = new HttpRequestMessage(HttpMethod.Post, relativeUrl)
        {
            Content = new StringContent(payload, Encoding.UTF8, "application/json")
        };

        request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        AddAuthHeaders(request);

        using var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
        var responseBody = await response.Content.ReadAsStringAsync().ConfigureAwait(false);

        TResponse? deserialized = null;
        try
        {
            deserialized = JsonConvert.DeserializeObject<TResponse>(responseBody);
        }
        catch (JsonException)
        {
            // handled below
        }

        if (deserialized is null)
        {
            var status = (int)response.StatusCode;
            var snippet = string.IsNullOrEmpty(responseBody)
                ? "<empty>"
                : responseBody.Length <= 2048
                    ? responseBody
                    : responseBody.Substring(0, 2048);

            if (!response.IsSuccessStatusCode)
            {
                throw new HttpRequestException(
                    $"AiDotNet.Serving returned HTTP {status} for {relativeUrl}. Body: {snippet}");
            }

            throw new InvalidOperationException(
                $"AiDotNet.Serving returned an empty/unparseable response (Status={status}) for {relativeUrl}. Body: {snippet}");
        }

        return deserialized;
    }

    private void AddAuthHeaders(HttpRequestMessage request)
    {
        if (!string.IsNullOrWhiteSpace(_options.ApiKey))
        {
            request.Headers.TryAddWithoutValidation(_options.ApiKeyHeaderName, _options.ApiKey);
        }

        if (!string.IsNullOrWhiteSpace(_options.BearerToken))
        {
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _options.BearerToken);
        }
    }
}
