using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.ProgramSynthesis.Serving;
using AiDotNet.ProgramSynthesis.Tokenization;
using AiDotNet.Reasoning.Benchmarks;
using AiDotNet.Reasoning.Benchmarks.Models;
using AiDotNet.Tokenization.Models;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Results;

public partial class PredictionModelResult<T, TInput, TOutput>
{
    internal IFullModel<T, Tensor<T>, Tensor<T>>? ProgramSynthesisModel { get; private set; }

    internal IProgramSynthesisServingClient? ProgramSynthesisServingClient { get; private set; }

    internal ProgramSynthesisServingClientOptions? ProgramSynthesisServingClientOptions { get; private set; }

    /// <summary>
    /// Executes a structured code task using the configured program synthesis model.
    /// </summary>
    public CodeTaskResultBase ExecuteCodeTask(CodeTaskRequestBase request)
    {
        if (request is null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        var model = GetCodeModelOrThrow();
        return model.PerformTask(request);
    }

    /// <summary>
    /// Executes a structured code task, optionally delegating to AiDotNet.Serving when configured and preferred.
    /// </summary>
    public Task<CodeTaskResultBase> ExecuteCodeTaskAsync(CodeTaskRequestBase request, CancellationToken cancellationToken = default)
    {
        if (request is null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        if (ShouldPreferServing() && ProgramSynthesisServingClient is not null)
        {
            return ProgramSynthesisServingClient.ExecuteCodeTaskAsync(request, cancellationToken);
        }

        return Task.FromResult(ExecuteCodeTask(request));
    }

    /// <summary>
    /// Executes a sandboxed program via AiDotNet.Serving.
    /// </summary>
    public Task<ProgramExecuteResponse> ExecuteProgramAsync(ProgramExecuteRequest request, CancellationToken cancellationToken = default)
    {
        if (request is null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        return GetServingClientOrThrow().ExecuteProgramAsync(request, cancellationToken);
    }

    /// <summary>
    /// Evaluates a program against input/output test cases via AiDotNet.Serving.
    /// </summary>
    public Task<ProgramEvaluateIoResponse> EvaluateProgramIoAsync(ProgramEvaluateIoRequest request, CancellationToken cancellationToken = default)
    {
        if (request is null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        return GetServingClientOrThrow().EvaluateProgramIoAsync(request, cancellationToken);
    }

    /// <summary>
    /// Executes SQL via AiDotNet.Serving.
    /// </summary>
    public Task<SqlExecuteResponse> ExecuteSqlAsync(SqlExecuteRequest request, CancellationToken cancellationToken = default)
    {
        if (request is null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        return GetServingClientOrThrow().ExecuteSqlAsync(request, cancellationToken);
    }

    /// <summary>
    /// Tokenizes code using the canonical code-tokenization pipeline (supports AST extraction when enabled).
    /// </summary>
    public CodeTokenizationResult TokenizeCode(
        string code,
        ProgramLanguage language,
        EncodingOptions? options = null,
        CodeTokenizationPipelineOptions? pipelineOptions = null)
    {
        var tokenizer = ProgramSynthesisTokenizerFactory.CreateDefault(language);
        var pipeline = new CodeTokenizationPipeline();

        return pipelineOptions is null
            ? pipeline.Tokenize(code ?? string.Empty, language, tokenizer, options)
            : pipeline.TokenizeWithStructure(code ?? string.Empty, language, tokenizer, pipelineOptions, options);
    }

    public CodeSummarizationResult SummarizeCode(CodeSummarizationRequest request)
        => ExecuteCodeTaskTyped<CodeSummarizationResult>(request);

    public CodeGenerationResult GenerateCode(CodeGenerationRequest request)
        => ExecuteCodeTaskTyped<CodeGenerationResult>(request);

    /// <summary>
    /// Evaluates the model on HumanEval using the configured dataset path (via env var) and returns a benchmark report.
    /// </summary>
    /// <remarks>
    /// This currently performs a single generation per prompt (pass@1 behavior). The <paramref name="passK"/> parameter is
    /// accepted to enable forward-compatible support for multi-sample evaluation.
    /// </remarks>
    public Task<BenchmarkResult<T>> EvaluateHumanEvalPassAtKAsync(
        int passK = 1,
        int? sampleSize = null,
        CancellationToken cancellationToken = default)
    {
        _ = passK;

        var benchmark = new HumanEvalBenchmark<T>();
        return benchmark.EvaluateAsync(
            prompt => Task.FromResult(
                GenerateCode(new CodeGenerationRequest
                {
                    Language = ProgramLanguage.Python,
                    Description = prompt
                }).GeneratedCode ?? string.Empty),
            sampleSize,
            cancellationToken);
    }

    private CodeTaskResultBase ExecuteCodeTaskOrThrow(CodeTaskRequestBase request)
        => ExecuteCodeTask(request);

    private TResult ExecuteCodeTaskTyped<TResult>(CodeTaskRequestBase request)
        where TResult : CodeTaskResultBase
    {
        var result = ExecuteCodeTaskOrThrow(request);
        if (result is TResult typed)
        {
            return typed;
        }

        throw new InvalidOperationException(
            $"Code task result type mismatch. Expected {typeof(TResult).Name}, got {result.GetType().Name}.");
    }

    private ICodeModel<T> GetCodeModelOrThrow()
    {
        if (ProgramSynthesisModel is ICodeModel<T> configuredCodeModel)
        {
            return configuredCodeModel;
        }

        if (Model is ICodeModel<T> modelCodeModel)
        {
            return modelCodeModel;
        }

        throw new InvalidOperationException(
            "No code model is configured for this PredictionModelResult. Configure an ICodeModel<T> via PredictionModelBuilder.ConfigureModel(...) or PredictionModelBuilder.ConfigureProgramSynthesis(...).");
    }

    private bool ShouldPreferServing()
        => (ProgramSynthesisServingClientOptions?.PreferServing).GetValueOrDefault();

    private IProgramSynthesisServingClient GetServingClientOrThrow()
    {
        if (ProgramSynthesisServingClient is not null)
        {
            return ProgramSynthesisServingClient;
        }

        throw new InvalidOperationException(
            "Program Synthesis Serving is not configured. Provide ProgramSynthesisServingClientOptions or ProgramSynthesisServingClient when building the model result.");
    }
}
