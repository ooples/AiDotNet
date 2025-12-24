using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.ProgramSynthesis.Tokenization;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNet.ProgramSynthesis.Engines;

/// <summary>
/// Base class for code models that provides shared tokenization, task dispatch, and structured outputs.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
public abstract class CodeModelBase<T> : NeuralNetworkBase<T>, ICodeModel<T>
{
    private static readonly TimeSpan RegexTimeout = TimeSpan.FromSeconds(1);
    private readonly ITokenizer _tokenizer;

    protected CodeSynthesisArchitecture<T> CodeArchitecture { get; }

    protected ITokenizer Tokenizer => _tokenizer;

    public ProgramLanguage TargetLanguage => CodeArchitecture.TargetLanguage;

    public int MaxSequenceLength => CodeArchitecture.MaxSequenceLength;

    public int VocabularySize => CodeArchitecture.VocabularySize;

    protected CodeModelBase(
        CodeSynthesisArchitecture<T> architecture,
        ILossFunction<T> lossFunction,
        ITokenizer? tokenizer = null)
        : base(architecture, lossFunction)
    {
        CodeArchitecture = architecture ?? throw new ArgumentNullException(nameof(architecture));
        _tokenizer = tokenizer ?? CreateDefaultTokenizer(CodeArchitecture.TargetLanguage);

        if (_tokenizer.VocabularySize > CodeArchitecture.VocabularySize)
        {
            throw new ArgumentException(
                $"Tokenizer vocabulary size ({_tokenizer.VocabularySize}) cannot exceed model vocabulary size ({CodeArchitecture.VocabularySize}).",
                nameof(tokenizer));
        }
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        SetTrainingMode(false);

        var output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }

        return output;
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        int index = 0;
        foreach (var layer in Layers)
        {
            int layerParameterCount = layer.ParameterCount;
            if (layerParameterCount <= 0)
            {
                continue;
            }

            var layerParameters = parameters.Slice(index, layerParameterCount);
            layer.UpdateParameters(layerParameters);
            index += layerParameterCount;
        }
    }

    protected void TrainWithOptimizer(
        Tensor<T> input,
        Tensor<T> expectedOutput,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer)
    {
        SetTrainingMode(true);

        var output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }

        LastLoss = LossFunction.CalculateLoss(output.ToVector(), expectedOutput.ToVector());

        var outputGradient = LossFunction.CalculateDerivative(output.ToVector(), expectedOutput.ToVector());
        var gradient = new Tensor<T>(output.Shape, outputGradient);

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradient = Layers[i].Backward(gradient);
        }

        optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    protected ModelMetadata<T> CreateTransformerModelMetadata(
        string modelName,
        IReadOnlyDictionary<string, object>? extraInfo,
        string optimizerName)
    {
        var info = new Dictionary<string, object>
        {
            { "ModelName", modelName },
            { "TargetLanguage", CodeArchitecture.TargetLanguage.ToString() },
            { "CodeTaskType", CodeArchitecture.CodeTaskType.ToString() },
            { "NumEncoderLayers", CodeArchitecture.NumEncoderLayers },
            { "NumHeads", CodeArchitecture.NumHeads },
            { "ModelDimension", CodeArchitecture.ModelDimension },
            { "FeedForwardDimension", CodeArchitecture.FeedForwardDimension },
            { "MaxSequenceLength", CodeArchitecture.MaxSequenceLength },
            { "VocabularySize", CodeArchitecture.VocabularySize },
            { "DropoutRate", CodeArchitecture.DropoutRate },
            { "UsePositionalEncoding", CodeArchitecture.UsePositionalEncoding },
            { "LayerCount", Layers.Count },
            { "ParameterCount", GetParameterCount() },
            { "LossFunction", LossFunction.GetType().Name },
            { "Optimizer", optimizerName }
        };

        if (extraInfo is not null)
        {
            foreach (var kvp in extraInfo)
            {
                info[kvp.Key] = kvp.Value;
            }
        }

        return new ModelMetadata<T>
        {
            ModelType = ModelType.Transformer,
            AdditionalInfo = info,
            ModelData = Serialize()
        };
    }

    public Tensor<T> EncodeCode(string code)
    {
        var tokenIds = TokenizeToIds(code);
        var input = ToTokenIdTensor(tokenIds);
        return Predict(input);
    }

    public string DecodeCode(Tensor<T> encoding)
    {
        if (encoding == null)
        {
            throw new ArgumentNullException(nameof(encoding));
        }

        var tokenIds = ExtractTokenIdsForDecode(encoding);
        return _tokenizer.Decode(tokenIds, skipSpecialTokens: true);
    }

    [Obsolete("Use PerformTask(CodeTaskRequestBase) for structured outputs.")]
    public string PerformTask(string code, CodeTask task)
    {
        var request = CreateLegacyRequest(code, task);
        var result = PerformTask(request);
        return FormatLegacyResult(result);
    }

    public CodeTaskResultBase PerformTask(CodeTaskRequestBase request)
    {
        if (request is null)
        {
            throw new ArgumentNullException(nameof(request));
        }

        var stopwatch = Stopwatch.StartNew();

        try
        {
            var result = request switch
            {
                CodeCompletionRequest completion => PerformCompletion(completion),
                CodeGenerationRequest generation => PerformGeneration(generation),
                CodeTranslationRequest translation => PerformTranslation(translation),
                CodeSummarizationRequest summarization => PerformSummarization(summarization),
                CodeBugDetectionRequest bugDetection => PerformBugDetection(bugDetection),
                CodeBugFixingRequest bugFixing => PerformBugFixing(bugFixing),
                CodeRefactoringRequest refactoring => PerformRefactoring(refactoring),
                CodeUnderstandingRequest understanding => PerformUnderstanding(understanding),
                CodeTestGenerationRequest testGeneration => PerformTestGeneration(testGeneration),
                CodeDocumentationRequest documentation => PerformDocumentation(documentation),
                CodeSearchRequest search => PerformSearch(search),
                CodeCloneDetectionRequest cloneDetection => PerformCloneDetection(cloneDetection),
                CodeReviewRequest codeReview => PerformCodeReview(codeReview),
                _ => CreateFailureResult(request.Task, request.Language, request.RequestId, "Unsupported request type.")
            };

            stopwatch.Stop();
            result.Language = request.Language;
            result.RequestId = request.RequestId;
            result.Success = string.IsNullOrEmpty(result.Error);
            result.Telemetry.ProcessingTimeMs = stopwatch.ElapsedMilliseconds;
            return result;
        }
        catch (ArgumentException ex)
        {
            stopwatch.Stop();
            var failure = CreateFailureResult(request.Task, request.Language, request.RequestId, ex.Message);
            failure.Success = false;
            failure.Error = ex.Message;
            failure.Telemetry.ProcessingTimeMs = stopwatch.ElapsedMilliseconds;
            return failure;
        }
        catch (InvalidOperationException ex)
        {
            stopwatch.Stop();
            var failure = CreateFailureResult(request.Task, request.Language, request.RequestId, ex.Message);
            failure.Success = false;
            failure.Error = ex.Message;
            failure.Telemetry.ProcessingTimeMs = stopwatch.ElapsedMilliseconds;
            return failure;
        }
        catch (NotSupportedException ex)
        {
            stopwatch.Stop();
            var failure = CreateFailureResult(request.Task, request.Language, request.RequestId, ex.Message);
            failure.Success = false;
            failure.Error = ex.Message;
            failure.Telemetry.ProcessingTimeMs = stopwatch.ElapsedMilliseconds;
            return failure;
        }
    }

    public virtual Tensor<T> GetEmbeddings(string code)
    {
        var tokenIds = TokenizeToIds(code);
        var input = ToTokenIdTensor(tokenIds);

        if (Layers.Count == 0)
        {
            return input;
        }

        return Layers[0].Forward(input);
    }

    private static ITokenizer CreateDefaultTokenizer(ProgramLanguage language)
    {
        return ProgramSynthesisTokenizerFactory.CreateDefault(language, splitIdentifiers: true);
    }

    private List<int> TokenizeToIds(string text)
    {
        var options = new EncodingOptions
        {
            AddSpecialTokens = true,
            Truncation = true,
            MaxLength = CodeArchitecture.MaxSequenceLength
        };

        var result = _tokenizer.Encode(text ?? string.Empty, options);
        return result.TokenIds;
    }

    private Tensor<T> ToTokenIdTensor(List<int> tokenIds)
    {
        int seqLen = Math.Max(1, Math.Min(CodeArchitecture.MaxSequenceLength, tokenIds.Count));

        var tensor = new Tensor<T>([seqLen, 1, 1]);
        for (int i = 0; i < seqLen; i++)
        {
            tensor[i, 0, 0] = NumOps.FromDouble(tokenIds[i]);
        }

        return tensor;
    }

    private List<int> ExtractTokenIdsForDecode(Tensor<T> encoding)
    {
        if (encoding.Shape.Length != 3)
        {
            throw new ArgumentException("Expected a rank-3 tensor [seq, batch, dim].", nameof(encoding));
        }

        var seq = encoding.Shape[0];
        var batch = encoding.Shape[1];
        var dim = encoding.Shape[2];

        if (batch <= 0)
        {
            throw new ArgumentException("Batch dimension must be >= 1.", nameof(encoding));
        }

        if (dim == 1)
        {
            var ids = new List<int>(seq);
            for (int t = 0; t < seq; t++)
            {
                ids.Add(Convert.ToInt32(encoding[t, 0, 0]));
            }

            return ids;
        }

        int limit = Math.Min(dim, _tokenizer.VocabularySize);
        var logitsIds = new List<int>(seq);
        for (int t = 0; t < seq; t++)
        {
            int bestIndex = 0;
            double bestValue = double.NegativeInfinity;

            for (int i = 0; i < limit; i++)
            {
                var value = NumOps.ToDouble(encoding[t, 0, i]);
                if (value > bestValue)
                {
                    bestValue = value;
                    bestIndex = i;
                }
            }

            logitsIds.Add(bestIndex);
        }

        return logitsIds;
    }

    protected virtual CodeCompletionResult PerformCompletion(CodeCompletionRequest request)
    {
        var candidates = GenerateHeuristicCompletions(request.Code, request.CursorOffset, request.MaxCandidates);
        return new CodeCompletionResult { Candidates = candidates };
    }

    protected virtual CodeGenerationResult PerformGeneration(CodeGenerationRequest request)
    {
        var generated = TryGenerateFromExamples(request.Language, request.Description, request.Examples)
                        ?? DecodeCode(EncodeCode(request.Description));

        return new CodeGenerationResult { GeneratedCode = generated };
    }

    protected virtual CodeTranslationResult PerformTranslation(CodeTranslationRequest request)
    {
        var translated = DecodeCode(EncodeCode(request.Code));
        return new CodeTranslationResult
        {
            SourceLanguage = request.SourceLanguage,
            TargetLanguage = request.TargetLanguage,
            TranslatedCode = translated
        };
    }

    protected virtual CodeSummarizationResult PerformSummarization(CodeSummarizationRequest request)
    {
        var summary = SummarizeCodeHeuristically(request.Code, request.Language);
        return new CodeSummarizationResult { Summary = summary };
    }

    protected virtual CodeBugDetectionResult PerformBugDetection(CodeBugDetectionRequest request)
    {
        return new CodeBugDetectionResult
        {
            Issues = AnalyzeIssues(request.Code, request.Language, filePath: null)
        };
    }

    protected virtual CodeBugFixingResult PerformBugFixing(CodeBugFixingRequest request)
    {
        var fixedCode = FixTrivialSyntaxIssues(request.Code, request.Language);

        return new CodeBugFixingResult
        {
            FixedCode = fixedCode,
            Diff = CreateWholeDocumentReplaceDiff(request.Code, fixedCode),
            FixedIssues = string.Equals(request.Code, fixedCode, StringComparison.Ordinal)
                ? new List<CodeIssue>()
                : new List<CodeIssue>
                {
                    new()
                    {
                        Severity = CodeIssueSeverity.Warning,
                        Category = CodeIssueCategory.Syntax,
                        Summary = "Applied trivial syntax fixups."
                    }
                }
        };
    }

    protected virtual CodeRefactoringResult PerformRefactoring(CodeRefactoringRequest request)
    {
        var refactored = NormalizeWhitespace(request.Code);
        return new CodeRefactoringResult
        {
            RefactoredCode = refactored,
            Diff = CreateWholeDocumentReplaceDiff(request.Code, refactored)
        };
    }

    protected virtual CodeUnderstandingResult PerformUnderstanding(CodeUnderstandingRequest request)
    {
        var symbols = ExtractSymbols(request.Code, request.Language, request.FilePath);
        var dependencies = ExtractDependencies(request.Code, request.Language);
        var complexity = EstimateComplexity(request.Code);
        var callGraph = BuildCallGraph(request.Code, symbols);
        var hotspots = BuildHotspots(symbols, complexity);
        var securityHotspots = FindSecurityHotspots(request.Code, request.Language, request.FilePath);

        return new CodeUnderstandingResult
        {
            Symbols = symbols,
            Dependencies = dependencies,
            Complexity = complexity,
            CallGraph = callGraph,
            Hotspots = hotspots,
            ControlFlowSummaries = new List<string> { $"Estimated cyclomatic complexity: {complexity.EstimatedCyclomaticComplexity}." },
            DataFlowSummaries = new List<string> { $"Estimated assignments occurrences: {(request.Code ?? string.Empty).Count(c => c == '=')}." },
            SecurityHotspots = securityHotspots
        };
    }

    protected virtual CodeTestGenerationResult PerformTestGeneration(CodeTestGenerationRequest request)
    {
        return new CodeTestGenerationResult
        {
            Tests = GenerateBasicTests(request.Code, request.Language)
        };
    }

    protected virtual CodeDocumentationResult PerformDocumentation(CodeDocumentationRequest request)
    {
        var documentation = GenerateBasicDocumentation(request.Code, request.Language);
        var updatedCode = ApplyBasicDocumentation(request.Code, request.Language, documentation);

        return new CodeDocumentationResult
        {
            Documentation = documentation,
            UpdatedCode = updatedCode
        };
    }

    protected virtual CodeSearchResult PerformSearch(CodeSearchRequest request)
    {
        return new CodeSearchResult
        {
            FiltersApplied = new List<string>(request.Filters),
            Results = SearchCorpus(request.Query, request.Corpus.Documents)
        };
    }

    protected virtual CodeCloneDetectionResult PerformCloneDetection(CodeCloneDetectionRequest request)
    {
        return new CodeCloneDetectionResult
        {
            CloneGroups = DetectClones(request.Corpus.Documents, request.MinSimilarity)
        };
    }

    protected virtual CodeReviewResult PerformCodeReview(CodeReviewRequest request)
    {
        var issues = AnalyzeIssues(request.Code, request.Language, request.FilePath);
        return new CodeReviewResult
        {
            Issues = issues,
            FixSuggestions = BuildFixSuggestions(issues),
            PrioritizedPlan = BuildPrioritizedPlan(issues)
        };
    }

    private static CodeTaskRequestBase CreateLegacyRequest(string code, CodeTask task)
    {
        return task switch
        {
            CodeTask.Completion => new CodeCompletionRequest { Code = code },
            CodeTask.Generation => new CodeGenerationRequest { Description = code },
            CodeTask.Translation => new CodeTranslationRequest { Code = code },
            CodeTask.Summarization => new CodeSummarizationRequest { Code = code },
            CodeTask.BugDetection => new CodeBugDetectionRequest { Code = code },
            CodeTask.BugFixing => new CodeBugFixingRequest { Code = code },
            CodeTask.Refactoring => new CodeRefactoringRequest { Code = code },
            CodeTask.Understanding => new CodeUnderstandingRequest { Code = code },
            CodeTask.TestGeneration => new CodeTestGenerationRequest { Code = code },
            CodeTask.Documentation => new CodeDocumentationRequest { Code = code },
            CodeTask.Search => new CodeSearchRequest { Query = code },
            CodeTask.CloneDetection => new CodeCloneDetectionRequest
            {
                Corpus = new CodeCorpusReference
                {
                    Documents = new List<CodeCorpusDocument>
                    {
                        new() { DocumentId = "doc0", Content = code }
                    }
                }
            },
            CodeTask.CodeReview => new CodeReviewRequest { Code = code },
            _ => new CodeSummarizationRequest { Code = code }
        };
    }

    private static string FormatLegacyResult(CodeTaskResultBase result)
    {
        if (!result.Success)
        {
            return $"Error: {result.Error}";
        }

        return result switch
        {
            CodeCompletionResult completion => string.Join("\n", completion.Candidates.Select(c => c.CompletionText)),
            CodeGenerationResult generation => generation.GeneratedCode,
            CodeTranslationResult translation => translation.TranslatedCode,
            CodeSummarizationResult summarization => summarization.Summary,
            CodeBugDetectionResult bugs => string.Join("\n", bugs.Issues.Select(i => i.Summary)),
            CodeBugFixingResult fixing => fixing.FixedCode,
            CodeRefactoringResult refactor => refactor.RefactoredCode,
            CodeUnderstandingResult understanding => $"Symbols: {understanding.Symbols.Count}, Dependencies: {understanding.Dependencies.Count}",
            CodeTestGenerationResult tests => string.Join("\n", tests.Tests),
            CodeDocumentationResult docs => docs.Documentation,
            CodeSearchResult search => string.Join("\n", search.Results.Select(r => r.SnippetText)),
            CodeCloneDetectionResult clones => $"CloneGroups: {clones.CloneGroups.Count}",
            CodeReviewResult review => string.Join("\n", review.Issues.Select(i => i.Summary)),
            _ => string.Empty
        };
    }

    private static CodeTaskResultBase CreateFailureResult(CodeTask task, ProgramLanguage language, string? requestId, string error)
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

    private static List<CodeCompletionCandidate> GenerateHeuristicCompletions(string code, int? cursorOffset, int maxCandidates)
    {
        var prefix = code ?? string.Empty;
        if (cursorOffset.HasValue && cursorOffset.Value >= 0 && cursorOffset.Value <= prefix.Length)
        {
            prefix = prefix.Substring(0, cursorOffset.Value);
        }

        var missing = ComputeMissingClosers(prefix);
        var candidates = new List<CodeCompletionCandidate>();

        if (missing.Length > 0)
        {
            candidates.Add(new CodeCompletionCandidate { CompletionText = missing, Score = 1.0 });
        }

        if (candidates.Count < maxCandidates)
        {
            candidates.Add(new CodeCompletionCandidate { CompletionText = "\n", Score = 0.5 });
        }

        while (candidates.Count > maxCandidates)
        {
            candidates.RemoveAt(candidates.Count - 1);
        }

        return candidates;
    }

    private static string ComputeMissingClosers(string text)
    {
        int paren = 0;
        int brace = 0;
        int bracket = 0;

        foreach (var ch in text)
        {
            switch (ch)
            {
                case '(':
                    paren++;
                    break;
                case ')':
                    paren = Math.Max(0, paren - 1);
                    break;
                case '{':
                    brace++;
                    break;
                case '}':
                    brace = Math.Max(0, brace - 1);
                    break;
                case '[':
                    bracket++;
                    break;
                case ']':
                    bracket = Math.Max(0, bracket - 1);
                    break;
            }
        }

        var sb = new StringBuilder();
        sb.Append(']', bracket);
        sb.Append('}', brace);
        sb.Append(')', paren);
        return sb.ToString();
    }

    private static string? TryGenerateFromExamples(
        ProgramLanguage language,
        string description,
        List<ProgramInputOutputExample> examples)
    {
        if (examples == null || examples.Count == 0)
        {
            return TryGenerateFromDescription(language, description);
        }

        if (TryMatchSumOperation(examples))
        {
            return GenerateSumSnippet(language);
        }

        if (TryMatchSortOperation(examples))
        {
            return GenerateSortSnippet(language);
        }

        if (TryMatchReverseStringOperation(examples))
        {
            return GenerateReverseStringSnippet(language);
        }

        return TryGenerateFromDescription(language, description);
    }

    private static string? TryGenerateFromDescription(ProgramLanguage language, string description)
    {
        if (string.IsNullOrWhiteSpace(description))
        {
            return null;
        }

        var d = description.ToLowerInvariant();
        if (d.Contains("sum") || d.Contains("add"))
        {
            return GenerateSumSnippet(language);
        }

        if (d.Contains("sort"))
        {
            return GenerateSortSnippet(language);
        }

        if (d.Contains("reverse") && d.Contains("string"))
        {
            return GenerateReverseStringSnippet(language);
        }

        return null;
    }

    private static bool TryMatchSumOperation(List<ProgramInputOutputExample> examples)
    {
        foreach (var example in examples)
        {
            if (!TryParseNumberList(example.Input, out var values))
            {
                return false;
            }

            if (!double.TryParse(example.ExpectedOutput, out var expected))
            {
                return false;
            }

            var sum = values.Sum();
            if (Math.Abs(sum - expected) > 1e-9)
            {
                return false;
            }
        }

        return true;
    }

    private static bool TryMatchSortOperation(List<ProgramInputOutputExample> examples)
    {
        foreach (var example in examples)
        {
            if (!TryParseNumberList(example.Input, out var values))
            {
                return false;
            }

            if (!TryParseNumberList(example.ExpectedOutput, out var expectedList))
            {
                return false;
            }

            var sorted = values.OrderBy(v => v).ToList();
            if (sorted.Count != expectedList.Count)
            {
                return false;
            }

            for (int i = 0; i < sorted.Count; i++)
            {
                if (Math.Abs(sorted[i] - expectedList[i]) > 1e-9)
                {
                    return false;
                }
            }
        }

        return true;
    }

    private static bool TryMatchReverseStringOperation(List<ProgramInputOutputExample> examples)
    {
        foreach (var example in examples)
        {
            var inTrim = example.Input?.Trim() ?? string.Empty;
            var outTrim = example.ExpectedOutput?.Trim() ?? string.Empty;

            if (inTrim.Length == 0)
            {
                return false;
            }

            var reversed = new string(inTrim.Reverse().ToArray());
            if (!string.Equals(reversed, outTrim, StringComparison.Ordinal))
            {
                return false;
            }
        }

        return true;
    }

    private static bool TryParseNumberList(string text, out List<double> values)
    {
        values = new List<double>();
        var trimmed = (text ?? string.Empty).Trim();
        if (!trimmed.StartsWith("[", StringComparison.Ordinal) || !trimmed.EndsWith("]", StringComparison.Ordinal))
        {
            return false;
        }

        var inner = trimmed.Substring(1, trimmed.Length - 2);
        if (string.IsNullOrWhiteSpace(inner))
        {
            return true;
        }

        var parts = inner.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (var p in parts)
        {
            if (!double.TryParse(p.Trim(), out var v))
            {
                values = new List<double>();
                return false;
            }

            values.Add(v);
        }

        return true;
    }

    private static string GenerateSumSnippet(ProgramLanguage language)
    {
        return language switch
        {
            ProgramLanguage.Python => "def solve(nums):\n    return sum(nums)\n",
            ProgramLanguage.CSharp => "public static class Solution\n{\n    public static double Solve(double[] nums)\n    {\n        double sum = 0;\n        foreach (var n in nums) sum += n;\n        return sum;\n    }\n}\n",
            ProgramLanguage.JavaScript => "function solve(nums) {\n  return nums.reduce((a, b) => a + b, 0);\n}\n",
            ProgramLanguage.TypeScript => "function solve(nums: number[]): number {\n  return nums.reduce((a, b) => a + b, 0);\n}\n",
            ProgramLanguage.Java => "public final class Solution {\n  public static double solve(double[] nums) {\n    double sum = 0;\n    for (double n : nums) sum += n;\n    return sum;\n  }\n}\n",
            ProgramLanguage.Go => "package main\n\nfunc solve(nums []float64) float64 {\n\tvar sum float64\n\tfor _, n := range nums {\n\t\tsum += n\n\t}\n\treturn sum\n}\n",
            ProgramLanguage.Rust => "pub fn solve(nums: &[f64]) -> f64 {\n    nums.iter().sum()\n}\n",
            ProgramLanguage.C => "double solve(const double* nums, int len) {\n    double sum = 0;\n    for (int i = 0; i < len; i++) sum += nums[i];\n    return sum;\n}\n",
            ProgramLanguage.CPlusPlus => "double solve(const std::vector<double>& nums) {\n    double sum = 0;\n    for (double n : nums) sum += n;\n    return sum;\n}\n",
            _ => "solve(nums)"
        };
    }

    private static string GenerateSortSnippet(ProgramLanguage language)
    {
        return language switch
        {
            ProgramLanguage.Python => "def solve(nums):\n    return sorted(nums)\n",
            ProgramLanguage.CSharp => "using System.Linq;\n\npublic static class Solution\n{\n    public static double[] Solve(double[] nums)\n    {\n        return nums.OrderBy(n => n).ToArray();\n    }\n}\n",
            ProgramLanguage.JavaScript => "function solve(nums) {\n  return [...nums].sort((a, b) => a - b);\n}\n",
            ProgramLanguage.TypeScript => "function solve(nums: number[]): number[] {\n  return [...nums].sort((a, b) => a - b);\n}\n",
            ProgramLanguage.Go => "package main\n\nimport \"sort\"\n\nfunc solve(nums []float64) []float64 {\n\tcopy := append([]float64{}, nums...)\n\tsort.Float64s(copy)\n\treturn copy\n}\n",
            ProgramLanguage.Rust => "pub fn solve(nums: &[f64]) -> Vec<f64> {\n    let mut v = nums.to_vec();\n    v.sort_by(|a, b| a.partial_cmp(b).unwrap());\n    v\n}\n",
            _ => "solve(nums)"
        };
    }

    private static string GenerateReverseStringSnippet(ProgramLanguage language)
    {
        return language switch
        {
            ProgramLanguage.Python => "def solve(s: str) -> str:\n    return s[::-1]\n",
            ProgramLanguage.CSharp => "using System.Linq;\n\npublic static class Solution\n{\n    public static string Solve(string s)\n    {\n        return new string(s.Reverse().ToArray());\n    }\n}\n",
            ProgramLanguage.JavaScript => "function solve(s) {\n  return s.split(\"\").reverse().join(\"\");\n}\n",
            ProgramLanguage.TypeScript => "function solve(s: string): string {\n  return s.split(\"\").reverse().join(\"\");\n}\n",
            ProgramLanguage.Go => "package main\n\nfunc solve(s string) string {\n\tr := []rune(s)\n\tfor i, j := 0, len(r)-1; i < j; i, j = i+1, j-1 {\n\t\tr[i], r[j] = r[j], r[i]\n\t}\n\treturn string(r)\n}\n",
            ProgramLanguage.Rust => "pub fn solve(s: &str) -> String {\n    s.chars().rev().collect()\n}\n",
            _ => "solve(s)"
        };
    }

    private static string SummarizeCodeHeuristically(string code, ProgramLanguage language)
    {
        var complexity = EstimateComplexity(code);
        var sb = new StringBuilder();
        sb.Append("Language=").Append(language).Append("; ");
        sb.Append("Lines=").Append(complexity.LineCount).Append("; ");
        sb.Append("Chars=").Append(complexity.CharacterCount).Append("; ");
        sb.Append("EstCyclomatic=").Append(complexity.EstimatedCyclomaticComplexity).Append(';');
        return sb.ToString();
    }

    private static List<CodeIssue> AnalyzeIssues(string code, ProgramLanguage language, string? filePath)
    {
        var issues = new List<CodeIssue>();
        var text = code ?? string.Empty;

        if (text.Length == 0)
        {
            issues.Add(new CodeIssue
            {
                Severity = CodeIssueSeverity.Warning,
                Category = CodeIssueCategory.Correctness,
                Summary = "Empty code input.",
                Details = "No code was provided to review.",
                Rationale = "A review requires at least one code unit (file/function) to analyze.",
                FixGuidance = "Provide the code to review (and optionally a file path for more accurate locations).",
                TestGuidance = "Add or run the relevant test suite once code is provided."
            });
            return issues;
        }

        foreach (var marker in new[] { "TODO", "FIXME" })
        {
            var idx = text.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
            if (idx >= 0)
            {
                issues.Add(new CodeIssue
                {
                    Severity = CodeIssueSeverity.Info,
                    Category = CodeIssueCategory.Maintainability,
                    Summary = $"Contains {marker} marker.",
                    Details = $"{marker} markers usually indicate incomplete work or deferred decisions.",
                    Rationale = "Leaving TODO/FIXME markers in shipped code can lead to unfinished features, hidden bugs, and maintenance drift.",
                    FixGuidance = $"Resolve the {marker} item or convert it into a tracked issue with clear acceptance criteria.",
                    TestGuidance = "Add tests for the behavior covered by the TODO/FIXME and run the full test suite after changes.",
                    Location = CreateLocationFromOffset(text, idx, filePath)
                });
            }
        }

        if ((language == ProgramLanguage.JavaScript || language == ProgramLanguage.TypeScript) &&
            text.IndexOf("eval(", StringComparison.OrdinalIgnoreCase) >= 0)
        {
            var idx = text.IndexOf("eval(", StringComparison.OrdinalIgnoreCase);
            issues.Add(new CodeIssue
            {
                Severity = CodeIssueSeverity.Critical,
                Category = CodeIssueCategory.Security,
                Summary = "Use of eval() detected.",
                Details = "eval() executes arbitrary code and can lead to remote code execution if input is user-controlled.",
                Rationale = "Dynamic code execution is a common injection vector and is rarely necessary in production code.",
                FixGuidance = "Replace eval() with a safer alternative (e.g., JSON.parse for JSON input, or a dedicated expression parser with a strict grammar).",
                TestGuidance = "Add negative tests for injection payloads and validate that untrusted input cannot influence executed code paths.",
                Location = CreateLocationFromOffset(text, idx, filePath)
            });
        }

        if (language == ProgramLanguage.CSharp && text.IndexOf("Process.Start", StringComparison.OrdinalIgnoreCase) >= 0)
        {
            var idx = text.IndexOf("Process.Start", StringComparison.OrdinalIgnoreCase);
            issues.Add(new CodeIssue
            {
                Severity = CodeIssueSeverity.Warning,
                Category = CodeIssueCategory.Security,
                Summary = "Process.Start usage can be dangerous with untrusted input.",
                Details = "Process.Start can execute arbitrary commands if arguments are constructed from untrusted input.",
                Rationale = "Command injection and privilege escalation risks increase when launching processes with user-controlled data.",
                FixGuidance = "Avoid launching external processes when possible; if required, use strict allow-lists and safe argument APIs (no shell), and validate inputs.",
                TestGuidance = "Add security tests that attempt command/argument injection and verify sanitization/allow-list behavior.",
                Location = CreateLocationFromOffset(text, idx, filePath)
            });
        }

        var complexity = EstimateComplexity(text);
        if (complexity.EstimatedCyclomaticComplexity > 25)
        {
            issues.Add(new CodeIssue
            {
                Severity = CodeIssueSeverity.Warning,
                Category = CodeIssueCategory.Maintainability,
                Summary = $"High estimated cyclomatic complexity ({complexity.EstimatedCyclomaticComplexity}).",
                Details = "High complexity often correlates with bugs and makes the code harder to test and maintain.",
                Rationale = "Complex control flow increases cognitive load and reduces confidence in changes.",
                FixGuidance = "Refactor into smaller functions, simplify branching logic, and extract reusable helpers.",
                TestGuidance = "Add tests that cover all branches/edge cases, and consider property-based tests for complex logic."
            });
        }

        return issues;
    }

    private static string FixTrivialSyntaxIssues(string code, ProgramLanguage language)
    {
        var fixedCode = code ?? string.Empty;
        var missing = ComputeMissingClosers(fixedCode);
        if (!string.IsNullOrEmpty(missing))
        {
            fixedCode += missing;
        }

        if (language == ProgramLanguage.Python && !fixedCode.EndsWith("\n", StringComparison.Ordinal))
        {
            fixedCode += "\n";
        }

        return fixedCode;
    }

    private static string NormalizeWhitespace(string code)
    {
        var lines = (code ?? string.Empty).Replace("\r\n", "\n").Replace('\r', '\n').Split('\n');
        for (int i = 0; i < lines.Length; i++)
        {
            lines[i] = lines[i].Replace("\t", "    ").TrimEnd();
        }
        return string.Join("\n", lines);
    }

    private static CodeTransformDiff CreateWholeDocumentReplaceDiff(string original, string updated)
    {
        if (string.Equals(original, updated, StringComparison.Ordinal))
        {
            return new CodeTransformDiff();
        }

        return new CodeTransformDiff
        {
            Edits =
            {
                new CodeEditOperation
                {
                    OperationType = CodeEditOperationType.Replace,
                    Span = new CodeSpan
                    {
                        Start = new CodePosition { Line = 1, Column = 1, Offset = 0 },
                        End = new CodePosition { Line = 1, Column = 1, Offset = (original ?? string.Empty).Length }
                    },
                    Text = updated
                }
            }
        };
    }

    private static CodeLocation CreateLocationFromOffset(string text, int offset, string? filePath)
    {
        var safeText = text ?? string.Empty;
        var safeOffset = Math.Max(0, Math.Min(offset, safeText.Length));

        int line = 1;
        int lastLineStart = 0;

        for (int i = 0; i < safeOffset; i++)
        {
            if (safeText[i] == '\n')
            {
                line++;
                lastLineStart = i + 1;
            }
        }

        var col = safeOffset - lastLineStart + 1;

        return new CodeLocation
        {
            FilePath = filePath,
            Span = new CodeSpan
            {
                Start = new CodePosition { Line = line, Column = col, Offset = safeOffset },
                End = new CodePosition { Line = line, Column = col, Offset = safeOffset }
            }
        };
    }

    private static CodeComplexityMetrics EstimateComplexity(string code)
    {
        var text = code ?? string.Empty;
        var lines = text.Replace("\r\n", "\n").Replace('\r', '\n').Split('\n');

        int cyclo = 1;
        foreach (var token in new[] { " if ", " for ", " while ", " case ", " catch ", "&&", "||", "?" })
        {
            var idx = 0;
            while (true)
            {
                idx = text.IndexOf(token, idx, StringComparison.OrdinalIgnoreCase);
                if (idx < 0)
                {
                    break;
                }
                cyclo++;
                idx += token.Length;
            }
        }

        return new CodeComplexityMetrics
        {
            LineCount = lines.Length,
            CharacterCount = text.Length,
            EstimatedCyclomaticComplexity = cyclo
        };
    }

    private static List<CodeSymbol> ExtractSymbols(string code, ProgramLanguage language, string? filePath)
    {
        var symbols = new List<CodeSymbol>();
        var text = code ?? string.Empty;

        IEnumerable<(string Pattern, CodeSymbolKind Kind)> patterns = language switch
        {
            ProgramLanguage.Python => new[]
            {
                (@"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", CodeSymbolKind.Function),
                (@"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b", CodeSymbolKind.Class)
            },
            ProgramLanguage.CSharp => new[]
            {
                (@"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b", CodeSymbolKind.Class),
                (@"\binterface\s+([A-Za-z_][A-Za-z0-9_]*)\b", CodeSymbolKind.Interface),
                (@"\benum\s+([A-Za-z_][A-Za-z0-9_]*)\b", CodeSymbolKind.Enum),
                (@"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)\b", CodeSymbolKind.Struct),
                (@"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", CodeSymbolKind.Method)
            },
            ProgramLanguage.JavaScript or ProgramLanguage.TypeScript => new[]
            {
                (@"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", CodeSymbolKind.Function),
                (@"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b", CodeSymbolKind.Class)
            },
            ProgramLanguage.Java => new[]
            {
                (@"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b", CodeSymbolKind.Class),
                (@"\binterface\s+([A-Za-z_][A-Za-z0-9_]*)\b", CodeSymbolKind.Interface),
                (@"\benum\s+([A-Za-z_][A-Za-z0-9_]*)\b", CodeSymbolKind.Enum)
            },
            ProgramLanguage.Go => new[]
            {
                (@"\bfunc\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", CodeSymbolKind.Function),
                (@"\btype\s+([A-Za-z_][A-Za-z0-9_]*)\s+struct\b", CodeSymbolKind.Struct)
            },
            ProgramLanguage.Rust => new[]
            {
                (@"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", CodeSymbolKind.Function),
                (@"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)\b", CodeSymbolKind.Struct),
                (@"\benum\s+([A-Za-z_][A-Za-z0-9_]*)\b", CodeSymbolKind.Enum)
            },
            _ => Array.Empty<(string, CodeSymbolKind)>()
        };

        foreach (var (pattern, kind) in patterns)
        {
            foreach (Match m in Regex.Matches(
                         text,
                         pattern,
                         RegexOptions.Multiline | RegexOptions.CultureInvariant,
                         RegexTimeout))
            {
                if (m.Groups.Count < 2)
                {
                    continue;
                }

                var name = m.Groups[1].Value;
                symbols.Add(new CodeSymbol
                {
                    Name = name,
                    Kind = kind,
                    Location = CreateLocationFromOffset(text, m.Index, filePath)
                });
            }
        }

        return symbols
            .GroupBy(s => (s.Name, s.Kind))
            .Select(g => g.First())
            .ToList();
    }

    private static List<CodeDependency> ExtractDependencies(string code, ProgramLanguage language)
    {
        var deps = new List<CodeDependency>();
        var text = code ?? string.Empty;
        var lines = text.Replace("\r\n", "\n").Replace('\r', '\n').Split('\n');

        foreach (var trimmed in lines.Select(line => line.Trim()))
        {
            if (trimmed.Length == 0)
            {
                continue;
            }

            switch (language)
            {
                case ProgramLanguage.Python:
                    if (trimmed.StartsWith("import ", StringComparison.Ordinal))
                    {
                        deps.Add(new CodeDependency { Name = trimmed.Substring("import ".Length).Trim(), Kind = "import" });
                    }
                    else if (trimmed.StartsWith("from ", StringComparison.Ordinal) &&
                             trimmed.IndexOf(" import ", StringComparison.Ordinal) >= 0)
                    {
                        deps.Add(new CodeDependency { Name = trimmed, Kind = "from-import" });
                    }
                    break;
                case ProgramLanguage.CSharp:
                    if (trimmed.StartsWith("using ", StringComparison.Ordinal) &&
                        trimmed.EndsWith(";", StringComparison.Ordinal))
                    {
                        deps.Add(new CodeDependency { Name = trimmed.Substring("using ".Length).TrimEnd(';').Trim(), Kind = "using" });
                    }
                    break;
                case ProgramLanguage.Java:
                case ProgramLanguage.JavaScript:
                case ProgramLanguage.TypeScript:
                case ProgramLanguage.Go:
                    if (trimmed.StartsWith("import ", StringComparison.Ordinal))
                    {
                        deps.Add(new CodeDependency { Name = trimmed, Kind = "import" });
                    }
                    break;
                case ProgramLanguage.Rust:
                    if (trimmed.StartsWith("use ", StringComparison.Ordinal))
                    {
                        deps.Add(new CodeDependency { Name = trimmed.TrimEnd(';'), Kind = "use" });
                    }
                    break;
            }
        }

        return deps;
    }

    private static List<CodeCallGraphEdge> BuildCallGraph(string code, List<CodeSymbol> symbols)
    {
        var edges = new List<CodeCallGraphEdge>();
        var text = code ?? string.Empty;
        var functionNames = symbols
            .Where(s => s.Kind is CodeSymbolKind.Function or CodeSymbolKind.Method)
            .Select(s => s.Name)
            .Distinct(StringComparer.Ordinal)
            .ToList();

        foreach (var caller in functionNames)
        {
            foreach (var callee in functionNames)
            {
                if (caller == callee)
                {
                    continue;
                }

                var pattern = callee + "(";
                if (text.IndexOf(pattern, StringComparison.Ordinal) >= 0)
                {
                    edges.Add(new CodeCallGraphEdge { Caller = caller, Callee = callee });
                }
            }
        }

        return edges;
    }

    private static List<CodeHotspot> BuildHotspots(List<CodeSymbol> symbols, CodeComplexityMetrics complexity)
    {
        var hotspots = new List<CodeHotspot>();
        if (complexity.EstimatedCyclomaticComplexity >= 15)
        {
            var top = symbols.FirstOrDefault();
            hotspots.Add(new CodeHotspot
            {
                SymbolName = top?.Name ?? "<module>",
                Reason = "High estimated cyclomatic complexity.",
                Score = Math.Min(1.0, complexity.EstimatedCyclomaticComplexity / 50.0)
            });
        }

        return hotspots;
    }

    private static List<CodeSecurityHotspot> FindSecurityHotspots(string code, ProgramLanguage language, string? filePath)
    {
        var hotspots = new List<CodeSecurityHotspot>();
        var text = code ?? string.Empty;

        void AddHotspot(string category, string needle, string summary)
        {
            var idx = text.IndexOf(needle, StringComparison.OrdinalIgnoreCase);
            if (idx >= 0)
            {
                hotspots.Add(new CodeSecurityHotspot
                {
                    Category = category,
                    Summary = summary,
                    Location = CreateLocationFromOffset(text, idx, filePath)
                });
            }
        }

        if (language == ProgramLanguage.JavaScript || language == ProgramLanguage.TypeScript)
        {
            AddHotspot("Injection", "eval(", "Dynamic evaluation can enable injection vulnerabilities.");
        }

        if (language == ProgramLanguage.CSharp)
        {
            AddHotspot("Execution", "Process.Start", "Starting external processes can be dangerous with untrusted input.");
        }

        AddHotspot("Secrets", "password", "Potential secret handling detected; ensure secrets are not logged or hard-coded.");

        return hotspots;
    }

    private static List<string> GenerateBasicTests(string code, ProgramLanguage language)
    {
        var symbols = ExtractSymbols(code, language, filePath: null);
        var firstFunc = symbols.FirstOrDefault(s => s.Kind is CodeSymbolKind.Function or CodeSymbolKind.Method)?.Name ?? "solve";

        return language switch
        {
            ProgramLanguage.CSharp => new List<string>
            {
                "using Xunit;",
                "",
                "public class GeneratedTests",
                "{",
                "    [Fact]",
                $"    public void {firstFunc}_BasicCase()",
                "    {",
                "        Assert.True(true);",
                "    }",
                "}"
            },
            ProgramLanguage.Python => new List<string>
            {
                "import unittest",
                "",
                "class GeneratedTests(unittest.TestCase):",
                $"    def test_{firstFunc}_basic(self):",
                "        self.assertTrue(True)",
                "",
                "if __name__ == '__main__':",
                "    unittest.main()"
            },
            _ => new List<string>
            {
                $"// Tests for {firstFunc}",
                "/* Add assertions here */"
            }
        };
    }

    private static string GenerateBasicDocumentation(string code, ProgramLanguage language)
    {
        var complexity = EstimateComplexity(code);
        return $"Auto-generated documentation: lines={complexity.LineCount}, complexity={complexity.EstimatedCyclomaticComplexity}.";
    }

    private static string ApplyBasicDocumentation(string code, ProgramLanguage language, string documentation)
    {
        if (string.IsNullOrEmpty(code))
        {
            return code ?? string.Empty;
        }

        return language switch
        {
            ProgramLanguage.CSharp => "/// <summary>\n/// " + documentation + "\n/// </summary>\n" + code,
            ProgramLanguage.Python => "\"\"\"" + documentation + "\"\"\"\n" + code,
            _ => "/* " + documentation + " */\n" + code
        };
    }

    private static List<CodeSearchHit> SearchCorpus(string query, List<CodeCorpusDocument> documents)
    {
        var hits = new List<CodeSearchHit>();
        if (string.IsNullOrWhiteSpace(query) || documents == null || documents.Count == 0)
        {
            return hits;
        }

        var terms = query.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
            .Select(t => t.Trim())
            .Where(t => t.Length > 0)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

        foreach (var doc in documents)
        {
            var content = doc.Content ?? string.Empty;
            var score = 0.0;
            var firstHit = -1;

            foreach (var term in terms)
            {
                var idx = content.IndexOf(term, StringComparison.OrdinalIgnoreCase);
                if (idx >= 0)
                {
                    score += 1.0;
                    if (firstHit < 0)
                    {
                        firstHit = idx;
                    }
                }
            }

            if (score <= 0)
            {
                continue;
            }

            var snippet = ExtractSnippet(content, firstHit, 120);
            hits.Add(new CodeSearchHit
            {
                Score = score,
                SnippetText = snippet,
                Location = CreateLocationFromOffset(content, Math.Max(0, firstHit), doc.FilePath),
                MatchType = CodeMatchType.Lexical,
                Provenance = doc.DocumentId.Length > 0 ? new CodeProvenance { SourcePath = doc.FilePath } : null,
                MatchExplanation = "Lexical match on query terms."
            });
        }

        return hits
            .OrderByDescending(h => h.Score)
            .Take(20)
            .ToList();
    }

    private static List<CodeCloneGroup> DetectClones(List<CodeCorpusDocument> documents, double minSimilarity)
    {
        var groups = new List<CodeCloneGroup>();
        if (documents == null || documents.Count < 2)
        {
            return groups;
        }

        var normalized = documents
            .Select(d => (Doc: d, Tokens: NormalizeForCloneDetection(d.Content)))
            .ToList();

        for (int i = 0; i < normalized.Count; i++)
        {
            for (int j = i + 1; j < normalized.Count; j++)
            {
                var sim = Jaccard(normalized[i].Tokens, normalized[j].Tokens);
                if (sim < minSimilarity)
                {
                    continue;
                }

                groups.Add(new CodeCloneGroup
                {
                    Similarity = sim,
                    CloneType = CodeCloneType.Type3,
                    MatchType = CodeMatchType.Structural,
                    NormalizationSummary = "lowercased + tokenized on non-alphanumerics",
                    Instances = new List<CodeCloneInstance>
                    {
                        new() { SnippetText = ExtractSnippet(normalized[i].Doc.Content, 0, 200), Location = new CodeLocation { FilePath = normalized[i].Doc.FilePath } },
                        new() { SnippetText = ExtractSnippet(normalized[j].Doc.Content, 0, 200), Location = new CodeLocation { FilePath = normalized[j].Doc.FilePath } }
                    },
                    RefactorSuggestions = new List<string> { "Consider extracting shared logic into a shared function/module." }
                });
            }
        }

        return groups;
    }

    private static HashSet<string> NormalizeForCloneDetection(string content)
    {
        var set = new HashSet<string>(StringComparer.Ordinal);
        var text = (content ?? string.Empty).ToLowerInvariant();
        var current = new StringBuilder();

        void Flush()
        {
            if (current.Length > 0)
            {
                set.Add(current.ToString());
                current.Clear();
            }
        }

        foreach (var ch in text)
        {
            if (char.IsLetterOrDigit(ch) || ch == '_')
            {
                current.Append(ch);
            }
            else
            {
                Flush();
            }
        }

        Flush();
        return set;
    }

    private static double Jaccard(HashSet<string> a, HashSet<string> b)
    {
        if (a.Count == 0 && b.Count == 0)
        {
            return 1.0;
        }

        var intersection = a.Intersect(b).Count();
        var union = a.Union(b).Count();
        return union == 0 ? 0.0 : (double)intersection / union;
    }

    private static string ExtractSnippet(string text, int startIndex, int maxLen)
    {
        var s = text ?? string.Empty;
        if (s.Length == 0)
        {
            return string.Empty;
        }

        var start = Math.Max(0, Math.Min(startIndex, s.Length - 1));
        var len = Math.Min(maxLen, s.Length - start);
        return s.Substring(start, len);
    }

    private static List<string> BuildPrioritizedPlan(List<CodeIssue> issues)
    {
        return issues
            .OrderByDescending(i => i.Severity)
            .ThenBy(i => i.Category)
            .Select(i => $"{i.Severity}: {i.Summary}")
            .ToList();
    }

    private static List<CodeFixSuggestion> BuildFixSuggestions(List<CodeIssue> issues)
    {
        return issues
            .Select(i => new CodeFixSuggestion
            {
                Summary = $"Address: {i.Summary}",
                Rationale = i.Rationale ?? i.Details,
                FixGuidance = i.FixGuidance,
                TestGuidance = i.TestGuidance
            })
            .ToList();
    }
}
