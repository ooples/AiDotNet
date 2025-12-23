using AiDotNet.ProgramSynthesis.Engines;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Options;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.ProgramSynthesis.Serving;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;
using AiDotNet.Tokenization.Vocabulary;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

public sealed class ProgramSynthesisCoverageSmokeTests
{
    [Fact]
    public void ProgramSynthesisOptions_HasSensibleDefaults()
    {
        var options = new ProgramSynthesisOptions();

        Assert.Equal(ProgramSynthesisModelKind.CodeT5, options.ModelKind);
        Assert.Equal(ProgramLanguage.Generic, options.TargetLanguage);
        Assert.Equal(CodeTask.Generation, options.DefaultTask);
        Assert.Equal(SynthesisType.Neural, options.SynthesisType);
        Assert.True(options.VocabularySize > 0);
        Assert.True(options.MaxSequenceLength > 0);
        Assert.True(options.NumEncoderLayers > 0);
        Assert.True(options.NumDecoderLayers > 0);
    }

    [Fact]
    public void CodeT5_GetModelMetadata_ProducesTransformerMetadata()
    {
        var tokenizer = new SimpleTestTokenizer(vocabularySize: 16);
        var architecture = CreateSmallArchitecture(CodeTask.Generation, useDataFlow: false, numDecoderLayers: 1, vocabularySize: 32);

        var model = new CodeT5<double>(architecture, tokenizer: tokenizer);
        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.Equal("CodeT5", metadata.AdditionalInfo["ModelName"]);
        Assert.True(metadata.ModelData.Length > 0);
    }

    [Fact]
    public void GraphCodeBERT_GetModelMetadata_IncludesUseDataFlow()
    {
        var tokenizer = new SimpleTestTokenizer(vocabularySize: 16);
        var architecture = CreateSmallArchitecture(CodeTask.BugDetection, useDataFlow: true, numDecoderLayers: 0, vocabularySize: 32);

        var model = new GraphCodeBERT<double>(architecture, tokenizer: tokenizer);
        var metadata = model.GetModelMetadata();

        Assert.NotNull(metadata);
        Assert.Equal("GraphCodeBERT", metadata.AdditionalInfo["ModelName"]);
        Assert.Equal(true, metadata.AdditionalInfo["UseDataFlow"]);
        Assert.True(metadata.ModelData.Length > 0);
    }

    [Fact]
    public void ServingProgramExecutionEngine_ReturnsTrue_OnSuccessfulExecution()
    {
        var engine = new ServingProgramExecutionEngine(new FakeServingClient(success: true));

        var ok = engine.TryExecute(
            ProgramLanguage.Python,
            sourceCode: "print('hi')",
            input: "",
            out var output,
            out var error,
            cancellationToken: CancellationToken.None);

        Assert.True(ok);
        Assert.Equal("ok", output);
        Assert.Null(error);
    }

    [Fact]
    public void ServingProgramExecutionEngine_ReturnsFalse_OnFailedExecution()
    {
        var engine = new ServingProgramExecutionEngine(new FakeServingClient(success: false));

        var ok = engine.TryExecute(
            ProgramLanguage.Python,
            sourceCode: "print('hi')",
            input: "",
            out var output,
            out var error,
            cancellationToken: CancellationToken.None);

        Assert.False(ok);
        Assert.Equal(string.Empty, output);
        Assert.Equal("fail", error);
    }

    private static CodeSynthesisArchitecture<double> CreateSmallArchitecture(
        CodeTask taskType,
        bool useDataFlow,
        int numDecoderLayers,
        int vocabularySize)
    {
        return new CodeSynthesisArchitecture<double>(
            synthesisType: SynthesisType.Neural,
            targetLanguage: ProgramLanguage.Generic,
            codeTaskType: taskType,
            numEncoderLayers: 1,
            numDecoderLayers: numDecoderLayers,
            numHeads: 2,
            modelDimension: 16,
            feedForwardDimension: 32,
            maxSequenceLength: 16,
            vocabularySize: vocabularySize,
            maxProgramLength: 8,
            dropoutRate: 0.0,
            usePositionalEncoding: true,
            useDataFlow: useDataFlow);
    }

    private sealed class FakeServingClient : IProgramSynthesisServingClient
    {
        private readonly bool _success;

        public FakeServingClient(bool success)
        {
            _success = success;
        }

        public Task<CodeTaskResultBase> ExecuteCodeTaskAsync(CodeTaskRequestBase request, CancellationToken cancellationToken)
            => throw new NotSupportedException();

        public Task<ProgramExecuteResponse> ExecuteProgramAsync(ProgramExecuteRequest request, CancellationToken cancellationToken)
        {
            return Task.FromResult(new ProgramExecuteResponse
            {
                Success = _success,
                Language = request.Language,
                ExitCode = _success ? 0 : 1,
                StdOut = _success ? "ok" : string.Empty,
                Error = _success ? null : "fail"
            });
        }

        public Task<ProgramEvaluateIoResponse> EvaluateProgramIoAsync(ProgramEvaluateIoRequest request, CancellationToken cancellationToken)
            => throw new NotSupportedException();

        public Task<SqlExecuteResponse> ExecuteSqlAsync(SqlExecuteRequest request, CancellationToken cancellationToken)
            => throw new NotSupportedException();
    }

    private sealed class SimpleTestTokenizer : ITokenizer
    {
        private readonly Vocabulary _vocabulary;
        private readonly SpecialTokens _specialTokens = SpecialTokens.Default();

        public SimpleTestTokenizer(int vocabularySize)
        {
            _vocabulary = new Vocabulary();
            for (int i = 0; i < vocabularySize; i++)
            {
                _vocabulary.AddToken($"t{i}");
            }
        }

        public IVocabulary Vocabulary => _vocabulary;

        public SpecialTokens SpecialTokens => _specialTokens;

        public int VocabularySize => _vocabulary.Size;

        public TokenizationResult Encode(string text, EncodingOptions? options = null)
        {
            var tokens = Tokenize(text);
            var ids = ConvertTokensToIds(tokens);
            return new TokenizationResult(tokens, ids);
        }

        public List<TokenizationResult> EncodeBatch(List<string> texts, EncodingOptions? options = null)
        {
            return texts.Select(text => Encode(text, options)).ToList();
        }

        public string Decode(List<int> tokenIds, bool skipSpecialTokens = true)
        {
            var tokens = ConvertIdsToTokens(tokenIds);
            return string.Join(" ", tokens);
        }

        public List<string> DecodeBatch(List<List<int>> tokenIdsBatch, bool skipSpecialTokens = true)
        {
            return tokenIdsBatch.Select(batch => Decode(batch, skipSpecialTokens)).ToList();
        }

        public List<string> Tokenize(string text)
        {
            return string.IsNullOrWhiteSpace(text)
                ? []
                : text.Split(new[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries).ToList();
        }

        public List<int> ConvertTokensToIds(List<string> tokens)
        {
            return tokens.Select(token => _vocabulary.AddToken(token)).ToList();
        }

        public List<string> ConvertIdsToTokens(List<int> ids)
        {
            return ids.Select(id => _vocabulary.GetToken(id) ?? _specialTokens.UnkToken).ToList();
        }
    }
}
