using System.Reflection;
using System.Threading;
using AiDotNet.ProgramSynthesis.Engines;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Execution;
using AiDotNet.ProgramSynthesis.Interfaces;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.Tests.UnitTests.ProgramSynthesis.Fakes;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

public sealed class NeuralProgramSynthesizerCoverageTests
{
    [Fact(Timeout = 60000)]
    public async Task ValidateProgram_RejectsEmptyCode_AndHonorsMaxProgramLength()
    {
        var synthesizer = CreateSynthesizer(executionEngine: null, maxProgramLength: 2);

        Assert.Throws<ArgumentNullException>(() => synthesizer.ValidateProgram(global::AiDotNet.Tests.TestNull.Of<AiDotNet.ProgramSynthesis.Models.Program<double>>()));

        var empty = new Program<double>("", ProgramLanguage.Generic, isValid: false, fitnessScore: 0.0, complexity: 0);
        Assert.False(synthesizer.ValidateProgram(empty));

        var tooComplex = new Program<double>("line1\nline2\nline3\n", ProgramLanguage.Generic, isValid: false, fitnessScore: 0.0, complexity: 3);
        Assert.False(synthesizer.ValidateProgram(tooComplex));
    }

    [Fact(Timeout = 60000)]
    public async Task ValidateProgram_ValidatesSqlAndGenericBrackets()
    {
        var synthesizer = CreateSynthesizer(executionEngine: null, maxProgramLength: 100);

        var sql = new Program<double>("SELECT 1;", ProgramLanguage.SQL, isValid: false, fitnessScore: 0.0, complexity: 1);
        Assert.True(synthesizer.ValidateProgram(sql));

        var bracketsOk = new Program<double>("function f(){ return (1); }", ProgramLanguage.Generic, isValid: false, fitnessScore: 0.0, complexity: 1);
        Assert.True(synthesizer.ValidateProgram(bracketsOk));

        var bracketsBad = new Program<double>("function f(){ return (1; }", ProgramLanguage.Generic, isValid: false, fitnessScore: 0.0, complexity: 1);
        Assert.False(synthesizer.ValidateProgram(bracketsBad));
    }

    [Fact(Timeout = 60000)]
    public async Task EvaluateProgram_ReturnsExpectedDefaults()
    {
        var synthesizer = CreateSynthesizer(executionEngine: null, maxProgramLength: 100);

        var invalid = new Program<double>("x", ProgramLanguage.Generic, isValid: false, fitnessScore: 0.0, complexity: 1);
        Assert.Equal(0.0, synthesizer.EvaluateProgram(invalid, new ProgramInput<double>()));

        var validNoExamples = new Program<double>("x", ProgramLanguage.Generic, isValid: true, fitnessScore: 0.0, complexity: 1);
        Assert.Equal(0.5, synthesizer.EvaluateProgram(validNoExamples, new ProgramInput<double> { Examples = new List<ProgramInputOutputExample>() }));
    }

    [Fact(Timeout = 60000)]
    public async Task SynthesizeProgram_WithExamples_StopsWhenNoExecutionEngineAvailable()
    {
        var synthesizer = CreateSynthesizer(executionEngine: null, maxProgramLength: 100);

        var input = new ProgramInput<double>
        {
            TargetLanguage = ProgramLanguage.Generic,
            Description = "Return the input unchanged.",
            Examples = new List<ProgramInputOutputExample>
            {
                new() { Input = "hello", ExpectedOutput = "hello" }
            }
        };

        var result = synthesizer.SynthesizeProgram(input);
        Assert.True(result.IsValid);
        Assert.Equal(0.0, result.FitnessScore, precision: 6);
    }

    [Fact(Timeout = 60000)]
    public async Task BuildFeedbackInput_ReturnsNull_WhenNoFailures()
    {
        var synthesizer = CreateSynthesizer(executionEngine: new EchoExecutionEngine(), maxProgramLength: 100);

        var program = new Program<double>(
            sourceCode: "x",
            language: ProgramLanguage.Generic,
            isValid: true,
            fitnessScore: 0.0,
            complexity: 1);

        var input = new ProgramInput<double>
        {
            TargetLanguage = ProgramLanguage.Generic,
            Examples = new List<ProgramInputOutputExample>
            {
                new() { Input = "a", ExpectedOutput = "a" }
            }
        };

        var method = typeof(NeuralProgramSynthesizer<double>)
            .GetMethod("BuildFeedbackInput", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(method);

        var feedback = method.Invoke(synthesizer, new object[] { program, input });
        Assert.Null(feedback);
    }

    [Fact]
    public void ValidateProgram_FallsBackToGenericChecks_OnlyWhenTheGrammarCannotLoad()
    {
        // A grammar library that cannot load, even wrapped by a type initializer, falls back to the generic checks.
        var wrapped = new TypeInitializationException("Grammar", new DllNotFoundException("tree-sitter-csharp"));
        var program = new Program<double>("public class C { }", ProgramLanguage.CSharp, isValid: false, fitnessScore: 0.0, complexity: 1);
        Assert.True(CreateSynthesizer(new ThrowingSyntaxProvider(wrapped)).ValidateProgram(program));

        // The provider's own failure is a defect, not a missing grammar, so it surfaces instead of passing generically.
        Assert.Throws<InvalidOperationException>(() =>
            CreateSynthesizer(new ThrowingSyntaxProvider(new InvalidOperationException("parser bug"))).ValidateProgram(program));
    }

    private sealed class ThrowingSyntaxProvider : AiDotNet.ProgramSynthesis.Interfaces.ICodeSyntaxProvider
    {
        private readonly Exception _failure;
        public ThrowingSyntaxProvider(Exception failure) => _failure = failure;
        public bool? TryValidate(ProgramLanguage language, string sourceCode) => throw _failure;
        public bool TryExtractAst(string code, ProgramLanguage language,
            AiDotNet.ProgramSynthesis.Tokenization.CodeTokenizationPipelineOptions options,
            out System.Collections.Generic.List<CodeAstNode> nodes,
            out System.Collections.Generic.List<CodeAstEdge> edges) => throw _failure;
    }

    private static NeuralProgramSynthesizer<double> CreateSynthesizer(AiDotNet.ProgramSynthesis.Interfaces.ICodeSyntaxProvider provider)
    {
        var architecture = new CodeSynthesisArchitecture<double>(
            synthesisType: SynthesisType.Neural,
            targetLanguage: ProgramLanguage.Generic,
            codeTaskType: CodeTask.Generation,
            numEncoderLayers: 0,
            numDecoderLayers: 0,
            numHeads: 1,
            modelDimension: 16,
            feedForwardDimension: 32,
            maxSequenceLength: 64,
            vocabularySize: 256,
            maxProgramLength: 64,
            dropoutRate: 0.0,
            usePositionalEncoding: false);

        return new NeuralProgramSynthesizer<double>(architecture, FakeCodeModel.CreateDefault(ProgramLanguage.Generic),
            codeSyntaxProvider: provider);
    }

    private static NeuralProgramSynthesizer<double> CreateSynthesizer(IProgramExecutionEngine? executionEngine, int maxProgramLength)
    {
        var architecture = new CodeSynthesisArchitecture<double>(
            synthesisType: SynthesisType.Neural,
            targetLanguage: ProgramLanguage.Generic,
            codeTaskType: CodeTask.Generation,
            numEncoderLayers: 0,
            numDecoderLayers: 0,
            numHeads: 1,
            modelDimension: 16,
            feedForwardDimension: 32,
            maxSequenceLength: 64,
            vocabularySize: 256,
            maxProgramLength: maxProgramLength,
            dropoutRate: 0.0,
            usePositionalEncoding: false);

        var codeModel = FakeCodeModel.CreateDefault(ProgramLanguage.Generic);
        return new NeuralProgramSynthesizer<double>(architecture, codeModel, executionEngine: executionEngine);
    }

    private sealed class EchoExecutionEngine : IProgramExecutionEngine
    {
        public bool TryExecute(
            ProgramLanguage language,
            string sourceCode,
            string input,
            out string output,
            out string? errorMessage,
            CancellationToken cancellationToken = default)
        {
            output = input;
            errorMessage = null;
            return true;
        }

        public Task<ProgramExecuteResponse> ExecuteAsync(
            ProgramExecuteRequest request,
            CancellationToken cancellationToken = default) =>
            Task.FromResult(new ProgramExecuteResponse
            {
                Success = true,
                Language = request.Language,
                ExitCode = 0,
                StdOut = request.StdIn ?? string.Empty
            });
    }
}
