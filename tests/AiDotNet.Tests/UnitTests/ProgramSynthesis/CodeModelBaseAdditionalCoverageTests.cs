using System.Text;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.ProgramSynthesis.Engines;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

public sealed class CodeModelBaseAdditionalCoverageTests
{
    [Fact]
    public void EncodeDecode_HandlesCommonShapesAndValidatesArguments()
    {
        var model = new MinimalCodeModel(CreateArchitecture());

        Assert.Throws<ArgumentNullException>(() => model.DecodeCode(null!));

        var rank2 = new Tensor<double>([2, 2]);
        Assert.Throws<ArgumentException>(() => model.DecodeCode(rank2));

        var dim1 = new Tensor<double>([2, 1, 1]);
        dim1[0, 0, 0] = 5;
        dim1[1, 0, 0] = 6;
        var decoded = model.DecodeCode(dim1);
        Assert.NotNull(decoded);

        var logits = new Tensor<double>([2, 1, 3]);
        logits[0, 0, 0] = 0.1;
        logits[0, 0, 1] = 0.9;
        logits[0, 0, 2] = 0.2;
        logits[1, 0, 0] = 0.3;
        logits[1, 0, 1] = 0.2;
        logits[1, 0, 2] = 0.4;
        decoded = model.DecodeCode(logits);
        Assert.NotNull(decoded);
    }

    [Fact]
    public void PerformCompletion_OffersMissingClosersAndCursorOffset()
    {
        var model = new MinimalCodeModel(CreateArchitecture());
        var request = new CodeCompletionRequest
        {
            Language = ProgramLanguage.CSharp,
            Code = "if (true) {",
            CursorOffset = "if (true) {".Length,
            MaxCandidates = 2
        };

        var result = model.PerformTask(request);
        var completion = Assert.IsType<CodeCompletionResult>(result);
        Assert.NotEmpty(completion.Candidates);
        Assert.Contains("}", completion.Candidates[0].CompletionText);
    }

    [Fact]
    public void PerformGeneration_Heuristics_HandleSortAndReverse()
    {
        var model = new MinimalCodeModel(CreateArchitecture());

        var sort = model.PerformTask(new CodeGenerationRequest
        {
            Language = ProgramLanguage.Python,
            Description = "sort numbers",
            Examples = new List<ProgramInputOutputExample>
            {
                new() { Input = "3,1,2", ExpectedOutput = "1,2,3" }
            }
        });

        var sortResult = Assert.IsType<CodeGenerationResult>(sort);
        Assert.Contains("sort", sortResult.GeneratedCode, StringComparison.OrdinalIgnoreCase);

        var reverse = model.PerformTask(new CodeGenerationRequest
        {
            Language = ProgramLanguage.Python,
            Description = "reverse string",
            Examples = new List<ProgramInputOutputExample>
            {
                new() { Input = "abc", ExpectedOutput = "cba" }
            }
        });

        var reverseResult = Assert.IsType<CodeGenerationResult>(reverse);
        Assert.Contains("return", reverseResult.GeneratedCode, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void PerformBugFixing_FixesTrivialClosersAndNormalizesPythonNewline()
    {
        var model = new MinimalCodeModel(CreateArchitecture());

        var request = new CodeBugFixingRequest
        {
            Language = ProgramLanguage.Python,
            Code = "def f(x):\n    if x > 0:\n        return x"
        };

        var result = Assert.IsType<CodeBugFixingResult>(model.PerformTask(request));
        Assert.True(result.Success);
        Assert.EndsWith("\n", result.FixedCode);
    }

    [Fact]
    public void PerformCodeReview_ProducesLocationsAndSecurityIssues()
    {
        var model = new MinimalCodeModel(CreateArchitecture());

        var code = new StringBuilder()
            .AppendLine("// TODO: remove")
            .AppendLine("using System;")
            .AppendLine("class C { void M() { System.Diagnostics.Process.Start(\"cmd.exe\"); } }")
            .AppendLine("if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ if(true){ } } } } } } } } } } } } } } } } } } } } } } } } } } }")
            .ToString();

        var result = Assert.IsType<CodeReviewResult>(model.PerformTask(new CodeReviewRequest
        {
            Language = ProgramLanguage.CSharp,
            Code = code,
            FilePath = "a.cs"
        }));

        Assert.True(result.Success);
        Assert.NotEmpty(result.Issues);
        Assert.Contains(result.Issues, issue => issue.Location?.FilePath == "a.cs");
    }

    [Fact]
    public void PerformTask_ReturnsFailureResult_ForUnsupportedRequestType()
    {
        var model = new MinimalCodeModel(CreateArchitecture());
        var unknown = new UnknownRequest();

        var result = model.PerformTask(unknown);
        Assert.False(result.Success);
        Assert.False(string.IsNullOrWhiteSpace(result.Error));
    }

    [Theory]
    [InlineData("arg")]
    [InlineData("invalid")]
    [InlineData("unsupported")]
    public void PerformTask_ConvertsKnownExceptionsToFailureResult(string kind)
    {
        var model = new ThrowingCodeModel(CreateArchitecture(), kind);

        var result = model.PerformTask(new CodeSummarizationRequest
        {
            Language = ProgramLanguage.CSharp,
            Code = "class C {}",
            RequestId = "x"
        });

        Assert.False(result.Success);
        Assert.Equal("x", result.RequestId);
        Assert.False(string.IsNullOrWhiteSpace(result.Error));
        Assert.True(result.Telemetry.ProcessingTimeMs >= 0);
    }

    private static CodeSynthesisArchitecture<double> CreateArchitecture() => new(
        synthesisType: SynthesisType.Neural,
        targetLanguage: ProgramLanguage.Generic,
        codeTaskType: CodeTask.Generation,
        maxSequenceLength: 64,
        vocabularySize: 256,
        numEncoderLayers: 0,
        numDecoderLayers: 0,
        dropoutRate: 0.0,
        usePositionalEncoding: false);

    private class MinimalCodeModel : CodeModelBase<double>
    {
        public MinimalCodeModel(CodeSynthesisArchitecture<double> architecture)
            : base(architecture, new CrossEntropyLoss<double>())
        {
            InitializeLayers();
        }

        protected override void InitializeLayers()
        {
        }

        public override void Train(Tensor<double> input, Tensor<double> expectedOutput)
        {
        }

        public override void UpdateParameters(Vector<double> parameters)
        {
        }

        public override ModelMetadata<double> GetModelMetadata() => new()
        {
            ModelType = ModelType.Transformer,
            AdditionalInfo = new Dictionary<string, object> { { "ModelName", nameof(MinimalCodeModel) } }
        };

        protected override void SerializeNetworkSpecificData(BinaryWriter writer)
        {
            writer.Write(0);
        }

        protected override void DeserializeNetworkSpecificData(BinaryReader reader)
        {
            _ = reader.ReadInt32();
        }

        protected override IFullModel<double, Tensor<double>, Tensor<double>> CreateNewInstance() => new MinimalCodeModel(CodeArchitecture);
    }

    private sealed class ThrowingCodeModel : MinimalCodeModel
    {
        private readonly string _kind;

        public ThrowingCodeModel(CodeSynthesisArchitecture<double> architecture, string kind)
            : base(architecture)
        {
            _kind = kind;
        }

        protected override CodeSummarizationResult PerformSummarization(CodeSummarizationRequest request)
        {
            return _kind switch
            {
                "arg" => throw new ArgumentException("bad arg"),
                "invalid" => throw new InvalidOperationException("invalid op"),
                _ => throw new NotSupportedException("not supported")
            };
        }
    }

    private sealed class UnknownRequest : CodeTaskRequestBase
    {
        public override CodeTask Task => (CodeTask)999;
    }
}
