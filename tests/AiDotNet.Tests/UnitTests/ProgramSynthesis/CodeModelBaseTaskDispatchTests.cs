using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.Tests.UnitTests.ProgramSynthesis.Fakes;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

public class CodeModelBaseTaskDispatchTests
{
    [Fact]
    public void PerformTask_AllTasks_ReturnsStructuredResults()
    {
        var model = FakeCodeModel.CreateDefault(targetLanguage: ProgramLanguage.CSharp);

        var requests = new CodeTaskRequestBase[]
        {
            new CodeCompletionRequest { Code = "class C {", Language = ProgramLanguage.CSharp, RequestId = "completion" },
            new CodeGenerationRequest
            {
                Description = "sum two numbers",
                Language = ProgramLanguage.Python,
                RequestId = "generation",
                Examples = new List<ProgramInputOutputExample>
                {
                    new() { Input = "1,2", ExpectedOutput = "3" }
                }
            },
            new CodeTranslationRequest
            {
                Code = "print('hello')",
                SourceLanguage = ProgramLanguage.Python,
                TargetLanguage = ProgramLanguage.CSharp,
                Language = ProgramLanguage.Python,
                RequestId = "translation"
            },
            new CodeSummarizationRequest { Code = "int Add(int a,int b){return a+b;}", Language = ProgramLanguage.CSharp, RequestId = "summarization" },
            new CodeBugDetectionRequest { Code = "var x = 1; // TODO", Language = ProgramLanguage.CSharp, RequestId = "bug-detect" },
            new CodeBugFixingRequest { Code = "if(true){\n", Language = ProgramLanguage.CSharp, RequestId = "bug-fix" },
            new CodeRefactoringRequest { Code = "int  x=1;\n", Language = ProgramLanguage.CSharp, RequestId = "refactor" },
            new CodeUnderstandingRequest { Code = "public class C { void M(){} }", Language = ProgramLanguage.CSharp, RequestId = "understand" },
            new CodeTestGenerationRequest { Code = "int Add(int a,int b){return a+b;}", Language = ProgramLanguage.CSharp, RequestId = "tests" },
            new CodeDocumentationRequest { Code = "int Add(int a,int b){return a+b;}", Language = ProgramLanguage.CSharp, RequestId = "docs" },
            new CodeSearchRequest
            {
                Query = "Add",
                Language = ProgramLanguage.CSharp,
                RequestId = "search",
                Corpus = new CodeCorpusReference
                {
                    Documents = new List<CodeCorpusDocument>
                    {
                        new CodeCorpusDocument { DocumentId = "doc1", FilePath = "a.cs", Content = "int Add(int a,int b){return a+b;}" }
                    }
                }
            },
            new CodeCloneDetectionRequest
            {
                Language = ProgramLanguage.CSharp,
                RequestId = "clones",
                Corpus = new CodeCorpusReference
                {
                    Documents = new List<CodeCorpusDocument>
                    {
                        new CodeCorpusDocument { DocumentId = "d1", FilePath = "a.cs", Content = "int Add(int a,int b){return a+b;}" },
                        new CodeCorpusDocument { DocumentId = "d2", FilePath = "b.cs", Content = "int Add(int x,int y){return x+y;}" }
                    }
                },
                MinSimilarity = 0.1
            },
            new CodeReviewRequest { Code = "var x = 1; // TODO", Language = ProgramLanguage.CSharp, RequestId = "review" }
        };

        foreach (var request in requests)
        {
            var result = model.PerformTask(request);
            Assert.NotNull(result);
            Assert.Equal(request.Task, result.Task);
            Assert.Equal(request.Language, result.Language);
            Assert.Equal(request.RequestId, result.RequestId);
            Assert.True(result.Success);
        }

        Assert.IsType<CodeCompletionResult>(model.PerformTask(requests[0]));
        Assert.IsType<CodeGenerationResult>(model.PerformTask(requests[1]));
        Assert.IsType<CodeTranslationResult>(model.PerformTask(requests[2]));
        Assert.IsType<CodeSummarizationResult>(model.PerformTask(requests[3]));
        Assert.IsType<CodeBugDetectionResult>(model.PerformTask(requests[4]));
        Assert.IsType<CodeBugFixingResult>(model.PerformTask(requests[5]));
        Assert.IsType<CodeRefactoringResult>(model.PerformTask(requests[6]));
        Assert.IsType<CodeUnderstandingResult>(model.PerformTask(requests[7]));
        Assert.IsType<CodeTestGenerationResult>(model.PerformTask(requests[8]));
        Assert.IsType<CodeDocumentationResult>(model.PerformTask(requests[9]));
        Assert.IsType<CodeSearchResult>(model.PerformTask(requests[10]));
        Assert.IsType<CodeCloneDetectionResult>(model.PerformTask(requests[11]));
        Assert.IsType<CodeReviewResult>(model.PerformTask(requests[12]));
    }
}
