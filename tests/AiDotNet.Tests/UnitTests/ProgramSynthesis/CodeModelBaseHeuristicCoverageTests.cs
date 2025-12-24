using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;
using AiDotNet.Tests.UnitTests.ProgramSynthesis.Fakes;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ProgramSynthesis;

public sealed class CodeModelBaseHeuristicCoverageTests
{
    [Fact]
    public void PerformTask_HeuristicRequests_ReturnStructuredOutputs()
    {
        var model = FakeCodeModel.CreateDefault(targetLanguage: ProgramLanguage.CSharp);

        var completion = (CodeCompletionResult)model.PerformTask(new CodeCompletionRequest
        {
            Code = "if (x",
            Language = ProgramLanguage.CSharp,
            CursorOffset = 5,
            MaxCandidates = 3
        });
        Assert.True(completion.Success);
        Assert.NotEmpty(completion.Candidates);

        var generation = (CodeGenerationResult)model.PerformTask(new CodeGenerationRequest
        {
            Language = ProgramLanguage.Python,
            Description = "Return the sum of two ints",
            Examples = new List<ProgramInputOutputExample>
            {
                new() { Input = "1 2", ExpectedOutput = "3" }
            }
        });
        Assert.True(generation.Success);
        Assert.NotNull(generation.GeneratedCode);

        var translation = (CodeTranslationResult)model.PerformTask(new CodeTranslationRequest
        {
            Language = ProgramLanguage.CSharp,
            SourceLanguage = ProgramLanguage.CSharp,
            TargetLanguage = ProgramLanguage.Python,
            Code = "int Add(int a,int b){return a+b;}"
        });
        Assert.True(translation.Success);

        var summarization = (CodeSummarizationResult)model.PerformTask(new CodeSummarizationRequest
        {
            Language = ProgramLanguage.CSharp,
            Code = "public class C { void M() { } }"
        });
        Assert.True(summarization.Success);
        Assert.False(string.IsNullOrWhiteSpace(summarization.Summary));

        var bugDetection = (CodeBugDetectionResult)model.PerformTask(new CodeBugDetectionRequest
        {
            Language = ProgramLanguage.CSharp,
            Code = "var x = eval(userInput);"
        });
        Assert.True(bugDetection.Success);
        Assert.NotNull(bugDetection.Issues);

        var bugFixing = (CodeBugFixingResult)model.PerformTask(new CodeBugFixingRequest
        {
            Language = ProgramLanguage.CSharp,
            Code = "class C { void M(){ int x=1; }"
        });
        Assert.True(bugFixing.Success);
        Assert.NotNull(bugFixing.FixedCode);

        var refactoring = (CodeRefactoringResult)model.PerformTask(new CodeRefactoringRequest
        {
            Language = ProgramLanguage.CSharp,
            Code = "class  C{  void  M( ){ } }"
        });
        Assert.True(refactoring.Success);

        var understanding = (CodeUnderstandingResult)model.PerformTask(new CodeUnderstandingRequest
        {
            Language = ProgramLanguage.CSharp,
            FilePath = "Demo.cs",
            Code = "using System; namespace N { class C { void M() { Console.WriteLine(1); } } }"
        });
        Assert.True(understanding.Success);
        Assert.NotNull(understanding.Symbols);
        Assert.NotNull(understanding.Dependencies);

        var testGeneration = (CodeTestGenerationResult)model.PerformTask(new CodeTestGenerationRequest
        {
            Language = ProgramLanguage.CSharp,
            Code = "int Add(int a,int b){return a+b;}"
        });
        Assert.True(testGeneration.Success);
        Assert.NotNull(testGeneration.Tests);

        var documentation = (CodeDocumentationResult)model.PerformTask(new CodeDocumentationRequest
        {
            Language = ProgramLanguage.CSharp,
            Code = "int Add(int a,int b){return a+b;}"
        });
        Assert.True(documentation.Success);
        Assert.NotNull(documentation.Documentation);

        var search = (CodeSearchResult)model.PerformTask(new CodeSearchRequest
        {
            Language = ProgramLanguage.CSharp,
            Query = "Add",
            Corpus = new CodeCorpusReference
            {
                Documents =
                [
                    new CodeCorpusDocument { DocumentId = "d1", Language = ProgramLanguage.CSharp, Content = "int Add(int a,int b){return a+b;}" }
                ]
            }
        });
        Assert.True(search.Success);
        Assert.NotNull(search.Results);

        var cloneDetection = (CodeCloneDetectionResult)model.PerformTask(new CodeCloneDetectionRequest
        {
            Language = ProgramLanguage.CSharp,
            Corpus = new CodeCorpusReference
            {
                Documents =
                [
                    new CodeCorpusDocument { DocumentId = "a", Language = ProgramLanguage.CSharp, Content = "int Add(int a,int b){return a+b;}" },
                    new CodeCorpusDocument { DocumentId = "b", Language = ProgramLanguage.CSharp, Content = "int Sum(int x,int y){return x+y;}" }
                ]
            },
            MinSimilarity = 0.5
        });
        Assert.True(cloneDetection.Success);
        Assert.NotNull(cloneDetection.CloneGroups);

        var review = (CodeReviewResult)model.PerformTask(new CodeReviewRequest
        {
            Language = ProgramLanguage.CSharp,
            FilePath = "Demo.cs",
            Code = "var password = \"123\"; // TODO"
        });
        Assert.True(review.Success);
        Assert.NotNull(review.Issues);
    }
}
