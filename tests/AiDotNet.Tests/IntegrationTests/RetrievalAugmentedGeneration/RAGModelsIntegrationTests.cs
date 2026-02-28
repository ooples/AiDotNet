using AiDotNet.RetrievalAugmentedGeneration.Models;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.RetrievalAugmentedGeneration;

/// <summary>
/// Integration tests for RAG model classes:
/// Document, VectorDocument, GroundedAnswer, ThoughtNode,
/// ToolInvocation, MultiStepReasoningResult, ReasoningStepResult,
/// ToolAugmentedResult, VerifiedReasoningResult, VerifiedReasoningStep.
/// </summary>
public class RAGModelsIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region Document

    [Fact]
    public void Document_DefaultConstructor_HasDefaults()
    {
        var doc = new Document<double>();
        Assert.Equal(string.Empty, doc.Id);
        Assert.Equal(string.Empty, doc.Content);
        Assert.NotNull(doc.Metadata);
        Assert.Empty(doc.Metadata);
        Assert.False(doc.HasRelevanceScore);
        Assert.Null(doc.Embedding);
    }

    [Fact]
    public void Document_IdContentConstructor_SetsValues()
    {
        var doc = new Document<double>("doc1", "Hello world");
        Assert.Equal("doc1", doc.Id);
        Assert.Equal("Hello world", doc.Content);
    }

    [Fact]
    public void Document_FullConstructor_SetsMetadata()
    {
        var metadata = new Dictionary<string, object> { { "author", "John" }, { "year", 2024 } };
        var doc = new Document<double>("doc2", "Content here", metadata);
        Assert.Equal("doc2", doc.Id);
        Assert.Equal("Content here", doc.Content);
        Assert.Equal("John", doc.Metadata["author"]);
        Assert.Equal(2024, doc.Metadata["year"]);
    }

    [Fact]
    public void Document_RelevanceScore_CanBeSet()
    {
        var doc = new Document<double>("doc3", "Test");
        doc.RelevanceScore = 0.95;
        doc.HasRelevanceScore = true;
        Assert.Equal(0.95, doc.RelevanceScore, Tolerance);
        Assert.True(doc.HasRelevanceScore);
    }

    [Fact]
    public void Document_Embedding_CanBeSet()
    {
        var doc = new Document<double>("doc4", "Test");
        var embedding = new Vector<double>(new double[] { 0.1, 0.2, 0.3 });
        doc.Embedding = embedding;
        Assert.NotNull(doc.Embedding);
        Assert.Equal(3, doc.Embedding.Length);
    }

    #endregion

    #region VectorDocument

    [Fact]
    public void VectorDocument_DefaultConstructor_HasDefaults()
    {
        var vDoc = new VectorDocument<double>();
        Assert.NotNull(vDoc.Document);
        Assert.NotNull(vDoc.Embedding);
    }

    [Fact]
    public void VectorDocument_Constructor_SetsDocumentAndEmbedding()
    {
        var doc = new Document<double>("vd1", "Vector doc content");
        var embedding = new Vector<double>(new double[] { 0.5, -0.3, 0.8 });
        var vDoc = new VectorDocument<double>(doc, embedding);

        Assert.Equal("vd1", vDoc.Document.Id);
        Assert.Equal("Vector doc content", vDoc.Document.Content);
        Assert.Equal(3, vDoc.Embedding.Length);
        Assert.Equal(0.5, vDoc.Embedding[0], Tolerance);
    }

    #endregion

    #region GroundedAnswer

    [Fact]
    public void GroundedAnswer_DefaultConstructor_HasDefaults()
    {
        var answer = new GroundedAnswer<double>();
        Assert.Equal(string.Empty, answer.Answer);
        Assert.Equal(string.Empty, answer.Query);
        Assert.NotNull(answer.SourceDocuments);
        Assert.Empty(answer.SourceDocuments);
        Assert.NotNull(answer.Citations);
        Assert.Empty(answer.Citations);
        Assert.Equal(0.0, answer.ConfidenceScore, Tolerance);
    }

    [Fact]
    public void GroundedAnswer_AnswerAndSources_Constructor()
    {
        var docs = new List<Document<double>>
        {
            new Document<double>("s1", "Source 1"),
            new Document<double>("s2", "Source 2")
        };
        var answer = new GroundedAnswer<double>("The answer is 42.", docs.AsReadOnly());

        Assert.Equal("The answer is 42.", answer.Answer);
        Assert.Equal(2, answer.SourceDocuments.Count);
    }

    [Fact]
    public void GroundedAnswer_FullConstructor_SetsAllProperties()
    {
        var docs = new List<Document<double>> { new Document<double>("s1", "Source") }.AsReadOnly();
        var citations = new List<string> { "[1] Source document" }.AsReadOnly();

        var answer = new GroundedAnswer<double>("What is AI?", "AI is machine learning.",
            docs, citations, 0.92);

        Assert.Equal("What is AI?", answer.Query);
        Assert.Equal("AI is machine learning.", answer.Answer);
        Assert.Single(answer.SourceDocuments);
        Assert.Single(answer.Citations);
        Assert.Equal(0.92, answer.ConfidenceScore, Tolerance);
    }

    #endregion

    #region ThoughtNode

    [Fact]
    public void ThoughtNode_DefaultProperties()
    {
        var node = new ThoughtNode<double>();
        Assert.Equal(string.Empty, node.Thought);
        Assert.NotNull(node.Children);
        Assert.Empty(node.Children);
        Assert.Equal(0.0, node.EvaluationScore, Tolerance);
        Assert.NotNull(node.RetrievedDocuments);
        Assert.Empty(node.RetrievedDocuments);
        Assert.Equal(0, node.Depth);
        Assert.Null(node.Parent);
    }

    [Fact]
    public void ThoughtNode_TreeStructure()
    {
        var root = new ThoughtNode<double> { Thought = "Root", Depth = 0 };
        var child1 = new ThoughtNode<double> { Thought = "Child1", Depth = 1, Parent = root };
        var child2 = new ThoughtNode<double> { Thought = "Child2", Depth = 1, Parent = root };
        root.Children.Add(child1);
        root.Children.Add(child2);

        Assert.Equal(2, root.Children.Count);
        Assert.Equal(root, child1.Parent);
        Assert.Equal(root, child2.Parent);
        Assert.Equal(1, child1.Depth);
    }

    [Fact]
    public void ThoughtNode_WithRetrievedDocuments()
    {
        var node = new ThoughtNode<double> { Thought = "Research", EvaluationScore = 0.8 };
        node.RetrievedDocuments.Add(new Document<double>("d1", "Document content"));

        Assert.Equal(0.8, node.EvaluationScore, Tolerance);
        Assert.Single(node.RetrievedDocuments);
    }

    #endregion

    #region ToolInvocation

    [Fact]
    public void ToolInvocation_DefaultProperties()
    {
        var tool = new ToolInvocation();
        Assert.Equal(string.Empty, tool.ToolName);
        Assert.Equal(string.Empty, tool.Input);
        Assert.Equal(string.Empty, tool.Output);
        Assert.False(tool.Success);
    }

    [Fact]
    public void ToolInvocation_SetProperties()
    {
        var tool = new ToolInvocation
        {
            ToolName = "Calculator",
            Input = "2+2",
            Output = "4",
            Success = true
        };

        Assert.Equal("Calculator", tool.ToolName);
        Assert.Equal("2+2", tool.Input);
        Assert.Equal("4", tool.Output);
        Assert.True(tool.Success);
    }

    #endregion
}
