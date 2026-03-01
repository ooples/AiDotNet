using AiDotNet.Document;
using AiDotNet.Document.GraphBased;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Document;

/// <summary>
/// Integration tests for graph-based document models.
/// </summary>
public class GraphBasedDocumentTests
{
    private static NeuralNetworkArchitecture<double> CreateArchitecture()
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 64,
            inputWidth: 64,
            inputDepth: 3,
            outputSize: 9);
    }

    private static Tensor<double> CreateSmallImage(int size = 64)
    {
        int totalSize = 1 * 3 * size * size;
        var data = new Vector<double>(totalSize);
        for (int i = 0; i < totalSize; i++)
            data[i] = 0.5;
        return new Tensor<double>(new[] { 1, 3, size, size }, data);
    }

    #region DocGCN Tests

    [Fact]
    public void DocGCN_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new DocGCN<double>(arch);
        Assert.NotNull(model);
    }

    [Fact]
    public void DocGCN_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new DocGCN<double>(arch);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void DocGCN_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new DocGCN<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("DocGCN", meta.Name);
    }

    #endregion

    #region PICK Tests

    [Fact]
    public void PICK_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new PICK<double>(arch);
        Assert.NotNull(model);
    }

    [Fact]
    public void PICK_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new PICK<double>(arch);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void PICK_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new PICK<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("PICK", meta.Name);
    }

    #endregion

    #region TRIE Tests

    [Fact]
    public void TRIE_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new TRIE<double>(arch);
        Assert.NotNull(model);
    }

    [Fact]
    public void TRIE_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new TRIE<double>(arch);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void TRIE_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new TRIE<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("TRIE", meta.Name);
    }

    #endregion

    #region LayoutGraph Tests

    [Fact]
    public void LayoutGraph_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new LayoutGraph<double>(arch);
        Assert.NotNull(model);
    }

    [Fact]
    public void LayoutGraph_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new LayoutGraph<double>(arch);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact]
    public void LayoutGraph_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new LayoutGraph<double>(arch);
        var meta = model.GetModelMetadata();
        Assert.Equal("LayoutGraph", meta.Name);
    }

    #endregion

    #region Cross-Model Tests

    [Fact]
    public void AllGraphBasedModels_SupportsTraining_InNativeMode()
    {
        var arch = CreateArchitecture();
        var models = new DocumentNeuralNetworkBase<double>[]
        {
            new DocGCN<double>(arch),
            new PICK<double>(arch),
            new TRIE<double>(arch),
            new LayoutGraph<double>(arch),
        };

        foreach (var model in models)
        {
            // All native mode models support training
            Assert.True(model.SupportsTraining);
        }
    }

    #endregion
}
