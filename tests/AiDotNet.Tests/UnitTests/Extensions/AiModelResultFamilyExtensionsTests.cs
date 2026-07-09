using System;
using AiDotNet.Diffusion.Extensions;
using AiDotNet.Graphs.Extensions;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Results;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Transformers.Extensions;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Extensions;

/// <summary>
/// Regression tests for #1836 — the transformer / diffusion / graph slice of family-specific
/// extension methods on <see cref="AiModelResult{T, TInput, TOutput}"/>. The radiance-field
/// slice has its own test file (<c>AiModelResultRadianceFieldExtensionsTests</c>); these three
/// families are grouped together because their tests all follow the same pattern
/// (wrong-family throws with a clear message).
/// </summary>
public class AiModelResultFamilyExtensionsTests
{
    private static AiModelResult<float, Matrix<float>, Vector<float>> BuildWrongFamilyResult()
    {
        var model = new SimpleRegression<float>();
        return new AiModelResult<float, Matrix<float>, Vector<float>> { Model = model };
    }

    // ---------------- Diffusion ----------------

    [Fact]
    public void Diffusion_Generate_OnNonDiffusionResult_ThrowsWithModelTypeNamed()
    {
        var result = BuildWrongFamilyResult();
        var ex = Assert.Throws<InvalidOperationException>(
            () => result.Generate(new[] { 1, 3, 8, 8 }, numInferenceSteps: 4));
        Assert.Contains("IDiffusionModel", ex.Message);
        Assert.Contains("SimpleRegression", ex.Message);
    }

    [Fact]
    public void Diffusion_EncodeToLatent_OnNonLatentDiffusionResult_ThrowsWithModelTypeNamed()
    {
        var result = BuildWrongFamilyResult();
        var image = new Tensor<float>(new[] { 1, 3, 8, 8 }, new Vector<float>(new float[192]));
        var ex = Assert.Throws<InvalidOperationException>(
            () => result.EncodeToLatent(image));
        Assert.Contains("ILatentDiffusionModel", ex.Message);
        Assert.Contains("SimpleRegression", ex.Message);
    }

    [Fact]
    public void Diffusion_Generate_NullResult_ThrowsArgumentNullException()
    {
        AiModelResult<float, Tensor<float>, Tensor<float>>? result = null;
        Assert.Throws<ArgumentNullException>(() => result!.Generate(new[] { 1, 3, 8, 8 }));
    }

    // ---------------- Transformer ----------------

    [Fact]
    public void Transformer_GenerateGreedy_OnWrongShapedResult_ThrowsWithGuidance()
    {
        var result = BuildWrongFamilyResult();
        var startTokens = new Tensor<float>(new[] { 1, 2 }, new Vector<float>(new float[] { 3f, 7f }));
        var ex = Assert.Throws<InvalidOperationException>(
            () => result.GenerateGreedy(startTokens, maxNewTokens: 2));
        // Wrong TInput/TOutput signature (Matrix / Vector, not Tensor / Tensor) → gate message.
        Assert.Contains("Tensor", ex.Message);
    }

    [Fact]
    public void Transformer_GenerateGreedy_NegativeTokenCount_Throws()
    {
        // Use a Tensor<T>-typed result so we get past the shape gate to argument validation.
        var startTokens = new Tensor<float>(new[] { 1, 2 }, new Vector<float>(new float[] { 3f, 7f }));
        var result = new AiModelResult<float, Tensor<float>, Tensor<float>>
        {
            Model = new AiDotNet.NeuralRadianceFields.Models.NeRF<float>(
                positionEncodingLevels: 2, directionEncodingLevels: 2, hiddenDim: 8, numLayers: 1,
                colorHiddenDim: 4, colorNumLayers: 1, useHierarchicalSampling: false,
                renderSamples: 2, renderNearBound: 1.0, renderFarBound: 3.0, learningRate: 1e-3),
        };
        Assert.Throws<ArgumentOutOfRangeException>(
            () => result.GenerateGreedy(startTokens, maxNewTokens: 0));
    }

    [Fact]
    public void Transformer_GenerateSampled_NullStartTokens_ThrowsArgumentNullException()
    {
        var result = new AiModelResult<float, Tensor<float>, Tensor<float>>
        {
            Model = new AiDotNet.NeuralRadianceFields.Models.NeRF<float>(
                positionEncodingLevels: 2, directionEncodingLevels: 2, hiddenDim: 8, numLayers: 1,
                colorHiddenDim: 4, colorNumLayers: 1, useHierarchicalSampling: false,
                renderSamples: 2, renderNearBound: 1.0, renderFarBound: 3.0, learningRate: 1e-3),
        };
        Assert.Throws<ArgumentNullException>(
            () => result.GenerateSampled(null!, maxNewTokens: 4, temperature: 1.0f));
    }

    // ---------------- Graph ----------------

    [Fact]
    public void Graph_PredictOnGraph_OnNonGraphResult_ThrowsWithModelTypeNamed()
    {
        var result = BuildWrongFamilyResult();
        var adj = new Tensor<float>(new[] { 3, 3 }, new Vector<float>(new float[9]));
        var feat = new Tensor<float>(new[] { 3, 2 }, new Vector<float>(new float[6]));
        var ex = Assert.Throws<InvalidOperationException>(
            () => result.PredictOnGraph(adj, feat));
        Assert.Contains("NodeClassificationModel", ex.Message);
        Assert.Contains("SimpleRegression", ex.Message);
    }

    [Fact]
    public void Graph_PredictLink_NullAdjacency_ThrowsArgumentNullException()
    {
        var result = BuildWrongFamilyResult();
        var feat = new Tensor<float>(new[] { 3, 2 }, new Vector<float>(new float[6]));
        // NOTE: type gate fires before the arg-null check because the model isn't a
        // NodeClassificationModel — that's fine; the test's job is to prove the extension
        // rejects the wrong-family call cleanly.
        Assert.Throws<InvalidOperationException>(
            () => result.PredictLink(null!, feat, sourceNode: 0, targetNode: 1));
    }
}
