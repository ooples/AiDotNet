using System;
using System.Threading.Tasks;
using AiDotNet;
using AiDotNet.Data.Loaders;
using AiDotNet.Regression;
using AiDotNet.RetrievalAugmentedGeneration.Embeddings;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Asserts that an embedding model passed to ConfigureEmbeddingModel is surfaced on the built result.
/// </summary>
/// <remarks>
/// An embedding model is a preprocessing/transform component (text → vector), not a trainable model, so
/// it is not routed through training; it is kept on the result for inference-time transforms. Before
/// this wiring the configured embedder was stored in a field nothing read and dropped on the floor.
/// </remarks>
public class ConfiguredEmbeddingModelTests
{
    private static (Matrix<double> X, Vector<double> Y) BuildData(int rows = 60, int cols = 3)
    {
        var x = new Matrix<double>(rows, cols);
        var y = new Vector<double>(rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) x[i, j] = Math.Sin((i + j) * 0.15) + (i * 0.01);
            y[i] = Math.Sin((i + cols) * 0.15) + (i * 0.01);
        }

        return (x, y);
    }

    [Fact(Timeout = 120000)]
    public async Task ConfiguredEmbeddingModel_IsSurfacedOnResult()
    {
        var (x, y) = BuildData();
        var embedder = new StubEmbeddingModel<double>(embeddingDimension: 16);

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureEmbeddingModel(embedder)
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        Assert.Same(embedder, result.EmbeddingModel);
    }

    [Fact(Timeout = 120000)]
    public async Task NoConfiguredEmbeddingModel_LeavesResultPropertyNull()
    {
        var (x, y) = BuildData();

        var result = await new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(new MultipleRegression<double>())
            .ConfigureDataLoader(new InMemoryDataLoader<double, Matrix<double>, Vector<double>>(x, y))
            .BuildAsync();

        Assert.Null(result.EmbeddingModel);
    }
}
