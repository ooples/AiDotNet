using AiDotNet.RetrievalAugmentedGeneration.Configuration;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.RetrievalAugmentedGeneration;

/// <summary>
/// Integration tests for RAG configuration classes:
/// RAGConfiguration, ChunkingConfig, DocumentStoreConfig,
/// EmbeddingConfig, RetrievalConfig, RerankingConfig,
/// QueryExpansionConfig, ContextCompressionConfig.
/// </summary>
public class RAGConfigurationIntegrationTests
{
    #region RAGConfiguration

    [Fact]
    public void RAGConfiguration_DefaultsAreNotNull()
    {
        var config = new RAGConfiguration<double>();
        Assert.NotNull(config.DocumentStore);
        Assert.NotNull(config.Chunking);
        Assert.NotNull(config.Embedding);
        Assert.NotNull(config.Retrieval);
        Assert.NotNull(config.Reranking);
        Assert.NotNull(config.QueryExpansion);
        Assert.NotNull(config.ContextCompression);
    }

    [Fact]
    public void RAGConfiguration_CanSetSubconfigs()
    {
        var config = new RAGConfiguration<double>();
        config.Chunking.ChunkSize = 2000;
        config.Chunking.ChunkOverlap = 400;

        Assert.Equal(2000, config.Chunking.ChunkSize);
        Assert.Equal(400, config.Chunking.ChunkOverlap);
    }

    #endregion

    #region ChunkingConfig

    [Fact]
    public void ChunkingConfig_DefaultValues()
    {
        var config = new ChunkingConfig();
        Assert.Equal(string.Empty, config.Strategy);
        Assert.Equal(1000, config.ChunkSize);
        Assert.Equal(200, config.ChunkOverlap);
        Assert.NotNull(config.Parameters);
        Assert.Empty(config.Parameters);
    }

    [Fact]
    public void ChunkingConfig_CanSetParameters()
    {
        var config = new ChunkingConfig
        {
            Strategy = "sentence",
            ChunkSize = 500,
            ChunkOverlap = 50
        };
        config.Parameters["maxSentences"] = 10;

        Assert.Equal("sentence", config.Strategy);
        Assert.Equal(500, config.ChunkSize);
        Assert.Equal(10, config.Parameters["maxSentences"]);
    }

    #endregion

    #region DocumentStoreConfig

    [Fact]
    public void DocumentStoreConfig_DefaultConstructor()
    {
        var config = new DocumentStoreConfig();
        Assert.NotNull(config);
    }

    #endregion

    #region EmbeddingConfig

    [Fact]
    public void EmbeddingConfig_DefaultConstructor()
    {
        var config = new EmbeddingConfig();
        Assert.NotNull(config);
    }

    #endregion

    #region RetrievalConfig

    [Fact]
    public void RetrievalConfig_DefaultConstructor()
    {
        var config = new RetrievalConfig();
        Assert.NotNull(config);
    }

    #endregion

    #region RerankingConfig

    [Fact]
    public void RerankingConfig_DefaultConstructor()
    {
        var config = new RerankingConfig();
        Assert.NotNull(config);
    }

    #endregion

    #region QueryExpansionConfig

    [Fact]
    public void QueryExpansionConfig_DefaultConstructor()
    {
        var config = new QueryExpansionConfig();
        Assert.NotNull(config);
    }

    #endregion

    #region ContextCompressionConfig

    [Fact]
    public void ContextCompressionConfig_DefaultConstructor()
    {
        var config = new ContextCompressionConfig();
        Assert.NotNull(config);
    }

    #endregion
}
