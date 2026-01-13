using System;
using System.Collections.Generic;
using System.Linq;
using Xunit;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tokenization.Interfaces;
using AiDotNet.Tokenization.Models;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks
{
    public class TransformerEmbeddingNetworkTests
    {
        private class MockTokenizer : ITokenizer
        {
            public IVocabulary Vocabulary => throw new NotImplementedException();
            public SpecialTokens SpecialTokens => new SpecialTokens();
            public int VocabularySize => 1000;

            public TokenizationResult Encode(string text, EncodingOptions? options = null)
            {
                // Simple deterministic encoding based on char codes
                var ids = new List<int>();
                for (int i = 0; i < text.Length; i++)
                {
                    ids.Add((text[i] % 1000));
                }
                return new TokenizationResult(ids.Select(id => id.ToString()).ToList(), ids);
            }

            public List<TokenizationResult> EncodeBatch(List<string> texts, EncodingOptions? options = null)
            {
                return texts.Select(t => Encode(t)).ToList();
            }

            public string Decode(List<int> tokenIds, bool skipSpecialTokens = true) => "";
            public List<string> DecodeBatch(List<List<int>> tokenIdsBatch, bool skipSpecialTokens = true) => new List<string>();
            public List<string> Tokenize(string text) => new List<string>();
            public List<int> ConvertTokensToIds(List<string> tokens) => new List<int>();
            public List<string> ConvertIdsToTokens(List<int> ids) => new List<string>();
        }

        [Fact]
        public void TransformerEmbeddingNetwork_Embed_ReturnsCorrectDimension()
        {
            // Arrange
            int embeddingDim = 64;
            int vocabSize = 1000;
            var architecture = new NeuralNetworkArchitecture<double>(
                InputType.OneDimensional,
                NeuralNetworkTaskType.Regression, // Used as generic feature extraction
                inputSize: 512, // Max sequence length
                outputSize: embeddingDim
            );
            var tokenizer = new MockTokenizer();
            
            var model = new TransformerEmbeddingNetwork<double>(
                architecture, 
                tokenizer,
                optimizer: null,
                vocabSize: vocabSize,
                embeddingDimension: embeddingDim,
                numLayers: 2, 
                numHeads: 4, 
                feedForwardDim: 128);

            // Act
            var embedding = model.Embed("hello world");

            // Assert
            Assert.NotNull(embedding);
            Assert.Equal(embeddingDim, embedding.Length);
        }

        [Fact]
        public void TransformerEmbeddingNetwork_EmbedBatch_ReturnsCorrectMatrix()
        {
            // Arrange
            int embeddingDim = 32;
            int vocabSize = 1000;
            var architecture = new NeuralNetworkArchitecture<double>(
                InputType.OneDimensional,
                NeuralNetworkTaskType.Regression,
                inputSize: 512,
                outputSize: embeddingDim
            );
            var tokenizer = new MockTokenizer();
            var texts = new List<string> { "text 1", "text 2", "text 3" };
            
            var model = new TransformerEmbeddingNetwork<double>(
                architecture, 
                tokenizer,
                optimizer: null,
                vocabSize: vocabSize,
                embeddingDimension: embeddingDim, 
                numLayers: 1, 
                numHeads: 2);

            // Act
            var embeddings = model.EmbedBatch(texts);

            // Assert
            Assert.NotNull(embeddings);
            Assert.Equal(3, embeddings.Rows);
            Assert.Equal(embeddingDim, embeddings.Columns);
        }
        
        [Fact]
        public void TransformerEmbeddingNetwork_PoolingStrategy_ClsToken_Works()
        {
            // Arrange
            int embeddingDim = 16;
            int vocabSize = 1000;
            var architecture = new NeuralNetworkArchitecture<double>(
                InputType.OneDimensional,
                NeuralNetworkTaskType.Regression,
                inputSize: 512,
                outputSize: embeddingDim
            );
            var tokenizer = new MockTokenizer();
            
            var model = new TransformerEmbeddingNetwork<double>(
                architecture, 
                tokenizer,
                optimizer: null,
                vocabSize: vocabSize,
                embeddingDimension: embeddingDim, 
                numLayers: 1, 
                numHeads: 2,
                poolingStrategy: TransformerEmbeddingNetwork<double>.PoolingStrategy.ClsToken);

            // Act
            var embedding = model.Embed("test");

            // Assert
            Assert.Equal(embeddingDim, embedding.Length);
        }
    }
}
