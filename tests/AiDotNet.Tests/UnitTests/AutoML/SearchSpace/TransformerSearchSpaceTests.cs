using System.Collections.Generic;
using AiDotNet.AutoML.SearchSpace;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.SearchSpace
{
    /// <summary>
    /// Unit tests for the TransformerSearchSpace class.
    /// </summary>
    public class TransformerSearchSpaceTests
    {
        [Fact]
        public void TransformerSearchSpace_Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert
            Assert.NotNull(searchSpace);
            Assert.NotNull(searchSpace.Operations);
        }

        [Fact]
        public void TransformerSearchSpace_Operations_ContainsTransformerOperations()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert
            Assert.Contains("identity", searchSpace.Operations);
            Assert.Contains("self_attention", searchSpace.Operations);
            Assert.Contains("multi_head_attention_4", searchSpace.Operations);
            Assert.Contains("multi_head_attention_8", searchSpace.Operations);
            Assert.Contains("multi_head_attention_16", searchSpace.Operations);
            Assert.Contains("feed_forward_2x", searchSpace.Operations);
            Assert.Contains("feed_forward_4x", searchSpace.Operations);
            Assert.Contains("layer_norm", searchSpace.Operations);
            Assert.Contains("glu", searchSpace.Operations);
        }

        [Fact]
        public void TransformerSearchSpace_Operations_HasNineOperations()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert
            Assert.Equal(9, searchSpace.Operations.Count);
        }

        [Fact]
        public void TransformerSearchSpace_MaxNodes_IsTwentyFour()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert
            Assert.Equal(24, searchSpace.MaxNodes);
        }

        [Fact]
        public void TransformerSearchSpace_InputChannels_IsSevenSixtyEight()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert - Common embedding dimension
            Assert.Equal(768, searchSpace.InputChannels);
        }

        [Fact]
        public void TransformerSearchSpace_OutputChannels_IsSevenSixtyEight()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert - Same as input for transformer encoder
            Assert.Equal(768, searchSpace.OutputChannels);
        }

        [Fact]
        public void TransformerSearchSpace_AttentionHeads_ContainsExpectedValues()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert
            Assert.NotNull(searchSpace.AttentionHeads);
            Assert.Equal(4, searchSpace.AttentionHeads.Count);
            Assert.Contains(4, searchSpace.AttentionHeads);
            Assert.Contains(8, searchSpace.AttentionHeads);
            Assert.Contains(12, searchSpace.AttentionHeads);
            Assert.Contains(16, searchSpace.AttentionHeads);
        }

        [Fact]
        public void TransformerSearchSpace_HiddenDimensions_ContainsExpectedValues()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert
            Assert.NotNull(searchSpace.HiddenDimensions);
            Assert.Equal(4, searchSpace.HiddenDimensions.Count);
            Assert.Contains(768, searchSpace.HiddenDimensions);
            Assert.Contains(1024, searchSpace.HiddenDimensions);
            Assert.Contains(2048, searchSpace.HiddenDimensions);
            Assert.Contains(3072, searchSpace.HiddenDimensions);
        }

        [Fact]
        public void TransformerSearchSpace_FeedForwardMultipliers_ContainsTwoAndFour()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert
            Assert.NotNull(searchSpace.FeedForwardMultipliers);
            Assert.Equal(2, searchSpace.FeedForwardMultipliers.Count);
            Assert.Contains(2, searchSpace.FeedForwardMultipliers);
            Assert.Contains(4, searchSpace.FeedForwardMultipliers);
        }

        [Fact]
        public void TransformerSearchSpace_DropoutRates_ContainsExpectedValues()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert
            Assert.NotNull(searchSpace.DropoutRates);
            Assert.Equal(4, searchSpace.DropoutRates.Count);
            Assert.Contains(0.0, searchSpace.DropoutRates);
            Assert.Contains(0.1, searchSpace.DropoutRates);
            Assert.Contains(0.2, searchSpace.DropoutRates);
            Assert.Contains(0.3, searchSpace.DropoutRates);
        }

        [Fact]
        public void TransformerSearchSpace_UsePreNorm_DefaultIsTrue()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert - Pre-LayerNorm is default
            Assert.True(searchSpace.UsePreNorm);
        }

        [Fact]
        public void TransformerSearchSpace_AttentionHeads_CanBeModified()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Act
            searchSpace.AttentionHeads = new List<int> { 2, 4, 6 };

            // Assert
            Assert.Equal(3, searchSpace.AttentionHeads.Count);
            Assert.Contains(2, searchSpace.AttentionHeads);
        }

        [Fact]
        public void TransformerSearchSpace_HiddenDimensions_CanBeModified()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Act
            searchSpace.HiddenDimensions = new List<int> { 256, 512, 1024 };

            // Assert
            Assert.Equal(3, searchSpace.HiddenDimensions.Count);
            Assert.Contains(256, searchSpace.HiddenDimensions);
        }

        [Fact]
        public void TransformerSearchSpace_FeedForwardMultipliers_CanBeModified()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Act
            searchSpace.FeedForwardMultipliers = new List<int> { 1, 2, 4, 8 };

            // Assert
            Assert.Equal(4, searchSpace.FeedForwardMultipliers.Count);
            Assert.Contains(8, searchSpace.FeedForwardMultipliers);
        }

        [Fact]
        public void TransformerSearchSpace_DropoutRates_CanBeModified()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Act
            searchSpace.DropoutRates = new List<double> { 0.0, 0.5 };

            // Assert
            Assert.Equal(2, searchSpace.DropoutRates.Count);
            Assert.Contains(0.5, searchSpace.DropoutRates);
        }

        [Fact]
        public void TransformerSearchSpace_UsePreNorm_CanBeModified()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Act
            searchSpace.UsePreNorm = false;

            // Assert - Post-LayerNorm
            Assert.False(searchSpace.UsePreNorm);
        }

        [Fact]
        public void TransformerSearchSpace_Float_WorksCorrectly()
        {
            // Arrange & Act
            var searchSpace = new TransformerSearchSpace<float>();

            // Assert
            Assert.NotNull(searchSpace);
            Assert.Equal(24, searchSpace.MaxNodes);
            Assert.Equal(9, searchSpace.Operations.Count);
        }

        [Fact]
        public void TransformerSearchSpace_InheritsFromSearchSpaceBase()
        {
            // Arrange & Act
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert - verify it inherits from SearchSpaceBase
            Assert.IsAssignableFrom<SearchSpaceBase<double>>(searchSpace);
        }

        [Fact]
        public void TransformerSearchSpace_Operations_IncludesAttentionMechanisms()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert - Transformer specific attention operations
            var attentionOps = searchSpace.Operations.FindAll(op => op.Contains("attention"));
            Assert.Equal(4, attentionOps.Count);
        }

        [Fact]
        public void TransformerSearchSpace_Operations_IncludesFeedForwardNetworks()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert
            var ffnOps = searchSpace.Operations.FindAll(op => op.StartsWith("feed_forward"));
            Assert.Equal(2, ffnOps.Count);
        }

        [Fact]
        public void TransformerSearchSpace_Operations_IncludesLayerNorm()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert
            Assert.Contains("layer_norm", searchSpace.Operations);
        }

        [Fact]
        public void TransformerSearchSpace_Operations_IncludesGatedLinearUnit()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert
            Assert.Contains("glu", searchSpace.Operations);
        }

        [Fact]
        public void TransformerSearchSpace_Operations_IncludesMultiHeadAttentionVariants()
        {
            // Arrange
            var searchSpace = new TransformerSearchSpace<double>();

            // Assert - Different head counts
            Assert.Contains("multi_head_attention_4", searchSpace.Operations);
            Assert.Contains("multi_head_attention_8", searchSpace.Operations);
            Assert.Contains("multi_head_attention_16", searchSpace.Operations);
        }
    }
}
