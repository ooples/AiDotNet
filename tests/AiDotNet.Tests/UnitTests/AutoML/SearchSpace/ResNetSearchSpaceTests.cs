using System.Collections.Generic;
using AiDotNet.AutoML.SearchSpace;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.SearchSpace
{
    /// <summary>
    /// Unit tests for the ResNetSearchSpace class.
    /// </summary>
    public class ResNetSearchSpaceTests
    {
        [Fact]
        public void ResNetSearchSpace_Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new ResNetSearchSpace<double>();

            // Assert
            Assert.NotNull(searchSpace);
            Assert.NotNull(searchSpace.Operations);
        }

        [Fact]
        public void ResNetSearchSpace_Operations_ContainsResNetOperations()
        {
            // Arrange
            var searchSpace = new ResNetSearchSpace<double>();

            // Assert
            Assert.Contains("identity", searchSpace.Operations);
            Assert.Contains("conv1x1", searchSpace.Operations);
            Assert.Contains("conv3x3", searchSpace.Operations);
            Assert.Contains("conv5x5", searchSpace.Operations);
            Assert.Contains("residual_block_basic", searchSpace.Operations);
            Assert.Contains("residual_block_bottleneck", searchSpace.Operations);
            Assert.Contains("maxpool3x3", searchSpace.Operations);
            Assert.Contains("avgpool3x3", searchSpace.Operations);
            Assert.Contains("grouped_conv3x3", searchSpace.Operations);
        }

        [Fact]
        public void ResNetSearchSpace_Operations_HasNineOperations()
        {
            // Arrange
            var searchSpace = new ResNetSearchSpace<double>();

            // Assert
            Assert.Equal(9, searchSpace.Operations.Count);
        }

        [Fact]
        public void ResNetSearchSpace_MaxNodes_IsSixteen()
        {
            // Arrange
            var searchSpace = new ResNetSearchSpace<double>();

            // Assert
            Assert.Equal(16, searchSpace.MaxNodes);
        }

        [Fact]
        public void ResNetSearchSpace_InputChannels_IsThree()
        {
            // Arrange
            var searchSpace = new ResNetSearchSpace<double>();

            // Assert - RGB images
            Assert.Equal(3, searchSpace.InputChannels);
        }

        [Fact]
        public void ResNetSearchSpace_OutputChannels_IsOneThousand()
        {
            // Arrange
            var searchSpace = new ResNetSearchSpace<double>();

            // Assert - ImageNet classes
            Assert.Equal(1000, searchSpace.OutputChannels);
        }

        [Fact]
        public void ResNetSearchSpace_BottleneckRatio_DefaultIsFour()
        {
            // Arrange
            var searchSpace = new ResNetSearchSpace<double>();

            // Assert
            Assert.Equal(4, searchSpace.BottleneckRatio);
        }

        [Fact]
        public void ResNetSearchSpace_GroupCount_DefaultIsThirtyTwo()
        {
            // Arrange
            var searchSpace = new ResNetSearchSpace<double>();

            // Assert - ResNeXt default
            Assert.Equal(32, searchSpace.GroupCount);
        }

        [Fact]
        public void ResNetSearchSpace_BlockDepths_ContainsExpectedValues()
        {
            // Arrange
            var searchSpace = new ResNetSearchSpace<double>();

            // Assert
            Assert.NotNull(searchSpace.BlockDepths);
            Assert.Equal(5, searchSpace.BlockDepths.Count);
            Assert.Contains(2, searchSpace.BlockDepths);
            Assert.Contains(3, searchSpace.BlockDepths);
            Assert.Contains(4, searchSpace.BlockDepths);
            Assert.Contains(6, searchSpace.BlockDepths);
            Assert.Contains(8, searchSpace.BlockDepths);
        }

        [Fact]
        public void ResNetSearchSpace_BottleneckRatio_CanBeModified()
        {
            // Arrange
            var searchSpace = new ResNetSearchSpace<double>();

            // Act
            searchSpace.BottleneckRatio = 2;

            // Assert
            Assert.Equal(2, searchSpace.BottleneckRatio);
        }

        [Fact]
        public void ResNetSearchSpace_GroupCount_CanBeModified()
        {
            // Arrange
            var searchSpace = new ResNetSearchSpace<double>();

            // Act
            searchSpace.GroupCount = 64;

            // Assert
            Assert.Equal(64, searchSpace.GroupCount);
        }

        [Fact]
        public void ResNetSearchSpace_BlockDepths_CanBeModified()
        {
            // Arrange
            var searchSpace = new ResNetSearchSpace<double>();

            // Act
            searchSpace.BlockDepths = new List<int> { 1, 2, 3 };

            // Assert
            Assert.Equal(3, searchSpace.BlockDepths.Count);
            Assert.Contains(1, searchSpace.BlockDepths);
        }

        [Fact]
        public void ResNetSearchSpace_Float_WorksCorrectly()
        {
            // Arrange & Act
            var searchSpace = new ResNetSearchSpace<float>();

            // Assert
            Assert.NotNull(searchSpace);
            Assert.Equal(16, searchSpace.MaxNodes);
            Assert.Equal(9, searchSpace.Operations.Count);
        }

        [Fact]
        public void ResNetSearchSpace_InheritsFromSearchSpaceBase()
        {
            // Arrange & Act
            var searchSpace = new ResNetSearchSpace<double>();

            // Assert - verify it inherits from SearchSpaceBase
            Assert.IsAssignableFrom<SearchSpaceBase<double>>(searchSpace);
        }

        [Fact]
        public void ResNetSearchSpace_Operations_IncludesResidualBlocks()
        {
            // Arrange
            var searchSpace = new ResNetSearchSpace<double>();

            // Assert - ResNet specific residual blocks
            var residualBlocks = searchSpace.Operations.FindAll(op => op.StartsWith("residual_block"));
            Assert.Equal(2, residualBlocks.Count);
        }

        [Fact]
        public void ResNetSearchSpace_Operations_IncludesBasicBlock()
        {
            // Arrange
            var searchSpace = new ResNetSearchSpace<double>();

            // Assert
            Assert.Contains("residual_block_basic", searchSpace.Operations);
        }

        [Fact]
        public void ResNetSearchSpace_Operations_IncludesBottleneckBlock()
        {
            // Arrange
            var searchSpace = new ResNetSearchSpace<double>();

            // Assert
            Assert.Contains("residual_block_bottleneck", searchSpace.Operations);
        }

        [Fact]
        public void ResNetSearchSpace_Operations_IncludesGroupedConvolution()
        {
            // Arrange
            var searchSpace = new ResNetSearchSpace<double>();

            // Assert - ResNeXt style
            Assert.Contains("grouped_conv3x3", searchSpace.Operations);
        }

        [Fact]
        public void ResNetSearchSpace_Operations_IncludesPoolingOperations()
        {
            // Arrange
            var searchSpace = new ResNetSearchSpace<double>();

            // Assert
            Assert.Contains("maxpool3x3", searchSpace.Operations);
            Assert.Contains("avgpool3x3", searchSpace.Operations);
        }
    }
}
