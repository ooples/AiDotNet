using System.Collections.Generic;
using AiDotNet.AutoML.SearchSpace;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.AutoML.SearchSpace
{
    /// <summary>
    /// Unit tests for the MobileNetSearchSpace class.
    /// </summary>
    public class MobileNetSearchSpaceTests
    {
        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new MobileNetSearchSpace<double>();

            // Assert
            Assert.NotNull(searchSpace);
            Assert.NotNull(searchSpace.Operations);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_Operations_ContainsMobileNetOperations()
        {
            // Arrange
            var searchSpace = new MobileNetSearchSpace<double>();

            // Assert
            Assert.Contains("identity", searchSpace.Operations);
            Assert.Contains("conv1x1", searchSpace.Operations);
            Assert.Contains("conv3x3", searchSpace.Operations);
            Assert.Contains("depthwise_conv3x3", searchSpace.Operations);
            Assert.Contains("inverted_residual_3x3_e3", searchSpace.Operations);
            Assert.Contains("inverted_residual_3x3_e6", searchSpace.Operations);
            Assert.Contains("inverted_residual_5x5_e3", searchSpace.Operations);
            Assert.Contains("inverted_residual_5x5_e6", searchSpace.Operations);
            Assert.Contains("se_block", searchSpace.Operations);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_Operations_HasNineOperations()
        {
            // Arrange
            var searchSpace = new MobileNetSearchSpace<double>();

            // Assert
            Assert.Equal(9, searchSpace.Operations.Count);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_MaxNodes_IsTwenty()
        {
            // Arrange
            var searchSpace = new MobileNetSearchSpace<double>();

            // Assert
            Assert.Equal(20, searchSpace.MaxNodes);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_InputChannels_IsThree()
        {
            // Arrange
            var searchSpace = new MobileNetSearchSpace<double>();

            // Assert - RGB images
            Assert.Equal(3, searchSpace.InputChannels);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_OutputChannels_IsOneThousand()
        {
            // Arrange
            var searchSpace = new MobileNetSearchSpace<double>();

            // Assert - ImageNet classes
            Assert.Equal(1000, searchSpace.OutputChannels);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_ExpansionRatios_ContainsThreeAndSix()
        {
            // Arrange
            var searchSpace = new MobileNetSearchSpace<double>();

            // Assert
            Assert.NotNull(searchSpace.ExpansionRatios);
            Assert.Equal(2, searchSpace.ExpansionRatios.Count);
            Assert.Contains(3, searchSpace.ExpansionRatios);
            Assert.Contains(6, searchSpace.ExpansionRatios);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_KernelSizes_ContainsThreeAndFive()
        {
            // Arrange
            var searchSpace = new MobileNetSearchSpace<double>();

            // Assert
            Assert.NotNull(searchSpace.KernelSizes);
            Assert.Equal(2, searchSpace.KernelSizes.Count);
            Assert.Contains(3, searchSpace.KernelSizes);
            Assert.Contains(5, searchSpace.KernelSizes);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_DepthMultiplier_DefaultIsOne()
        {
            // Arrange
            var searchSpace = new MobileNetSearchSpace<double>();

            // Assert
            Assert.Equal(1.0, searchSpace.DepthMultiplier);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_WidthMultiplier_DefaultIsOne()
        {
            // Arrange
            var searchSpace = new MobileNetSearchSpace<double>();

            // Assert
            Assert.Equal(1.0, searchSpace.WidthMultiplier);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_ExpansionRatios_CanBeModified()
        {
            // Arrange
            var searchSpace = new MobileNetSearchSpace<double>();

            // Act
            searchSpace.ExpansionRatios = new List<int> { 2, 4, 8 };

            // Assert
            Assert.Equal(3, searchSpace.ExpansionRatios.Count);
            Assert.Contains(2, searchSpace.ExpansionRatios);
            Assert.Contains(4, searchSpace.ExpansionRatios);
            Assert.Contains(8, searchSpace.ExpansionRatios);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_KernelSizes_CanBeModified()
        {
            // Arrange
            var searchSpace = new MobileNetSearchSpace<double>();

            // Act
            searchSpace.KernelSizes = new List<int> { 3, 5, 7 };

            // Assert
            Assert.Equal(3, searchSpace.KernelSizes.Count);
            Assert.Contains(7, searchSpace.KernelSizes);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_DepthMultiplier_CanBeModified()
        {
            // Arrange
            var searchSpace = new MobileNetSearchSpace<double>();

            // Act
            searchSpace.DepthMultiplier = 1.5;

            // Assert
            Assert.Equal(1.5, searchSpace.DepthMultiplier);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_WidthMultiplier_CanBeModified()
        {
            // Arrange
            var searchSpace = new MobileNetSearchSpace<double>();

            // Act
            searchSpace.WidthMultiplier = 0.75;

            // Assert
            Assert.Equal(0.75, searchSpace.WidthMultiplier);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_Float_WorksCorrectly()
        {
            // Arrange & Act
            var searchSpace = new MobileNetSearchSpace<float>();

            // Assert
            Assert.NotNull(searchSpace);
            Assert.Equal(20, searchSpace.MaxNodes);
            Assert.Equal(9, searchSpace.Operations.Count);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_InheritsFromSearchSpaceBase()
        {
            // Arrange & Act
            var searchSpace = new MobileNetSearchSpace<double>();

            // Assert - verify it inherits from SearchSpaceBase
            Assert.IsAssignableFrom<SearchSpaceBase<double>>(searchSpace);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_Operations_IncludesInvertedResiduals()
        {
            // Arrange
            var searchSpace = new MobileNetSearchSpace<double>();

            // Assert - MobileNet specific inverted residual blocks
            var invertedResiduals = searchSpace.Operations.FindAll(op => op.StartsWith("inverted_residual"));
            Assert.Equal(4, invertedResiduals.Count);
        }

        [Fact(Timeout = 60000)]
        public async Task MobileNetSearchSpace_Operations_IncludesSqueezeExcitation()
        {
            // Arrange
            var searchSpace = new MobileNetSearchSpace<double>();

            // Assert
            Assert.Contains("se_block", searchSpace.Operations);
        }
    }
}
