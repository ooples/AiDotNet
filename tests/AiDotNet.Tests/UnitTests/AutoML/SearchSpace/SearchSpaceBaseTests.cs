using System.Collections.Generic;
using AiDotNet.AutoML.SearchSpace;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.SearchSpace
{
    /// <summary>
    /// Unit tests for the SearchSpaceBase class.
    /// </summary>
    public class SearchSpaceBaseTests
    {
        [Fact]
        public void SearchSpaceBase_Constructor_InitializesWithDefaultValues()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();

            // Assert
            Assert.NotNull(searchSpace);
            Assert.NotNull(searchSpace.Operations);
        }

        [Fact]
        public void SearchSpaceBase_DefaultOperations_ContainsExpectedOperations()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();

            // Act
            var operations = searchSpace.Operations;

            // Assert
            Assert.Contains("identity", operations);
            Assert.Contains("conv3x3", operations);
            Assert.Contains("conv5x5", operations);
            Assert.Contains("maxpool3x3", operations);
            Assert.Contains("avgpool3x3", operations);
        }

        [Fact]
        public void SearchSpaceBase_DefaultOperations_HasFiveOperations()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();

            // Assert
            Assert.Equal(5, searchSpace.Operations.Count);
        }

        [Fact]
        public void SearchSpaceBase_DefaultMaxNodes_IsEight()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();

            // Assert
            Assert.Equal(8, searchSpace.MaxNodes);
        }

        [Fact]
        public void SearchSpaceBase_DefaultInputChannels_IsOne()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();

            // Assert
            Assert.Equal(1, searchSpace.InputChannels);
        }

        [Fact]
        public void SearchSpaceBase_DefaultOutputChannels_IsOne()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();

            // Assert
            Assert.Equal(1, searchSpace.OutputChannels);
        }

        [Fact]
        public void SearchSpaceBase_Operations_CanBeModified()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var newOperations = new List<string> { "custom_op1", "custom_op2" };

            // Act
            searchSpace.Operations = newOperations;

            // Assert
            Assert.Equal(2, searchSpace.Operations.Count);
            Assert.Contains("custom_op1", searchSpace.Operations);
            Assert.Contains("custom_op2", searchSpace.Operations);
        }

        [Fact]
        public void SearchSpaceBase_MaxNodes_CanBeModified()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();

            // Act
            searchSpace.MaxNodes = 16;

            // Assert
            Assert.Equal(16, searchSpace.MaxNodes);
        }

        [Fact]
        public void SearchSpaceBase_InputChannels_CanBeModified()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();

            // Act
            searchSpace.InputChannels = 3;

            // Assert
            Assert.Equal(3, searchSpace.InputChannels);
        }

        [Fact]
        public void SearchSpaceBase_OutputChannels_CanBeModified()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();

            // Act
            searchSpace.OutputChannels = 1000;

            // Assert
            Assert.Equal(1000, searchSpace.OutputChannels);
        }

        [Fact]
        public void SearchSpaceBase_Float_WorksCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<float>();

            // Assert
            Assert.NotNull(searchSpace);
            Assert.Equal(8, searchSpace.MaxNodes);
        }

        [Fact]
        public void SearchSpaceBase_Operations_CanAddItems()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            int initialCount = searchSpace.Operations.Count;

            // Act
            searchSpace.Operations.Add("new_operation");

            // Assert
            Assert.Equal(initialCount + 1, searchSpace.Operations.Count);
            Assert.Contains("new_operation", searchSpace.Operations);
        }

        [Fact]
        public void SearchSpaceBase_Operations_CanRemoveItems()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            int initialCount = searchSpace.Operations.Count;

            // Act
            searchSpace.Operations.Remove("identity");

            // Assert
            Assert.Equal(initialCount - 1, searchSpace.Operations.Count);
            Assert.DoesNotContain("identity", searchSpace.Operations);
        }

        [Fact]
        public void SearchSpaceBase_Operations_CanClear()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();

            // Act
            searchSpace.Operations.Clear();

            // Assert
            Assert.Empty(searchSpace.Operations);
        }
    }
}
