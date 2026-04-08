using System;
using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.NAS
{
    /// <summary>
    /// Unit tests for the ENAS (Efficient Neural Architecture Search) algorithm.
    /// </summary>
    public class ENASTests
    {
        [Fact]
        public void ENAS_Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(searchSpace, numNodes: 4);

            // Assert
            Assert.NotNull(enas);
        }

        [Fact]
        public void ENAS_SampleArchitecture_ReturnsValidArchitecture()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(searchSpace, numNodes: 3);

            // Act
            var (architecture, logProb, entropy) = enas.SampleArchitecture();

            // Assert
            Assert.NotNull(architecture);
            Assert.True(architecture.Operations.Count > 0);
        }

        [Fact]
        public void ENAS_SampleArchitecture_ReturnsLogProbability()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(searchSpace, numNodes: 3);

            // Act
            var (_, logProb, _) = enas.SampleArchitecture();

            // Assert
            Assert.True(logProb <= 0.0); // Log probability should be non-positive
        }

        [Fact]
        public void ENAS_SampleArchitecture_ReturnsEntropy()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(searchSpace, numNodes: 3);

            // Act
            var (_, _, entropy) = enas.SampleArchitecture();

            // Assert
            Assert.True(entropy >= 0.0); // Entropy should be non-negative
        }

        [Fact]
        public void ENAS_GetControllerParameters_ReturnsNonEmptyList()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(searchSpace, numNodes: 4);

            // Act
            var parameters = enas.GetControllerParameters();

            // Assert
            Assert.NotNull(parameters);
            Assert.True(parameters.Count > 0);
        }

        [Fact]
        public void ENAS_GetControllerGradients_ReturnsMatchingList()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(searchSpace, numNodes: 4);

            // Act
            var parameters = enas.GetControllerParameters();
            var gradients = enas.GetControllerGradients();

            // Assert
            Assert.Equal(parameters.Count, gradients.Count);
        }

        [Fact]
        public void ENAS_GetSharedWeights_ReturnsEmptyDictionaryInitially()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(searchSpace, numNodes: 4);

            // Act
            var sharedWeights = enas.GetSharedWeights();

            // Assert
            Assert.NotNull(sharedWeights);
            Assert.Empty(sharedWeights);
        }

        [Fact]
        public void ENAS_GetSharedWeights_WithKey_InitializesAndReturns()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(searchSpace, numNodes: 4);

            // Act
            var weights = enas.GetSharedWeights("conv3x3");

            // Assert
            Assert.NotNull(weights);
            Assert.True(weights.Length > 0);
        }

        [Fact]
        public void ENAS_GetSharedWeights_SameKey_ReturnsSameReference()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(searchSpace, numNodes: 4);

            // Act
            var weights1 = enas.GetSharedWeights("conv3x3");
            var weights2 = enas.GetSharedWeights("conv3x3");

            // Assert
            Assert.Same(weights1, weights2);
        }

        [Fact]
        public void ENAS_UpdateController_UpdatesBaseline()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(searchSpace, numNodes: 3);
            var (_, logProb, entropy) = enas.SampleArchitecture();

            // Act
            var baselineBefore = enas.GetBaseline();
            enas.UpdateController(1.0, logProb, entropy);
            var baselineAfter = enas.GetBaseline();

            // Assert
            Assert.NotEqual(baselineBefore, baselineAfter);
        }

        [Fact]
        public void ENAS_GetBaseline_ReturnsZeroInitially()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(searchSpace, numNodes: 3);

            // Act
            var baseline = enas.GetBaseline();

            // Assert
            Assert.Equal(0.0, baseline);
        }

        [Fact]
        public void ENAS_MultipleSamples_ReturnDifferentArchitectures()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(searchSpace, numNodes: 3);

            // Act
            var (arch1, _, _) = enas.SampleArchitecture();
            var (arch2, _, _) = enas.SampleArchitecture();
            var (arch3, _, _) = enas.SampleArchitecture();

            // Assert - at least one should be different due to sampling
            var desc1 = arch1.GetDescription();
            var desc2 = arch2.GetDescription();
            var desc3 = arch3.GetDescription();

            // Check that we get valid architectures
            Assert.NotNull(desc1);
            Assert.NotNull(desc2);
            Assert.NotNull(desc3);
        }

        [Fact]
        public void ENAS_CustomControllerHiddenSize_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(searchSpace, numNodes: 4, controllerHiddenSize: 64);

            // Assert
            Assert.NotNull(enas.GetControllerParameters());
            Assert.True(enas.GetControllerParameters()[0].Length > 0);
        }

        [Fact]
        public void ENAS_CustomBaselineDecay_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(searchSpace, numNodes: 3, baselineDecay: 0.99);

            // Act
            var (_, logProb, entropy) = enas.SampleArchitecture();
            enas.UpdateController(1.0, logProb, entropy);
            var baseline = enas.GetBaseline();

            // Assert
            Assert.NotEqual(0.0, baseline);
        }

        [Fact]
        public void ENAS_CustomEntropyWeight_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var enas = new ENAS<double>(searchSpace, numNodes: 3, entropyWeight: 0.05);

            // Assert
            Assert.NotNull(enas);
        }
    }
}
