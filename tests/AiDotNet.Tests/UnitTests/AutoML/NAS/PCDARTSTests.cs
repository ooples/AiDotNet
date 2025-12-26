using System;
using System.Linq;
using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.NAS
{
    /// <summary>
    /// Unit tests for the PC-DARTS (Partial Channel DARTS) algorithm.
    /// </summary>
    public class PCDARTSTests
    {
        [Fact]
        public void PCDARTS_Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 4);

            // Assert
            Assert.NotNull(pcdarts);
        }

        [Fact]
        public void PCDARTS_GetArchitectureParameters_ReturnsValidList()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 4);

            // Act
            var params_ = pcdarts.GetArchitectureParameters();

            // Assert
            Assert.NotNull(params_);
            Assert.Equal(4, params_.Count);
        }

        [Fact]
        public void PCDARTS_GetChannelSamplingRatio_ReturnsConfiguredValue()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 4, channelSamplingRatio: 0.5);

            // Act
            var ratio = pcdarts.GetChannelSamplingRatio();

            // Assert
            Assert.Equal(0.5, ratio);
        }

        [Fact]
        public void PCDARTS_SampleChannels_ReturnsSampledSubset()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 4, channelSamplingRatio: 0.25);

            // Act
            var sampledChannels = pcdarts.SampleChannels(totalChannels: 16);

            // Assert
            Assert.NotNull(sampledChannels);
            Assert.Equal(4, sampledChannels.Count); // 25% of 16 = 4
        }

        [Fact]
        public void PCDARTS_SampleChannels_ReturnsAtLeastOneChannel()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 4, channelSamplingRatio: 0.01);

            // Act
            var sampledChannels = pcdarts.SampleChannels(totalChannels: 4);

            // Assert
            Assert.True(sampledChannels.Count >= 1);
        }

        [Fact]
        public void PCDARTS_SampleChannels_ReturnsSortedChannels()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 4, channelSamplingRatio: 0.5);

            // Act
            var sampledChannels = pcdarts.SampleChannels(totalChannels: 10);

            // Assert
            for (int i = 1; i < sampledChannels.Count; i++)
            {
                Assert.True(sampledChannels[i] > sampledChannels[i - 1]);
            }
        }

        [Fact]
        public void PCDARTS_SampleChannels_ReturnsValidIndices()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 4, channelSamplingRatio: 0.5);
            int totalChannels = 16;

            // Act
            var sampledChannels = pcdarts.SampleChannels(totalChannels);

            // Assert
            Assert.All(sampledChannels, ch => Assert.InRange(ch, 0, totalChannels - 1));
        }

        [Fact]
        public void PCDARTS_ApplyEdgeNormalization_ReturnsValidProbabilities()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 4, useEdgeNormalization: true);
            var alpha = new Matrix<double>(3, 5);
            var random = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < alpha.Rows; i++)
                for (int j = 0; j < alpha.Columns; j++)
                    alpha[i, j] = random.NextDouble() - 0.5;

            // Act
            var normalized = pcdarts.ApplyEdgeNormalization(alpha);

            // Assert
            Assert.NotNull(normalized);
            Assert.Equal(alpha.Rows, normalized.Rows);
            Assert.Equal(alpha.Columns, normalized.Columns);
        }

        [Fact]
        public void PCDARTS_GetMemorySavingsRatio_CalculatesCorrectly()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 4, channelSamplingRatio: 0.25);

            // Act
            var savings = pcdarts.GetMemorySavingsRatio();

            // Assert
            Assert.Equal(0.75, savings); // 1.0 - 0.25
        }

        [Fact]
        public void PCDARTS_DeriveArchitecture_ReturnsValidArchitecture()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 3);

            // Act
            var architecture = pcdarts.DeriveArchitecture();

            // Assert
            Assert.NotNull(architecture);
            Assert.True(architecture.Operations.Count > 0);
        }

        [Fact]
        public void PCDARTS_DeriveArchitecture_SelectsTopTwoEdges()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 4);

            // Act
            var architecture = pcdarts.DeriveArchitecture();

            // Assert - each node (except first) should have at most 2 incoming edges
            // The total number of operations should be at most numNodes * 2 (8 for 4 nodes)
            Assert.True(architecture.Operations.Count <= 8);
        }

        [Fact]
        public void PCDARTS_WithoutEdgeNormalization_StillWorks()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 3, useEdgeNormalization: false);

            // Act
            var architecture = pcdarts.DeriveArchitecture();

            // Assert
            Assert.NotNull(architecture);
        }

        [Fact]
        public void PCDARTS_GetArchitectureGradients_MatchesParameters()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var pcdarts = new PCDARTS<double>(searchSpace, numNodes: 4);

            // Act
            var params_ = pcdarts.GetArchitectureParameters();
            var gradients = pcdarts.GetArchitectureGradients();

            // Assert
            Assert.Equal(params_.Count, gradients.Count);
            for (int i = 0; i < params_.Count; i++)
            {
                Assert.Equal(params_[i].Rows, gradients[i].Rows);
                Assert.Equal(params_[i].Columns, gradients[i].Columns);
            }
        }
    }
}
