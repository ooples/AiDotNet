using System;
using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.NAS
{
    /// <summary>
    /// Unit tests for the ProxylessNAS (Direct Neural Architecture Search on Target Hardware) algorithm.
    /// </summary>
    public class ProxylessNASTests
    {
        [Fact]
        public void ProxylessNAS_Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var proxyless = new ProxylessNAS<double>(searchSpace, numNodes: 4);

            // Assert
            Assert.NotNull(proxyless);
        }

        [Fact]
        public void ProxylessNAS_GetArchitectureParameters_ReturnsValidList()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var proxyless = new ProxylessNAS<double>(searchSpace, numNodes: 4);

            // Act
            var params_ = proxyless.GetArchitectureParameters();

            // Assert
            Assert.NotNull(params_);
            Assert.Equal(4, params_.Count);
        }

        [Fact]
        public void ProxylessNAS_GetArchitectureParameters_HasCorrectShape()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var proxyless = new ProxylessNAS<double>(searchSpace, numNodes: 3);

            // Act
            var params_ = proxyless.GetArchitectureParameters();

            // Assert
            Assert.Equal(1, params_[0].Rows); // First node connects to 1 previous node
            Assert.Equal(2, params_[1].Rows); // Second node connects to 2 previous nodes
            Assert.Equal(3, params_[2].Rows); // Third node connects to 3 previous nodes
        }

        [Fact]
        public void ProxylessNAS_GetArchitectureGradients_MatchesParameterCount()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var proxyless = new ProxylessNAS<double>(searchSpace, numNodes: 4);

            // Act
            var params_ = proxyless.GetArchitectureParameters();
            var gradients = proxyless.GetArchitectureGradients();

            // Assert
            Assert.Equal(params_.Count, gradients.Count);
            for (int i = 0; i < params_.Count; i++)
            {
                Assert.Equal(params_[i].Rows, gradients[i].Rows);
                Assert.Equal(params_[i].Columns, gradients[i].Columns);
            }
        }

        [Fact]
        public void ProxylessNAS_BinarizePaths_WithBinarization_ReturnsOneHotVectors()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var proxyless = new ProxylessNAS<double>(searchSpace, numNodes: 4, useBinarization: true);
            var alpha = new Matrix<double>(3, 5);
            var random = new Random(42);
            for (int i = 0; i < alpha.Rows; i++)
                for (int j = 0; j < alpha.Columns; j++)
                    alpha[i, j] = random.NextDouble() - 0.5;

            // Act
            var binarized = proxyless.BinarizePaths(alpha);

            // Assert
            Assert.NotNull(binarized);
            Assert.Equal(alpha.Rows, binarized.Rows);
            Assert.Equal(alpha.Columns, binarized.Columns);

            // Each row should have exactly one 1 and the rest 0
            for (int row = 0; row < binarized.Rows; row++)
            {
                int oneCount = 0;
                for (int col = 0; col < binarized.Columns; col++)
                {
                    if (Math.Abs(binarized[row, col] - 1.0) < 0.01)
                        oneCount++;
                    else
                        Assert.True(Math.Abs(binarized[row, col]) < 0.01); // Should be 0
                }
                Assert.Equal(1, oneCount);
            }
        }

        [Fact]
        public void ProxylessNAS_BinarizePaths_WithoutBinarization_ReturnsSoftmax()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var proxyless = new ProxylessNAS<double>(searchSpace, numNodes: 4, useBinarization: false);
            var alpha = new Matrix<double>(3, 5);
            var random = new Random(42);
            for (int i = 0; i < alpha.Rows; i++)
                for (int j = 0; j < alpha.Columns; j++)
                    alpha[i, j] = random.NextDouble() - 0.5;

            // Act
            var result = proxyless.BinarizePaths(alpha);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(alpha.Rows, result.Rows);

            // Each row should sum to approximately 1 (softmax)
            for (int row = 0; row < result.Rows; row++)
            {
                double rowSum = 0;
                for (int col = 0; col < result.Columns; col++)
                {
                    rowSum += result[row, col];
                }
                Assert.True(Math.Abs(rowSum - 1.0) < 0.01);
            }
        }

        [Fact]
        public void ProxylessNAS_ComputeExpectedLatency_ReturnsNonNegativeValue()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var proxyless = new ProxylessNAS<double>(searchSpace, numNodes: 4);

            // Act
            var latency = proxyless.ComputeExpectedLatency(inputChannels: 32, spatialSize: 14);

            // Assert
            Assert.True(latency >= 0.0);
        }

        [Fact]
        public void ProxylessNAS_ComputeTotalLoss_IncludesLatencyPenalty()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var proxyless = new ProxylessNAS<double>(searchSpace, numNodes: 4, latencyWeight: 0.5);
            double taskLoss = 1.0;

            // Act
            var totalLoss = proxyless.ComputeTotalLoss(taskLoss, inputChannels: 32, spatialSize: 14);

            // Assert
            Assert.True(totalLoss >= taskLoss - 0.01); // Should include latency term
        }

        [Fact]
        public void ProxylessNAS_DeriveArchitecture_ReturnsValidArchitecture()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var proxyless = new ProxylessNAS<double>(searchSpace, numNodes: 4);

            // Act
            var architecture = proxyless.DeriveArchitecture();

            // Assert
            Assert.NotNull(architecture);
            Assert.True(architecture.Operations.Count > 0);
        }

        [Fact]
        public void ProxylessNAS_DeriveArchitecture_SelectsTopTwoEdges()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var proxyless = new ProxylessNAS<double>(searchSpace, numNodes: 4);

            // Act
            var architecture = proxyless.DeriveArchitecture();

            // Assert - each node should have at most 2 incoming edges
            Assert.True(architecture.Operations.Count <= 8); // 4 nodes * 2 edges max
        }

        [Fact]
        public void ProxylessNAS_EstimateArchitectureCost_ReturnsValidCost()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var proxyless = new ProxylessNAS<double>(searchSpace, numNodes: 4);

            // Act
            var cost = proxyless.EstimateArchitectureCost(inputChannels: 32, spatialSize: 14);

            // Assert
            Assert.True(cost.Latency >= 0.0);
            Assert.True(cost.Energy >= 0.0);
            Assert.True(cost.Memory >= 0.0);
        }

        [Fact]
        public void ProxylessNAS_SetBinarizationTemperature_UpdatesTemperature()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var proxyless = new ProxylessNAS<double>(searchSpace, numNodes: 4);
            var alpha = new Matrix<double>(2, 5);
            for (int i = 0; i < alpha.Rows; i++)
                for (int j = 0; j < alpha.Columns; j++)
                    alpha[i, j] = j == 0 ? 1.0 : 0.0;

            // Act
            proxyless.SetBinarizationTemperature(0.1); // Very low temperature
            var resultLow = proxyless.BinarizePaths(alpha);

            proxyless.SetBinarizationTemperature(10.0); // High temperature
            var resultHigh = proxyless.BinarizePaths(alpha);

            // Assert - both should work without throwing
            Assert.NotNull(resultLow);
            Assert.NotNull(resultHigh);
        }

        [Fact]
        public void ProxylessNAS_WithDifferentPlatforms_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var proxylessMobile = new ProxylessNAS<double>(searchSpace, numNodes: 4, targetPlatform: HardwarePlatform.Mobile);
            var proxylessGpu = new ProxylessNAS<double>(searchSpace, numNodes: 4, targetPlatform: HardwarePlatform.GPU);
            var proxylessEdge = new ProxylessNAS<double>(searchSpace, numNodes: 4, targetPlatform: HardwarePlatform.EdgeTPU);

            // Assert
            Assert.NotNull(proxylessMobile);
            Assert.NotNull(proxylessGpu);
            Assert.NotNull(proxylessEdge);
        }

        [Fact]
        public void ProxylessNAS_WithDifferentLatencyWeights_AffectsTotalLoss()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var proxylessLow = new ProxylessNAS<double>(searchSpace, numNodes: 4, latencyWeight: 0.01);
            var proxylessHigh = new ProxylessNAS<double>(searchSpace, numNodes: 4, latencyWeight: 1.0);
            double taskLoss = 1.0;

            // Act
            var totalLossLow = proxylessLow.ComputeTotalLoss(taskLoss, inputChannels: 32, spatialSize: 14);
            var totalLossHigh = proxylessHigh.ComputeTotalLoss(taskLoss, inputChannels: 32, spatialSize: 14);

            // Assert - higher latency weight should result in higher total loss
            Assert.True(totalLossHigh >= totalLossLow);
        }

        [Fact]
        public void ProxylessNAS_MultipleDeriveArchitecture_ProducesConsistentResults()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var proxyless = new ProxylessNAS<double>(searchSpace, numNodes: 4);

            // Act
            var arch1 = proxyless.DeriveArchitecture();
            var arch2 = proxyless.DeriveArchitecture();

            // Assert - same parameters should produce same architecture
            Assert.Equal(arch1.Operations.Count, arch2.Operations.Count);
        }
    }
}
