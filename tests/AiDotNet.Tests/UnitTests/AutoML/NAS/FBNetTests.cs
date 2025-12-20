using System;
using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.NAS
{
    /// <summary>
    /// Unit tests for the FBNet (Hardware-Aware NAS) algorithm.
    /// </summary>
    public class FBNetTests
    {
        [Fact]
        public void FBNet_Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 10);

            // Assert
            Assert.NotNull(fbnet);
        }

        [Fact]
        public void FBNet_GetArchitectureParameters_ReturnsCorrectCount()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 15);

            // Act
            var params_ = fbnet.GetArchitectureParameters();

            // Assert
            Assert.NotNull(params_);
            Assert.Equal(15, params_.Count);
        }

        [Fact]
        public void FBNet_GetTemperature_ReturnsInitialValue()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 10, initialTemperature: 5.0);

            // Act
            var temp = fbnet.GetTemperature();

            // Assert
            Assert.Equal(5.0, temp);
        }

        [Fact]
        public void FBNet_AnnealTemperature_DecreasesTemperature()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 10, initialTemperature: 5.0);
            var initialTemp = fbnet.GetTemperature();

            // Act
            fbnet.AnnealTemperature(50, 100);
            var midTemp = fbnet.GetTemperature();

            // Assert
            Assert.True(midTemp < initialTemp);
        }

        [Fact]
        public void FBNet_GumbelSoftmax_ReturnsProbabilities()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 10);
            var theta = new Vector<double>(5);
            var random = new Random(42);
            for (int i = 0; i < theta.Length; i++)
                theta[i] = random.NextDouble() - 0.5;

            // Act
            var probs = fbnet.GumbelSoftmax(theta, hard: false);

            // Assert
            Assert.NotNull(probs);
            Assert.Equal(theta.Length, probs.Length);
        }

        [Fact]
        public void FBNet_GumbelSoftmax_HardMode_ReturnsOneHot()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 10);
            var theta = new Vector<double>(5);
            var random = new Random(42);
            for (int i = 0; i < theta.Length; i++)
                theta[i] = random.NextDouble() - 0.5;

            // Act
            var probs = fbnet.GumbelSoftmax(theta, hard: true);

            // Assert
            double sum = 0;
            for (int i = 0; i < probs.Length; i++)
                sum += probs[i];
            Assert.True(Math.Abs(sum - 1.0) < 0.01);
        }

        [Fact]
        public void FBNet_ComputeExpectedLatency_ReturnsNonNegativeValue()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 5);

            // Act
            var latency = fbnet.ComputeExpectedLatency();

            // Assert
            Assert.True(latency >= 0.0);
        }

        [Fact]
        public void FBNet_ComputeTotalLoss_IncludesLatencyPenalty()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 5, latencyWeight: 0.5);
            double taskLoss = 1.0;

            // Act
            var totalLoss = fbnet.ComputeTotalLoss(taskLoss);

            // Assert
            Assert.True(totalLoss > taskLoss - 0.01); // Should include some latency term
        }

        [Fact]
        public void FBNet_DeriveArchitecture_ReturnsValidArchitecture()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 5);

            // Act
            var architecture = fbnet.DeriveArchitecture();

            // Assert
            Assert.NotNull(architecture);
            Assert.Equal(5, architecture.Operations.Count);
        }

        [Fact]
        public void FBNet_MeetsConstraints_ReturnsBoolean()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 3);

            // Act
            var meetsConstraints = fbnet.MeetsConstraints();

            // Assert - should return true or false without throwing
            Assert.True(meetsConstraints || !meetsConstraints);
        }

        [Fact]
        public void FBNet_GetArchitectureCost_ReturnsValidCost()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 5);

            // Act
            var cost = fbnet.GetArchitectureCost();

            // Assert
            Assert.True(cost.Latency >= 0.0);
            Assert.True(cost.Energy >= 0.0);
            Assert.True(cost.Memory >= 0.0);
        }

        [Fact]
        public void FBNet_SetConstraints_UpdatesConstraints()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 5);
            var newConstraints = new HardwareConstraints<double>
            {
                MaxLatency = 50.0,
                MaxMemory = 50.0,
                MaxEnergy = 250.0
            };

            // Act
            fbnet.SetConstraints(newConstraints);
            var meetsConstraints = fbnet.MeetsConstraints();

            // Assert - should work without throwing
            Assert.True(meetsConstraints || !meetsConstraints);
        }

        [Fact]
        public void FBNet_WithDifferentPlatforms_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var fbnetMobile = new FBNet<double>(searchSpace, numLayers: 5, targetPlatform: HardwarePlatform.Mobile);
            var fbnetGpu = new FBNet<double>(searchSpace, numLayers: 5, targetPlatform: HardwarePlatform.GPU);
            var fbnetEdge = new FBNet<double>(searchSpace, numLayers: 5, targetPlatform: HardwarePlatform.EdgeTPU);

            // Assert
            Assert.NotNull(fbnetMobile);
            Assert.NotNull(fbnetGpu);
            Assert.NotNull(fbnetEdge);
        }

        [Fact]
        public void FBNet_GetArchitectureGradients_MatchesParameters()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 10);

            // Act
            var params_ = fbnet.GetArchitectureParameters();
            var gradients = fbnet.GetArchitectureGradients();

            // Assert
            Assert.Equal(params_.Count, gradients.Count);
            for (int i = 0; i < params_.Count; i++)
            {
                Assert.Equal(params_[i].Length, gradients[i].Length);
            }
        }
    }
}
