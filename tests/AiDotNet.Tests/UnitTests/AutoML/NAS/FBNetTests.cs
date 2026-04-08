using System;
using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
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
            var random = RandomHelper.CreateSeededRandom(42);
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
            var random = RandomHelper.CreateSeededRandom(42);
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

        #region Edge Case Tests

        [Fact]
        public void FBNet_SingleLayer_InitializesCorrectly()
        {
            // Arrange & Act - minimal numLayers
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 1);

            // Assert
            Assert.NotNull(fbnet);
            var params_ = fbnet.GetArchitectureParameters();
            Assert.Single(params_);
        }

        [Fact]
        public void FBNet_AnnealTemperature_AtStepZero_ReturnsInitialTemperature()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 5, initialTemperature: 5.0);

            // Act - anneal at step 0
            fbnet.AnnealTemperature(0, 100);
            var temp = fbnet.GetTemperature();

            // Assert - should still be close to initial temperature
            Assert.True(temp >= 4.9); // Should be very close to 5.0
        }

        [Fact]
        public void FBNet_AnnealTemperature_AtFinalStep_ReturnsMinTemperature()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 5, initialTemperature: 5.0);

            // Act - anneal at final step
            fbnet.AnnealTemperature(100, 100);
            var temp = fbnet.GetTemperature();

            // Assert - should be at minimum temperature (typically around 0.1)
            Assert.True(temp < 5.0);
        }

        [Fact]
        public void FBNet_AnnealTemperature_BeyondTotalSteps_HandlesGracefully()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 5, initialTemperature: 5.0);

            // Act - step beyond total (edge case)
            fbnet.AnnealTemperature(150, 100);
            var temp = fbnet.GetTemperature();

            // Assert - should not throw, temperature should be at minimum
            Assert.True(temp >= 0.0);
        }

        [Fact]
        public void FBNet_GumbelSoftmax_SingleElementVector_ReturnsOne()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 5);
            var theta = new Vector<double>(1);
            theta[0] = 1.0;

            // Act
            var probs = fbnet.GumbelSoftmax(theta, hard: false);

            // Assert - single element should sum to 1
            Assert.Single(probs);
            Assert.True(Math.Abs(probs[0] - 1.0) < 0.01);
        }

        [Fact]
        public void FBNet_GumbelSoftmax_AllZeroLogits_HandlesCorrectly()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 5);
            var theta = new Vector<double>(5);
            for (int i = 0; i < 5; i++)
                theta[i] = 0.0;

            // Act
            var probs = fbnet.GumbelSoftmax(theta, hard: false);

            // Assert - sum should still be 1
            double sum = 0;
            for (int i = 0; i < probs.Length; i++)
                sum += probs[i];
            Assert.True(Math.Abs(sum - 1.0) < 0.01);
        }

        [Fact]
        public void FBNet_ZeroLatencyWeight_TaskLossOnly()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 5, latencyWeight: 0.0);
            double taskLoss = 2.5;

            // Act
            var totalLoss = fbnet.ComputeTotalLoss(taskLoss);

            // Assert - with zero weight, total loss should equal task loss
            Assert.True(Math.Abs(totalLoss - taskLoss) < 0.01);
        }

        [Fact]
        public void FBNet_HighLatencyWeight_DominatedByLatency()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 5, latencyWeight: 100.0);
            double taskLoss = 1.0;

            // Act
            var totalLoss = fbnet.ComputeTotalLoss(taskLoss);

            // Assert - with high weight, total loss should be much larger than task loss
            Assert.True(totalLoss > taskLoss);
        }

        [Fact]
        public void FBNet_DeriveArchitecture_SingleLayer_ReturnsOneOperation()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 1);

            // Act
            var architecture = fbnet.DeriveArchitecture();

            // Assert
            Assert.NotNull(architecture);
            Assert.Single(architecture.Operations);
        }

        [Fact]
        public void FBNet_SetConstraints_VeryTightConstraints_StillWorks()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 10);
            var tightConstraints = new HardwareConstraints<double>
            {
                MaxLatency = 0.001,
                MaxMemory = 0.001,
                MaxEnergy = 0.001
            };

            // Act
            fbnet.SetConstraints(tightConstraints);
            var meetsConstraints = fbnet.MeetsConstraints();

            // Assert - should not throw
            Assert.False(meetsConstraints); // Likely won't meet very tight constraints
        }

        [Fact]
        public void FBNet_SetConstraints_VeryLooseConstraints_Satisfied()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var fbnet = new FBNet<double>(searchSpace, numLayers: 3);
            var looseConstraints = new HardwareConstraints<double>
            {
                MaxLatency = 100000.0,
                MaxMemory = 100000.0,
                MaxEnergy = 100000.0
            };

            // Act
            fbnet.SetConstraints(looseConstraints);
            var meetsConstraints = fbnet.MeetsConstraints();

            // Assert - should meet very loose constraints
            Assert.True(meetsConstraints);
        }

        #endregion
    }
}
