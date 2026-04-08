using System;
using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.NAS
{
    /// <summary>
    /// Unit tests for the GDAS (Gradient-based Differentiable Architecture Search) algorithm.
    /// </summary>
    public class GDASTests
    {
        [Fact]
        public void GDAS_Constructor_InitializesCorrectly()
        {
            // Arrange & Act
            var searchSpace = new SearchSpaceBase<double>();
            var gdas = new GDAS<double>(searchSpace, numNodes: 4);

            // Assert
            Assert.NotNull(gdas);
        }

        [Fact]
        public void GDAS_GetArchitectureParameters_ReturnsValidList()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var gdas = new GDAS<double>(searchSpace, numNodes: 4);

            // Act
            var params_ = gdas.GetArchitectureParameters();

            // Assert
            Assert.NotNull(params_);
            Assert.Equal(4, params_.Count);
        }

        [Fact]
        public void GDAS_GetArchitectureParameters_HasCorrectShape()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var gdas = new GDAS<double>(searchSpace, numNodes: 3);

            // Act
            var params_ = gdas.GetArchitectureParameters();

            // Assert
            Assert.Equal(1, params_[0].Rows); // First node connects to 1 previous node
            Assert.Equal(2, params_[1].Rows); // Second node connects to 2 previous nodes
            Assert.Equal(3, params_[2].Rows); // Third node connects to 3 previous nodes
        }

        [Fact]
        public void GDAS_GetArchitectureGradients_MatchesParameterCount()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var gdas = new GDAS<double>(searchSpace, numNodes: 4);

            // Act
            var params_ = gdas.GetArchitectureParameters();
            var gradients = gdas.GetArchitectureGradients();

            // Assert
            Assert.Equal(params_.Count, gradients.Count);
        }

        [Fact]
        public void GDAS_GetTemperature_ReturnsInitialTemperature()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var gdas = new GDAS<double>(searchSpace, numNodes: 4, initialTemperature: 5.0);

            // Act
            var temperature = gdas.GetTemperature();

            // Assert
            Assert.Equal(5.0, temperature);
        }

        [Fact]
        public void GDAS_AnnealTemperature_DecreasesTemperature()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var gdas = new GDAS<double>(searchSpace, numNodes: 4, initialTemperature: 5.0, finalTemperature: 0.1);
            var initialTemp = gdas.GetTemperature();

            // Act
            gdas.AnnealTemperature(50, 100); // Halfway through training
            var midTemp = gdas.GetTemperature();

            // Assert
            Assert.True(midTemp < initialTemp);
        }

        [Fact]
        public void GDAS_AnnealTemperature_AtEnd_ApproachesFinalTemperature()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var gdas = new GDAS<double>(searchSpace, numNodes: 4, initialTemperature: 5.0, finalTemperature: 0.1);

            // Act
            gdas.AnnealTemperature(99, 100); // Near end of training
            var finalTemp = gdas.GetTemperature();

            // Assert
            Assert.True(finalTemp < 0.5); // Should be close to 0.1
        }

        [Fact]
        public void GDAS_GumbelSoftmax_ReturnsValidProbabilities()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var gdas = new GDAS<double>(searchSpace, numNodes: 4);
            var alpha = new Matrix<double>(2, 5);
            var random = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < alpha.Rows; i++)
                for (int j = 0; j < alpha.Columns; j++)
                    alpha[i, j] = random.NextDouble() - 0.5;

            // Act
            var probs = gdas.GumbelSoftmax(alpha, hard: false);

            // Assert
            Assert.NotNull(probs);
            Assert.Equal(alpha.Rows, probs.Rows);
            Assert.Equal(alpha.Columns, probs.Columns);
        }

        [Fact]
        public void GDAS_GumbelSoftmax_HardMode_ReturnsOneHotVectors()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var gdas = new GDAS<double>(searchSpace, numNodes: 4);
            var alpha = new Matrix<double>(2, 5);
            var random = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < alpha.Rows; i++)
                for (int j = 0; j < alpha.Columns; j++)
                    alpha[i, j] = random.NextDouble() - 0.5;

            // Act
            var probs = gdas.GumbelSoftmax(alpha, hard: true);

            // Assert - each row should have exactly one 1 (or near 1) and rest 0
            for (int i = 0; i < probs.Rows; i++)
            {
                double rowSum = 0;
                for (int j = 0; j < probs.Columns; j++)
                {
                    rowSum += probs[i, j];
                }
                Assert.True(Math.Abs(rowSum - 1.0) < 0.01); // Row should sum to approximately 1
            }
        }

        [Fact]
        public void GDAS_DeriveArchitecture_ReturnsValidArchitecture()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var gdas = new GDAS<double>(searchSpace, numNodes: 3);

            // Act
            var architecture = gdas.DeriveArchitecture();

            // Assert
            Assert.NotNull(architecture);
            Assert.True(architecture.Operations.Count > 0);
        }

        [Fact]
        public void GDAS_DeriveArchitecture_ContainsOperationsFromSearchSpace()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var gdas = new GDAS<double>(searchSpace, numNodes: 3);

            // Act
            var architecture = gdas.DeriveArchitecture();
            var description = architecture.GetDescription();

            // Assert
            Assert.NotNull(description);
            Assert.Contains("Architecture with", description);
        }

        [Fact]
        public void GDAS_WithDifferentNumNodes_HasDifferentParameterCounts()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var gdas3 = new GDAS<double>(searchSpace, numNodes: 3);
            var gdas5 = new GDAS<double>(searchSpace, numNodes: 5);

            // Act
            var params3 = gdas3.GetArchitectureParameters();
            var params5 = gdas5.GetArchitectureParameters();

            // Assert
            Assert.Equal(3, params3.Count);
            Assert.Equal(5, params5.Count);
        }

        [Fact]
        public void GDAS_TemperatureAnnealing_MonotonicallyDecreases()
        {
            // Arrange
            var searchSpace = new SearchSpaceBase<double>();
            var gdas = new GDAS<double>(searchSpace, numNodes: 4, initialTemperature: 5.0, finalTemperature: 0.1);

            // Act & Assert
            double prevTemp = gdas.GetTemperature();
            for (int epoch = 10; epoch <= 100; epoch += 10)
            {
                gdas.AnnealTemperature(epoch, 100);
                double currentTemp = gdas.GetTemperature();
                Assert.True(currentTemp <= prevTemp);
                prevTemp = currentTemp;
            }
        }
    }
}
