using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Training.Factories;
using Xunit;

namespace AiDotNetTests.UnitTests.Training
{
    public class LossFunctionFactoryTests
    {
        [Theory]
        [InlineData(LossType.MeanSquaredError)]
        [InlineData(LossType.MeanAbsoluteError)]
        [InlineData(LossType.RootMeanSquaredError)]
        [InlineData(LossType.Huber)]
        [InlineData(LossType.CrossEntropy)]
        [InlineData(LossType.BinaryCrossEntropy)]
        [InlineData(LossType.CategoricalCrossEntropy)]
        [InlineData(LossType.SparseCategoricalCrossEntropy)]
        [InlineData(LossType.Focal)]
        [InlineData(LossType.Hinge)]
        [InlineData(LossType.SquaredHinge)]
        [InlineData(LossType.LogCosh)]
        [InlineData(LossType.Quantile)]
        [InlineData(LossType.Poisson)]
        [InlineData(LossType.KullbackLeiblerDivergence)]
        [InlineData(LossType.CosineSimilarity)]
        [InlineData(LossType.Contrastive)]
        [InlineData(LossType.Triplet)]
        [InlineData(LossType.Dice)]
        [InlineData(LossType.Jaccard)]
        [InlineData(LossType.ElasticNet)]
        [InlineData(LossType.Exponential)]
        [InlineData(LossType.ModifiedHuber)]
        [InlineData(LossType.Charbonnier)]
        [InlineData(LossType.MeanBiasError)]
        [InlineData(LossType.Wasserstein)]
        [InlineData(LossType.Margin)]
        [InlineData(LossType.CTC)]
        [InlineData(LossType.NoiseContrastiveEstimation)]
        [InlineData(LossType.OrdinalRegression)]
        [InlineData(LossType.WeightedCrossEntropy)]
        [InlineData(LossType.ScaleInvariantDepth)]
        [InlineData(LossType.Quantum)]
        public void Create_WithDefaultParams_ReturnsNonNullLossFunction(LossType lossType)
        {
            // Act
            var lossFunction = LossFunctionFactory<double>.Create(lossType);

            // Assert
            Assert.NotNull(lossFunction);
            Assert.IsAssignableFrom<ILossFunction<double>>(lossFunction);
        }

        [Fact]
        public void Create_HuberWithCustomDelta_ReturnsLossFunction()
        {
            // Arrange
            var parameters = new Dictionary<string, object> { { "delta", 2.5 } };

            // Act
            var lossFunction = LossFunctionFactory<double>.Create(LossType.Huber, parameters);

            // Assert
            Assert.NotNull(lossFunction);
        }

        [Fact]
        public void Create_FocalWithCustomParams_ReturnsLossFunction()
        {
            // Arrange
            var parameters = new Dictionary<string, object>
            {
                { "gamma", 3.0 },
                { "alpha", 0.5 }
            };

            // Act
            var lossFunction = LossFunctionFactory<double>.Create(LossType.Focal, parameters);

            // Assert
            Assert.NotNull(lossFunction);
        }

        [Fact]
        public void Create_QuantileWithCustomValue_ReturnsLossFunction()
        {
            // Arrange
            var parameters = new Dictionary<string, object> { { "quantile", 0.9 } };

            // Act
            var lossFunction = LossFunctionFactory<double>.Create(LossType.Quantile, parameters);

            // Assert
            Assert.NotNull(lossFunction);
        }

        [Fact]
        public void Create_ByNameString_ReturnsLossFunction()
        {
            // Act
            var lossFunction = LossFunctionFactory<double>.Create("MeanSquaredError");

            // Assert
            Assert.NotNull(lossFunction);
            Assert.IsAssignableFrom<ILossFunction<double>>(lossFunction);
        }

        [Fact]
        public void Create_ByNameString_CaseInsensitive()
        {
            // Act
            var lossFunction = LossFunctionFactory<double>.Create("meansquarederror");

            // Assert
            Assert.NotNull(lossFunction);
        }

        [Fact]
        public void Create_WithInvalidName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                LossFunctionFactory<double>.Create("NonExistentLoss"));
        }

        [Fact]
        public void Create_WithEmptyName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                LossFunctionFactory<double>.Create(""));
        }

        [Fact]
        public void Create_WithNullName_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() =>
                LossFunctionFactory<double>.Create((string)null));
        }

        [Fact]
        public void Create_ElasticNetWithCustomParams_ReturnsLossFunction()
        {
            // Arrange
            var parameters = new Dictionary<string, object>
            {
                { "l1Ratio", 0.7 },
                { "alpha", 0.05 }
            };

            // Act
            var lossFunction = LossFunctionFactory<double>.Create(LossType.ElasticNet, parameters);

            // Assert
            Assert.NotNull(lossFunction);
        }

        [Fact]
        public void Create_MarginWithCustomParams_ReturnsLossFunction()
        {
            // Arrange
            var parameters = new Dictionary<string, object>
            {
                { "mPlus", 0.8 },
                { "mMinus", 0.2 },
                { "lambda", 0.3 }
            };

            // Act
            var lossFunction = LossFunctionFactory<double>.Create(LossType.Margin, parameters);

            // Assert
            Assert.NotNull(lossFunction);
        }

        [Fact]
        public void Create_WithStringParams_ConvertsCorrectly()
        {
            // Arrange - YAML deserializer may produce strings for numeric values
            var parameters = new Dictionary<string, object> { { "delta", "1.5" } };

            // Act
            var lossFunction = LossFunctionFactory<double>.Create(LossType.Huber, parameters);

            // Assert
            Assert.NotNull(lossFunction);
        }
    }
}
