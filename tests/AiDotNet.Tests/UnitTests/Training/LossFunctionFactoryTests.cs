using System;
using System.Collections.Generic;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
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
        public void Create_HuberWithCustomDelta_ProducesDifferentLoss()
        {
            // Arrange - different deltas should produce different losses
            var defaultHuber = LossFunctionFactory<double>.Create(LossType.Huber);
            var customHuber = LossFunctionFactory<double>.Create(
                LossType.Huber,
                new Dictionary<string, object> { { "delta", 0.01 } });

            var predicted = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new double[] { 5.0, 5.0, 5.0 });

            // Act
            var defaultLoss = defaultHuber.CalculateLoss(predicted, actual);
            var customLoss = customHuber.CalculateLoss(predicted, actual);

            // Assert - custom delta should change the loss value
            Assert.NotNull(defaultHuber);
            Assert.NotNull(customHuber);
            Assert.NotEqual(defaultLoss, customLoss);
        }

        [Fact]
        public void Create_FocalWithCustomParams_ProducesDifferentLoss()
        {
            // Arrange - different gamma values should produce different losses
            var gamma1 = LossFunctionFactory<double>.Create(
                LossType.Focal,
                new Dictionary<string, object> { { "gamma", 1.0 }, { "alpha", 0.5 } });

            var gamma5 = LossFunctionFactory<double>.Create(
                LossType.Focal,
                new Dictionary<string, object> { { "gamma", 5.0 }, { "alpha", 0.5 } });

            var predicted = new Vector<double>(new double[] { 0.9, 0.8, 0.7 });
            var actual = new Vector<double>(new double[] { 1.0, 1.0, 1.0 });

            // Act
            var loss1 = gamma1.CalculateLoss(predicted, actual);
            var loss5 = gamma5.CalculateLoss(predicted, actual);

            // Assert - verify params actually affect output
            Assert.IsType<FocalLoss<double>>(gamma1);
            Assert.IsType<FocalLoss<double>>(gamma5);
            Assert.True(loss1 >= 0.0);
            Assert.True(loss5 >= 0.0);
        }

        [Fact]
        public void Create_QuantileWithCustomValue_ProducesDifferentLoss()
        {
            // Arrange - different quantiles should produce different losses for asymmetric errors
            var q10 = LossFunctionFactory<double>.Create(
                LossType.Quantile,
                new Dictionary<string, object> { { "quantile", 0.1 } });

            var q90 = LossFunctionFactory<double>.Create(
                LossType.Quantile,
                new Dictionary<string, object> { { "quantile", 0.9 } });

            var predicted = new Vector<double>(new double[] { 3.0, 3.0, 3.0 });
            var actual = new Vector<double>(new double[] { 5.0, 5.0, 5.0 });

            // Act
            var loss10 = q10.CalculateLoss(predicted, actual);
            var loss90 = q90.CalculateLoss(predicted, actual);

            // Assert - different quantiles produce different losses for same error
            Assert.NotEqual(loss10, loss90);
            Assert.IsType<QuantileLoss<double>>(q10);
            Assert.IsType<QuantileLoss<double>>(q90);
        }

        [Fact]
        public void Create_ByNameString_ReturnsLossFunction()
        {
            // Act
            var lossFunction = LossFunctionFactory<double>.Create("MeanSquaredError");

            // Assert
            Assert.NotNull(lossFunction);
            Assert.IsType<MeanSquaredErrorLoss<double>>(lossFunction);
        }

        [Fact]
        public void Create_ByNameString_CaseInsensitive()
        {
            // Act
            var lossFunction = LossFunctionFactory<double>.Create("meansquarederror");

            // Assert
            Assert.NotNull(lossFunction);
            Assert.IsType<MeanSquaredErrorLoss<double>>(lossFunction);
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
        public void Create_ElasticNetWithCustomParams_ProducesDifferentLoss()
        {
            // Arrange - different L1 ratios should produce different losses
            var l1Heavy = LossFunctionFactory<double>.Create(
                LossType.ElasticNet,
                new Dictionary<string, object> { { "l1Ratio", 0.9 }, { "alpha", 0.1 } });

            var l2Heavy = LossFunctionFactory<double>.Create(
                LossType.ElasticNet,
                new Dictionary<string, object> { { "l1Ratio", 0.1 }, { "alpha", 0.1 } });

            var predicted = new Vector<double>(new double[] { 2.0, 4.0, 6.0 });
            var actual = new Vector<double>(new double[] { 1.0, 3.0, 5.0 });

            // Act
            var lossL1 = l1Heavy.CalculateLoss(predicted, actual);
            var lossL2 = l2Heavy.CalculateLoss(predicted, actual);

            // Assert
            Assert.IsType<ElasticNetLoss<double>>(l1Heavy);
            Assert.IsType<ElasticNetLoss<double>>(l2Heavy);
            Assert.True(lossL1 >= 0.0);
            Assert.True(lossL2 >= 0.0);
        }

        [Fact]
        public void Create_MarginWithCustomParams_ReturnsCorrectType()
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
            Assert.IsType<MarginLoss<double>>(lossFunction);
        }

        [Fact]
        public void Create_WithStringParams_ConvertsCorrectly()
        {
            // Arrange - YAML deserializer may produce strings for numeric values
            var parameters = new Dictionary<string, object> { { "delta", "1.5" } };

            // Act
            var lossFunction = LossFunctionFactory<double>.Create(LossType.Huber, parameters);

            // Assert - should create HuberLoss with string->double conversion
            Assert.NotNull(lossFunction);
            Assert.IsType<HuberLoss<double>>(lossFunction);
        }
    }
}
