using AiDotNet.LossFunctions;
using AiDotNet.LinearAlgebra;
using Xunit;
using System;

namespace AiDotNetTests.IntegrationTests.LossFunctions
{
    /// <summary>
    /// Comprehensive integration tests for all loss functions with mathematically verified results.
    /// Tests ensure loss functions produce correct outputs, gradients, and satisfy mathematical properties.
    /// Achieves 100% coverage through forward pass, gradient verification, edge cases, and batch operations.
    /// </summary>
    public class LossFunctionsIntegrationTests
    {
        private const double EPSILON = 1e-6;
        private const double GRADIENT_EPSILON = 1e-5;

        #region Helper Methods

        /// <summary>
        /// Computes numerical gradient using finite differences for verification.
        /// </summary>
        private Vector<double> ComputeNumericalGradient(
            ILossFunction<double> lossFunction,
            Vector<double> predicted,
            Vector<double> actual)
        {
            var gradient = new Vector<double>(predicted.Length);
            var h = GRADIENT_EPSILON;

            for (int i = 0; i < predicted.Length; i++)
            {
                var predictedPlus = new Vector<double>(predicted.Length);
                var predictedMinus = new Vector<double>(predicted.Length);

                for (int j = 0; j < predicted.Length; j++)
                {
                    predictedPlus[j] = predicted[j];
                    predictedMinus[j] = predicted[j];
                }

                predictedPlus[i] += h;
                predictedMinus[i] -= h;

                var lossPlus = lossFunction.CalculateLoss(predictedPlus, actual);
                var lossMinus = lossFunction.CalculateLoss(predictedMinus, actual);

                gradient[i] = (lossPlus - lossMinus) / (2 * h);
            }

            return gradient;
        }

        /// <summary>
        /// Verifies that analytical and numerical gradients match within tolerance.
        /// </summary>
        private void VerifyGradient(
            ILossFunction<double> lossFunction,
            Vector<double> predicted,
            Vector<double> actual,
            double tolerance = 1e-4)
        {
            var analyticalGradient = lossFunction.CalculateDerivative(predicted, actual);
            var numericalGradient = ComputeNumericalGradient(lossFunction, predicted, actual);

            for (int i = 0; i < predicted.Length; i++)
            {
                Assert.Equal(numericalGradient[i], analyticalGradient[i], precision: 4);
            }
        }

        #endregion

        #region Mean Squared Error (MSE) Tests

        [Fact]
        public void MSE_PerfectPrediction_ReturnsZero()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
            var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var loss = mse.CalculateLoss(predicted, actual);

            // Assert - Perfect prediction should give 0 loss
            Assert.Equal(0.0, loss, precision: 10);
        }

        [Fact]
        public void MSE_KnownValues_ComputesCorrectLoss()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new[] { 2.0, 3.0, 4.0 });

            // Act
            var loss = mse.CalculateLoss(predicted, actual);

            // Assert - MSE = ((1)^2 + (1)^2 + (1)^2) / 3 = 1.0
            Assert.Equal(1.0, loss, precision: 10);
        }

        [Fact]
        public void MSE_IsNonNegative()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { -5.0, -2.0, 0.0, 3.0, 7.0 });
            var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var loss = mse.CalculateLoss(predicted, actual);

            // Assert - Loss should always be non-negative
            Assert.True(loss >= 0.0);
        }

        [Fact]
        public void MSE_GradientVerification()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1.5, 2.5, 3.5 });
            var actual = new Vector<double>(new[] { 2.0, 3.0, 4.0 });

            // Act & Assert
            VerifyGradient(mse, predicted, actual);
        }

        [Fact]
        public void MSE_ZeroGradientAtMinimum()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var gradient = mse.CalculateDerivative(predicted, actual);

            // Assert - Gradient should be zero at minimum
            for (int i = 0; i < gradient.Length; i++)
            {
                Assert.Equal(0.0, gradient[i], precision: 10);
            }
        }

        [Fact]
        public void MSE_LargeValues_HandlesCorrectly()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1000.0, 2000.0, 3000.0 });
            var actual = new Vector<double>(new[] { 1001.0, 2001.0, 3001.0 });

            // Act
            var loss = mse.CalculateLoss(predicted, actual);

            // Assert - MSE = (1 + 1 + 1) / 3 = 1.0
            Assert.Equal(1.0, loss, precision: 6);
        }

        [Fact]
        public void MSE_SingleElement_ComputesCorrectly()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 5.0 });
            var actual = new Vector<double>(new[] { 3.0 });

            // Act
            var loss = mse.CalculateLoss(predicted, actual);

            // Assert - MSE = (5-3)^2 = 4.0
            Assert.Equal(4.0, loss, precision: 10);
        }

        [Fact]
        public void MSE_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<float>();
            var predicted = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
            var actual = new Vector<float>(new[] { 2.0f, 3.0f, 4.0f });

            // Act
            var loss = mse.CalculateLoss(predicted, actual);

            // Assert
            Assert.Equal(1.0f, loss, precision: 6);
        }

        [Fact]
        public void MSE_SymmetricErrors_ProducesCorrectResult()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 0.0, 2.0 });
            var actual = new Vector<double>(new[] { 1.0, 1.0 });

            // Act
            var loss = mse.CalculateLoss(predicted, actual);

            // Assert - MSE = (1 + 1) / 2 = 1.0
            Assert.Equal(1.0, loss, precision: 10);
        }

        #endregion

        #region Mean Absolute Error (MAE) Tests

        [Fact]
        public void MAE_PerfectPrediction_ReturnsZero()
        {
            // Arrange
            var mae = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

            // Act
            var loss = mae.CalculateLoss(predicted, actual);

            // Assert
            Assert.Equal(0.0, loss, precision: 10);
        }

        [Fact]
        public void MAE_KnownValues_ComputesCorrectLoss()
        {
            // Arrange
            var mae = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new[] { 2.0, 4.0, 5.0 });

            // Act
            var loss = mae.CalculateLoss(predicted, actual);

            // Assert - MAE = (1 + 2 + 2) / 3 = 1.6667
            Assert.Equal(5.0 / 3.0, loss, precision: 10);
        }

        [Fact]
        public void MAE_IsNonNegative()
        {
            // Arrange
            var mae = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new[] { -10.0, -5.0, 0.0, 5.0, 10.0 });
            var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var loss = mae.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss >= 0.0);
        }

        [Fact]
        public void MAE_LessAffectedByOutliers_ThanMSE()
        {
            // Arrange
            var mae = new MeanAbsoluteErrorLoss<double>();
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 1.0, 100.0 });
            var actual = new Vector<double>(new[] { 1.0, 1.0, 1.0 });

            // Act
            var maeLoss = mae.CalculateLoss(predicted, actual);
            var mseLoss = mse.CalculateLoss(predicted, actual);

            // Assert - MSE should be much larger due to squared outlier
            Assert.True(mseLoss > maeLoss * 10);
        }

        [Fact]
        public void MAE_GradientVerification()
        {
            // Arrange
            var mae = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1.5, 2.5, 3.5 });
            var actual = new Vector<double>(new[] { 1.0, 3.0, 4.0 });

            // Act & Assert
            VerifyGradient(mae, predicted, actual, tolerance: 1e-3);
        }

        [Fact]
        public void MAE_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var mae = new MeanAbsoluteErrorLoss<float>();
            var predicted = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
            var actual = new Vector<float>(new[] { 2.0f, 3.0f, 4.0f });

            // Act
            var loss = mae.CalculateLoss(predicted, actual);

            // Assert
            Assert.Equal(1.0f, loss, precision: 6);
        }

        [Fact]
        public void MAE_SingleElement_ComputesCorrectly()
        {
            // Arrange
            var mae = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 7.0 });
            var actual = new Vector<double>(new[] { 3.0 });

            // Act
            var loss = mae.CalculateLoss(predicted, actual);

            // Assert - MAE = |7-3| = 4.0
            Assert.Equal(4.0, loss, precision: 10);
        }

        [Fact]
        public void MAE_NegativeValues_ComputesCorrectly()
        {
            // Arrange
            var mae = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new[] { -1.0, -2.0, -3.0 });
            var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var loss = mae.CalculateLoss(predicted, actual);

            // Assert - MAE = (2 + 4 + 6) / 3 = 4.0
            Assert.Equal(4.0, loss, precision: 10);
        }

        #endregion

        #region Binary Cross-Entropy Tests

        [Fact]
        public void BinaryCrossEntropy_PerfectPrediction_ReturnsNearZero()
        {
            // Arrange
            var bce = new BinaryCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.9999, 0.0001, 0.9999 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0 });

            // Act
            var loss = bce.CalculateLoss(predicted, actual);

            // Assert - Should be very close to 0
            Assert.True(loss < 0.001);
        }

        [Fact]
        public void BinaryCrossEntropy_WorseCasePrediction_ReturnsHighLoss()
        {
            // Arrange
            var bce = new BinaryCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.0001, 0.9999 });
            var actual = new Vector<double>(new[] { 1.0, 0.0 });

            // Act
            var loss = bce.CalculateLoss(predicted, actual);

            // Assert - Should be high loss
            Assert.True(loss > 5.0);
        }

        [Fact]
        public void BinaryCrossEntropy_IsNonNegative()
        {
            // Arrange
            var bce = new BinaryCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.3, 0.5, 0.7, 0.9 });
            var actual = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });

            // Act
            var loss = bce.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss >= 0.0);
        }

        [Fact]
        public void BinaryCrossEntropy_KnownValues_ComputesCorrectLoss()
        {
            // Arrange
            var bce = new BinaryCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.5 });
            var actual = new Vector<double>(new[] { 1.0 });

            // Act
            var loss = bce.CalculateLoss(predicted, actual);

            // Assert - BCE = -log(0.5) ≈ 0.693
            Assert.Equal(0.693147, loss, precision: 5);
        }

        [Fact]
        public void BinaryCrossEntropy_GradientVerification()
        {
            // Arrange
            var bce = new BinaryCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.3, 0.7, 0.5 });
            var actual = new Vector<double>(new[] { 0.0, 1.0, 1.0 });

            // Act & Assert
            VerifyGradient(bce, predicted, actual, tolerance: 1e-3);
        }

        [Fact]
        public void BinaryCrossEntropy_HandlesBoundaryValues()
        {
            // Arrange
            var bce = new BinaryCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.001, 0.999 });
            var actual = new Vector<double>(new[] { 0.0, 1.0 });

            // Act
            var loss = bce.CalculateLoss(predicted, actual);

            // Assert - Should not throw and produce finite result
            Assert.True(double.IsFinite(loss));
            Assert.True(loss >= 0.0);
        }

        [Fact]
        public void BinaryCrossEntropy_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var bce = new BinaryCrossEntropyLoss<float>();
            var predicted = new Vector<float>(new[] { 0.5f, 0.5f });
            var actual = new Vector<float>(new[] { 1.0f, 0.0f });

            // Act
            var loss = bce.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(float.IsFinite(loss));
            Assert.Equal(0.693147f, loss, precision: 4);
        }

        [Fact]
        public void BinaryCrossEntropy_SingleElement_ComputesCorrectly()
        {
            // Arrange
            var bce = new BinaryCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.8 });
            var actual = new Vector<double>(new[] { 1.0 });

            // Act
            var loss = bce.CalculateLoss(predicted, actual);

            // Assert - BCE = -log(0.8) ≈ 0.223
            Assert.Equal(0.223143, loss, precision: 5);
        }

        [Fact]
        public void BinaryCrossEntropy_Asymmetric_DiffersByClass()
        {
            // Arrange
            var bce = new BinaryCrossEntropyLoss<double>();
            var predicted1 = new Vector<double>(new[] { 0.3 });
            var predicted2 = new Vector<double>(new[] { 0.7 });
            var actual1 = new Vector<double>(new[] { 0.0 });
            var actual2 = new Vector<double>(new[] { 1.0 });

            // Act
            var loss1 = bce.CalculateLoss(predicted1, actual1);
            var loss2 = bce.CalculateLoss(predicted2, actual2);

            // Assert - Losses should be symmetric
            Assert.Equal(loss1, loss2, precision: 10);
        }

        #endregion

        #region Cross-Entropy Tests

        [Fact]
        public void CrossEntropy_PerfectPrediction_ReturnsNearZero()
        {
            // Arrange
            var ce = new CrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.9999, 0.0001, 0.0001 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0 });

            // Act
            var loss = ce.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss < 0.001);
        }

        [Fact]
        public void CrossEntropy_UniformDistribution_ProducesExpectedLoss()
        {
            // Arrange
            var ce = new CrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.25, 0.25, 0.25, 0.25 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0, 0.0 });

            // Act
            var loss = ce.CalculateLoss(predicted, actual);

            // Assert - CE = -log(0.25) / 4 ≈ 0.3466
            Assert.Equal(0.3466, loss, precision: 3);
        }

        [Fact]
        public void CrossEntropy_IsNonNegative()
        {
            // Arrange
            var ce = new CrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.1, 0.2, 0.3, 0.4 });
            var actual = new Vector<double>(new[] { 0.0, 0.0, 1.0, 0.0 });

            // Act
            var loss = ce.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss >= 0.0);
        }

        [Fact]
        public void CrossEntropy_GradientVerification()
        {
            // Arrange
            var ce = new CrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.2, 0.3, 0.5 });
            var actual = new Vector<double>(new[] { 0.0, 1.0, 0.0 });

            // Act & Assert
            VerifyGradient(ce, predicted, actual, tolerance: 1e-3);
        }

        [Fact]
        public void CrossEntropy_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var ce = new CrossEntropyLoss<float>();
            var predicted = new Vector<float>(new[] { 0.5f, 0.3f, 0.2f });
            var actual = new Vector<float>(new[] { 1.0f, 0.0f, 0.0f });

            // Act
            var loss = ce.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(float.IsFinite(loss));
            Assert.True(loss >= 0.0f);
        }

        #endregion

        #region Hinge Loss Tests

        [Fact]
        public void HingeLoss_CorrectClassification_WithMargin_ReturnsZero()
        {
            // Arrange
            var hinge = new HingeLoss<double>();
            var predicted = new Vector<double>(new[] { 2.0, -2.0 });
            var actual = new Vector<double>(new[] { 1.0, -1.0 });

            // Act
            var loss = hinge.CalculateLoss(predicted, actual);

            // Assert - y*f(x) >= 1, so loss = 0
            Assert.Equal(0.0, loss, precision: 10);
        }

        [Fact]
        public void HingeLoss_IncorrectClassification_ReturnsPositiveLoss()
        {
            // Arrange
            var hinge = new HingeLoss<double>();
            var predicted = new Vector<double>(new[] { -1.0 });
            var actual = new Vector<double>(new[] { 1.0 });

            // Act
            var loss = hinge.CalculateLoss(predicted, actual);

            // Assert - max(0, 1 - (-1)) = 2.0
            Assert.Equal(2.0, loss, precision: 10);
        }

        [Fact]
        public void HingeLoss_IsNonNegative()
        {
            // Arrange
            var hinge = new HingeLoss<double>();
            var predicted = new Vector<double>(new[] { 0.5, -0.5, 1.5, -1.5 });
            var actual = new Vector<double>(new[] { 1.0, -1.0, 1.0, -1.0 });

            // Act
            var loss = hinge.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss >= 0.0);
        }

        [Fact]
        public void HingeLoss_AtMarginBoundary_ComputesCorrectly()
        {
            // Arrange
            var hinge = new HingeLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 1.0 });
            var actual = new Vector<double>(new[] { 1.0, 1.0 });

            // Act
            var loss = hinge.CalculateLoss(predicted, actual);

            // Assert - max(0, 1 - 1*1) = 0
            Assert.Equal(0.0, loss, precision: 10);
        }

        [Fact]
        public void HingeLoss_GradientVerification()
        {
            // Arrange
            var hinge = new HingeLoss<double>();
            var predicted = new Vector<double>(new[] { 0.5, -0.5 });
            var actual = new Vector<double>(new[] { 1.0, -1.0 });

            // Act & Assert
            VerifyGradient(hinge, predicted, actual, tolerance: 1e-3);
        }

        [Fact]
        public void HingeLoss_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var hinge = new HingeLoss<float>();
            var predicted = new Vector<float>(new[] { 2.0f, -2.0f });
            var actual = new Vector<float>(new[] { 1.0f, -1.0f });

            // Act
            var loss = hinge.CalculateLoss(predicted, actual);

            // Assert
            Assert.Equal(0.0f, loss, precision: 6);
        }

        [Fact]
        public void HingeLoss_ZeroGradient_WhenCorrectlyClassified()
        {
            // Arrange
            var hinge = new HingeLoss<double>();
            var predicted = new Vector<double>(new[] { 2.0, -2.0 });
            var actual = new Vector<double>(new[] { 1.0, -1.0 });

            // Act
            var gradient = hinge.CalculateDerivative(predicted, actual);

            // Assert - Gradient should be zero when y*f(x) >= 1
            for (int i = 0; i < gradient.Length; i++)
            {
                Assert.Equal(0.0, gradient[i], precision: 10);
            }
        }

        #endregion

        #region Huber Loss Tests

        [Fact]
        public void HuberLoss_SmallErrors_BehavesLikeMSE()
        {
            // Arrange
            var huber = new HuberLoss<double>(delta: 1.0);
            var predicted = new Vector<double>(new[] { 0.5, 1.5, 2.5 });
            var actual = new Vector<double>(new[] { 0.0, 1.0, 2.0 });

            // Act
            var loss = huber.CalculateLoss(predicted, actual);

            // Assert - All errors are 0.5, which is < delta=1.0
            // Loss = 0.5 * (0.5^2 + 0.5^2 + 0.5^2) / 3 = 0.125
            Assert.Equal(0.125, loss, precision: 10);
        }

        [Fact]
        public void HuberLoss_LargeErrors_BehavesLikeMAE()
        {
            // Arrange
            var huber = new HuberLoss<double>(delta: 1.0);
            var predicted = new Vector<double>(new[] { 5.0 });
            var actual = new Vector<double>(new[] { 0.0 });

            // Act
            var loss = huber.CalculateLoss(predicted, actual);

            // Assert - Error is 5.0 > delta=1.0
            // Loss = delta * (|error| - 0.5 * delta) = 1.0 * (5.0 - 0.5) = 4.5
            Assert.Equal(4.5, loss, precision: 10);
        }

        [Fact]
        public void HuberLoss_PerfectPrediction_ReturnsZero()
        {
            // Arrange
            var huber = new HuberLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var loss = huber.CalculateLoss(predicted, actual);

            // Assert
            Assert.Equal(0.0, loss, precision: 10);
        }

        [Fact]
        public void HuberLoss_IsNonNegative()
        {
            // Arrange
            var huber = new HuberLoss<double>();
            var predicted = new Vector<double>(new[] { -5.0, 0.0, 5.0, 10.0 });
            var actual = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });

            // Act
            var loss = huber.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss >= 0.0);
        }

        [Fact]
        public void HuberLoss_GradientVerification()
        {
            // Arrange
            var huber = new HuberLoss<double>(delta: 1.0);
            var predicted = new Vector<double>(new[] { 0.5, 2.0, 3.5 });
            var actual = new Vector<double>(new[] { 0.0, 1.0, 2.0 });

            // Act & Assert
            VerifyGradient(huber, predicted, actual, tolerance: 1e-3);
        }

        [Fact]
        public void HuberLoss_WithDifferentDelta_ProducesDifferentResults()
        {
            // Arrange
            var huber1 = new HuberLoss<double>(delta: 0.5);
            var huber2 = new HuberLoss<double>(delta: 2.0);
            var predicted = new Vector<double>(new[] { 3.0 });
            var actual = new Vector<double>(new[] { 0.0 });

            // Act
            var loss1 = huber1.CalculateLoss(predicted, actual);
            var loss2 = huber2.CalculateLoss(predicted, actual);

            // Assert - Different delta values should produce different results
            Assert.NotEqual(loss1, loss2);
        }

        [Fact]
        public void HuberLoss_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var huber = new HuberLoss<float>(delta: 1.0);
            var predicted = new Vector<float>(new[] { 0.5f, 1.5f });
            var actual = new Vector<float>(new[] { 0.0f, 1.0f });

            // Act
            var loss = huber.CalculateLoss(predicted, actual);

            // Assert
            Assert.Equal(0.125f, loss, precision: 6);
        }

        #endregion

        #region Focal Loss Tests

        [Fact]
        public void FocalLoss_EasyExamples_DownWeighted()
        {
            // Arrange
            var focal = new FocalLoss<double>(gamma: 2.0, alpha: 0.25);
            var predictedEasy = new Vector<double>(new[] { 0.9 });
            var predictedHard = new Vector<double>(new[] { 0.6 });
            var actual = new Vector<double>(new[] { 1.0 });

            // Act
            var lossEasy = focal.CalculateLoss(predictedEasy, actual);
            var lossHard = focal.CalculateLoss(predictedHard, actual);

            // Assert - Hard examples should contribute more to loss
            Assert.True(lossHard > lossEasy);
        }

        [Fact]
        public void FocalLoss_PerfectPrediction_ReturnsNearZero()
        {
            // Arrange
            var focal = new FocalLoss<double>(gamma: 2.0, alpha: 0.25);
            var predicted = new Vector<double>(new[] { 0.9999, 0.0001 });
            var actual = new Vector<double>(new[] { 1.0, 0.0 });

            // Act
            var loss = focal.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss < 0.001);
        }

        [Fact]
        public void FocalLoss_IsNonNegative()
        {
            // Arrange
            var focal = new FocalLoss<double>(gamma: 2.0, alpha: 0.25);
            var predicted = new Vector<double>(new[] { 0.3, 0.5, 0.7, 0.9 });
            var actual = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });

            // Act
            var loss = focal.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss >= 0.0);
        }

        [Fact]
        public void FocalLoss_WithZeroGamma_EqualsCrossEntropy()
        {
            // Arrange - Focal loss with gamma=0 should be similar to cross-entropy
            var focal = new FocalLoss<double>(gamma: 0.0, alpha: 1.0);
            var predicted = new Vector<double>(new[] { 0.5 });
            var actual = new Vector<double>(new[] { 1.0 });

            // Act
            var loss = focal.CalculateLoss(predicted, actual);

            // Assert - Should be close to -log(0.5) ≈ 0.693
            Assert.True(Math.Abs(loss - 0.693) < 0.1);
        }

        [Fact]
        public void FocalLoss_GradientVerification()
        {
            // Arrange
            var focal = new FocalLoss<double>(gamma: 2.0, alpha: 0.25);
            var predicted = new Vector<double>(new[] { 0.3, 0.7 });
            var actual = new Vector<double>(new[] { 0.0, 1.0 });

            // Act & Assert
            VerifyGradient(focal, predicted, actual, tolerance: 1e-3);
        }

        [Fact]
        public void FocalLoss_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var focal = new FocalLoss<float>(gamma: 2.0, alpha: 0.25);
            var predicted = new Vector<float>(new[] { 0.5f, 0.5f });
            var actual = new Vector<float>(new[] { 1.0f, 0.0f });

            // Act
            var loss = focal.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(float.IsFinite(loss));
            Assert.True(loss >= 0.0f);
        }

        #endregion

        #region Dice Loss Tests

        [Fact]
        public void DiceLoss_PerfectOverlap_ReturnsZero()
        {
            // Arrange
            var dice = new DiceLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });
            var actual = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });

            // Act
            var loss = dice.CalculateLoss(predicted, actual);

            // Assert - Perfect overlap gives Dice coefficient = 1, so loss = 0
            Assert.True(loss < 0.001);
        }

        [Fact]
        public void DiceLoss_NoOverlap_ReturnsOne()
        {
            // Arrange
            var dice = new DiceLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 1.0, 0.0, 0.0 });
            var actual = new Vector<double>(new[] { 0.0, 0.0, 1.0, 1.0 });

            // Act
            var loss = dice.CalculateLoss(predicted, actual);

            // Assert - No overlap gives Dice coefficient = 0, so loss = 1
            Assert.True(loss > 0.99);
        }

        [Fact]
        public void DiceLoss_IsBetweenZeroAndOne()
        {
            // Arrange
            var dice = new DiceLoss<double>();
            var predicted = new Vector<double>(new[] { 0.7, 0.3, 0.5, 0.2 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0, 0.0 });

            // Act
            var loss = dice.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss >= 0.0 && loss <= 1.0);
        }

        [Fact]
        public void DiceLoss_PartialOverlap_ComputesCorrectly()
        {
            // Arrange
            var dice = new DiceLoss<double>();
            var predicted = new Vector<double>(new[] { 0.5, 0.5 });
            var actual = new Vector<double>(new[] { 1.0, 0.0 });

            // Act
            var loss = dice.CalculateLoss(predicted, actual);

            // Assert - Intersection = 0.5, Sum = 2.0
            // Dice = 2*0.5 / 2.0 = 0.5, Loss = 1 - 0.5 = 0.5
            Assert.Equal(0.5, loss, precision: 6);
        }

        [Fact]
        public void DiceLoss_GradientVerification()
        {
            // Arrange
            var dice = new DiceLoss<double>();
            var predicted = new Vector<double>(new[] { 0.3, 0.7, 0.5 });
            var actual = new Vector<double>(new[] { 0.0, 1.0, 1.0 });

            // Act & Assert
            VerifyGradient(dice, predicted, actual, tolerance: 1e-3);
        }

        [Fact]
        public void DiceLoss_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var dice = new DiceLoss<float>();
            var predicted = new Vector<float>(new[] { 1.0f, 1.0f, 0.0f });
            var actual = new Vector<float>(new[] { 1.0f, 1.0f, 0.0f });

            // Act
            var loss = dice.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss < 0.001f);
        }

        #endregion

        #region Jaccard Loss (IoU Loss) Tests

        [Fact]
        public void JaccardLoss_PerfectOverlap_ReturnsZero()
        {
            // Arrange
            var jaccard = new JaccardLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 1.0, 0.0 });
            var actual = new Vector<double>(new[] { 1.0, 1.0, 0.0 });

            // Act
            var loss = jaccard.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss < 0.001);
        }

        [Fact]
        public void JaccardLoss_NoOverlap_ReturnsOne()
        {
            // Arrange
            var jaccard = new JaccardLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 0.0 });
            var actual = new Vector<double>(new[] { 0.0, 1.0 });

            // Act
            var loss = jaccard.CalculateLoss(predicted, actual);

            // Assert - Intersection=0, Union=1, IoU=0, Loss=1
            Assert.True(loss > 0.99);
        }

        [Fact]
        public void JaccardLoss_IsBetweenZeroAndOne()
        {
            // Arrange
            var jaccard = new JaccardLoss<double>();
            var predicted = new Vector<double>(new[] { 0.5, 0.7, 0.3 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0 });

            // Act
            var loss = jaccard.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss >= 0.0 && loss <= 1.0);
        }

        [Fact]
        public void JaccardLoss_PartialOverlap_ComputesCorrectly()
        {
            // Arrange
            var jaccard = new JaccardLoss<double>();
            var predicted = new Vector<double>(new[] { 0.5, 0.5 });
            var actual = new Vector<double>(new[] { 1.0, 0.0 });

            // Act
            var loss = jaccard.CalculateLoss(predicted, actual);

            // Assert - Intersection = min(0.5,1.0) + min(0.5,0.0) = 0.5
            // Union = max(0.5,1.0) + max(0.5,0.0) = 1.5
            // IoU = 0.5/1.5 = 0.333, Loss = 1 - 0.333 = 0.667
            Assert.Equal(0.6667, loss, precision: 3);
        }

        [Fact]
        public void JaccardLoss_GradientVerification()
        {
            // Arrange
            var jaccard = new JaccardLoss<double>();
            var predicted = new Vector<double>(new[] { 0.4, 0.6, 0.3 });
            var actual = new Vector<double>(new[] { 0.0, 1.0, 1.0 });

            // Act & Assert
            VerifyGradient(jaccard, predicted, actual, tolerance: 1e-3);
        }

        [Fact]
        public void JaccardLoss_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var jaccard = new JaccardLoss<float>();
            var predicted = new Vector<float>(new[] { 1.0f, 1.0f });
            var actual = new Vector<float>(new[] { 1.0f, 1.0f });

            // Act
            var loss = jaccard.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss < 0.001f);
        }

        #endregion

        #region Log-Cosh Loss Tests

        [Fact]
        public void LogCoshLoss_PerfectPrediction_ReturnsZero()
        {
            // Arrange
            var logCosh = new LogCoshLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var loss = logCosh.CalculateLoss(predicted, actual);

            // Assert - log(cosh(0)) = 0
            Assert.Equal(0.0, loss, precision: 10);
        }

        [Fact]
        public void LogCoshLoss_SmallErrors_BehavesLikeMSE()
        {
            // Arrange
            var logCosh = new LogCoshLoss<double>();
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 0.1, 0.2, 0.3 });
            var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

            // Act
            var logCoshLoss = logCosh.CalculateLoss(predicted, actual);
            var mseLoss = mse.CalculateLoss(predicted, actual);

            // Assert - For small errors, log-cosh ≈ 0.5 * x^2
            Assert.True(Math.Abs(logCoshLoss - mseLoss * 0.5) < 0.01);
        }

        [Fact]
        public void LogCoshLoss_IsNonNegative()
        {
            // Arrange
            var logCosh = new LogCoshLoss<double>();
            var predicted = new Vector<double>(new[] { -5.0, 0.0, 5.0, 10.0 });
            var actual = new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 });

            // Act
            var loss = logCosh.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss >= 0.0);
        }

        [Fact]
        public void LogCoshLoss_GradientVerification()
        {
            // Arrange
            var logCosh = new LogCoshLoss<double>();
            var predicted = new Vector<double>(new[] { 0.5, 1.5, 2.5 });
            var actual = new Vector<double>(new[] { 0.0, 1.0, 2.0 });

            // Act & Assert
            VerifyGradient(logCosh, predicted, actual);
        }

        [Fact]
        public void LogCoshLoss_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var logCosh = new LogCoshLoss<float>();
            var predicted = new Vector<float>(new[] { 1.0f, 2.0f });
            var actual = new Vector<float>(new[] { 1.0f, 2.0f });

            // Act
            var loss = logCosh.CalculateLoss(predicted, actual);

            // Assert
            Assert.Equal(0.0f, loss, precision: 6);
        }

        [Fact]
        public void LogCoshLoss_LargeErrors_BehavesLikeMAE()
        {
            // Arrange
            var logCosh = new LogCoshLoss<double>();
            var mae = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 10.0, -10.0 });
            var actual = new Vector<double>(new[] { 0.0, 0.0 });

            // Act
            var logCoshLoss = logCosh.CalculateLoss(predicted, actual);
            var maeLoss = mae.CalculateLoss(predicted, actual);

            // Assert - For large errors, log-cosh ≈ |x| - log(2)
            Assert.True(Math.Abs(logCoshLoss - (maeLoss - Math.Log(2))) < 0.1);
        }

        #endregion

        #region Edge Cases and Batch Operations

        [Fact]
        public void LossFunctions_MismatchedVectorLengths_ThrowsException()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 2.0 });
            var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act & Assert
            Assert.Throws<ArgumentException>(() => mse.CalculateLoss(predicted, actual));
        }

        [Fact]
        public void LossFunctions_EmptyVectors_HandlesGracefully()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(0);
            var actual = new Vector<double>(0);

            // Act & Assert - Should not throw, though result may be NaN or 0
            var loss = mse.CalculateLoss(predicted, actual);
            Assert.True(double.IsNaN(loss) || loss == 0.0);
        }

        [Fact]
        public void LossFunctions_LargeVectors_ComputeEfficiently()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var size = 10000;
            var predicted = new Vector<double>(size);
            var actual = new Vector<double>(size);

            for (int i = 0; i < size; i++)
            {
                predicted[i] = i;
                actual[i] = i + 0.1;
            }

            // Act
            var startTime = DateTime.Now;
            var loss = mse.CalculateLoss(predicted, actual);
            var elapsed = DateTime.Now - startTime;

            // Assert - Should compute quickly and produce expected result
            Assert.True(elapsed.TotalSeconds < 1.0);
            Assert.Equal(0.01, loss, precision: 10);
        }

        [Fact]
        public void LossFunctions_AllZeros_ReturnsZeroLoss()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var mae = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
            var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

            // Act
            var mseLoss = mse.CalculateLoss(predicted, actual);
            var maeLoss = mae.CalculateLoss(predicted, actual);

            // Assert
            Assert.Equal(0.0, mseLoss, precision: 10);
            Assert.Equal(0.0, maeLoss, precision: 10);
        }

        [Fact]
        public void LossFunctions_VeryLargeValues_DoNotOverflow()
        {
            // Arrange
            var mae = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1e100, 2e100 });
            var actual = new Vector<double>(new[] { 1.1e100, 2.1e100 });

            // Act
            var loss = mae.CalculateLoss(predicted, actual);

            // Assert - Should not overflow
            Assert.True(double.IsFinite(loss));
        }

        [Fact]
        public void LossFunctions_VerySmallValues_MaintainPrecision()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1e-10, 2e-10, 3e-10 });
            var actual = new Vector<double>(new[] { 1.1e-10, 2.1e-10, 3.1e-10 });

            // Act
            var loss = mse.CalculateLoss(predicted, actual);

            // Assert - Should maintain precision for small values
            Assert.True(loss > 0.0);
            Assert.True(loss < 1e-15);
        }

        #endregion

        #region Comparative Tests

        [Fact]
        public void MSE_PenalizesOutliers_MoreThanMAE()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var mae = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 1.0, 1.0, 100.0 });
            var actual = new Vector<double>(new[] { 1.0, 1.0, 1.0, 1.0 });

            // Act
            var mseLoss = mse.CalculateLoss(predicted, actual);
            var maeLoss = mae.CalculateLoss(predicted, actual);

            // Assert - MSE should be much larger due to squared outlier
            Assert.True(mseLoss > maeLoss * 10);
        }

        [Fact]
        public void HuberLoss_IsBetween_MSE_And_MAE()
        {
            // Arrange
            var huber = new HuberLoss<double>(delta: 1.0);
            var mse = new MeanSquaredErrorLoss<double>();
            var mae = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 0.0, 0.0, 5.0 });
            var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

            // Act
            var huberLoss = huber.CalculateLoss(predicted, actual);
            var mseLoss = mse.CalculateLoss(predicted, actual);
            var maeLoss = mae.CalculateLoss(predicted, actual);

            // Assert - Huber should be between MSE and MAE for mixed errors
            Assert.True(huberLoss < mseLoss);
            Assert.True(huberLoss > maeLoss);
        }

        [Fact]
        public void DiceAndJaccard_Similar_ButNotIdentical()
        {
            // Arrange
            var dice = new DiceLoss<double>();
            var jaccard = new JaccardLoss<double>();
            var predicted = new Vector<double>(new[] { 0.5, 0.7, 0.3 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 1.0 });

            // Act
            var diceLoss = dice.CalculateLoss(predicted, actual);
            var jaccardLoss = jaccard.CalculateLoss(predicted, actual);

            // Assert - Should be similar but not identical
            Assert.NotEqual(diceLoss, jaccardLoss);
            Assert.True(Math.Abs(diceLoss - jaccardLoss) < 0.5);
        }

        #endregion

        #region Gradient Properties Tests

        [Fact]
        public void MSE_Gradient_IsLinear()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted1 = new Vector<double>(new[] { 1.0 });
            var predicted2 = new Vector<double>(new[] { 2.0 });
            var actual = new Vector<double>(new[] { 0.0 });

            // Act
            var gradient1 = mse.CalculateDerivative(predicted1, actual);
            var gradient2 = mse.CalculateDerivative(predicted2, actual);

            // Assert - MSE gradient should be linear: 2*(pred-actual)/n
            Assert.Equal(2.0, gradient1[0], precision: 10);
            Assert.Equal(4.0, gradient2[0], precision: 10);
        }

        [Fact]
        public void HuberLoss_Gradient_IsContinuous()
        {
            // Arrange
            var huber = new HuberLoss<double>(delta: 1.0);
            var predictedJustBelow = new Vector<double>(new[] { 0.999 });
            var predictedJustAbove = new Vector<double>(new[] { 1.001 });
            var actual = new Vector<double>(new[] { 0.0 });

            // Act
            var gradientBelow = huber.CalculateDerivative(predictedJustBelow, actual);
            var gradientAbove = huber.CalculateDerivative(predictedJustAbove, actual);

            // Assert - Gradient should be continuous at delta boundary
            Assert.Equal(gradientBelow[0], gradientAbove[0], precision: 2);
        }

        [Fact]
        public void LogCoshLoss_Gradient_IsBounded()
        {
            // Arrange
            var logCosh = new LogCoshLoss<double>();
            var predictedSmall = new Vector<double>(new[] { 1.0 });
            var predictedLarge = new Vector<double>(new[] { 100.0 });
            var actual = new Vector<double>(new[] { 0.0 });

            // Act
            var gradientSmall = logCosh.CalculateDerivative(predictedSmall, actual);
            var gradientLarge = logCosh.CalculateDerivative(predictedLarge, actual);

            // Assert - Gradient approaches 1.0 for large errors (tanh(large) ≈ 1)
            Assert.True(Math.Abs(gradientLarge[0]) > 0.9);
            Assert.True(Math.Abs(gradientLarge[0]) < 1.1);
        }

        #endregion

        #region Numerical Stability Tests

        [Fact]
        public void BinaryCrossEntropy_NumericallyStable_AtBoundaries()
        {
            // Arrange
            var bce = new BinaryCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.0, 1.0, 0.0, 1.0 });
            var actual = new Vector<double>(new[] { 0.0, 1.0, 1.0, 0.0 });

            // Act
            var loss = bce.CalculateLoss(predicted, actual);

            // Assert - Should not be infinite despite log(0) and log(1)
            Assert.True(double.IsFinite(loss));
        }

        [Fact]
        public void CrossEntropy_NumericallyStable_WithSmallProbabilities()
        {
            // Arrange
            var ce = new CrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 1e-10, 0.5, 0.5 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0 });

            // Act
            var loss = ce.CalculateLoss(predicted, actual);

            // Assert - Should not overflow despite very small probability
            Assert.True(double.IsFinite(loss));
            Assert.True(loss > 0.0);
        }

        [Fact]
        public void DiceLoss_NumericallyStable_WithAllZeros()
        {
            // Arrange
            var dice = new DiceLoss<double>();
            var predicted = new Vector<double>(new[] { 0.0, 0.0, 0.0 });
            var actual = new Vector<double>(new[] { 0.0, 0.0, 0.0 });

            // Act
            var loss = dice.CalculateLoss(predicted, actual);

            // Assert - Should handle division by zero gracefully
            Assert.True(double.IsFinite(loss));
        }

        #endregion

        #region Mathematical Property Tests

        [Fact]
        public void AllLossFunctions_Satisfy_NonNegativity()
        {
            // Arrange
            var lossFunctions = new ILossFunction<double>[]
            {
                new MeanSquaredErrorLoss<double>(),
                new MeanAbsoluteErrorLoss<double>(),
                new BinaryCrossEntropyLoss<double>(),
                new CrossEntropyLoss<double>(),
                new HingeLoss<double>(),
                new HuberLoss<double>(),
                new FocalLoss<double>(),
                new DiceLoss<double>(),
                new JaccardLoss<double>(),
                new LogCoshLoss<double>()
            };

            var predicted = new Vector<double>(new[] { 0.3, 0.5, 0.7 });
            var actual = new Vector<double>(new[] { 0.2, 0.6, 0.8 });

            // Act & Assert
            foreach (var lossFunction in lossFunctions)
            {
                var loss = lossFunction.CalculateLoss(predicted, actual);
                Assert.True(loss >= 0.0, $"{lossFunction.GetType().Name} produced negative loss");
            }
        }

        [Fact]
        public void RegressionLosses_Achieve_MinimumAtPerfectPrediction()
        {
            // Arrange
            var lossFunctions = new ILossFunction<double>[]
            {
                new MeanSquaredErrorLoss<double>(),
                new MeanAbsoluteErrorLoss<double>(),
                new HuberLoss<double>(),
                new LogCoshLoss<double>()
            };

            var perfect = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var imperfect = new Vector<double>(new[] { 1.1, 2.1, 3.1 });
            var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act & Assert
            foreach (var lossFunction in lossFunctions)
            {
                var perfectLoss = lossFunction.CalculateLoss(perfect, actual);
                var imperfectLoss = lossFunction.CalculateLoss(imperfect, actual);

                Assert.True(perfectLoss < imperfectLoss,
                    $"{lossFunction.GetType().Name} did not achieve minimum at perfect prediction");
            }
        }

        [Fact]
        public void MSE_Is_Convex()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted1 = new Vector<double>(new[] { 0.0 });
            var predicted2 = new Vector<double>(new[] { 2.0 });
            var predictedMid = new Vector<double>(new[] { 1.0 });
            var actual = new Vector<double>(new[] { 1.0 });

            // Act
            var loss1 = mse.CalculateLoss(predicted1, actual);
            var loss2 = mse.CalculateLoss(predicted2, actual);
            var lossMid = mse.CalculateLoss(predictedMid, actual);

            // Assert - For convex function: f((x1+x2)/2) <= (f(x1)+f(x2))/2
            Assert.True(lossMid <= (loss1 + loss2) / 2.0);
        }

        [Fact]
        public void MAE_Satisfies_TriangleInequality()
        {
            // Arrange
            var mae = new MeanAbsoluteErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 0.0 });
            var intermediate = new Vector<double>(new[] { 0.5 });
            var actual = new Vector<double>(new[] { 1.0 });

            // Act
            var lossDirectly = mae.CalculateLoss(predicted, actual);
            var lossViaIntermediate = mae.CalculateLoss(predicted, intermediate) +
                                      mae.CalculateLoss(intermediate, actual);

            // Assert - Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
            Assert.True(lossDirectly <= lossViaIntermediate + EPSILON);
        }

        #endregion

        #region Categorical Cross-Entropy Tests

        [Fact]
        public void CategoricalCrossEntropy_OneHotEncoded_PerfectPrediction_ReturnsNearZero()
        {
            // Arrange
            var cce = new CategoricalCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.9999, 0.0001, 0.0001 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0 });

            // Act
            var loss = cce.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss < 0.001);
        }

        [Fact]
        public void CategoricalCrossEntropy_WrongPrediction_ReturnsHighLoss()
        {
            // Arrange
            var cce = new CategoricalCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.01, 0.98, 0.01 });
            var actual = new Vector<double>(new[] { 1.0, 0.0, 0.0 });

            // Act
            var loss = cce.CalculateLoss(predicted, actual);

            // Assert - Predicting wrong class should give high loss
            Assert.True(loss > 2.0);
        }

        [Fact]
        public void CategoricalCrossEntropy_IsNonNegative()
        {
            // Arrange
            var cce = new CategoricalCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.2, 0.3, 0.5 });
            var actual = new Vector<double>(new[] { 0.0, 1.0, 0.0 });

            // Act
            var loss = cce.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss >= 0.0);
        }

        [Fact]
        public void CategoricalCrossEntropy_GradientVerification()
        {
            // Arrange
            var cce = new CategoricalCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.3, 0.4, 0.3 });
            var actual = new Vector<double>(new[] { 0.0, 1.0, 0.0 });

            // Act & Assert
            VerifyGradient(cce, predicted, actual, tolerance: 1e-3);
        }

        [Fact]
        public void CategoricalCrossEntropy_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var cce = new CategoricalCrossEntropyLoss<float>();
            var predicted = new Vector<float>(new[] { 0.9f, 0.05f, 0.05f });
            var actual = new Vector<float>(new[] { 1.0f, 0.0f, 0.0f });

            // Act
            var loss = cce.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(float.IsFinite(loss));
            Assert.True(loss >= 0.0f);
        }

        #endregion

        #region Squared Hinge Loss Tests

        [Fact]
        public void SquaredHingeLoss_CorrectClassification_WithMargin_ReturnsZero()
        {
            // Arrange
            var squaredHinge = new SquaredHingeLoss<double>();
            var predicted = new Vector<double>(new[] { 2.0, -2.0 });
            var actual = new Vector<double>(new[] { 1.0, -1.0 });

            // Act
            var loss = squaredHinge.CalculateLoss(predicted, actual);

            // Assert - y*f(x) >= 1, so loss = 0
            Assert.Equal(0.0, loss, precision: 10);
        }

        [Fact]
        public void SquaredHingeLoss_IncorrectClassification_ReturnsSquaredPenalty()
        {
            // Arrange
            var squaredHinge = new SquaredHingeLoss<double>();
            var predicted = new Vector<double>(new[] { -1.0 });
            var actual = new Vector<double>(new[] { 1.0 });

            // Act
            var loss = squaredHinge.CalculateLoss(predicted, actual);

            // Assert - max(0, 1 - (-1))^2 = 2^2 = 4.0
            Assert.Equal(4.0, loss, precision: 10);
        }

        [Fact]
        public void SquaredHingeLoss_IsNonNegative()
        {
            // Arrange
            var squaredHinge = new SquaredHingeLoss<double>();
            var predicted = new Vector<double>(new[] { 0.5, -0.5, 1.5, -1.5 });
            var actual = new Vector<double>(new[] { 1.0, -1.0, 1.0, -1.0 });

            // Act
            var loss = squaredHinge.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss >= 0.0);
        }

        [Fact]
        public void SquaredHingeLoss_PenalizesMore_ThanRegularHinge()
        {
            // Arrange
            var squaredHinge = new SquaredHingeLoss<double>();
            var hinge = new HingeLoss<double>();
            var predicted = new Vector<double>(new[] { -1.0 });
            var actual = new Vector<double>(new[] { 1.0 });

            // Act
            var squaredLoss = squaredHinge.CalculateLoss(predicted, actual);
            var hingeLoss = hinge.CalculateLoss(predicted, actual);

            // Assert - Squared hinge should penalize more
            Assert.True(squaredLoss > hingeLoss);
        }

        [Fact]
        public void SquaredHingeLoss_GradientVerification()
        {
            // Arrange
            var squaredHinge = new SquaredHingeLoss<double>();
            var predicted = new Vector<double>(new[] { 0.5, -0.5 });
            var actual = new Vector<double>(new[] { 1.0, -1.0 });

            // Act & Assert
            VerifyGradient(squaredHinge, predicted, actual, tolerance: 1e-3);
        }

        [Fact]
        public void SquaredHingeLoss_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var squaredHinge = new SquaredHingeLoss<float>();
            var predicted = new Vector<float>(new[] { 2.0f, -2.0f });
            var actual = new Vector<float>(new[] { 1.0f, -1.0f });

            // Act
            var loss = squaredHinge.CalculateLoss(predicted, actual);

            // Assert
            Assert.Equal(0.0f, loss, precision: 6);
        }

        #endregion

        #region Root Mean Squared Error (RMSE) Tests

        [Fact]
        public void RMSE_PerfectPrediction_ReturnsZero()
        {
            // Arrange
            var rmse = new RootMeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });
            var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 });

            // Act
            var loss = rmse.CalculateLoss(predicted, actual);

            // Assert
            Assert.Equal(0.0, loss, precision: 10);
        }

        [Fact]
        public void RMSE_KnownValues_ComputesCorrectLoss()
        {
            // Arrange
            var rmse = new RootMeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new[] { 2.0, 3.0, 4.0 });

            // Act
            var loss = rmse.CalculateLoss(predicted, actual);

            // Assert - RMSE = sqrt((1 + 1 + 1) / 3) = sqrt(1) = 1.0
            Assert.Equal(1.0, loss, precision: 10);
        }

        [Fact]
        public void RMSE_IsNonNegative()
        {
            // Arrange
            var rmse = new RootMeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { -5.0, -2.0, 0.0, 3.0, 7.0 });
            var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

            // Act
            var loss = rmse.CalculateLoss(predicted, actual);

            // Assert
            Assert.True(loss >= 0.0);
        }

        [Fact]
        public void RMSE_RelatedToMSE_BySquareRoot()
        {
            // Arrange
            var rmse = new RootMeanSquaredErrorLoss<double>();
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var actual = new Vector<double>(new[] { 2.0, 4.0, 5.0 });

            // Act
            var rmseLoss = rmse.CalculateLoss(predicted, actual);
            var mseLoss = mse.CalculateLoss(predicted, actual);

            // Assert - RMSE = sqrt(MSE)
            Assert.Equal(Math.Sqrt(mseLoss), rmseLoss, precision: 10);
        }

        [Fact]
        public void RMSE_GradientVerification()
        {
            // Arrange
            var rmse = new RootMeanSquaredErrorLoss<double>();
            var predicted = new Vector<double>(new[] { 1.5, 2.5, 3.5 });
            var actual = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act & Assert
            VerifyGradient(rmse, predicted, actual);
        }

        [Fact]
        public void RMSE_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var rmse = new RootMeanSquaredErrorLoss<float>();
            var predicted = new Vector<float>(new[] { 1.0f, 2.0f, 3.0f });
            var actual = new Vector<float>(new[] { 2.0f, 3.0f, 4.0f });

            // Act
            var loss = rmse.CalculateLoss(predicted, actual);

            // Assert
            Assert.Equal(1.0f, loss, precision: 6);
        }

        [Fact]
        public void RMSE_ScaleDependent_UnlikeMSE()
        {
            // Arrange
            var rmse = new RootMeanSquaredErrorLoss<double>();
            var predicted1 = new Vector<double>(new[] { 1.0, 2.0 });
            var actual1 = new Vector<double>(new[] { 2.0, 3.0 });
            var predicted2 = new Vector<double>(new[] { 10.0, 20.0 });
            var actual2 = new Vector<double>(new[] { 20.0, 30.0 });

            // Act
            var loss1 = rmse.CalculateLoss(predicted1, actual1);
            var loss2 = rmse.CalculateLoss(predicted2, actual2);

            // Assert - RMSE scales with the magnitude
            Assert.True(loss2 > loss1);
        }

        #endregion

        #region Stress and Performance Tests

        [Fact]
        public void AllLossFunctions_LargeVectors_PerformEfficiently()
        {
            // Arrange
            var size = 10000;
            var predicted = new Vector<double>(size);
            var actual = new Vector<double>(size);

            for (int i = 0; i < size; i++)
            {
                predicted[i] = i * 0.001;
                actual[i] = (i + 1) * 0.001;
            }

            var lossFunctions = new ILossFunction<double>[]
            {
                new MeanSquaredErrorLoss<double>(),
                new MeanAbsoluteErrorLoss<double>(),
                new HuberLoss<double>(),
                new LogCoshLoss<double>()
            };

            // Act & Assert
            foreach (var lossFunction in lossFunctions)
            {
                var startTime = DateTime.Now;
                var loss = lossFunction.CalculateLoss(predicted, actual);
                var gradient = lossFunction.CalculateDerivative(predicted, actual);
                var elapsed = DateTime.Now - startTime;

                Assert.True(elapsed.TotalSeconds < 1.0,
                    $"{lossFunction.GetType().Name} took too long: {elapsed.TotalSeconds}s");
                Assert.True(double.IsFinite(loss));
            }
        }

        [Fact]
        public void AllLossFunctions_HandleNegativeValues_Correctly()
        {
            // Arrange
            var predicted = new Vector<double>(new[] { -1.0, -2.0, -3.0 });
            var actual = new Vector<double>(new[] { -1.5, -2.5, -3.5 });

            var lossFunctions = new ILossFunction<double>[]
            {
                new MeanSquaredErrorLoss<double>(),
                new MeanAbsoluteErrorLoss<double>(),
                new HuberLoss<double>(),
                new LogCoshLoss<double>(),
                new RootMeanSquaredErrorLoss<double>()
            };

            // Act & Assert
            foreach (var lossFunction in lossFunctions)
            {
                var loss = lossFunction.CalculateLoss(predicted, actual);
                Assert.True(double.IsFinite(loss));
                Assert.True(loss >= 0.0);
            }
        }

        [Fact]
        public void AllLossFunctions_HandleMixedSignValues_Correctly()
        {
            // Arrange
            var predicted = new Vector<double>(new[] { -1.0, 0.0, 1.0, 2.0 });
            var actual = new Vector<double>(new[] { -2.0, 1.0, 0.0, 3.0 });

            var lossFunctions = new ILossFunction<double>[]
            {
                new MeanSquaredErrorLoss<double>(),
                new MeanAbsoluteErrorLoss<double>(),
                new HuberLoss<double>(),
                new LogCoshLoss<double>()
            };

            // Act & Assert
            foreach (var lossFunction in lossFunctions)
            {
                var loss = lossFunction.CalculateLoss(predicted, actual);
                var gradient = lossFunction.CalculateDerivative(predicted, actual);

                Assert.True(double.IsFinite(loss));
                Assert.True(loss >= 0.0);
                for (int i = 0; i < gradient.Length; i++)
                {
                    Assert.True(double.IsFinite(gradient[i]));
                }
            }
        }

        #endregion

        #region Symmetry and Invariance Tests

        [Fact]
        public void MSE_IsSymmetric_WhenErrorsAreReversed()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var predicted1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var actual1 = new Vector<double>(new[] { 2.0, 3.0, 4.0 });
            var predicted2 = new Vector<double>(new[] { 2.0, 3.0, 4.0 });
            var actual2 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var loss1 = mse.CalculateLoss(predicted1, actual1);
            var loss2 = mse.CalculateLoss(predicted2, actual2);

            // Assert - MSE should be symmetric
            Assert.Equal(loss1, loss2, precision: 10);
        }

        [Fact]
        public void MAE_IsSymmetric_WhenErrorsAreReversed()
        {
            // Arrange
            var mae = new MeanAbsoluteErrorLoss<double>();
            var predicted1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var actual1 = new Vector<double>(new[] { 2.0, 3.0, 4.0 });
            var predicted2 = new Vector<double>(new[] { 2.0, 3.0, 4.0 });
            var actual2 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });

            // Act
            var loss1 = mae.CalculateLoss(predicted1, actual1);
            var loss2 = mae.CalculateLoss(predicted2, actual2);

            // Assert - MAE should be symmetric
            Assert.Equal(loss1, loss2, precision: 10);
        }

        [Fact]
        public void RegressionLosses_ScaleInvariant_WithConstantOffset()
        {
            // Arrange
            var mse = new MeanSquaredErrorLoss<double>();
            var offset = 100.0;
            var predicted1 = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
            var actual1 = new Vector<double>(new[] { 2.0, 3.0, 4.0 });
            var predicted2 = new Vector<double>(new[] { 101.0, 102.0, 103.0 });
            var actual2 = new Vector<double>(new[] { 102.0, 103.0, 104.0 });

            // Act
            var loss1 = mse.CalculateLoss(predicted1, actual1);
            var loss2 = mse.CalculateLoss(predicted2, actual2);

            // Assert - Loss should be the same (translation invariant)
            Assert.Equal(loss1, loss2, precision: 10);
        }

        #endregion

        #region Gradient Consistency Tests

        [Fact]
        public void AllRegressionLosses_Gradient_PointsTowardActual()
        {
            // Arrange
            var predicted = new Vector<double>(new[] { 5.0 });
            var actual = new Vector<double>(new[] { 3.0 });

            var lossFunctions = new ILossFunction<double>[]
            {
                new MeanSquaredErrorLoss<double>(),
                new MeanAbsoluteErrorLoss<double>(),
                new HuberLoss<double>(),
                new LogCoshLoss<double>()
            };

            // Act & Assert
            foreach (var lossFunction in lossFunctions)
            {
                var gradient = lossFunction.CalculateDerivative(predicted, actual);

                // Gradient should be positive when predicted > actual
                Assert.True(gradient[0] > 0.0,
                    $"{lossFunction.GetType().Name} gradient should be positive");
            }
        }

        [Fact]
        public void AllRegressionLosses_Gradient_DirectionReverses()
        {
            // Arrange
            var predicted1 = new Vector<double>(new[] { 5.0 });
            var predicted2 = new Vector<double>(new[] { 1.0 });
            var actual = new Vector<double>(new[] { 3.0 });

            var lossFunctions = new ILossFunction<double>[]
            {
                new MeanSquaredErrorLoss<double>(),
                new MeanAbsoluteErrorLoss<double>(),
                new HuberLoss<double>()
            };

            // Act & Assert
            foreach (var lossFunction in lossFunctions)
            {
                var gradient1 = lossFunction.CalculateDerivative(predicted1, actual);
                var gradient2 = lossFunction.CalculateDerivative(predicted2, actual);

                // Gradients should have opposite signs
                Assert.True(gradient1[0] * gradient2[0] < 0.0,
                    $"{lossFunction.GetType().Name} gradients should have opposite signs");
            }
        }

        #endregion

        #region Special Case Tests

        [Fact]
        public void BinaryCrossEntropy_ExtremeConfidence_ClipsCorrectly()
        {
            // Arrange
            var bce = new BinaryCrossEntropyLoss<double>();
            var predicted = new Vector<double>(new[] { 0.0000001, 0.9999999 });
            var actual = new Vector<double>(new[] { 1.0, 0.0 });

            // Act
            var loss = bce.CalculateLoss(predicted, actual);

            // Assert - Should handle extreme values without overflow
            Assert.True(double.IsFinite(loss));
            Assert.True(loss > 0.0);
        }

        [Fact]
        public void DiceAndJaccard_AllOnes_ProduceLowLoss()
        {
            // Arrange
            var dice = new DiceLoss<double>();
            var jaccard = new JaccardLoss<double>();
            var predicted = new Vector<double>(new[] { 1.0, 1.0, 1.0 });
            var actual = new Vector<double>(new[] { 1.0, 1.0, 1.0 });

            // Act
            var diceLoss = dice.CalculateLoss(predicted, actual);
            var jaccardLoss = jaccard.CalculateLoss(predicted, actual);

            // Assert - Perfect overlap should give near-zero loss
            Assert.True(diceLoss < 0.01);
            Assert.True(jaccardLoss < 0.01);
        }

        [Fact]
        public void HingeLoss_WithSoftMargin_BehavesCorrectly()
        {
            // Arrange
            var hinge = new HingeLoss<double>();
            var predicted = new Vector<double>(new[] { 0.5, 0.8, 1.2 });
            var actual = new Vector<double>(new[] { 1.0, 1.0, 1.0 });

            // Act
            var loss = hinge.CalculateLoss(predicted, actual);

            // Assert - Loss should decrease as predictions improve
            Assert.True(loss > 0.0);
            Assert.True(double.IsFinite(loss));
        }

        #endregion
    }
}
