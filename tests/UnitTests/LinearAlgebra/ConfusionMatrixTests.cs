using System;
using AiDotNet.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.LinearAlgebra
{
    public class ConfusionMatrixTests
    {
        [Fact]
        public void Constructor_WithDimension_InitializesCorrectly()
        {
            // Arrange & Act
            var cm = new ConfusionMatrix<double>(3);

            // Assert
            Assert.Equal(3, cm.Dimension);
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.Equal(0.0, cm[i, j]);
                }
            }
        }

        [Fact]
        public void Constructor_WithZeroDimension_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new ConfusionMatrix<double>(0));
        }

        [Fact]
        public void Constructor_WithNegativeDimension_ThrowsArgumentException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new ConfusionMatrix<double>(-1));
        }

        [Fact]
        public void Indexer_GetAndSet_WorksCorrectly()
        {
            // Arrange
            var cm = new ConfusionMatrix<double>(3);

            // Act
            cm[0, 0] = 10.0;
            cm[1, 2] = 5.0;
            cm[2, 1] = 3.0;

            // Assert
            Assert.Equal(10.0, cm[0, 0]);
            Assert.Equal(5.0, cm[1, 2]);
            Assert.Equal(3.0, cm[2, 1]);
        }

        [Fact]
        public void Increment_IncreasesValueByOne()
        {
            // Arrange
            var cm = new ConfusionMatrix<int>(2);
            cm[0, 0] = 5;

            // Act
            cm.Increment(0, 0);

            // Assert
            Assert.Equal(6, cm[0, 0]);
        }

        [Fact]
        public void GetTruePositives_BinaryClassification_ReturnsCorrectValue()
        {
            // Arrange
            var cm = new ConfusionMatrix<double>(2);
            cm[0, 0] = 50.0; // True Negative
            cm[0, 1] = 10.0; // False Positive
            cm[1, 0] = 5.0;  // False Negative
            cm[1, 1] = 35.0; // True Positive

            // Act
            var tp = cm.GetTruePositives(1);

            // Assert
            Assert.Equal(35.0, tp);
        }

        [Fact]
        public void GetTrueNegatives_BinaryClassification_ReturnsCorrectValue()
        {
            // Arrange
            var cm = new ConfusionMatrix<double>(2);
            cm[0, 0] = 50.0; // True Negative
            cm[0, 1] = 10.0; // False Positive
            cm[1, 0] = 5.0;  // False Negative
            cm[1, 1] = 35.0; // True Positive

            // Act
            var tn = cm.GetTrueNegatives(1);

            // Assert
            Assert.Equal(50.0, tn);
        }

        [Fact]
        public void GetFalsePositives_BinaryClassification_ReturnsCorrectValue()
        {
            // Arrange
            var cm = new ConfusionMatrix<double>(2);
            cm[0, 0] = 50.0; // True Negative
            cm[0, 1] = 10.0; // False Positive
            cm[1, 0] = 5.0;  // False Negative
            cm[1, 1] = 35.0; // True Positive

            // Act
            var fp = cm.GetFalsePositives(1);

            // Assert
            Assert.Equal(10.0, fp);
        }

        [Fact]
        public void GetFalseNegatives_BinaryClassification_ReturnsCorrectValue()
        {
            // Arrange
            var cm = new ConfusionMatrix<double>(2);
            cm[0, 0] = 50.0; // True Negative
            cm[0, 1] = 10.0; // False Positive
            cm[1, 0] = 5.0;  // False Negative
            cm[1, 1] = 35.0; // True Positive

            // Act
            var fn = cm.GetFalseNegatives(1);

            // Assert
            Assert.Equal(5.0, fn);
        }

        [Fact]
        public void GetAccuracy_BinaryClassification_ReturnsCorrectValue()
        {
            // Arrange
            var cm = new ConfusionMatrix<double>(2);
            cm[0, 0] = 50.0; // True Negative
            cm[0, 1] = 10.0; // False Positive
            cm[1, 0] = 5.0;  // False Negative
            cm[1, 1] = 35.0; // True Positive

            // Act
            var accuracy = cm.GetAccuracy();

            // Assert
            // Accuracy = (TP + TN) / (TP + TN + FP + FN) = (35 + 50) / (35 + 50 + 10 + 5) = 85 / 100 = 0.85
            Assert.Equal(0.85, accuracy, 5);
        }

        [Fact]
        public void GetPrecision_BinaryClassification_ReturnsCorrectValue()
        {
            // Arrange
            var cm = new ConfusionMatrix<double>(2);
            cm[0, 0] = 50.0; // True Negative
            cm[0, 1] = 10.0; // False Positive
            cm[1, 0] = 5.0;  // False Negative
            cm[1, 1] = 35.0; // True Positive

            // Act
            var precision = cm.GetPrecision(1);

            // Assert
            // Precision = TP / (TP + FP) = 35 / (35 + 10) = 35 / 45 = 0.7777...
            Assert.Equal(0.7777, precision, 4);
        }

        [Fact]
        public void GetRecall_BinaryClassification_ReturnsCorrectValue()
        {
            // Arrange
            var cm = new ConfusionMatrix<double>(2);
            cm[0, 0] = 50.0; // True Negative
            cm[0, 1] = 10.0; // False Positive
            cm[1, 0] = 5.0;  // False Negative
            cm[1, 1] = 35.0; // True Positive

            // Act
            var recall = cm.GetRecall(1);

            // Assert
            // Recall = TP / (TP + FN) = 35 / (35 + 5) = 35 / 40 = 0.875
            Assert.Equal(0.875, recall, 5);
        }

        [Fact]
        public void GetF1Score_BinaryClassification_ReturnsCorrectValue()
        {
            // Arrange
            var cm = new ConfusionMatrix<double>(2);
            cm[0, 0] = 50.0; // True Negative
            cm[0, 1] = 10.0; // False Positive
            cm[1, 0] = 5.0;  // False Negative
            cm[1, 1] = 35.0; // True Positive

            // Act
            var f1 = cm.GetF1Score(1);

            // Assert
            // Precision = 35/45 = 0.7777
            // Recall = 35/40 = 0.875
            // F1 = 2 * (P * R) / (P + R) = 2 * (0.7777 * 0.875) / (0.7777 + 0.875) = 0.8235
            Assert.Equal(0.8235, f1, 4);
        }

        [Fact]
        public void GetSpecificity_BinaryClassification_ReturnsCorrectValue()
        {
            // Arrange
            var cm = new ConfusionMatrix<double>(2);
            cm[0, 0] = 50.0; // True Negative
            cm[0, 1] = 10.0; // False Positive
            cm[1, 0] = 5.0;  // False Negative
            cm[1, 1] = 35.0; // True Positive

            // Act
            var specificity = cm.GetSpecificity(1);

            // Assert
            // Specificity = TN / (TN + FP) = 50 / (50 + 10) = 50 / 60 = 0.8333
            Assert.Equal(0.8333, specificity, 4);
        }

        [Fact]
        public void GetTotal_ReturnsSumOfAllElements()
        {
            // Arrange
            var cm = new ConfusionMatrix<double>(2);
            cm[0, 0] = 50.0;
            cm[0, 1] = 10.0;
            cm[1, 0] = 5.0;
            cm[1, 1] = 35.0;

            // Act
            var total = cm.GetTotal();

            // Assert
            Assert.Equal(100.0, total);
        }

        [Fact]
        public void Clear_ResetsAllValues()
        {
            // Arrange
            var cm = new ConfusionMatrix<double>(3);
            cm[0, 0] = 10.0;
            cm[1, 1] = 20.0;
            cm[2, 2] = 30.0;

            // Act
            cm.Clear();

            // Assert
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    Assert.Equal(0.0, cm[i, j]);
                }
            }
        }

        [Fact]
        public void MulticlassConfusionMatrix_CalculatesCorrectAccuracy()
        {
            // Arrange
            var cm = new ConfusionMatrix<double>(3);
            cm[0, 0] = 20.0;
            cm[0, 1] = 5.0;
            cm[0, 2] = 3.0;
            cm[1, 0] = 2.0;
            cm[1, 1] = 25.0;
            cm[1, 2] = 1.0;
            cm[2, 0] = 1.0;
            cm[2, 1] = 2.0;
            cm[2, 2] = 18.0;

            // Act
            var accuracy = cm.GetAccuracy();

            // Assert
            // Accuracy = (20 + 25 + 18) / 77 = 63 / 77 = 0.8181
            Assert.Equal(0.8181, accuracy, 4);
        }

        [Fact]
        public void GetRowSum_ReturnsCorrectSum()
        {
            // Arrange
            var cm = new ConfusionMatrix<double>(3);
            cm[1, 0] = 2.0;
            cm[1, 1] = 25.0;
            cm[1, 2] = 1.0;

            // Act
            var rowSum = cm.GetRowSum(1);

            // Assert
            Assert.Equal(28.0, rowSum);
        }

        [Fact]
        public void GetColumnSum_ReturnsCorrectSum()
        {
            // Arrange
            var cm = new ConfusionMatrix<double>(3);
            cm[0, 1] = 5.0;
            cm[1, 1] = 25.0;
            cm[2, 1] = 2.0;

            // Act
            var colSum = cm.GetColumnSum(1);

            // Assert
            Assert.Equal(32.0, colSum);
        }

        [Fact]
        public void IntConfusionMatrix_WorksCorrectly()
        {
            // Arrange & Act
            var cm = new ConfusionMatrix<int>(2);
            cm[0, 0] = 50;
            cm[0, 1] = 10;
            cm[1, 0] = 5;
            cm[1, 1] = 35;

            // Assert
            Assert.Equal(50, cm[0, 0]);
            Assert.Equal(35, cm[1, 1]);
            Assert.Equal(100, cm.GetTotal());
        }
    }
}
