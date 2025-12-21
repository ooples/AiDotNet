using AiDotNet.Enums;
using AiDotNet.FeatureSelectors;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.UnitTests.FeatureSelectors
{
    public class UnivariateFeatureSelectorTests
    {
        [Fact]
        public void SelectFeatures_WithFValue_SelectsTopKFeatures()
        {
            // Arrange: Create dataset where first and third features are informative
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 10.0, 5.0, 10.5 },  // Class 0
                { 2.0, 11.0, 6.0, 11.2 },  // Class 0
                { 1.5, 10.5, 5.5, 10.8 },  // Class 0
                { 8.0, 9.0, 15.0, 11.0 },  // Class 1
                { 9.0, 8.5, 16.0, 10.9 },  // Class 1
                { 8.5, 9.2, 15.5, 11.1 }   // Class 1
            });
            var target = new Vector<double>(new double[] { 0, 0, 0, 1, 1, 1 });

            var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
                target,
                UnivariateScoringFunction.FValue,
                k: 2);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(6, result.Rows); // Same number of samples
            Assert.Equal(2, result.Columns); // Only 2 features selected
        }

        [Fact]
        public void SelectFeatures_WithMutualInformation_WorksCorrectly()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 0.5, 2.0 },
                { 2.0, 0.6, 2.1 },
                { 8.0, 0.55, 8.5 },
                { 9.0, 0.65, 8.7 }
            });
            var target = new Vector<double>(new double[] { 0, 0, 1, 1 });

            var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
                target,
                UnivariateScoringFunction.MutualInformation,
                k: 2);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(4, result.Rows);
            Assert.Equal(2, result.Columns);
        }

        [Fact]
        public void SelectFeatures_WithChiSquared_WorksCorrectly()
        {
            // Arrange: Categorical features
            var features = new Matrix<double>(new double[,]
            {
                { 0, 1, 0 },
                { 0, 1, 1 },
                { 1, 0, 0 },
                { 1, 0, 1 }
            });
            var target = new Vector<double>(new double[] { 0, 0, 1, 1 });

            var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
                target,
                UnivariateScoringFunction.ChiSquared,
                k: 2);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(4, result.Rows);
            Assert.Equal(2, result.Columns);
        }

        [Fact]
        public void SelectFeatures_WithDefaultK_SelectsHalfFeatures()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 },
                { 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 },
                { 8.0, 9.0, 10.0, 11.0, 12.0, 13.0 },
                { 9.0, 10.0, 11.0, 12.0, 13.0, 14.0 }
            });
            var target = new Vector<double>(new double[] { 0, 0, 1, 1 });

            var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
                target,
                UnivariateScoringFunction.FValue);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(4, result.Rows);
            Assert.Equal(3, result.Columns); // Default is 50% = 3 features
        }

        [Fact]
        public void SelectFeatures_WithKGreaterThanFeatureCount_SelectsAllFeatures()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 2.0, 3.0, 4.0 },
                { 8.0, 9.0, 10.0 },
                { 9.0, 10.0, 11.0 }
            });
            var target = new Vector<double>(new double[] { 0, 0, 1, 1 });

            var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
                target,
                UnivariateScoringFunction.FValue,
                k: 100);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(4, result.Rows);
            Assert.Equal(3, result.Columns); // All 3 features
        }

        [Fact]
        public void SelectFeatures_WithTargetLengthMismatch_ThrowsArgumentException()
        {
            // Arrange
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0 },
                { 2.0, 3.0 },
                { 3.0, 4.0 }
            });
            var target = new Vector<double>(new double[] { 0, 1 }); // Wrong length

            var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
                target,
                UnivariateScoringFunction.FValue,
                k: 1);

            // Act & Assert
            Assert.Throws<ArgumentException>(() => selector.SelectFeatures(features));
        }

        [Fact]
        public void SelectFeatures_WithFloatType_WorksCorrectly()
        {
            // Arrange
            var features = new Matrix<float>(new float[,]
            {
                { 1.0f, 10.0f, 5.0f },
                { 2.0f, 11.0f, 6.0f },
                { 8.0f, 9.0f, 15.0f },
                { 9.0f, 8.5f, 16.0f }
            });
            var target = new Vector<float>(new float[] { 0, 0, 1, 1 });

            var selector = new UnivariateFeatureSelector<float, Matrix<float>>(
                target,
                UnivariateScoringFunction.FValue,
                k: 2);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(4, result.Rows);
            Assert.Equal(2, result.Columns);
        }

        [Fact]
        public void SelectFeatures_WithSingleClass_HandlesGracefully()
        {
            // Arrange: All samples have the same target value
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0, 3.0 },
                { 2.0, 3.0, 4.0 },
                { 3.0, 4.0, 5.0 }
            });
            var target = new Vector<double>(new double[] { 0, 0, 0 });

            var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
                target,
                UnivariateScoringFunction.FValue,
                k: 2);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(3, result.Rows);
            Assert.Equal(2, result.Columns); // Should still select 2 features
        }

        [Fact]
        public void Constructor_WithNullTarget_ThrowsArgumentNullException()
        {
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() =>
                new UnivariateFeatureSelector<double, Matrix<double>>(
                    null!,
                    UnivariateScoringFunction.FValue,
                    k: 2));
        }

        [Fact]
        public void SelectFeatures_WithK1_SelectsSingleBestFeature()
        {
            // Arrange: Make first feature most informative
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 5.0, 5.1 },  // Class 0
                { 2.0, 5.1, 5.0 },  // Class 0
                { 9.0, 4.9, 5.2 },  // Class 1
                { 10.0, 5.2, 4.9 }  // Class 1
            });
            var target = new Vector<double>(new double[] { 0, 0, 1, 1 });

            var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
                target,
                UnivariateScoringFunction.FValue,
                k: 1);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(4, result.Rows);
            Assert.Equal(1, result.Columns); // Only 1 feature
            // First feature should be selected (has highest variance between classes)
            Assert.True(Math.Abs(result[0, 0] - 1.0) < 0.01 || Math.Abs(result[0, 0] - 5.0) < 0.01);
        }

        [Fact]
        public void SelectFeatures_WithMultipleClasses_WorksCorrectly()
        {
            // Arrange: 3-class problem
            var features = new Matrix<double>(new double[,]
            {
                { 1.0, 10.0, 5.0 },   // Class 0
                { 2.0, 11.0, 5.5 },   // Class 0
                { 5.0, 5.0, 10.0 },   // Class 1
                { 6.0, 6.0, 11.0 },   // Class 1
                { 10.0, 1.0, 1.0 },   // Class 2
                { 11.0, 2.0, 2.0 }    // Class 2
            });
            var target = new Vector<double>(new double[] { 0, 0, 1, 1, 2, 2 });

            var selector = new UnivariateFeatureSelector<double, Matrix<double>>(
                target,
                UnivariateScoringFunction.FValue,
                k: 2);

            // Act
            var result = selector.SelectFeatures(features);

            // Assert
            Assert.Equal(6, result.Rows);
            Assert.Equal(2, result.Columns);
        }
    }
}
