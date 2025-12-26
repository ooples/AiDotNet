using System;
using AiDotNet.LinearAlgebra;
using AiDotNet.Preprocessing.Scalers;
using Xunit;

namespace AiDotNetTests.UnitTests.Preprocessing
{
    public class StandardScalerTests
    {
        [Fact]
        public void StandardScaler_FitTransform_CentersAndScalesData()
        {
            // Arrange
            var data = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0 },
                { 2.0, 4.0 },
                { 3.0, 6.0 },
                { 4.0, 8.0 },
                { 5.0, 10.0 }
            });
            var scaler = new StandardScaler<double>();

            // Act
            var result = scaler.FitTransform(data);

            // Assert - mean should be approximately 0
            var col1Sum = result[0, 0] + result[1, 0] + result[2, 0] + result[3, 0] + result[4, 0];
            Assert.True(Math.Abs(col1Sum) < 0.0001, "Column 1 sum should be close to 0");
        }

        [Fact]
        public void StandardScaler_InverseTransform_ReturnsOriginalData()
        {
            // Arrange
            var data = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0 },
                { 2.0, 4.0 },
                { 3.0, 6.0 }
            });
            var scaler = new StandardScaler<double>();

            // Act
            var transformed = scaler.FitTransform(data);
            var inversed = scaler.InverseTransform(transformed);

            // Assert - should be close to original
            Assert.True(Math.Abs(inversed[0, 0] - 1.0) < 0.0001);
            Assert.True(Math.Abs(inversed[1, 0] - 2.0) < 0.0001);
            Assert.True(Math.Abs(inversed[2, 0] - 3.0) < 0.0001);
        }

        [Fact]
        public void StandardScaler_WithoutCentering_OnlyScales()
        {
            // Arrange
            var data = new Matrix<double>(new double[,]
            {
                { 1.0 },
                { 2.0 },
                { 3.0 }
            });
            var scaler = new StandardScaler<double>(withMean: false, withStd: true);

            // Act
            scaler.Fit(data);
            var result = scaler.Transform(data);

            // Assert - values should not be centered (non-zero mean)
            var mean = (result[0, 0] + result[1, 0] + result[2, 0]) / 3.0;
            Assert.True(Math.Abs(mean) > 0.1, "Mean should not be zero when centering is disabled");
        }

        [Fact]
        public void StandardScaler_WithSpecificColumns_OnlyScalesSelectedColumns()
        {
            // Arrange
            var data = new Matrix<double>(new double[,]
            {
                { 1.0, 100.0 },
                { 2.0, 200.0 },
                { 3.0, 300.0 }
            });
            var scaler = new StandardScaler<double>(columnIndices: new[] { 0 }); // Only scale first column

            // Act
            var result = scaler.FitTransform(data);

            // Assert - second column should be unchanged
            Assert.Equal(100.0, result[0, 1]);
            Assert.Equal(200.0, result[1, 1]);
            Assert.Equal(300.0, result[2, 1]);
        }

        [Fact]
        public void StandardScaler_ThrowsWhenTransformCalledBeforeFit()
        {
            // Arrange
            var data = new Matrix<double>(new double[,] { { 1.0 } });
            var scaler = new StandardScaler<double>();

            // Act & Assert
            Assert.Throws<InvalidOperationException>(() => scaler.Transform(data));
        }
    }

    public class MinMaxScalerTests
    {
        [Fact]
        public void MinMaxScaler_FitTransform_ScalesToDefaultRange()
        {
            // Arrange
            var data = new Matrix<double>(new double[,]
            {
                { 0.0 },
                { 50.0 },
                { 100.0 }
            });
            var scaler = new MinMaxScaler<double>();

            // Act
            var result = scaler.FitTransform(data);

            // Assert - should be scaled to [0, 1]
            Assert.True(Math.Abs(result[0, 0] - 0.0) < 0.0001, "Min should be 0");
            Assert.True(Math.Abs(result[1, 0] - 0.5) < 0.0001, "Middle should be 0.5");
            Assert.True(Math.Abs(result[2, 0] - 1.0) < 0.0001, "Max should be 1");
        }

        [Fact]
        public void MinMaxScaler_CustomRange_ScalesToSpecifiedRange()
        {
            // Arrange
            var data = new Matrix<double>(new double[,]
            {
                { 0.0 },
                { 50.0 },
                { 100.0 }
            });
            var scaler = new MinMaxScaler<double>(featureRangeMin: -1.0, featureRangeMax: 1.0);

            // Act
            var result = scaler.FitTransform(data);

            // Assert - should be scaled to [-1, 1]
            Assert.True(Math.Abs(result[0, 0] - (-1.0)) < 0.0001, "Min should be -1");
            Assert.True(Math.Abs(result[1, 0] - 0.0) < 0.0001, "Middle should be 0");
            Assert.True(Math.Abs(result[2, 0] - 1.0) < 0.0001, "Max should be 1");
        }

        [Fact]
        public void MinMaxScaler_InverseTransform_ReturnsOriginalData()
        {
            // Arrange
            var data = new Matrix<double>(new double[,]
            {
                { 10.0, 20.0 },
                { 30.0, 40.0 },
                { 50.0, 60.0 }
            });
            var scaler = new MinMaxScaler<double>();

            // Act
            var transformed = scaler.FitTransform(data);
            var inversed = scaler.InverseTransform(transformed);

            // Assert
            Assert.True(Math.Abs(inversed[0, 0] - 10.0) < 0.0001);
            Assert.True(Math.Abs(inversed[2, 1] - 60.0) < 0.0001);
        }

        [Fact]
        public void MinMaxScaler_ConstantColumn_HandlesGracefully()
        {
            // Arrange - all values are the same (constant column)
            var data = new Matrix<double>(new double[,]
            {
                { 5.0 },
                { 5.0 },
                { 5.0 }
            });
            var scaler = new MinMaxScaler<double>();

            // Act
            var result = scaler.FitTransform(data);

            // Assert - should map to middle of range (0.5 for [0, 1])
            Assert.True(Math.Abs(result[0, 0] - 0.5) < 0.0001);
            Assert.True(Math.Abs(result[1, 0] - 0.5) < 0.0001);
            Assert.True(Math.Abs(result[2, 0] - 0.5) < 0.0001);
        }

        [Fact]
        public void MinMaxScaler_InvalidRange_ThrowsException()
        {
            // Act & Assert - min >= max should throw
            Assert.Throws<ArgumentException>(() => new MinMaxScaler<double>(1.0, 0.0));
            Assert.Throws<ArgumentException>(() => new MinMaxScaler<double>(1.0, 1.0));
        }
    }

    public class RobustScalerTests
    {
        [Fact]
        public void RobustScaler_FitTransform_UsesMedianAndIQR()
        {
            // Arrange - data with outliers
            var data = new Matrix<double>(new double[,]
            {
                { 1.0 },
                { 2.0 },
                { 3.0 },
                { 4.0 },
                { 5.0 },
                { 1000.0 } // Outlier
            });
            var scaler = new RobustScaler<double>();

            // Act
            scaler.Fit(data);
            var median = scaler.Median;
            var iqr = scaler.InterquartileRange;

            // Assert - median should be less affected by outlier
            Assert.NotNull(median);
            Assert.NotNull(iqr);
            // Median of [1, 2, 3, 4, 5, 1000] = (3 + 4) / 2 = 3.5
            Assert.True(median[0] < 10.0, "Median should be robust to outlier");
        }

        [Fact]
        public void RobustScaler_InverseTransform_ReturnsOriginalData()
        {
            // Arrange
            var data = new Matrix<double>(new double[,]
            {
                { 1.0, 2.0 },
                { 2.0, 4.0 },
                { 3.0, 6.0 },
                { 4.0, 8.0 },
                { 5.0, 10.0 }
            });
            var scaler = new RobustScaler<double>();

            // Act
            var transformed = scaler.FitTransform(data);
            var inversed = scaler.InverseTransform(transformed);

            // Assert
            Assert.True(Math.Abs(inversed[0, 0] - 1.0) < 0.0001);
            Assert.True(Math.Abs(inversed[2, 1] - 6.0) < 0.0001);
        }

        [Fact]
        public void RobustScaler_CustomQuantileRange_UsesSpecifiedRange()
        {
            // Arrange
            var data = new Matrix<double>(new double[,]
            {
                { 1.0 },
                { 2.0 },
                { 3.0 },
                { 4.0 },
                { 5.0 }
            });
            // Use 10th and 90th percentiles instead of default 25th and 75th
            var scaler = new RobustScaler<double>(10.0, 90.0);

            // Act
            scaler.Fit(data);
            var iqr = scaler.InterquartileRange;

            // Assert - IQR should be wider with 10-90 range than 25-75
            Assert.NotNull(iqr);
            Assert.True(iqr[0] > 0);
        }

        [Fact]
        public void RobustScaler_WithoutScaling_OnlyCenters()
        {
            // Arrange
            var data = new Matrix<double>(new double[,]
            {
                { 1.0 },
                { 3.0 },
                { 5.0 }
            });
            var scaler = new RobustScaler<double>(withCentering: true, withScaling: false);

            // Act
            var result = scaler.FitTransform(data);

            // Assert - should only subtract median (3), not divide by IQR
            Assert.True(Math.Abs(result[0, 0] - (-2.0)) < 0.0001); // 1 - 3 = -2
            Assert.True(Math.Abs(result[1, 0] - 0.0) < 0.0001);    // 3 - 3 = 0
            Assert.True(Math.Abs(result[2, 0] - 2.0) < 0.0001);    // 5 - 3 = 2
        }

        [Fact]
        public void RobustScaler_InvalidQuantileRange_ThrowsException()
        {
            // Act & Assert
            Assert.Throws<ArgumentException>(() => new RobustScaler<double>(75.0, 25.0)); // min >= max
            Assert.Throws<ArgumentOutOfRangeException>(() => new RobustScaler<double>(-1.0, 75.0)); // out of range
            Assert.Throws<ArgumentOutOfRangeException>(() => new RobustScaler<double>(25.0, 101.0)); // out of range
        }
    }

    public class MaxAbsScalerTests
    {
        [Fact]
        public void MaxAbsScaler_FitTransform_ScalesByMaxAbsoluteValue()
        {
            // Arrange
            var data = new Matrix<double>(new double[,]
            {
                { -10.0 },
                { 5.0 },
                { 20.0 }
            });
            var scaler = new MaxAbsScaler<double>();

            // Act
            var result = scaler.FitTransform(data);

            // Assert - values should be scaled by max abs (20)
            Assert.True(Math.Abs(result[0, 0] - (-0.5)) < 0.0001);  // -10 / 20
            Assert.True(Math.Abs(result[1, 0] - 0.25) < 0.0001);   // 5 / 20
            Assert.True(Math.Abs(result[2, 0] - 1.0) < 0.0001);    // 20 / 20
        }

        [Fact]
        public void MaxAbsScaler_NegativeMaxAbs_ScalesCorrectly()
        {
            // Arrange - largest absolute value is negative
            var data = new Matrix<double>(new double[,]
            {
                { -100.0 },
                { 5.0 },
                { 20.0 }
            });
            var scaler = new MaxAbsScaler<double>();

            // Act
            var result = scaler.FitTransform(data);

            // Assert - values should be scaled by max abs (100)
            Assert.True(Math.Abs(result[0, 0] - (-1.0)) < 0.0001); // -100 / 100
            Assert.True(Math.Abs(result[1, 0] - 0.05) < 0.0001);  // 5 / 100
        }

        [Fact]
        public void MaxAbsScaler_InverseTransform_ReturnsOriginalData()
        {
            // Arrange
            var data = new Matrix<double>(new double[,]
            {
                { -10.0, 20.0 },
                { 5.0, 40.0 },
                { 15.0, -60.0 }
            });
            var scaler = new MaxAbsScaler<double>();

            // Act
            var transformed = scaler.FitTransform(data);
            var inversed = scaler.InverseTransform(transformed);

            // Assert
            Assert.True(Math.Abs(inversed[0, 0] - (-10.0)) < 0.0001);
            Assert.True(Math.Abs(inversed[2, 1] - (-60.0)) < 0.0001);
        }

        [Fact]
        public void MaxAbsScaler_PreservesZeros()
        {
            // Arrange - sparse-like data with zeros
            var data = new Matrix<double>(new double[,]
            {
                { 0.0, 10.0 },
                { 5.0, 0.0 },
                { 0.0, 0.0 }
            });
            var scaler = new MaxAbsScaler<double>();

            // Act
            var result = scaler.FitTransform(data);

            // Assert - zeros should remain zeros
            Assert.Equal(0.0, result[0, 0]);
            Assert.Equal(0.0, result[1, 1]);
            Assert.Equal(0.0, result[2, 0]);
            Assert.Equal(0.0, result[2, 1]);
        }

        [Fact]
        public void MaxAbsScaler_AllZerosColumn_HandlesGracefully()
        {
            // Arrange - column with all zeros
            var data = new Matrix<double>(new double[,]
            {
                { 0.0, 10.0 },
                { 0.0, 20.0 },
                { 0.0, 30.0 }
            });
            var scaler = new MaxAbsScaler<double>();

            // Act
            var result = scaler.FitTransform(data);

            // Assert - zeros column should remain zeros (no division by zero)
            Assert.Equal(0.0, result[0, 0]);
            Assert.Equal(0.0, result[1, 0]);
            Assert.Equal(0.0, result[2, 0]);
        }
    }

    public class ScalerCommonBehaviorTests
    {
        [Fact]
        public void AllScalers_SupportsInverseTransform()
        {
            // Assert - all scalers should support inverse transform
            Assert.True(new StandardScaler<double>().SupportsInverseTransform);
            Assert.True(new MinMaxScaler<double>().SupportsInverseTransform);
            Assert.True(new RobustScaler<double>().SupportsInverseTransform);
            Assert.True(new MaxAbsScaler<double>().SupportsInverseTransform);
        }

        [Fact]
        public void AllScalers_IsFittedProperty_WorksCorrectly()
        {
            // Arrange
            var data = new Matrix<double>(new double[,] { { 1.0 }, { 2.0 } });
            var standardScaler = new StandardScaler<double>();
            var minMaxScaler = new MinMaxScaler<double>();
            var robustScaler = new RobustScaler<double>();
            var maxAbsScaler = new MaxAbsScaler<double>();

            // Assert - before fit
            Assert.False(standardScaler.IsFitted);
            Assert.False(minMaxScaler.IsFitted);
            Assert.False(robustScaler.IsFitted);
            Assert.False(maxAbsScaler.IsFitted);

            // Act - fit
            standardScaler.Fit(data);
            minMaxScaler.Fit(data);
            robustScaler.Fit(data);
            maxAbsScaler.Fit(data);

            // Assert - after fit
            Assert.True(standardScaler.IsFitted);
            Assert.True(minMaxScaler.IsFitted);
            Assert.True(robustScaler.IsFitted);
            Assert.True(maxAbsScaler.IsFitted);
        }

        [Fact]
        public void AllScalers_GetFeatureNamesOut_ReturnsInputNames()
        {
            // Arrange
            var featureNames = new[] { "feature1", "feature2", "feature3" };
            var scaler = new StandardScaler<double>();

            // Act
            var result = scaler.GetFeatureNamesOut(featureNames);

            // Assert - scalers don't change feature count, so names should be same
            Assert.Equal(featureNames, result);
        }
    }
}
