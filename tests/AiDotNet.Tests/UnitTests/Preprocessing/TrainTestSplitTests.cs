using AiDotNet.Preprocessing;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Preprocessing;

/// <summary>
/// Unit tests for TrainTestSplit utility class.
/// </summary>
public class TrainTestSplitTests
{
    private const double Tolerance = 1e-10;

    #region Split Tests (Two-way)

    [Fact]
    public void Split_DefaultParameters_Returns80_20Split()
    {
        // Arrange
        var X = CreateMatrix(100, 3);
        var y = CreateVector(100);

        // Act
        var (XTrain, XTest, yTrain, yTest) = TrainTestSplit<double>.Split(X, y);

        // Assert
        Assert.Equal(80, XTrain.Rows);
        Assert.Equal(20, XTest.Rows);
        Assert.Equal(80, yTrain.Length);
        Assert.Equal(20, yTest.Length);
    }

    [Fact]
    public void Split_CustomTestSize_CorrectSplit()
    {
        // Arrange
        var X = CreateMatrix(100, 2);
        var y = CreateVector(100);

        // Act
        var (XTrain, XTest, yTrain, yTest) = TrainTestSplit<double>.Split(X, y, testSize: 0.3);

        // Assert
        Assert.Equal(70, XTrain.Rows);
        Assert.Equal(30, XTest.Rows);
        Assert.Equal(70, yTrain.Length);
        Assert.Equal(30, yTest.Length);
    }

    [Fact]
    public void Split_PreservesFeatureCount()
    {
        // Arrange
        var X = CreateMatrix(50, 5);
        var y = CreateVector(50);

        // Act
        var (XTrain, XTest, _, _) = TrainTestSplit<double>.Split(X, y);

        // Assert
        Assert.Equal(5, XTrain.Columns);
        Assert.Equal(5, XTest.Columns);
    }

    [Fact]
    public void Split_NoDataLoss_AllSamplesInOutput()
    {
        // Arrange
        var X = CreateMatrix(10, 2);
        var y = CreateVector(10);

        // Act
        var (XTrain, XTest, yTrain, yTest) = TrainTestSplit<double>.Split(X, y, testSize: 0.2);

        // Assert
        Assert.Equal(10, XTrain.Rows + XTest.Rows);
        Assert.Equal(10, yTrain.Length + yTest.Length);
    }

    [Fact]
    public void Split_SameSeed_ProducesSameResults()
    {
        // Arrange
        var X = CreateMatrix(100, 3);
        var y = CreateVector(100);

        // Act
        var (XTrain1, XTest1, yTrain1, yTest1) = TrainTestSplit<double>.Split(X, y, randomSeed: 123);
        var (XTrain2, XTest2, yTrain2, yTest2) = TrainTestSplit<double>.Split(X, y, randomSeed: 123);

        // Assert
        for (int i = 0; i < XTrain1.Rows; i++)
        {
            for (int j = 0; j < XTrain1.Columns; j++)
            {
                Assert.Equal(XTrain1[i, j], XTrain2[i, j], Tolerance);
            }
        }
    }

    [Fact]
    public void Split_DifferentSeeds_ProducesDifferentResults()
    {
        // Arrange
        var X = CreateMatrix(100, 3);
        var y = CreateVector(100);

        // Act
        var (XTrain1, _, _, _) = TrainTestSplit<double>.Split(X, y, randomSeed: 1);
        var (XTrain2, _, _, _) = TrainTestSplit<double>.Split(X, y, randomSeed: 2);

        // Assert - At least some values should differ
        bool anyDifferent = false;
        for (int i = 0; i < XTrain1.Rows && !anyDifferent; i++)
        {
            for (int j = 0; j < XTrain1.Columns && !anyDifferent; j++)
            {
                if (Math.Abs(XTrain1[i, j] - XTrain2[i, j]) > Tolerance)
                {
                    anyDifferent = true;
                }
            }
        }
        Assert.True(anyDifferent);
    }

    [Fact]
    public void Split_NoShuffle_PreservesOrder()
    {
        // Arrange - Create data with sequential values
        var X = new Matrix<double>(10, 1);
        var y = new Vector<double>(10);
        for (int i = 0; i < 10; i++)
        {
            X[i, 0] = i;
            y[i] = i;
        }

        // Act
        var (XTrain, XTest, yTrain, yTest) = TrainTestSplit<double>.Split(X, y, testSize: 0.2, shuffle: false);

        // Assert - First 8 should be in train, last 2 in test
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(i, XTrain[i, 0], Tolerance);
        }
        Assert.Equal(8.0, XTest[0, 0], Tolerance);
        Assert.Equal(9.0, XTest[1, 0], Tolerance);
    }

    [Fact]
    public void Split_InvalidTestSize_ThrowsException()
    {
        // Arrange
        var X = CreateMatrix(10, 2);
        var y = CreateVector(10);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => TrainTestSplit<double>.Split(X, y, testSize: 0));
        Assert.Throws<ArgumentException>(() => TrainTestSplit<double>.Split(X, y, testSize: 1));
        Assert.Throws<ArgumentException>(() => TrainTestSplit<double>.Split(X, y, testSize: -0.1));
        Assert.Throws<ArgumentException>(() => TrainTestSplit<double>.Split(X, y, testSize: 1.5));
    }

    [Fact]
    public void Split_NullInputs_ThrowsException()
    {
        // Arrange
        var X = CreateMatrix(10, 2);
        var y = CreateVector(10);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => TrainTestSplit<double>.Split(null!, y));
        Assert.Throws<ArgumentNullException>(() => TrainTestSplit<double>.Split(X, null!));
    }

    [Fact]
    public void Split_MismatchedSizes_ThrowsException()
    {
        // Arrange
        var X = CreateMatrix(10, 2);
        var y = CreateVector(5); // Mismatched size

        // Act & Assert
        Assert.Throws<ArgumentException>(() => TrainTestSplit<double>.Split(X, y));
    }

    #endregion

    #region SplitThreeWay Tests

    [Fact]
    public void SplitThreeWay_DefaultParameters_CorrectSplit()
    {
        // Arrange
        var X = CreateMatrix(100, 3);
        var y = CreateVector(100);

        // Act
        var (XTrain, XVal, XTest, yTrain, yVal, yTest) =
            TrainTestSplit<double>.SplitThreeWay(X, y);

        // Assert - Default is 70% train, 15% validation, 15% test
        Assert.Equal(70, XTrain.Rows);
        Assert.Equal(15, XVal.Rows);
        Assert.Equal(15, XTest.Rows);
        Assert.Equal(70, yTrain.Length);
        Assert.Equal(15, yVal.Length);
        Assert.Equal(15, yTest.Length);
    }

    [Fact]
    public void SplitThreeWay_CustomSizes_CorrectSplit()
    {
        // Arrange
        var X = CreateMatrix(100, 2);
        var y = CreateVector(100);

        // Act
        var (XTrain, XVal, XTest, yTrain, yVal, yTest) =
            TrainTestSplit<double>.SplitThreeWay(X, y, trainSize: 0.6, validationSize: 0.2);

        // Assert - 60% train, 20% validation, 20% test
        Assert.Equal(60, XTrain.Rows);
        Assert.Equal(20, XVal.Rows);
        Assert.Equal(20, XTest.Rows);
    }

    [Fact]
    public void SplitThreeWay_NoDataLoss_AllSamplesInOutput()
    {
        // Arrange
        var X = CreateMatrix(50, 2);
        var y = CreateVector(50);

        // Act
        var (XTrain, XVal, XTest, _, _, _) =
            TrainTestSplit<double>.SplitThreeWay(X, y);

        // Assert
        Assert.Equal(50, XTrain.Rows + XVal.Rows + XTest.Rows);
    }

    [Fact]
    public void SplitThreeWay_InvalidSizes_ThrowsException()
    {
        // Arrange
        var X = CreateMatrix(10, 2);
        var y = CreateVector(10);

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            TrainTestSplit<double>.SplitThreeWay(X, y, trainSize: 0));
        Assert.Throws<ArgumentException>(() =>
            TrainTestSplit<double>.SplitThreeWay(X, y, trainSize: 1));
        Assert.Throws<ArgumentException>(() =>
            TrainTestSplit<double>.SplitThreeWay(X, y, validationSize: 0));
        Assert.Throws<ArgumentException>(() =>
            TrainTestSplit<double>.SplitThreeWay(X, y, trainSize: 0.7, validationSize: 0.4)); // Sum >= 1
    }

    [Fact]
    public void SplitThreeWay_SameSeed_ProducesSameResults()
    {
        // Arrange
        var X = CreateMatrix(100, 3);
        var y = CreateVector(100);

        // Act
        var result1 = TrainTestSplit<double>.SplitThreeWay(X, y, randomSeed: 456);
        var result2 = TrainTestSplit<double>.SplitThreeWay(X, y, randomSeed: 456);

        // Assert
        Assert.Equal(result1.XTrain[0, 0], result2.XTrain[0, 0], Tolerance);
    }

    #endregion

    #region SplitX Tests

    [Fact]
    public void SplitX_BasicSplit_Works()
    {
        // Arrange
        var X = CreateMatrix(100, 3);

        // Act
        var (XTrain, XTest) = TrainTestSplit<double>.SplitX(X, testSize: 0.2);

        // Assert
        Assert.Equal(80, XTrain.Rows);
        Assert.Equal(20, XTest.Rows);
        Assert.Equal(3, XTrain.Columns);
        Assert.Equal(3, XTest.Columns);
    }

    [Fact]
    public void SplitX_NullInput_ThrowsException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => TrainTestSplit<double>.SplitX(null!));
    }

    [Fact]
    public void SplitX_InvalidTestSize_ThrowsException()
    {
        // Arrange
        var X = CreateMatrix(10, 2);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => TrainTestSplit<double>.SplitX(X, testSize: 0));
        Assert.Throws<ArgumentException>(() => TrainTestSplit<double>.SplitX(X, testSize: 1));
    }

    #endregion

    #region KFoldSplit Tests

    [Fact]
    public void KFoldSplit_DefaultK_ReturnsFiveFolds()
    {
        // Arrange
        var X = CreateMatrix(100, 3);
        var y = CreateVector(100);

        // Act
        var folds = TrainTestSplit<double>.KFoldSplit(X, y);

        // Assert
        Assert.Equal(5, folds.Count);
    }

    [Fact]
    public void KFoldSplit_CustomK_ReturnsCorrectFolds()
    {
        // Arrange
        var X = CreateMatrix(100, 2);
        var y = CreateVector(100);

        // Act
        var folds = TrainTestSplit<double>.KFoldSplit(X, y, k: 10);

        // Assert
        Assert.Equal(10, folds.Count);
    }

    [Fact]
    public void KFoldSplit_EachFoldHasCorrectTestSize()
    {
        // Arrange
        var X = CreateMatrix(100, 2);
        var y = CreateVector(100);

        // Act
        var folds = TrainTestSplit<double>.KFoldSplit(X, y, k: 5);

        // Assert - Each fold's test set should be ~20% (100/5 = 20)
        foreach (var (_, XTest, _, yTest) in folds)
        {
            Assert.Equal(20, XTest.Rows);
            Assert.Equal(20, yTest.Length);
        }
    }

    [Fact]
    public void KFoldSplit_TrainingSetCorrectSize()
    {
        // Arrange
        var X = CreateMatrix(100, 2);
        var y = CreateVector(100);

        // Act
        var folds = TrainTestSplit<double>.KFoldSplit(X, y, k: 5);

        // Assert - Each fold's training set should be 80% (100 - 20 = 80)
        foreach (var (XTrain, _, yTrain, _) in folds)
        {
            Assert.Equal(80, XTrain.Rows);
            Assert.Equal(80, yTrain.Length);
        }
    }

    [Fact]
    public void KFoldSplit_InvalidK_ThrowsException()
    {
        // Arrange
        var X = CreateMatrix(10, 2);
        var y = CreateVector(10);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => TrainTestSplit<double>.KFoldSplit(X, y, k: 1));
        Assert.Throws<ArgumentException>(() => TrainTestSplit<double>.KFoldSplit(X, y, k: 0));
        Assert.Throws<ArgumentException>(() => TrainTestSplit<double>.KFoldSplit(X, y, k: -1));
    }

    [Fact]
    public void KFoldSplit_KGreaterThanSamples_ThrowsException()
    {
        // Arrange
        var X = CreateMatrix(5, 2);
        var y = CreateVector(5);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => TrainTestSplit<double>.KFoldSplit(X, y, k: 10));
    }

    [Fact]
    public void KFoldSplit_NullInputs_ThrowsException()
    {
        // Arrange
        var X = CreateMatrix(10, 2);
        var y = CreateVector(10);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => TrainTestSplit<double>.KFoldSplit(null!, y));
        Assert.Throws<ArgumentNullException>(() => TrainTestSplit<double>.KFoldSplit(X, null!));
    }

    [Fact]
    public void KFoldSplit_SameSeed_ProducesSameResults()
    {
        // Arrange
        var X = CreateMatrix(50, 2);
        var y = CreateVector(50);

        // Act
        var folds1 = TrainTestSplit<double>.KFoldSplit(X, y, k: 5, randomSeed: 789);
        var folds2 = TrainTestSplit<double>.KFoldSplit(X, y, k: 5, randomSeed: 789);

        // Assert
        for (int f = 0; f < 5; f++)
        {
            Assert.Equal(folds1[f].XTrain[0, 0], folds2[f].XTrain[0, 0], Tolerance);
        }
    }

    [Fact]
    public void KFoldSplit_HandlesUnevenDivision()
    {
        // Arrange - 23 samples with k=5 doesn't divide evenly
        var X = CreateMatrix(23, 2);
        var y = CreateVector(23);

        // Act
        var folds = TrainTestSplit<double>.KFoldSplit(X, y, k: 5);

        // Assert - Last fold should get the remaining samples
        int totalTestSamples = folds.Sum(f => f.XTest.Rows);
        Assert.Equal(23, totalTestSamples);
    }

    #endregion

    #region Helper Methods

    private static Matrix<double> CreateMatrix(int rows, int cols)
    {
        var matrix = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                matrix[i, j] = i * cols + j;
            }
        }
        return matrix;
    }

    private static Vector<double> CreateVector(int length)
    {
        var vector = new Vector<double>(length);
        for (int i = 0; i < length; i++)
        {
            vector[i] = i;
        }
        return vector;
    }

    #endregion
}
