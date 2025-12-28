using Xunit;
using AiDotNet.Helpers;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for OutlierRemovalHelper to verify data type conversion operations.
/// </summary>
public class OutlierRemovalHelperIntegrationTests
{
    #region ConvertToMatrixVector Tests - Matrix/Vector Input

    [Fact]
    public void ConvertToMatrixVector_MatrixVectorInput_ReturnsSameObjects()
    {
        var inputMatrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });
        var outputVector = new Vector<double>(new[] { 10.0, 20.0, 30.0 });

        var (resultMatrix, resultVector) = OutlierRemovalHelper<double, Matrix<double>, Vector<double>>
            .ConvertToMatrixVector(inputMatrix, outputVector);

        Assert.Same(inputMatrix, resultMatrix);
        Assert.Same(outputVector, resultVector);
    }

    [Fact]
    public void ConvertToMatrixVector_MatrixVectorInput_PreservesValues()
    {
        var inputMatrix = new Matrix<double>(new double[,]
        {
            { 1.5, 2.5 },
            { 3.5, 4.5 }
        });
        var outputVector = new Vector<double>(new[] { 100.0, 200.0 });

        var (resultMatrix, resultVector) = OutlierRemovalHelper<double, Matrix<double>, Vector<double>>
            .ConvertToMatrixVector(inputMatrix, outputVector);

        Assert.Equal(1.5, resultMatrix[0, 0]);
        Assert.Equal(2.5, resultMatrix[0, 1]);
        Assert.Equal(3.5, resultMatrix[1, 0]);
        Assert.Equal(4.5, resultMatrix[1, 1]);
        Assert.Equal(100.0, resultVector[0]);
        Assert.Equal(200.0, resultVector[1]);
    }

    [Fact]
    public void ConvertToMatrixVector_MatrixVectorInput_PreservesShape()
    {
        var inputMatrix = new Matrix<double>(5, 3);
        var outputVector = new Vector<double>(5);

        var (resultMatrix, resultVector) = OutlierRemovalHelper<double, Matrix<double>, Vector<double>>
            .ConvertToMatrixVector(inputMatrix, outputVector);

        Assert.Equal(5, resultMatrix.Rows);
        Assert.Equal(3, resultMatrix.Columns);
        Assert.Equal(5, resultVector.Length);
    }

    [Fact]
    public void ConvertToMatrixVector_MatrixVectorInput_Float_WorksCorrectly()
    {
        var inputMatrix = new Matrix<float>(new float[,]
        {
            { 1f, 2f },
            { 3f, 4f }
        });
        var outputVector = new Vector<float>(new[] { 5f, 6f });

        var (resultMatrix, resultVector) = OutlierRemovalHelper<float, Matrix<float>, Vector<float>>
            .ConvertToMatrixVector(inputMatrix, outputVector);

        Assert.Equal(1f, resultMatrix[0, 0]);
        Assert.Equal(5f, resultVector[0]);
    }

    #endregion

    #region ConvertToMatrixVector Tests - Tensor Input

    [Fact]
    public void ConvertToMatrixVector_TensorInput_ConvertsTensorToMatrix()
    {
        var inputTensor = new Tensor<double>(new[] { 3, 2 });
        inputTensor[0, 0] = 1; inputTensor[0, 1] = 2;
        inputTensor[1, 0] = 3; inputTensor[1, 1] = 4;
        inputTensor[2, 0] = 5; inputTensor[2, 1] = 6;

        var outputTensor = new Tensor<double>(new[] { 3 });
        outputTensor[0] = 10; outputTensor[1] = 20; outputTensor[2] = 30;

        var (resultMatrix, resultVector) = OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
            .ConvertToMatrixVector(inputTensor, outputTensor);

        Assert.Equal(3, resultMatrix.Rows);
        Assert.Equal(2, resultMatrix.Columns);
        Assert.Equal(1.0, resultMatrix[0, 0]);
        Assert.Equal(2.0, resultMatrix[0, 1]);
        Assert.Equal(3.0, resultMatrix[1, 0]);
        Assert.Equal(4.0, resultMatrix[1, 1]);
    }

    [Fact]
    public void ConvertToMatrixVector_TensorInput_ConvertsOutputTensorToVector()
    {
        var inputTensor = new Tensor<double>(new[] { 2, 3 });
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                inputTensor[i, j] = i * 3 + j;

        var outputTensor = new Tensor<double>(new[] { 2 });
        outputTensor[0] = 100; outputTensor[1] = 200;

        var (resultMatrix, resultVector) = OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
            .ConvertToMatrixVector(inputTensor, outputTensor);

        Assert.Equal(2, resultVector.Length);
        Assert.Equal(100.0, resultVector[0]);
        Assert.Equal(200.0, resultVector[1]);
    }

    [Fact]
    public void ConvertToMatrixVector_TensorInput_PreservesAllValues()
    {
        var inputTensor = new Tensor<double>(new[] { 4, 5 });
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 5; j++)
                inputTensor[i, j] = i * 10 + j;

        var outputTensor = new Tensor<double>(new[] { 4 });
        for (int i = 0; i < 4; i++)
            outputTensor[i] = i * 100;

        var (resultMatrix, resultVector) = OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
            .ConvertToMatrixVector(inputTensor, outputTensor);

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                Assert.Equal(i * 10 + j, resultMatrix[i, j]);
            }
            Assert.Equal(i * 100, resultVector[i]);
        }
    }

    [Fact]
    public void ConvertToMatrixVector_TensorInput_Float_WorksCorrectly()
    {
        var inputTensor = new Tensor<float>(new[] { 2, 2 });
        inputTensor[0, 0] = 1.5f; inputTensor[0, 1] = 2.5f;
        inputTensor[1, 0] = 3.5f; inputTensor[1, 1] = 4.5f;

        var outputTensor = new Tensor<float>(new[] { 2 });
        outputTensor[0] = 10.5f; outputTensor[1] = 20.5f;

        var (resultMatrix, resultVector) = OutlierRemovalHelper<float, Tensor<float>, Tensor<float>>
            .ConvertToMatrixVector(inputTensor, outputTensor);

        Assert.Equal(1.5f, resultMatrix[0, 0]);
        Assert.Equal(10.5f, resultVector[0]);
    }

    #endregion

    #region ConvertToMatrixVector Tests - Error Cases

    [Fact]
    public void ConvertToMatrixVector_TensorNot2D_ThrowsInvalidOperationException()
    {
        var inputTensor = new Tensor<double>(new[] { 3 }); // 1D tensor
        var outputTensor = new Tensor<double>(new[] { 3 });

        Assert.Throws<InvalidOperationException>(() =>
            OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
                .ConvertToMatrixVector(inputTensor, outputTensor));
    }

    [Fact]
    public void ConvertToMatrixVector_Tensor3D_ThrowsInvalidOperationException()
    {
        var inputTensor = new Tensor<double>(new[] { 2, 3, 4 }); // 3D tensor
        var outputTensor = new Tensor<double>(new[] { 2 });

        Assert.Throws<InvalidOperationException>(() =>
            OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
                .ConvertToMatrixVector(inputTensor, outputTensor));
    }

    [Fact]
    public void ConvertToMatrixVector_OutputTensorNot1D_ThrowsInvalidOperationException()
    {
        var inputTensor = new Tensor<double>(new[] { 2, 3 });
        var outputTensor = new Tensor<double>(new[] { 2, 2 }); // 2D tensor

        Assert.Throws<InvalidOperationException>(() =>
            OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
                .ConvertToMatrixVector(inputTensor, outputTensor));
    }

    [Fact]
    public void ConvertToMatrixVector_UnsupportedTypes_ThrowsInvalidOperationException()
    {
        var inputs = new List<double> { 1, 2, 3 };
        var outputs = new List<double> { 4, 5, 6 };

        Assert.Throws<InvalidOperationException>(() =>
            OutlierRemovalHelper<double, List<double>, List<double>>
                .ConvertToMatrixVector(inputs, outputs));
    }

    #endregion

    #region ConvertToOriginalTypes Tests - Matrix/Vector Types

    [Fact]
    public void ConvertToOriginalTypes_MatrixVectorTypes_ReturnsSameObjects()
    {
        var cleanedMatrix = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 }
        });
        var cleanedVector = new Vector<double>(new[] { 10.0, 20.0 });

        var (resultInputs, resultOutputs) = OutlierRemovalHelper<double, Matrix<double>, Vector<double>>
            .ConvertToOriginalTypes(
                cleanedMatrix,
                cleanedVector,
                typeof(Matrix<double>),
                typeof(Vector<double>));

        Assert.Same(cleanedMatrix, resultInputs);
        Assert.Same(cleanedVector, resultOutputs);
    }

    [Fact]
    public void ConvertToOriginalTypes_MatrixVectorTypes_PreservesValues()
    {
        var cleanedMatrix = new Matrix<double>(new double[,]
        {
            { 1.5, 2.5, 3.5 },
            { 4.5, 5.5, 6.5 }
        });
        var cleanedVector = new Vector<double>(new[] { 100.5, 200.5 });

        var (resultInputs, resultOutputs) = OutlierRemovalHelper<double, Matrix<double>, Vector<double>>
            .ConvertToOriginalTypes(
                cleanedMatrix,
                cleanedVector,
                typeof(Matrix<double>),
                typeof(Vector<double>));

        Assert.Equal(1.5, resultInputs[0, 0]);
        Assert.Equal(6.5, resultInputs[1, 2]);
        Assert.Equal(100.5, resultOutputs[0]);
    }

    [Fact]
    public void ConvertToOriginalTypes_MatrixVectorTypes_Float_WorksCorrectly()
    {
        var cleanedMatrix = new Matrix<float>(new float[,]
        {
            { 1f, 2f },
            { 3f, 4f }
        });
        var cleanedVector = new Vector<float>(new[] { 5f, 6f });

        var (resultInputs, resultOutputs) = OutlierRemovalHelper<float, Matrix<float>, Vector<float>>
            .ConvertToOriginalTypes(
                cleanedMatrix,
                cleanedVector,
                typeof(Matrix<float>),
                typeof(Vector<float>));

        Assert.Equal(1f, resultInputs[0, 0]);
        Assert.Equal(5f, resultOutputs[0]);
    }

    #endregion

    #region ConvertToOriginalTypes Tests - Tensor Types

    [Fact]
    public void ConvertToOriginalTypes_TensorTypes_ConvertsMatrixToTensor()
    {
        var cleanedMatrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 }
        });
        var cleanedVector = new Vector<double>(new[] { 10.0, 20.0 });

        var (resultInputs, resultOutputs) = OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
            .ConvertToOriginalTypes(
                cleanedMatrix,
                cleanedVector,
                typeof(Tensor<double>),
                typeof(Tensor<double>));

        Assert.Equal(2, resultInputs.Shape.Length);
        Assert.Equal(2, resultInputs.Shape[0]);
        Assert.Equal(3, resultInputs.Shape[1]);
    }

    [Fact]
    public void ConvertToOriginalTypes_TensorTypes_ConvertsVectorToTensor()
    {
        var cleanedMatrix = new Matrix<double>(new double[,]
        {
            { 1, 2 },
            { 3, 4 },
            { 5, 6 }
        });
        var cleanedVector = new Vector<double>(new[] { 100.0, 200.0, 300.0 });

        var (resultInputs, resultOutputs) = OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
            .ConvertToOriginalTypes(
                cleanedMatrix,
                cleanedVector,
                typeof(Tensor<double>),
                typeof(Tensor<double>));

        Assert.Equal(1, resultOutputs.Shape.Length);
        Assert.Equal(3, resultOutputs.Shape[0]);
    }

    [Fact]
    public void ConvertToOriginalTypes_TensorTypes_PreservesAllInputValues()
    {
        var cleanedMatrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 }
        });
        var cleanedVector = new Vector<double>(new[] { 10.0, 20.0, 30.0 });

        var (resultInputs, resultOutputs) = OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
            .ConvertToOriginalTypes(
                cleanedMatrix,
                cleanedVector,
                typeof(Tensor<double>),
                typeof(Tensor<double>));

        Assert.Equal(1.0, resultInputs[0, 0]);
        Assert.Equal(2.0, resultInputs[0, 1]);
        Assert.Equal(3.0, resultInputs[0, 2]);
        Assert.Equal(9.0, resultInputs[2, 2]);
    }

    [Fact]
    public void ConvertToOriginalTypes_TensorTypes_PreservesAllOutputValues()
    {
        var cleanedMatrix = new Matrix<double>(2, 3);
        var cleanedVector = new Vector<double>(new[] { 111.0, 222.0 });

        var (resultInputs, resultOutputs) = OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
            .ConvertToOriginalTypes(
                cleanedMatrix,
                cleanedVector,
                typeof(Tensor<double>),
                typeof(Tensor<double>));

        Assert.Equal(111.0, resultOutputs[0]);
        Assert.Equal(222.0, resultOutputs[1]);
    }

    [Fact]
    public void ConvertToOriginalTypes_TensorTypes_Float_WorksCorrectly()
    {
        var cleanedMatrix = new Matrix<float>(new float[,]
        {
            { 1.5f, 2.5f },
            { 3.5f, 4.5f }
        });
        var cleanedVector = new Vector<float>(new[] { 10.5f, 20.5f });

        var (resultInputs, resultOutputs) = OutlierRemovalHelper<float, Tensor<float>, Tensor<float>>
            .ConvertToOriginalTypes(
                cleanedMatrix,
                cleanedVector,
                typeof(Tensor<float>),
                typeof(Tensor<float>));

        Assert.Equal(1.5f, resultInputs[0, 0]);
        Assert.Equal(10.5f, resultOutputs[0]);
    }

    #endregion

    #region ConvertToOriginalTypes Tests - Error Cases

    [Fact]
    public void ConvertToOriginalTypes_UnsupportedTypes_ThrowsInvalidOperationException()
    {
        var cleanedMatrix = new Matrix<double>(2, 2);
        var cleanedVector = new Vector<double>(2);

        Assert.Throws<InvalidOperationException>(() =>
            OutlierRemovalHelper<double, List<double>, List<double>>
                .ConvertToOriginalTypes(
                    cleanedMatrix,
                    cleanedVector,
                    typeof(List<double>),
                    typeof(List<double>)));
    }

    [Fact]
    public void ConvertToOriginalTypes_MismatchedTypes_ThrowsInvalidOperationException()
    {
        var cleanedMatrix = new Matrix<double>(2, 2);
        var cleanedVector = new Vector<double>(2);

        // Matrix input with Tensor output type - not supported
        Assert.Throws<InvalidOperationException>(() =>
            OutlierRemovalHelper<double, Matrix<double>, Tensor<double>>
                .ConvertToOriginalTypes(
                    cleanedMatrix,
                    cleanedVector,
                    typeof(Matrix<double>),
                    typeof(Tensor<double>)));
    }

    #endregion

    #region Round-Trip Tests

    [Fact]
    public void RoundTrip_MatrixVector_PreservesData()
    {
        var originalMatrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 },
            { 10, 11, 12 }
        });
        var originalVector = new Vector<double>(new[] { 100.0, 200.0, 300.0, 400.0 });

        // Convert to matrix/vector (passthrough)
        var (matrix, vector) = OutlierRemovalHelper<double, Matrix<double>, Vector<double>>
            .ConvertToMatrixVector(originalMatrix, originalVector);

        // Convert back
        var (resultMatrix, resultVector) = OutlierRemovalHelper<double, Matrix<double>, Vector<double>>
            .ConvertToOriginalTypes(
                matrix,
                vector,
                typeof(Matrix<double>),
                typeof(Vector<double>));

        Assert.Same(originalMatrix, resultMatrix);
        Assert.Same(originalVector, resultVector);
    }

    [Fact]
    public void RoundTrip_Tensor_PreservesData()
    {
        var originalInputTensor = new Tensor<double>(new[] { 3, 4 });
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                originalInputTensor[i, j] = i * 10 + j;

        var originalOutputTensor = new Tensor<double>(new[] { 3 });
        for (int i = 0; i < 3; i++)
            originalOutputTensor[i] = i * 100;

        // Convert to matrix/vector
        var (matrix, vector) = OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
            .ConvertToMatrixVector(originalInputTensor, originalOutputTensor);

        // Convert back to tensor
        var (resultInputTensor, resultOutputTensor) = OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
            .ConvertToOriginalTypes(
                matrix,
                vector,
                typeof(Tensor<double>),
                typeof(Tensor<double>));

        // Verify input values preserved
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                Assert.Equal(originalInputTensor[i, j], resultInputTensor[i, j]);
            }
        }

        // Verify output values preserved
        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(originalOutputTensor[i], resultOutputTensor[i]);
        }
    }

    [Fact]
    public void RoundTrip_Tensor_PreservesShape()
    {
        var originalInputTensor = new Tensor<double>(new[] { 5, 7 });
        var originalOutputTensor = new Tensor<double>(new[] { 5 });

        var (matrix, vector) = OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
            .ConvertToMatrixVector(originalInputTensor, originalOutputTensor);

        var (resultInputTensor, resultOutputTensor) = OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
            .ConvertToOriginalTypes(
                matrix,
                vector,
                typeof(Tensor<double>),
                typeof(Tensor<double>));

        Assert.Equal(2, resultInputTensor.Shape.Length);
        Assert.Equal(5, resultInputTensor.Shape[0]);
        Assert.Equal(7, resultInputTensor.Shape[1]);
        Assert.Equal(1, resultOutputTensor.Shape.Length);
        Assert.Equal(5, resultOutputTensor.Shape[0]);
    }

    #endregion

    #region Large Dataset Tests

    [Fact]
    public void ConvertToMatrixVector_LargeMatrix_HandlesCorrectly()
    {
        int rows = 1000;
        int cols = 50;
        var inputMatrix = new Matrix<double>(rows, cols);
        var outputVector = new Vector<double>(rows);

        var random = new Random(42);
        for (int i = 0; i < rows; i++)
        {
            outputVector[i] = random.NextDouble() * 100;
            for (int j = 0; j < cols; j++)
            {
                inputMatrix[i, j] = random.NextDouble() * 10;
            }
        }

        var (resultMatrix, resultVector) = OutlierRemovalHelper<double, Matrix<double>, Vector<double>>
            .ConvertToMatrixVector(inputMatrix, outputVector);

        Assert.Equal(rows, resultMatrix.Rows);
        Assert.Equal(cols, resultMatrix.Columns);
        Assert.Equal(rows, resultVector.Length);
    }

    [Fact]
    public void ConvertToMatrixVector_LargeTensor_HandlesCorrectly()
    {
        int rows = 500;
        int cols = 30;
        var inputTensor = new Tensor<double>(new[] { rows, cols });
        var outputTensor = new Tensor<double>(new[] { rows });

        var random = new Random(42);
        for (int i = 0; i < rows; i++)
        {
            outputTensor[i] = random.NextDouble() * 100;
            for (int j = 0; j < cols; j++)
            {
                inputTensor[i, j] = random.NextDouble() * 10;
            }
        }

        var (resultMatrix, resultVector) = OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
            .ConvertToMatrixVector(inputTensor, outputTensor);

        Assert.Equal(rows, resultMatrix.Rows);
        Assert.Equal(cols, resultMatrix.Columns);
        Assert.Equal(rows, resultVector.Length);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void ConvertToMatrixVector_SingleRowMatrix_WorksCorrectly()
    {
        var inputMatrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3, 4, 5 }
        });
        var outputVector = new Vector<double>(new[] { 100.0 });

        var (resultMatrix, resultVector) = OutlierRemovalHelper<double, Matrix<double>, Vector<double>>
            .ConvertToMatrixVector(inputMatrix, outputVector);

        Assert.Equal(1, resultMatrix.Rows);
        Assert.Equal(5, resultMatrix.Columns);
        Assert.Equal(1, resultVector.Length);
    }

    [Fact]
    public void ConvertToMatrixVector_SingleColumnMatrix_WorksCorrectly()
    {
        var inputMatrix = new Matrix<double>(new double[,]
        {
            { 1 },
            { 2 },
            { 3 }
        });
        var outputVector = new Vector<double>(new[] { 10.0, 20.0, 30.0 });

        var (resultMatrix, resultVector) = OutlierRemovalHelper<double, Matrix<double>, Vector<double>>
            .ConvertToMatrixVector(inputMatrix, outputVector);

        Assert.Equal(3, resultMatrix.Rows);
        Assert.Equal(1, resultMatrix.Columns);
        Assert.Equal(3, resultVector.Length);
    }

    [Fact]
    public void ConvertToMatrixVector_SingleElementTensor_WorksCorrectly()
    {
        var inputTensor = new Tensor<double>(new[] { 1, 1 });
        inputTensor[0, 0] = 42;

        var outputTensor = new Tensor<double>(new[] { 1 });
        outputTensor[0] = 100;

        var (resultMatrix, resultVector) = OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
            .ConvertToMatrixVector(inputTensor, outputTensor);

        Assert.Equal(1, resultMatrix.Rows);
        Assert.Equal(1, resultMatrix.Columns);
        Assert.Equal(42.0, resultMatrix[0, 0]);
        Assert.Equal(1, resultVector.Length);
        Assert.Equal(100.0, resultVector[0]);
    }

    [Fact]
    public void ConvertToOriginalTypes_SingleRowMatrix_WorksCorrectly()
    {
        var cleanedMatrix = new Matrix<double>(new double[,]
        {
            { 1, 2, 3 }
        });
        var cleanedVector = new Vector<double>(new[] { 10.0 });

        var (resultInputs, resultOutputs) = OutlierRemovalHelper<double, Tensor<double>, Tensor<double>>
            .ConvertToOriginalTypes(
                cleanedMatrix,
                cleanedVector,
                typeof(Tensor<double>),
                typeof(Tensor<double>));

        Assert.Equal(2, resultInputs.Shape.Length);
        Assert.Equal(1, resultInputs.Shape[0]);
        Assert.Equal(3, resultInputs.Shape[1]);
    }

    [Fact]
    public void ConvertToMatrixVector_NegativeValues_PreservesSign()
    {
        var inputMatrix = new Matrix<double>(new double[,]
        {
            { -1, -2 },
            { 3, -4 }
        });
        var outputVector = new Vector<double>(new[] { -100.0, 200.0 });

        var (resultMatrix, resultVector) = OutlierRemovalHelper<double, Matrix<double>, Vector<double>>
            .ConvertToMatrixVector(inputMatrix, outputVector);

        Assert.Equal(-1.0, resultMatrix[0, 0]);
        Assert.Equal(-2.0, resultMatrix[0, 1]);
        Assert.Equal(3.0, resultMatrix[1, 0]);
        Assert.Equal(-4.0, resultMatrix[1, 1]);
        Assert.Equal(-100.0, resultVector[0]);
        Assert.Equal(200.0, resultVector[1]);
    }

    [Fact]
    public void ConvertToMatrixVector_SpecialDoubleValues_HandlesCorrectly()
    {
        var inputMatrix = new Matrix<double>(new double[,]
        {
            { double.MaxValue, double.MinValue },
            { double.Epsilon, 0 }
        });
        var outputVector = new Vector<double>(new[] { double.MaxValue, double.MinValue });

        var (resultMatrix, resultVector) = OutlierRemovalHelper<double, Matrix<double>, Vector<double>>
            .ConvertToMatrixVector(inputMatrix, outputVector);

        Assert.Equal(double.MaxValue, resultMatrix[0, 0]);
        Assert.Equal(double.MinValue, resultMatrix[0, 1]);
        Assert.Equal(double.Epsilon, resultMatrix[1, 0]);
        Assert.Equal(0.0, resultMatrix[1, 1]);
    }

    #endregion
}
