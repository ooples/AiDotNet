using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public class TensorMatMulTransposeTests
{
    private const float FloatTolerance = 1e-5f;
    private const double DoubleTolerance = 1e-10;

    #region TensorTranspose Tests

    [Fact]
    public void TensorTranspose_SquareMatrix_Float_TransposesCorrectly()
    {
        // Arrange
        var engine = new CpuEngine();
        var input = new Tensor<float>([2, 2]);
        input[0, 0] = 1f; input[0, 1] = 2f;
        input[1, 0] = 3f; input[1, 1] = 4f;

        // Act
        var result = engine.TensorTranspose(input);

        // Assert
        Assert.Equal(new[] { 2, 2 }, result.Shape);
        Assert.Equal(1f, result[0, 0], FloatTolerance);
        Assert.Equal(3f, result[0, 1], FloatTolerance);
        Assert.Equal(2f, result[1, 0], FloatTolerance);
        Assert.Equal(4f, result[1, 1], FloatTolerance);
    }

    [Fact]
    public void TensorTranspose_SquareMatrix_Double_TransposesCorrectly()
    {
        // Arrange
        var engine = new CpuEngine();
        var input = new Tensor<double>([2, 2]);
        input[0, 0] = 1.0; input[0, 1] = 2.0;
        input[1, 0] = 3.0; input[1, 1] = 4.0;

        // Act
        var result = engine.TensorTranspose(input);

        // Assert
        Assert.Equal(new[] { 2, 2 }, result.Shape);
        Assert.Equal(1.0, result[0, 0], DoubleTolerance);
        Assert.Equal(3.0, result[0, 1], DoubleTolerance);
        Assert.Equal(2.0, result[1, 0], DoubleTolerance);
        Assert.Equal(4.0, result[1, 1], DoubleTolerance);
    }

    [Fact]
    public void TensorTranspose_NonSquareMatrix_SwapsDimensions()
    {
        // Arrange
        var engine = new CpuEngine();
        var input = new Tensor<float>([2, 3]);
        input[0, 0] = 1f; input[0, 1] = 2f; input[0, 2] = 3f;
        input[1, 0] = 4f; input[1, 1] = 5f; input[1, 2] = 6f;

        // Act
        var result = engine.TensorTranspose(input);

        // Assert
        Assert.Equal(new[] { 3, 2 }, result.Shape);
        Assert.Equal(1f, result[0, 0], FloatTolerance);
        Assert.Equal(4f, result[0, 1], FloatTolerance);
        Assert.Equal(2f, result[1, 0], FloatTolerance);
        Assert.Equal(5f, result[1, 1], FloatTolerance);
        Assert.Equal(3f, result[2, 0], FloatTolerance);
        Assert.Equal(6f, result[2, 1], FloatTolerance);
    }

    [Fact]
    public void TensorTranspose_1x1Matrix_ReturnsSameValue()
    {
        // Arrange
        var engine = new CpuEngine();
        var input = new Tensor<float>([1, 1]);
        input[0, 0] = 42f;

        // Act
        var result = engine.TensorTranspose(input);

        // Assert
        Assert.Equal(new[] { 1, 1 }, result.Shape);
        Assert.Equal(42f, result[0, 0], FloatTolerance);
    }

    [Fact]
    public void TensorTranspose_NullInput_ThrowsArgumentNullException()
    {
        // Arrange
        var engine = new CpuEngine();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => engine.TensorTranspose<float>(null!));
    }

    [Fact]
    public void TensorTranspose_Non2DTensor_ThrowsArgumentException()
    {
        // Arrange
        var engine = new CpuEngine();
        var input = new Tensor<float>([2, 2, 2]); // 3D tensor

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => engine.TensorTranspose(input));
        Assert.Contains("2D tensor", ex.Message);
    }

    [Fact]
    public void TensorTranspose_DoubleTranspose_ReturnsOriginal()
    {
        // Arrange
        var engine = new CpuEngine();
        var input = new Tensor<float>([3, 4]);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                input[i, j] = i * 4 + j;

        // Act
        var transposed = engine.TensorTranspose(input);
        var doubleTransposed = engine.TensorTranspose(transposed);

        // Assert
        Assert.Equal(input.Shape, doubleTransposed.Shape);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                Assert.Equal(input[i, j], doubleTransposed[i, j], FloatTolerance);
    }

    #endregion

    #region TensorMatMul Tests

    [Fact]
    public void TensorMatMul_SquareMatrices_Float_ComputesCorrectly()
    {
        // Arrange
        var engine = new CpuEngine();
        var a = new Tensor<float>([2, 2]);
        a[0, 0] = 1f; a[0, 1] = 2f;
        a[1, 0] = 3f; a[1, 1] = 4f;

        var b = new Tensor<float>([2, 2]);
        b[0, 0] = 5f; b[0, 1] = 6f;
        b[1, 0] = 7f; b[1, 1] = 8f;

        // Act
        var result = engine.TensorMatMul(a, b);

        // Assert
        // [1,2] * [5,6] = [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3,4]   [7,8]   [3*5+4*7, 3*6+4*8]   [43, 50]
        Assert.Equal(new[] { 2, 2 }, result.Shape);
        Assert.Equal(19f, result[0, 0], FloatTolerance);
        Assert.Equal(22f, result[0, 1], FloatTolerance);
        Assert.Equal(43f, result[1, 0], FloatTolerance);
        Assert.Equal(50f, result[1, 1], FloatTolerance);
    }

    [Fact]
    public void TensorMatMul_SquareMatrices_Double_ComputesCorrectly()
    {
        // Arrange
        var engine = new CpuEngine();
        var a = new Tensor<double>([2, 2]);
        a[0, 0] = 1.0; a[0, 1] = 2.0;
        a[1, 0] = 3.0; a[1, 1] = 4.0;

        var b = new Tensor<double>([2, 2]);
        b[0, 0] = 5.0; b[0, 1] = 6.0;
        b[1, 0] = 7.0; b[1, 1] = 8.0;

        // Act
        var result = engine.TensorMatMul(a, b);

        // Assert
        Assert.Equal(new[] { 2, 2 }, result.Shape);
        Assert.Equal(19.0, result[0, 0], DoubleTolerance);
        Assert.Equal(22.0, result[0, 1], DoubleTolerance);
        Assert.Equal(43.0, result[1, 0], DoubleTolerance);
        Assert.Equal(50.0, result[1, 1], DoubleTolerance);
    }

    [Fact]
    public void TensorMatMul_NonSquareMatrices_ComputesCorrectDimensions()
    {
        // Arrange
        var engine = new CpuEngine();
        var a = new Tensor<float>([2, 3]); // 2x3
        a[0, 0] = 1f; a[0, 1] = 2f; a[0, 2] = 3f;
        a[1, 0] = 4f; a[1, 1] = 5f; a[1, 2] = 6f;

        var b = new Tensor<float>([3, 2]); // 3x2
        b[0, 0] = 7f; b[0, 1] = 8f;
        b[1, 0] = 9f; b[1, 1] = 10f;
        b[2, 0] = 11f; b[2, 1] = 12f;

        // Act
        var result = engine.TensorMatMul(a, b);

        // Assert - result should be 2x2
        Assert.Equal(new[] { 2, 2 }, result.Shape);
        // [1,2,3] * [7,8]   = [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4,5,6]   [9,10]    [4*7+5*9+6*11, 4*8+5*10+6*12]   [139, 154]
        //           [11,12]
        Assert.Equal(58f, result[0, 0], FloatTolerance);
        Assert.Equal(64f, result[0, 1], FloatTolerance);
        Assert.Equal(139f, result[1, 0], FloatTolerance);
        Assert.Equal(154f, result[1, 1], FloatTolerance);
    }

    [Fact]
    public void TensorMatMul_1x1Matrices_ComputesCorrectly()
    {
        // Arrange
        var engine = new CpuEngine();
        var a = new Tensor<float>([1, 1]);
        a[0, 0] = 3f;

        var b = new Tensor<float>([1, 1]);
        b[0, 0] = 4f;

        // Act
        var result = engine.TensorMatMul(a, b);

        // Assert
        Assert.Equal(new[] { 1, 1 }, result.Shape);
        Assert.Equal(12f, result[0, 0], FloatTolerance);
    }

    [Fact]
    public void TensorMatMul_RowTimesColumn_ComputesDotProduct()
    {
        // Arrange
        var engine = new CpuEngine();
        var a = new Tensor<float>([1, 3]); // Row vector
        a[0, 0] = 1f; a[0, 1] = 2f; a[0, 2] = 3f;

        var b = new Tensor<float>([3, 1]); // Column vector
        b[0, 0] = 4f; b[1, 0] = 5f; b[2, 0] = 6f;

        // Act
        var result = engine.TensorMatMul(a, b);

        // Assert - should be 1x1 with dot product
        Assert.Equal(new[] { 1, 1 }, result.Shape);
        Assert.Equal(32f, result[0, 0], FloatTolerance); // 1*4 + 2*5 + 3*6 = 32
    }

    [Fact]
    public void TensorMatMul_ColumnTimesRow_ComputesOuterProduct()
    {
        // Arrange
        var engine = new CpuEngine();
        var a = new Tensor<float>([3, 1]); // Column vector
        a[0, 0] = 1f; a[1, 0] = 2f; a[2, 0] = 3f;

        var b = new Tensor<float>([1, 2]); // Row vector
        b[0, 0] = 4f; b[0, 1] = 5f;

        // Act
        var result = engine.TensorMatMul(a, b);

        // Assert - should be 3x2 outer product
        Assert.Equal(new[] { 3, 2 }, result.Shape);
        Assert.Equal(4f, result[0, 0], FloatTolerance);  // 1*4
        Assert.Equal(5f, result[0, 1], FloatTolerance);  // 1*5
        Assert.Equal(8f, result[1, 0], FloatTolerance);  // 2*4
        Assert.Equal(10f, result[1, 1], FloatTolerance); // 2*5
        Assert.Equal(12f, result[2, 0], FloatTolerance); // 3*4
        Assert.Equal(15f, result[2, 1], FloatTolerance); // 3*5
    }

    [Fact]
    public void TensorMatMul_NullFirstInput_ThrowsArgumentNullException()
    {
        // Arrange
        var engine = new CpuEngine();
        var b = new Tensor<float>([2, 2]);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => engine.TensorMatMul<float>(null!, b));
    }

    [Fact]
    public void TensorMatMul_NullSecondInput_ThrowsArgumentNullException()
    {
        // Arrange
        var engine = new CpuEngine();
        var a = new Tensor<float>([2, 2]);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => engine.TensorMatMul(a, null!));
    }

    [Fact]
    public void TensorMatMul_3Dx2D_ReturnsBatchedResult()
    {
        // Arrange - TensorMatMul now supports ND x 2D (industry standard batched matmul)
        // [batch, M, K] @ [K, N] = [batch, M, N]
        var engine = new CpuEngine();
        var a = new Tensor<float>([2, 3, 4]); // 3D tensor: batch=2, 3x4 matrices
        var b = new Tensor<float>([4, 5]);    // 2D tensor: 4x5 weight matrix

        // Act
        var result = engine.TensorMatMul(a, b);

        // Assert - output should be [2, 3, 5]
        Assert.Equal(3, result.Rank);
        Assert.Equal(2, result.Shape[0]);
        Assert.Equal(3, result.Shape[1]);
        Assert.Equal(5, result.Shape[2]);
    }

    [Fact]
    public void TensorMatMul_IncompatibleDimensions_ThrowsArgumentException()
    {
        // Arrange
        var engine = new CpuEngine();
        var a = new Tensor<float>([2, 3]); // 2x3
        var b = new Tensor<float>([4, 2]); // 4x2 - incompatible (3 != 4)

        // Act & Assert
        var ex = Assert.Throws<ArgumentException>(() => engine.TensorMatMul(a, b));
        Assert.Contains("incompatible", ex.Message.ToLower());
    }

    [Fact]
    public void TensorMatMul_IdentityMatrix_ReturnsOriginal()
    {
        // Arrange
        var engine = new CpuEngine();
        var a = new Tensor<float>([3, 3]);
        a[0, 0] = 1f; a[0, 1] = 2f; a[0, 2] = 3f;
        a[1, 0] = 4f; a[1, 1] = 5f; a[1, 2] = 6f;
        a[2, 0] = 7f; a[2, 1] = 8f; a[2, 2] = 9f;

        var identity = new Tensor<float>([3, 3]);
        identity[0, 0] = 1f; identity[0, 1] = 0f; identity[0, 2] = 0f;
        identity[1, 0] = 0f; identity[1, 1] = 1f; identity[1, 2] = 0f;
        identity[2, 0] = 0f; identity[2, 1] = 0f; identity[2, 2] = 1f;

        // Act
        var result = engine.TensorMatMul(a, identity);

        // Assert
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(a[i, j], result[i, j], FloatTolerance);
    }

    [Fact]
    public void TensorMatMul_ZeroMatrix_ReturnsZeros()
    {
        // Arrange
        var engine = new CpuEngine();
        var a = new Tensor<float>([2, 2]);
        a[0, 0] = 1f; a[0, 1] = 2f;
        a[1, 0] = 3f; a[1, 1] = 4f;

        var zero = new Tensor<float>([2, 2]); // All zeros by default

        // Act
        var result = engine.TensorMatMul(a, zero);

        // Assert
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                Assert.Equal(0f, result[i, j], FloatTolerance);
    }

    #endregion

    #region Combined TensorMatMul and TensorTranspose Tests

    [Fact]
    public void TensorMatMul_TransposeProperty_ABTranspose_Equals_BTransposeATranspose()
    {
        // (AB)^T = B^T * A^T
        var engine = new CpuEngine();
        var a = new Tensor<float>([2, 3]);
        a[0, 0] = 1f; a[0, 1] = 2f; a[0, 2] = 3f;
        a[1, 0] = 4f; a[1, 1] = 5f; a[1, 2] = 6f;

        var b = new Tensor<float>([3, 2]);
        b[0, 0] = 7f; b[0, 1] = 8f;
        b[1, 0] = 9f; b[1, 1] = 10f;
        b[2, 0] = 11f; b[2, 1] = 12f;

        // Act
        var ab = engine.TensorMatMul(a, b);
        var abTranspose = engine.TensorTranspose(ab);

        var aTranspose = engine.TensorTranspose(a);
        var bTranspose = engine.TensorTranspose(b);
        var btAt = engine.TensorMatMul(bTranspose, aTranspose);

        // Assert
        Assert.Equal(abTranspose.Shape, btAt.Shape);
        for (int i = 0; i < abTranspose.Shape[0]; i++)
            for (int j = 0; j < abTranspose.Shape[1]; j++)
                Assert.Equal(abTranspose[i, j], btAt[i, j], FloatTolerance);
    }

    #endregion
}
