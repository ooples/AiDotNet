using AiDotNet.Extensions;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LinearAlgebra;

/// <summary>
/// Integration tests for TensorExtensions methods.
/// </summary>
public class TensorExtensionsIntegrationTests
{
    private const double Tolerance = 1e-10;

    #region ConvertToMatrix Tests

    [Fact]
    public void ConvertToMatrix_2DTensor_CorrectDimensions()
    {
        var tensor = new Tensor<double>([2, 3]);
        tensor[[0, 0]] = 1; tensor[[0, 1]] = 2; tensor[[0, 2]] = 3;
        tensor[[1, 0]] = 4; tensor[[1, 1]] = 5; tensor[[1, 2]] = 6;

        var matrix = tensor.ConvertToMatrix();

        Assert.Equal(2, matrix.Rows);
        Assert.Equal(3, matrix.Columns);
        Assert.Equal(1, matrix[0, 0]);
        Assert.Equal(2, matrix[0, 1]);
        Assert.Equal(3, matrix[0, 2]);
        Assert.Equal(4, matrix[1, 0]);
        Assert.Equal(5, matrix[1, 1]);
        Assert.Equal(6, matrix[1, 2]);
    }

    [Fact]
    public void ConvertToMatrix_1DTensor_ReturnsColumnMatrix()
    {
        var tensor = new Tensor<double>([3]);
        tensor[[0]] = 1; tensor[[1]] = 2; tensor[[2]] = 3;

        var matrix = tensor.ConvertToMatrix();

        Assert.Equal(3, matrix.Rows);
        Assert.Equal(1, matrix.Columns);
        Assert.Equal(1, matrix[0, 0]);
        Assert.Equal(2, matrix[1, 0]);
        Assert.Equal(3, matrix[2, 0]);
    }

    [Fact]
    public void ConvertToMatrix_3DTensor_ThrowsException()
    {
        var tensor = new Tensor<double>([2, 3, 4]);

        Assert.Throws<ArgumentException>(() => tensor.ConvertToMatrix());
    }

    #endregion

    #region Unflatten Tests

    [Fact]
    public void Unflatten_FlatVector_RestoresShape()
    {
        var shape = new[] { 2, 3 };
        var tensor = new Tensor<double>(shape);
        var flattened = new Vector<double>([1, 2, 3, 4, 5, 6]);

        var result = tensor.Unflatten(flattened);

        Assert.Equal(shape, result.Shape);
        Assert.Equal(1, result[[0, 0]]);
        Assert.Equal(2, result[[0, 1]]);
        Assert.Equal(3, result[[0, 2]]);
        Assert.Equal(4, result[[1, 0]]);
        Assert.Equal(5, result[[1, 1]]);
        Assert.Equal(6, result[[1, 2]]);
    }

    [Fact]
    public void Unflatten_WrongSize_ThrowsException()
    {
        var tensor = new Tensor<double>([2, 3]); // Size 6
        var flattened = new Vector<double>([1, 2, 3, 4, 5]); // Size 5

        Assert.Throws<ArgumentException>(() => tensor.Unflatten(flattened));
    }

    #endregion

    #region ForEachPosition Tests

    [Fact]
    public void ForEachPosition_VisitsAllPositions()
    {
        var tensor = new Tensor<double>([2, 3]);
        for (int i = 0; i < 6; i++)
            tensor[i] = i + 1;

        var visited = new List<(int[], double)>();
        tensor.ForEachPosition((pos, val) =>
        {
            visited.Add((pos.ToArray(), val));
            return true;
        });

        Assert.Equal(6, visited.Count);
        Assert.Contains(visited, v => v.Item1.SequenceEqual(new[] { 0, 0 }) && v.Item2 == 1);
        Assert.Contains(visited, v => v.Item1.SequenceEqual(new[] { 1, 2 }) && v.Item2 == 6);
    }

    [Fact]
    public void ForEachPosition_EarlyStop_StopsIteration()
    {
        var tensor = new Tensor<double>([3, 3]);
        int count = 0;

        tensor.ForEachPosition((pos, val) =>
        {
            count++;
            return count < 5; // Stop after 5 iterations
        });

        Assert.Equal(5, count);
    }

    #endregion

    #region TensorEquals Tests

    [Fact]
    public void TensorEquals_IdenticalTensors_ReturnsTrue()
    {
        var a = new Tensor<double>([2, 2]);
        a[[0, 0]] = 1; a[[0, 1]] = 2; a[[1, 0]] = 3; a[[1, 1]] = 4;

        var b = new Tensor<double>([2, 2]);
        b[[0, 0]] = 1; b[[0, 1]] = 2; b[[1, 0]] = 3; b[[1, 1]] = 4;

        Assert.True(a.TensorEquals(b));
    }

    [Fact]
    public void TensorEquals_DifferentValues_ReturnsFalse()
    {
        var a = new Tensor<double>([2, 2]);
        a[[0, 0]] = 1; a[[0, 1]] = 2; a[[1, 0]] = 3; a[[1, 1]] = 4;

        var b = new Tensor<double>([2, 2]);
        b[[0, 0]] = 1; b[[0, 1]] = 2; b[[1, 0]] = 3; b[[1, 1]] = 5; // Different

        Assert.False(a.TensorEquals(b));
    }

    [Fact]
    public void TensorEquals_DifferentShapes_ReturnsFalse()
    {
        var a = new Tensor<double>([2, 3]);
        var b = new Tensor<double>([3, 2]);

        Assert.False(a.TensorEquals(b));
    }

    #endregion

    #region ConcatenateTensors Tests

    [Fact]
    public void ConcatenateTensors_2D_ConcatenatesAlongLastDimension()
    {
        var a = new Tensor<double>([2, 2]);
        a[[0, 0]] = 1; a[[0, 1]] = 2;
        a[[1, 0]] = 3; a[[1, 1]] = 4;

        var b = new Tensor<double>([2, 3]);
        b[[0, 0]] = 5; b[[0, 1]] = 6; b[[0, 2]] = 7;
        b[[1, 0]] = 8; b[[1, 1]] = 9; b[[1, 2]] = 10;

        var result = a.ConcatenateTensors(b);

        Assert.Equal(new[] { 2, 5 }, result.Shape);
        Assert.Equal(1, result[[0, 0]]);
        Assert.Equal(2, result[[0, 1]]);
        Assert.Equal(5, result[[0, 2]]);
        Assert.Equal(6, result[[0, 3]]);
        Assert.Equal(7, result[[0, 4]]);
    }

    [Fact]
    public void ConcatenateTensors_DifferentRanks_ThrowsException()
    {
        var a = new Tensor<double>([2, 2]);
        var b = new Tensor<double>([2, 2, 2]);

        Assert.Throws<ArgumentException>(() => a.ConcatenateTensors(b));
    }

    #endregion

    #region CreateXavierInitializedTensor Tests

    [Fact]
    public void CreateXavierInitializedTensor_CorrectShape()
    {
        var shape = new[] { 3, 4 };
        var random = new Random(42);
        var stddev = 0.5;

        var tensor = TensorExtensions.CreateXavierInitializedTensor<double>(shape, stddev, random);

        Assert.Equal(shape, tensor.Shape);
        Assert.Equal(12, tensor.Length);
    }

    [Fact]
    public void CreateXavierInitializedTensor_ValuesBoundedByStddev()
    {
        var shape = new[] { 100, 100 };
        var random = new Random(42);
        var stddev = 0.1;

        var tensor = TensorExtensions.CreateXavierInitializedTensor<double>(shape, stddev, random);

        // All values should be within [-stddev, stddev]
        for (int i = 0; i < tensor.Length; i++)
        {
            Assert.True(tensor[i] >= -stddev && tensor[i] <= stddev);
        }
    }

    #endregion

    #region CreateOnesTensor Tests

    [Fact]
    public void CreateOnesTensor_AllOnes()
    {
        var tensor = TensorExtensions.CreateOnesTensor<double>(5);

        Assert.Equal(new[] { 5 }, tensor.Shape);
        for (int i = 0; i < 5; i++)
        {
            Assert.Equal(1, tensor[i]);
        }
    }

    #endregion

    #region HeStddev Tests

    [Fact]
    public void HeStddev_CalculatesCorrectly()
    {
        // He initialization: sqrt(2 / fanIn)
        int fanIn = 100;
        var stddev = TensorExtensions.HeStddev(fanIn);

        Assert.Equal(Math.Sqrt(2.0 / 100.0), stddev, Tolerance);
    }

    #endregion

    #region XavierStddev Tests

    [Fact]
    public void XavierStddev_CalculatesCorrectly()
    {
        // Xavier initialization: sqrt(2 / (fanIn + fanOut))
        int fanIn = 100;
        int fanOut = 50;
        var stddev = TensorExtensions.XavierStddev(fanIn, fanOut);

        Assert.Equal(Math.Sqrt(2.0 / 150.0), stddev, Tolerance);
    }

    #endregion
}
