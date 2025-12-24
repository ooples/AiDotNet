using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

public class CpuEngineOperationsTests
{
    private readonly CpuEngine _engine = new();

    [Fact]
    public void TensorFloor_ReturnsFloorValues()
    {
        var tensor = new Tensor<float>(new[] { 4 });
        tensor[0] = 1.5f;
        tensor[1] = 2.3f;
        tensor[2] = -1.7f;
        tensor[3] = 3.9f;

        var result = _engine.TensorFloor(tensor);

        Assert.Equal(1f, result[0]);
        Assert.Equal(2f, result[1]);
        Assert.Equal(-2f, result[2]);
        Assert.Equal(3f, result[3]);
    }

    [Fact]
    public void TensorCeiling_ReturnsCeilingValues()
    {
        var tensor = new Tensor<float>(new[] { 4 });
        tensor[0] = 1.5f;
        tensor[1] = 2.3f;
        tensor[2] = -1.7f;
        tensor[3] = 3.9f;

        var result = _engine.TensorCeiling(tensor);

        Assert.Equal(2f, result[0]);
        Assert.Equal(3f, result[1]);
        Assert.Equal(-1f, result[2]);
        Assert.Equal(4f, result[3]);
    }

    [Fact]
    public void TensorFrac_ReturnsFractionalPart()
    {
        var tensor = new Tensor<float>(new[] { 3 });
        tensor[0] = 1.5f;
        tensor[1] = 2.3f;
        tensor[2] = 3.9f;

        var result = _engine.TensorFrac(tensor);

        Assert.Equal(0.5f, result[0], 5);
        Assert.Equal(0.3f, result[1], 1);
        Assert.Equal(0.9f, result[2], 1);
    }

    [Fact]
    public void TensorSin_MatchesMathSin()
    {
        var tensor = new Tensor<float>(new[] { 4 });
        tensor[0] = 0f;
        tensor[1] = (float)(Math.PI / 2);
        tensor[2] = (float)Math.PI;
        tensor[3] = (float)(3 * Math.PI / 2);

        var result = _engine.TensorSin(tensor);

        Assert.Equal((float)Math.Sin(0), result[0], 5);
        Assert.Equal((float)Math.Sin(Math.PI / 2), result[1], 5);
        Assert.Equal((float)Math.Sin(Math.PI), result[2], 5);
        Assert.Equal((float)Math.Sin(3 * Math.PI / 2), result[3], 5);
    }

    [Fact]
    public void TensorCos_MatchesMathCos()
    {
        var tensor = new Tensor<float>(new[] { 4 });
        tensor[0] = 0f;
        tensor[1] = (float)(Math.PI / 2);
        tensor[2] = (float)Math.PI;
        tensor[3] = (float)(3 * Math.PI / 2);

        var result = _engine.TensorCos(tensor);

        Assert.Equal((float)Math.Cos(0), result[0], 5);
        Assert.Equal((float)Math.Cos(Math.PI / 2), result[1], 5);
        Assert.Equal((float)Math.Cos(Math.PI), result[2], 5);
        Assert.Equal((float)Math.Cos(3 * Math.PI / 2), result[3], 5);
    }

    [Fact]
    public void TensorTrilinearInterpolate_AtCenter_ReturnsAverage()
    {
        // Create a 2x2x2 grid with 1 channel
        var grid = new Tensor<float>(new[] { 2, 2, 2, 1 });
        grid[0, 0, 0, 0] = 0f;
        grid[0, 0, 1, 0] = 1f;
        grid[0, 1, 0, 0] = 2f;
        grid[0, 1, 1, 0] = 3f;
        grid[1, 0, 0, 0] = 4f;
        grid[1, 0, 1, 0] = 5f;
        grid[1, 1, 0, 0] = 6f;
        grid[1, 1, 1, 0] = 7f;

        // Sample at center (0.5, 0.5, 0.5)
        var positions = new Tensor<float>(new[] { 1, 3 });
        positions[0, 0] = 0.5f;
        positions[0, 1] = 0.5f;
        positions[0, 2] = 0.5f;

        var result = _engine.TensorTrilinearInterpolate(grid, positions);

        // Average of all 8 corners: (0+1+2+3+4+5+6+7)/8 = 3.5
        Assert.Equal(3.5f, result[0, 0], 5);
    }

    [Fact]
    public void TensorFloor_PreservesShape()
    {
        var tensor = new Tensor<float>(new[] { 2, 3 });
        for (int i = 0; i < 6; i++)
            tensor.SetFlat(i, i + 0.5f);

        var result = _engine.TensorFloor(tensor);

        Assert.Equal(new[] { 2, 3 }, result.Shape);
    }

    [Fact]
    public void PairwiseDistanceSquared_ComputesCorrectDistances()
    {
        // Points: (0,0), (1,0), (0,1)
        var x = new Tensor<float>(new[] { 2, 2 });
        x[0, 0] = 0f; x[0, 1] = 0f;  // Point (0,0)
        x[1, 0] = 1f; x[1, 1] = 0f;  // Point (1,0)

        var y = new Tensor<float>(new[] { 2, 2 });
        y[0, 0] = 0f; y[0, 1] = 1f;  // Point (0,1)
        y[1, 0] = 1f; y[1, 1] = 1f;  // Point (1,1)

        var result = _engine.PairwiseDistanceSquared(x, y);

        // dist^2((0,0), (0,1)) = 1
        Assert.Equal(1f, result[0, 0], 5);
        // dist^2((0,0), (1,1)) = 2
        Assert.Equal(2f, result[0, 1], 5);
        // dist^2((1,0), (0,1)) = 2
        Assert.Equal(2f, result[1, 0], 5);
        // dist^2((1,0), (1,1)) = 1
        Assert.Equal(1f, result[1, 1], 5);
    }

    [Fact]
    public void PairwiseDistance_ComputesEuclideanDistances()
    {
        var x = new Tensor<float>(new[] { 1, 2 });
        x[0, 0] = 0f; x[0, 1] = 0f;

        var y = new Tensor<float>(new[] { 1, 2 });
        y[0, 0] = 3f; y[0, 1] = 4f;

        var result = _engine.PairwiseDistance(x, y);

        // dist((0,0), (3,4)) = 5
        Assert.Equal(5f, result[0, 0], 5);
    }

    [Fact]
    public void TopK_ReturnsLargestElements()
    {
        var input = new Tensor<float>(new[] { 5 });
        input[0] = 3f;
        input[1] = 1f;
        input[2] = 4f;
        input[3] = 1f;
        input[4] = 5f;

        var (values, indices) = _engine.TopK(input, 3, axis: 0, largest: true);

        // Top 3 largest: 5, 4, 3
        Assert.Equal(5f, values[0], 5);
        Assert.Equal(4f, values[1], 5);
        Assert.Equal(3f, values[2], 5);
    }

    [Fact]
    public void TopK_ReturnsSmallestElements()
    {
        var input = new Tensor<float>(new[] { 5 });
        input[0] = 3f;
        input[1] = 1f;
        input[2] = 4f;
        input[3] = 2f;
        input[4] = 5f;

        var (values, indices) = _engine.TopK(input, 2, axis: 0, largest: false);

        // Top 2 smallest: 1, 2
        Assert.Equal(1f, values[0], 5);
        Assert.Equal(2f, values[1], 5);
    }

    [Fact]
    public void ArgSort_ReturnsCorrectIndices()
    {
        var input = new Tensor<float>(new[] { 4 });
        input[0] = 3f;
        input[1] = 1f;
        input[2] = 4f;
        input[3] = 2f;

        var result = _engine.ArgSort(input, axis: 0, descending: false);

        // Sorted order: 1, 2, 3, 4 -> indices: 1, 3, 0, 2
        Assert.Equal(1, result[0]);
        Assert.Equal(3, result[1]);
        Assert.Equal(0, result[2]);
        Assert.Equal(2, result[3]);
    }

    [Fact]
    public void Gather_GathersCorrectElements()
    {
        var input = new Tensor<float>(new[] { 4 });
        input[0] = 10f;
        input[1] = 20f;
        input[2] = 30f;
        input[3] = 40f;

        var indices = new Tensor<int>(new[] { 3 });
        indices[0] = 2;
        indices[1] = 0;
        indices[2] = 3;

        var result = _engine.Gather(input, indices, axis: 0);

        Assert.Equal(30f, result[0], 5);
        Assert.Equal(10f, result[1], 5);
        Assert.Equal(40f, result[2], 5);
    }

    [Fact]
    public void TensorCosh_MatchesMathCosh()
    {
        var tensor = new Tensor<float>(new[] { 3 });
        tensor[0] = 0f;
        tensor[1] = 1f;
        tensor[2] = -1f;

        var result = _engine.TensorCosh(tensor);

        Assert.Equal((float)Math.Cosh(0), result[0], 5);
        Assert.Equal((float)Math.Cosh(1), result[1], 5);
        Assert.Equal((float)Math.Cosh(-1), result[2], 5);
    }

    [Fact]
    public void TensorSinh_MatchesMathSinh()
    {
        var tensor = new Tensor<float>(new[] { 3 });
        tensor[0] = 0f;
        tensor[1] = 1f;
        tensor[2] = -1f;

        var result = _engine.TensorSinh(tensor);

        Assert.Equal((float)Math.Sinh(0), result[0], 5);
        Assert.Equal((float)Math.Sinh(1), result[1], 5);
        Assert.Equal((float)Math.Sinh(-1), result[2], 5);
    }

    [Fact]
    public void TensorOuter_ComputesOuterProduct()
    {
        var a = new Tensor<float>(new[] { 2 });
        a[0] = 1f;
        a[1] = 2f;

        var b = new Tensor<float>(new[] { 3 });
        b[0] = 10f;
        b[1] = 20f;
        b[2] = 30f;

        var result = _engine.TensorOuter(a, b);

        Assert.Equal(new[] { 2, 3 }, result.Shape);
        Assert.Equal(10f, result[0, 0], 5);  // 1*10
        Assert.Equal(20f, result[0, 1], 5);  // 1*20
        Assert.Equal(30f, result[0, 2], 5);  // 1*30
        Assert.Equal(20f, result[1, 0], 5);  // 2*10
        Assert.Equal(40f, result[1, 1], 5);  // 2*20
        Assert.Equal(60f, result[1, 2], 5);  // 2*30
    }
}
