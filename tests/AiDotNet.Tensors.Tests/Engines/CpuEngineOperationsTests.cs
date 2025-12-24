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
}
