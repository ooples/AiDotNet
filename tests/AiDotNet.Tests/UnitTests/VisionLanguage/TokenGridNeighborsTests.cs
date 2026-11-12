using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.VisionLanguage;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

/// <summary>
/// Neighbour lookups used by the Show-o / Show-o2 mask-predict context: a token at a row edge has no
/// neighbour across the edge (a flat index +/- 1 would wrap into the adjacent row).
/// </summary>
public class TokenGridNeighborsTests
{
    private static Tensor<double> Grid3x3()
    {
        var values = new Tensor<double>([9]);
        for (int i = 0; i < 9; i++)
            values[i] = i + 1;
        return values;
    }

    [Fact(Timeout = 30000)]
    public async Task LeftNeighbour_AtColumnZero_IsAbsentNotPreviousRowEnd()
    {
        await Task.Yield();
        // Grid  1 2 3 / 4 5 6 / 7 8 9. Token 3 (value 4) is at row 1, column 0: it has no left
        // neighbour, so the result is 0, not 3 (the last token of row 0).
        var left = TokenGridNeighbors<double>.Shift(AiDotNetEngine.Current, Grid3x3(), 3, 0, -1);
        double[] expected = [0, 1, 2, 0, 4, 5, 0, 7, 8];
        for (int i = 0; i < 9; i++)
            Assert.Equal(expected[i], left[i]);
    }

    [Fact(Timeout = 30000)]
    public async Task RightNeighbour_AtLastColumn_IsAbsentNotNextRowStart()
    {
        await Task.Yield();
        var right = TokenGridNeighbors<double>.Shift(AiDotNetEngine.Current, Grid3x3(), 3, 0, 1);
        double[] expected = [2, 3, 0, 5, 6, 0, 8, 9, 0];
        for (int i = 0; i < 9; i++)
            Assert.Equal(expected[i], right[i]);
    }

    [Fact(Timeout = 30000)]
    public async Task DiagonalNeighbours_RespectRowAndColumnEdges()
    {
        await Task.Yield();
        var downRight = TokenGridNeighbors<double>.Shift(AiDotNetEngine.Current, Grid3x3(), 3, 1, 1);
        double[] expectedDownRight = [5, 6, 0, 8, 9, 0, 0, 0, 0];
        var upLeft = TokenGridNeighbors<double>.Shift(AiDotNetEngine.Current, Grid3x3(), 3, -1, -1);
        double[] expectedUpLeft = [0, 0, 0, 0, 1, 2, 0, 4, 5];
        for (int i = 0; i < 9; i++)
        {
            Assert.Equal(expectedDownRight[i], downRight[i]);
            Assert.Equal(expectedUpLeft[i], upLeft[i]);
        }
    }

    [Fact(Timeout = 30000)]
    public async Task VerticalNeighbours_AreZeroOutsideTheGrid()
    {
        await Task.Yield();
        var up = TokenGridNeighbors<double>.Shift(AiDotNetEngine.Current, Grid3x3(), 3, -1, 0);
        double[] expected = [0, 0, 0, 1, 2, 3, 4, 5, 6];
        for (int i = 0; i < 9; i++)
            Assert.Equal(expected[i], up[i]);
    }
}
