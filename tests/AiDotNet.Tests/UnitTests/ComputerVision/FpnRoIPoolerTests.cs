using AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Pins the FPN RoI level assignment (Lin et al. 2017, eq. 1) and that multi-level pooling returns
/// each box's features from its own level, in the caller's order.
/// </summary>
public class FpnRoIPoolerTests
{
    private static readonly int[] Strides = { 4, 8, 16, 32 };

    private static Tensor<double> Boxes(params double[] xyxy)
    {
        var t = new Tensor<double>(new[] { xyxy.Length / 4, 4 });
        for (int i = 0; i < xyxy.Length; i++) t[i] = xyxy[i];
        return t;
    }

    [Theory]
    [InlineData(224.0, 2)] // canonical size -> level 4 = P4, index 2 of P2..P5
    [InlineData(112.0, 1)] // half -> P3
    [InlineData(56.0, 0)]  // quarter -> P2
    [InlineData(448.0, 3)] // double -> P5
    [InlineData(10.0, 0)]  // below P2 clamps to the finest level
    [InlineData(2000.0, 3)] // above P5 clamps to the coarsest level
    [InlineData(223.0, 1)] // floor: just under 224 is still level 3
    public async Task AssignLevels_FollowsTheFpnRule(double side, int expectedIndex)
    {
        await Task.Yield();
        var levels = FpnRoIPooler<double>.AssignLevels(Boxes(10, 10, 10 + side, 10 + side), Strides);
        Assert.Equal(expectedIndex, levels[0]);
    }

    [Fact]
    public async Task Pool_ReturnsEachBoxFromItsLevelInCallerOrder()
    {
        await Task.Yield();
        var r = new Random(5);
        var levels = new List<Tensor<double>>();
        for (int l = 0; l < Strides.Length; l++)
        {
            int size = 256 / Strides[l];
            var map = new Tensor<double>(new[] { 1, 3, size, size });
            for (int i = 0; i < map.Length; i++) map[i] = r.NextDouble();
            levels.Add(map);
        }

        // Interleave sizes so the level groups are NOT contiguous in caller order.
        var boxes = Boxes(
            0, 0, 240, 240,   // P4
            8, 8, 40, 40,     // P2
            10, 20, 250, 250, // P4
            30, 30, 150, 140, // P3
            0, 0, 255, 255);  // P4
        var align = new RoIAlign<double>(outputSize: 3, samplingRatio: 2);

        var pooled = FpnRoIPooler<double>.Pool(align, levels, Strides, boxes);
        var assignment = FpnRoIPooler<double>.AssignLevels(boxes, Strides);
        var expectedAssignment = new[] { 2, 0, 2, 1, 2 };
        Assert.Equal(expectedAssignment, assignment);

        Assert.Equal(new[] { 5, 3, 3, 3 }, Enumerable.Range(0, 4).Select(i => pooled.Shape[i]).ToArray());
        int per = 3 * 3 * 3;
        for (int b = 0; b < boxes.Shape[0]; b++)
        {
            var single = Boxes(boxes[b, 0], boxes[b, 1], boxes[b, 2], boxes[b, 3]);
            var expected = align.Forward(levels[expectedAssignment[b]], single, 1.0 / Strides[expectedAssignment[b]]);
            for (int k = 0; k < per; k++)
            {
                Assert.Equal(expected[k], pooled[b * per + k], 12);
            }
        }
    }
}
