using AiDotNet.Augmentation.Image;
using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;
using AiDotNet.ComputerVision.Tracking;
using AiDotNet.ComputerVision.Visualization;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>
/// Element-type reference tests for the tracking and visualization code converted to engine ops: gates that must
/// survive Half rounding, index maps that must stay exact past 2048, and constant tables that must be computed in
/// double and rounded once.
/// </summary>
public class TrackingElementTypeTests
{
    [Fact(Timeout = 60000)]
    public async Task MinCostMatching_Half_RejectsAPairBeyondTheGate()
    {
        await Task.CompletedTask;

        // deep_sort min_cost_matching: cost 0.5 > max_distance 0.2, so the pair is not a match. In Half the cap
        // max_distance + 1e-5 rounds back to 0.2, so a rejection test on the capped cost accepted the pair.
        var cost = new Tensor<Half>(new[] { (Half)0.5 }, new[] { 1, 1 });

        var (matches, unmatchedRows, unmatchedCols) = TrackingAssociation<Half>.MinCostMatching(cost, (Half)0.2);

        Assert.Empty(matches);
        Assert.Equal(new[] { 0 }, unmatchedRows.ToArray());
        Assert.Equal(new[] { 0 }, unmatchedCols.ToArray());
    }

    [Fact(Timeout = 60000)]
    public async Task MinCostMatching_Half_KeepsAPairInsideTheGate()
    {
        await Task.CompletedTask;
        var cost = new Tensor<Half>(new[] { (Half)0.125 }, new[] { 1, 1 });

        var (matches, unmatchedRows, unmatchedCols) = TrackingAssociation<Half>.MinCostMatching(cost, (Half)0.2);

        Assert.Equal(new[] { (0, 0) }, matches.Select(m => (m.Row, m.Col)).ToArray());
        Assert.Empty(unmatchedRows);
        Assert.Empty(unmatchedCols);
    }

    [Fact(Timeout = 60000)]
    public async Task ReIdProjection_Float_IsTheDoubleTableRoundedOnce()
    {
        await Task.CompletedTask;

        // projection[f, c] = sin((f + 1) * c * 0.1). With 64 channels the angle reaches 806; forming it in float
        // (0.1f, then the product) moves it by about 1e-5, far more than one float rounding of the sine.
        const int featureDim = 128;
        const int channels = 64;
        var projection = ReIDNetwork<float>.Projection(featureDim, channels);

        Assert.Equal(new[] { featureDim, channels }, projection.Shape.ToArray());
        for (int f = 0; f < featureDim; f++)
        {
            for (int c = 0; c < channels; c++)
            {
                double expected = Math.Sin((f + 1) * c * 0.1);
                Assert.True(Math.Abs(expected - projection[f, c]) <= 1.2e-7,
                    $"projection[{f}, {c}] = {projection[f, c]}, expected {expected}.");
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task InstanceIdMap_Half_KeepsInstanceIndicesPast2048()
    {
        await Task.CompletedTask;

        // 2052 one-pixel instances all covering the pixel: the last one (index 2051) wins. 2051 is not a Half, so a
        // map held in Half reported 2052.
        const int count = 2052;
        var result = new InstanceSegmentationResult<Half>();
        for (int i = 0; i < count; i++)
        {
            var box = new BoundingBox<Half>((Half)0, (Half)0, (Half)1, (Half)1, BoundingBoxFormat.XYXY);
            var mask = new Tensor<Half>(new[] { (Half)1 }, new[] { 1, 1 });
            result.Instances.Add(new InstanceMask<Half>(box, mask, 0, (Half)1));
        }

        var ids = new MaskVisualizer<Half>().CreateInstanceIdMap(result, 1, 2);

        Assert.Equal(count - 1, ids[0, 0]);
        Assert.Equal(-1, ids[0, 1]);
    }

    [Fact(Timeout = 60000)]
    public async Task Sort_Half_BoxAreaPastHalfMaxKeepsTheTrackedBox()
    {
        await Task.CompletedTask;

        // A 300 x 300 box has area 90000, past Half's 65504. Held in a Half Kalman state the area was infinite, the
        // predicted box NaN, the track dropped and the box re-created as track 2. With the filter state in double
        // the same detection twice keeps track 1 at (0, 0, 300, 300).
        var tracker = new SORT<Half>(new TrackingOptions<Half> { MinHits = 1 });
        Detection<Half> Box() => new(
            new BoundingBox<Half>((Half)0, (Half)0, (Half)300, (Half)300, BoundingBoxFormat.XYXY), 0, (Half)0.9);

        tracker.Update(new List<Detection<Half>> { Box() });
        var track = Assert.Single(tracker.Update(new List<Detection<Half>> { Box() }).Tracks);

        Assert.Equal(1, track.TrackId);
        Assert.Equal(0.0, (double)track.Box.X1, 0.5);
        Assert.Equal(0.0, (double)track.Box.Y1, 0.5);
        Assert.Equal(300.0, (double)track.Box.X2, 0.5);
        Assert.Equal(300.0, (double)track.Box.Y2, 0.5);
    }

    [Fact(Timeout = 60000)]
    public async Task IouMatrix_Half_LargeBoxesGiveAFiniteIou()
    {
        await Task.CompletedTask;

        // Boxes of 300 x 300 overlapping by half: IoU = 45000 / 135000 = 1/3. The Half areas were infinite and the
        // IoU NaN.
        var a = new Tensor<Half>(new[] { (Half)0, (Half)0, (Half)300, (Half)300 }, new[] { 1, 4 });
        var b = new Tensor<Half>(new[] { (Half)150, (Half)0, (Half)450, (Half)300 }, new[] { 1, 4 });

        var iou = TrackingAssociation<Half>.IouMatrix(a, b);

        Assert.Equal((double)(Half)(1.0 / 3.0), (double)iou[0, 0]);
    }

    [Fact(Timeout = 60000)]
    public async Task DeepSortNormalizeRows_ZeroDescriptorStaysZero()
    {
        await Task.CompletedTask;

        // A black crop gives a zero descriptor: 0 / 0 made its cosine costs NaN and the assignment solver threw.
        // The zero row stays zero (cosine distance 1 to anything); other rows are unit length.
        var rows = new Tensor<float>(new[] { 0f, 0f, 3f, 4f }, new[] { 2, 2 });

        var normalized = DeepSORT<float>.NormalizeRows(rows);

        Assert.Equal(new[] { 0f, 0f, 0.6f, 0.8f }, normalized.ToArray());
    }
}
