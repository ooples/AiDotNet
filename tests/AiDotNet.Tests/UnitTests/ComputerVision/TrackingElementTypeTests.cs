using AiDotNet.Augmentation.Image;
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
}
