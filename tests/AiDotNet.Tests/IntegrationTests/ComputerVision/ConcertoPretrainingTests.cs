using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.ComputerVision.Segmentation.PointCloud;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Covers Concerto's self-supervised pretraining path.
/// </summary>
/// <remarks>
/// <para>
/// The point of these tests is that pretraining actually <b>trains</b>. Before this work the two
/// objectives evaluated in raw <c>double</c> and returned a bare scalar, so a loop built on them
/// would have run, reported a loss, and left every weight untouched -- and every "does it run"
/// assertion would have passed against it. So the load-bearing assertion here is that the loss
/// falls and the parameters move, not that the call returns.
/// </para>
/// </remarks>
public class ConcertoPretrainingTests
{
    // The encoder downsamples 32x, so a 128x128 input yields a 4x4 decoder feature map: 16 spatial
    // positions, hence 16 points. A 32x32 input gives exactly ONE position, which makes the
    // cross-modal objective degenerate -- the first version of this test used it and the point
    // count silently disagreed with the feature map by three orders of magnitude.
    private const int Grid = 4;
    private const int Side = 128;
    private const int FeatureGrid = Side / 32;

    private static NeuralNetworkArchitecture<double> Arch()
        => new(InputType.ThreeDimensional, NeuralNetworkTaskType.Regression,
               NetworkComplexity.Deep, 0, Side, Side, 3, 0);

    private static Tensor<double> Rand(int seed, params int[] shape)
    {
        int total = 1;
        foreach (int s in shape) total *= s;
        var data = new double[total];
        var rng = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < total; i++) data[i] = (rng.NextDouble() * 2.0) - 1.0;
        return new Tensor<double>(shape, new Vector<double>(data));
    }

    /// <summary>
    /// Builds a sample whose cameras see the points, so the cross-modal objective has real matches
    /// rather than degenerating to the empty-match zero.
    /// </summary>
    private static ConcertoPretrainingSample<double> Sample(Concerto<double> model, int seed, int featureWidth)
    {
        // One point per spatial position of the decoder output.
        int points = FeatureGrid * FeatureGrid;

        var coords = new Tensor<double>(new[] { points, 3 });
        var depth = new Tensor<double>(new[] { FeatureGrid, FeatureGrid });
        for (int y = 0; y < FeatureGrid; y++)
        {
            for (int x = 0; x < FeatureGrid; x++)
            {
                int p = (y * FeatureGrid) + x;
                // Place each point exactly on the pixel it will project to, one metre out, so the
                // identity camera below maps it back to (x, y) and the depth check passes. Every
                // point is therefore visible and lands in its own patch, which is what gives the
                // cross-modal term real matches instead of the empty-match zero.
                coords[p, 0] = x;
                coords[p, 1] = y;
                coords[p, 2] = 1.0;
                depth[y, x] = 1.0;
            }
        }

        var intrinsics = new Tensor<double>(new[] { 3, 3 });
        intrinsics[0, 0] = 1.0; intrinsics[1, 1] = 1.0; intrinsics[2, 2] = 1.0;

        var extrinsics = new Tensor<double>(new[] { 4, 4 });
        for (int i = 0; i < 4; i++) extrinsics[i, i] = 1.0;

        var view = new ConcertoPairedView<double>
        {
            ImagePatchFeatures = Rand(seed + 1, Grid * Grid, featureWidth),
            Intrinsics = intrinsics,
            Extrinsics = extrinsics,
            DepthMap = depth,
            PatchGridSize = Grid
        };

        return new ConcertoPretrainingSample<double>
        {
            Input = Rand(seed, 1, 3, Side, Side),
            PointCoordinates = coords,
            Views = new[] { view }
        };
    }

    private static Concerto<double> Model(int epochs, int numClasses)
        => new(Arch(), options: new ConcertoOptions
        {
            NumClasses = numClasses,
            PretrainingEpochs = epochs,
            // The decoder head here is three convolutions, so level 3 is its final output and
            // level 2 the layer before it.
            IntraModalUpcastLevel = 3,
            CrossModalUpcastLevel = 3,
            // The fixture pairs a single view per cloud (the paper's default is 4). Pretrain holds
            // a sample to the configured pairing count, so the count is stated rather than left to
            // coincide with the default.
            ImagesPerPointCloud = 1,
            LearningRate = 0.01
        });

    [Fact(Timeout = 300000)]
    public async Task Pretrain_ReducesLoss()
    {
        await Task.Yield();

        int numClasses = 8;
        var model = Model(epochs: 1, numClasses: numClasses);
        var samples = new[] { Sample(model, seed: 7, featureWidth: numClasses) };

        double first = model.Pretrain(samples);
        double later = 0.0;
        for (int i = 0; i < 6; i++) later = model.Pretrain(samples);

        // double.IsFinite does not exist on net471, which this project also targets.
        Assert.True(
            !double.IsNaN(first) && !double.IsInfinity(first),
            $"First epoch loss was not finite: {first}");
        Assert.True(
            !double.IsNaN(later) && !double.IsInfinity(later),
            $"Later epoch loss was not finite: {later}");
        Assert.True(
            later < first,
            $"Pretraining did not reduce the loss: started at {first}, ended at {later}. "
            + "A loop whose objectives carry no gradient graph produces exactly this — it runs and "
            + "reports a number while the weights never move.");
    }

    [Fact(Timeout = 300000)]
    public async Task Pretrain_MovesParameters()
    {
        await Task.Yield();

        int numClasses = 8;
        var model = Model(epochs: 2, numClasses: numClasses);
        var samples = new[] { Sample(model, seed: 11, featureWidth: numClasses) };

        var before = model.GetParameters();
        var snapshot = new double[before.Length];
        for (int i = 0; i < before.Length; i++) snapshot[i] = before[i];

        model.Pretrain(samples);

        var after = model.GetParameters();
        int moved = 0;
        for (int i = 0; i < after.Length; i++)
        {
            if (Math.Abs(after[i] - snapshot[i]) > 1e-12) moved++;
        }

        Assert.True(moved > 0, "Pretraining left every parameter untouched.");
    }

    [Fact(Timeout = 300000)]
    public async Task Pretrain_RejectsMismatchedPatchWidth()
    {
        await Task.Yield();

        var model = Model(epochs: 1, numClasses: 8);
        // Paired views one channel narrower than the decoder level feeding the cross-modal loss.
        var samples = new[] { Sample(model, seed: 13, featureWidth: 7) };

        var error = Assert.Throws<ArgumentException>(() => model.Pretrain(samples));
        Assert.Contains("cosine similarity", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Timeout = 300000)]
    public async Task Pretrain_RejectsEmptySampleSet()
    {
        await Task.Yield();

        var model = Model(epochs: 1, numClasses: 8);
        Assert.Throws<ArgumentException>(() => model.Pretrain(Array.Empty<ConcertoPretrainingSample<double>>()));
    }
}
