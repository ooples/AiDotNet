using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralRadianceFields.Data;
using AiDotNet.NeuralRadianceFields.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralRadianceFields;

/// <summary>
/// Regression test for #1835 — densification must actually FIRE during a real training run.
///
/// The existing coverage does not establish this:
///   * <c>GaussianSplattingDensificationTests</c> drives the internal
///     <c>RunDensifyAndPruneForTest</c> hook, so it proves the split/prune maths and the
///     schedule-window gates, but not that the training loop ever calls them.
///   * <c>ImageTrainingPathTests.GaussianSplatting_TrainOnImageBatch_ReturnsFiniteLoss</c>
///     explicitly sets <c>EnableDensification = false</c>.
///
/// So densification could be wired, configurable, serialized and unit-tested while never
/// running during training, and every existing test would still pass. This test closes that
/// gap by training through the real image-space path and asserting the cloud changed size.
/// </summary>
public class GaussianSplattingDensificationFiresDuringTrainingTests
{
    private static ImageView<float>[] BuildViews(int count = 2, int height = 4, int width = 4)
    {
        var views = new ImageView<float>[count];
        for (int v = 0; v < count; v++)
        {
            var photo = new float[height * width * 3];
            for (int i = 0; i < photo.Length; i++)
            {
                // Non-uniform target so the photometric loss produces a real gradient signal.
                photo[i] = (i % 7) / 7.0f;
            }

            var pose = new float[] { 0f, 0f, v * -1f };
            var rotation = new Matrix<float>(3, 3);
            rotation[0, 0] = 1f;
            rotation[1, 1] = 1f;
            rotation[2, 2] = 1f;

            views[v] = new ImageView<float>(
                new Tensor<float>([height, width, 3], new Vector<float>(photo)),
                new Vector<float>(pose),
                rotation,
                focalLength: 0f);
        }

        return views;
    }

    [Fact]
    public void TrainOnImageBatch_WithDensificationEnabled_MutatesTheGaussianCloud()
    {
        var model = new GaussianSplatting<float>(new GaussianSplattingOptions
        {
            ShDegree = 0,
            EnableSpatialIndex = false,
            EnableDensification = true,
            MaxGaussians = 64,

            // Explicit window so the run does not depend on the short-run auto-scaling that
            // #1835 added — this test is about whether the loop fires, not about scheduling.
            DensificationStartIteration = 1,
            DensificationEndIteration = 10_000,
            DensificationInterval = 1,

            // Paper-realistic prune threshold (0.005, Kerbl et al.) so prune does NOT empty the
            // cloud, paired with a near-zero split threshold so any real gradient triggers a split.
            // An emptied cloud would be a fixture artifact, not evidence about the training loop.
            OpacityPruneThreshold = 0.005,
            GradientNormThreshold = 1e-9,
        });

        int startCount = model.GaussianCount;
        Assert.True(startCount > 0, "Fixture must start with a non-empty cloud to be meaningful.");

        var loader = ImageTrainingDataLoaders.FromViews(BuildViews(), seed: 17);
        for (int step = 0; step < 8; step++)
        {
            model.TrainOnImageBatch(loader, raysPerBatch: 4, optimizerOptions: null);
        }

        // The invariant is that the cloud CHANGED — split may grow it, prune may shrink it,
        // and the ordering under the MaxGaussians cap decides which wins. A count that never
        // moved means DensifyAndPrune was never reached from the training loop.
        Assert.NotEqual(startCount, model.GaussianCount);
    }

    [Fact]
    public void TrainOnImageBatch_WithDensificationDisabled_LeavesTheCloudUntouched()
    {
        // Control arm. Identical to the test above except EnableDensification = false. If this
        // ALSO changed the count, the assertion above would be measuring ordinary training
        // churn rather than densification, and would not discriminate.
        var model = new GaussianSplatting<float>(new GaussianSplattingOptions
        {
            ShDegree = 0,
            EnableSpatialIndex = false,
            EnableDensification = false,
            MaxGaussians = 64,
            DensificationStartIteration = 1,
            DensificationEndIteration = 10_000,
            DensificationInterval = 1,
            OpacityPruneThreshold = 0.005,
            GradientNormThreshold = 1e-9,
        });

        int startCount = model.GaussianCount;

        var loader = ImageTrainingDataLoaders.FromViews(BuildViews(), seed: 17);
        for (int step = 0; step < 8; step++)
        {
            model.TrainOnImageBatch(loader, raysPerBatch: 4, optimizerOptions: null);
        }

        Assert.Equal(startCount, model.GaussianCount);
    }
}
