using AiDotNet.Helpers;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Training;

/// <summary>
/// The model's gradient clip covers its layer-owned parameters only. A custom objective that also trains model-owned
/// parameters is not fused, and the eager step it falls back to must clip just the covered gradients.
/// </summary>
public class ScopedGradientClipTests
{
    private static Tensor<double> Scalar(double value) => new(new[] { 1 }, new Vector<double>(new[] { value }));

    [Fact]
    public void A_scoped_clip_scales_only_the_covered_gradients_by_their_own_norm()
    {
        var layerOwned = Scalar(0);
        var modelOwned = Scalar(0);
        var gradients = new Dictionary<Tensor<double>, Tensor<double>>(TensorReferenceComparer<Tensor<double>>.Instance)
        {
            [layerOwned] = Scalar(3),
            [modelOwned] = Scalar(4),
        };

        TapeTrainingStepper<double>.ClipByGlobalNorm(gradients, 1.0, new[] { layerOwned });

        // The covered gradient has norm 3 and is scaled to the limit; the other neither enters the norm nor is scaled.
        Assert.Equal(1.0, gradients[layerOwned][0], 5);
        Assert.Equal(4.0, gradients[modelOwned][0], 12);
    }

    [Fact]
    public void An_unscoped_clip_still_scales_every_gradient_by_the_global_norm()
    {
        var first = Scalar(0);
        var second = Scalar(0);
        var gradients = new Dictionary<Tensor<double>, Tensor<double>>(TensorReferenceComparer<Tensor<double>>.Instance)
        {
            [first] = Scalar(3),
            [second] = Scalar(4),
        };

        TapeTrainingStepper<double>.ClipByGlobalNorm(gradients, 1.0);

        Assert.Equal(0.6, gradients[first][0], 5);
        Assert.Equal(0.8, gradients[second][0], 5);
    }

    [Fact]
    public void A_scope_whose_gradients_are_within_the_limit_changes_nothing()
    {
        var layerOwned = Scalar(0);
        var modelOwned = Scalar(0);
        var gradients = new Dictionary<Tensor<double>, Tensor<double>>(TensorReferenceComparer<Tensor<double>>.Instance)
        {
            [layerOwned] = Scalar(0.5),
            [modelOwned] = Scalar(40),
        };

        TapeTrainingStepper<double>.ClipByGlobalNorm(gradients, 1.0, new[] { layerOwned });

        Assert.Equal(0.5, gradients[layerOwned][0], 12);
        Assert.Equal(40.0, gradients[modelOwned][0], 12);
    }
}
