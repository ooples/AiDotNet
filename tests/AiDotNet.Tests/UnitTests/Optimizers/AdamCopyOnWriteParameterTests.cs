using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Adam updates parameters in place through their managed arrays. A model snapshot - the facade's per-epoch
/// evaluation DeepCopy - shares parameter storage copy-on-write, so that in-place write must first give the
/// parameter its own storage: otherwise the update also lands in the snapshot, and once the parameter detaches, the
/// cached array Adam keeps writing belongs to the snapshot alone and the parameter stops training.
/// Measured through the facade: the flat training path with the per-epoch copy trained a different model than the
/// same run without it.
/// </summary>
public class AdamCopyOnWriteParameterTests
{
    private const int N = 8;

    private static AdamOptimizer<float, Tensor<float>, Tensor<float>> Adam() =>
        new(model: null, options: new AdamOptimizerOptions<float, Tensor<float>, Tensor<float>>
        {
            InitialLearningRate = 0.01,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            EnableGradientClipping = false,
            AnomalyGuardMode = AdamAnomalyGuardMode.Never,
        });

    private static Tensor<float> Filled(float start, float step)
    {
        var t = new Tensor<float>(new[] { N });
        for (int i = 0; i < N; i++) t[i] = start + step * i;
        return t;
    }

    private static void Step(AdamOptimizer<float, Tensor<float>, Tensor<float>> adam, Tensor<float> parameter)
    {
        var gradients = new Dictionary<Tensor<float>, Tensor<float>>(
            AiDotNet.Helpers.TensorReferenceComparer<Tensor<float>>.Instance)
        {
            [parameter] = Filled(0.5f, -0.1f),
        };
        adam.Step(new TapeStepContext<float>(new[] { parameter }, gradients, 0f));
    }

    [Fact]
    public void A_step_after_a_snapshot_leaves_the_snapshot_unchanged()
    {
        var adam = Adam();
        var parameter = Filled(1f, 0.25f);
        Step(adam, parameter);   // first step fills Adam's steady-state array cache

        var snapshot = (Tensor<float>)parameter.CloneShared();
        var snapshotValues = snapshot.ToArray();
        var before = parameter.ToArray();

        Step(adam, parameter);

        Assert.Equal(snapshotValues, snapshot.ToArray());
        Assert.NotEqual(before, parameter.ToArray());
    }

    [Fact]
    public void A_parameter_that_detached_from_a_snapshot_keeps_training()
    {
        var adam = Adam();
        var parameter = Filled(1f, 0.25f);
        Step(adam, parameter);

        var snapshot = (Tensor<float>)parameter.CloneShared();
        var snapshotValues = snapshot.ToArray();
        parameter[0] = parameter[0];   // an ordinary tensor write: the parameter takes its own storage
        var before = parameter.ToArray();

        Step(adam, parameter);

        Assert.NotEqual(before, parameter.ToArray());
        Assert.Equal(snapshotValues, snapshot.ToArray());
    }
}
