using System.Collections.Generic;
using AiDotNet.Models.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Optimizers;

/// <summary>
/// Pins <see cref="AdamWOptimizerOptions{T, TInput, TOutput}.LearningRateMask"/>: a parameter's multiplier scales its whole
/// update (the Adam step and the decoupled decay), as a PyTorch parameter group's own learning rate does.
/// </summary>
public class AdamWLearningRateMaskTests
{
    private const double LearningRate = 0.1, WeightDecay = 0.1, Epsilon = 1e-8;

    // One step from theta = 1 with gradient 1: the Adam step is lr / (1 + eps) and the decay lr * wd.
    private static double Moved(double rate) => rate * (LearningRate / (1 + Epsilon) + LearningRate * WeightDecay);

    [Fact]
    public void FlatUpdate_ScalesEachParametersStepAndDecay()
    {
        var optimizer = new AdamWOptimizer<double, Matrix<double>, Vector<double>>(null, new AdamWOptimizerOptions<double, Matrix<double>, Vector<double>>
        {
            InitialLearningRate = LearningRate, WeightDecay = WeightDecay, Epsilon = Epsilon,
            LearningRateMask = new Vector<double>(new[] { 1.0, 0.5, 0.0 }),
        });
        var updated = optimizer.UpdateParameters(new Vector<double>(new[] { 1.0, 1.0, 1.0 }), new Vector<double>(new[] { 1.0, 1.0, 1.0 }));
        Assert.Equal(1.0 - Moved(1.0), updated[0], 12);
        Assert.Equal(1.0 - Moved(0.5), updated[1], 12);
        Assert.Equal(1.0, updated[2], 12);
    }

    [Fact]
    public void TapeStep_ScalesEachParametersStepAndDecay()
    {
        var engine = AiDotNetEngine.Current;
        var first = new Tensor<double>(new[] { 1.0, 1.0 }, new[] { 2 });
        var second = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var optimizer = new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(null, new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = LearningRate, WeightDecay = WeightDecay, Epsilon = Epsilon, UseAdaptiveLearningRate = false,
            EnableGradientClipping = false,
            LearningRateMask = new Vector<double>(new[] { 1.0, 0.25, 0.5 }),
        });
        TapeTrainingStepper<double>.EagerObjectiveStep(new List<Tensor<double>> { first, second },
            () => engine.TensorAdd(engine.ReduceSum(first, null, keepDims: false), engine.ReduceSum(second, null, keepDims: false)), optimizer);
        Assert.Equal(1.0 - Moved(1.0), first[0], 12);
        Assert.Equal(1.0 - Moved(0.25), first[1], 12);
        Assert.Equal(1.0 - Moved(0.5), second[0], 12);
    }

    [Fact]
    public void MaskParameters_FindEachSteppedParametersEntries_WhenTheStepReachesOnlySome()
    {
        // The loss uses only `second`, so the step reaches it alone; its entries are the mask's last.
        var engine = AiDotNetEngine.Current;
        var first = new Tensor<double>(new[] { 1.0, 1.0 }, new[] { 2 });
        var second = new Tensor<double>(new[] { 1.0 }, new[] { 1 });
        var optimizer = new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(null, new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = LearningRate, WeightDecay = WeightDecay, Epsilon = Epsilon, UseAdaptiveLearningRate = false,
            EnableGradientClipping = false,
            LearningRateMask = new Vector<double>(new[] { 1.0, 1.0, 0.5 }),
            WeightDecayMask = new Vector<double>(new[] { 1.0, 1.0, 1.0 }),
            MaskParameters = new[] { first, second },
        });
        TapeTrainingStepper<double>.EagerObjectiveStep(new List<Tensor<double>> { first, second },
            () => engine.ReduceSum(second, null, keepDims: false), optimizer);
        Assert.Equal(new[] { 1.0, 1.0 }, first.ToArray());
        Assert.Equal(1.0 - Moved(0.5), second[0], 12);
    }

    [Fact]
    public void Mask_RejectsNegativeEntries_AndDeclinesFusedCompilation()
    {
        Assert.Throws<System.ArgumentOutOfRangeException>(() => new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            LearningRateMask = new Vector<double>(new[] { 1.0, -0.1 }),
        });
        var optimizer = new AdamWOptimizer<double, Tensor<double>, Tensor<double>>(null, new AdamWOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            UseAdaptiveLearningRate = false, LearningRateMask = new Vector<double>(new[] { 1.0 }),
        });
        Assert.False(((AiDotNet.Optimizers.Fused.IFusedOptimizerSpec)optimizer).TryGetFusedOptimizerConfig(out _));
    }
}
