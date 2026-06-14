using System;
using System.Threading.Tasks;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Paper-faithful invariant tests for MobileNetV2 per Sandler et al. 2018,
/// "MobileNetV2: Inverted Residuals and Linear Bottlenecks".
/// </summary>
public class MobileNetV2NetworkTests : NeuralNetworkModelTestBase
{
    private const int Channels = 3;
    private const int Height = 32;
    private const int Width = 32;
    private const int NumClasses = 10;

    // MobileNetV2 supports both width and resolution multipliers. The test
    // uses a CIFAR-sized resolution and alpha=0.35 so all inverted-residual
    // stages are still exercised without turning CI invariants into a
    // full ImageNet training run.
    protected override int[] InputShape => [Channels, Height, Width];
    protected override int[] OutputShape => [NumClasses];

    protected override int TrainingIterations => 1;
    protected override int MoreDataShortIterations => 1;
    protected override int MoreDataLongIterations => 2;
    protected override int MemorizationTaskIterations => 4;
    protected override double MemorizationTaskLossThreshold => 0.99999;

    // The parameterless MobileNetV2Network() constructor defaults to 1000
    // ImageNet classes and 224x224 input — incompatible with this test's
    // small 3x32x32 probe and 10-class OutputShape assertion. Use the
    // architecture-aware overload so the network classifier head matches
    // the test's expected output dimension.
    protected override INeuralNetworkModel<double> CreateNetwork()
        => new MobileNetV2Network<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.ThreeDimensional,
                taskType: NeuralNetworkTaskType.MultiClassClassification,
                inputHeight: Height, inputWidth: Width, inputDepth: Channels,
                outputSize: 10),
            new MobileNetV2Configuration(
                MobileNetV2WidthMultiplier.Alpha035,
                NumClasses,
                inputHeight: Height,
                inputWidth: Width,
                inputChannels: Channels));

    // MobileNetV2 is a raw-logit classifier trained with CrossEntropyWithLogitsLoss
    // (PyTorch nn.CrossEntropyLoss semantics). The base scaffold's continuous
    // [0,1) targets are regression targets, not legal class distributions.
    protected override Tensor<double> CreateRandomTargetTensor(int[] shape, Random rng)
    {
        var target = new Tensor<double>(shape);
        int classCount = shape[^1];
        int samples = target.Length / classCount;
        for (int sample = 0; sample < samples; sample++)
        {
            int classIndex = rng.Next(classCount);
            target[sample * classCount + classIndex] = 1.0;
        }

        return target;
    }

    [Fact(Timeout = 120000)]
    public override async Task Training_ShouldReduceLoss()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        using var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);
        var target = CreateRandomTargetTensor(EffectiveOutputShape, rng);

        double initialLoss = ComputeCrossEntropy(network.Predict(input), target);

        for (int i = 0; i < TrainingIterations * 3; i++)
            network.Train(input, target);

        double finalLoss = ComputeCrossEntropy(network.Predict(input), target);

        Assert.True(finalLoss <= initialLoss + 1e-6,
            $"Cross-entropy did not reduce: initial={initialLoss:F6}, final={finalLoss:F6}. " +
            "Gradient computation or parameter update may be broken.");
    }

    [Fact(Timeout = 120000)]
    public override async Task MoreData_ShouldNotDegrade()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng1 = ModelTestHelpers.CreateSeededRandom(42);
        var rng2 = ModelTestHelpers.CreateSeededRandom(42);

        var network1 = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng1);
        var target = CreateRandomTargetTensor(EffectiveOutputShape, rng1);
        var input2 = CreateRandomTensor(InputShape, rng2);
        var target2 = CreateRandomTargetTensor(EffectiveOutputShape, rng2);

        network1.Predict(input);
        var network2 = (INeuralNetworkModel<double>)network1.Clone();

        for (int i = 0; i < MoreDataShortIterations; i++)
            network1.Train(input, target);
        double lossShort = ComputeCrossEntropy(network1.Predict(input), target);

        for (int i = 0; i < MoreDataLongIterations; i++)
            network2.Train(input2, target2);
        double lossLong = ComputeCrossEntropy(network2.Predict(input2), target2);

        Assert.True(lossLong <= lossShort + 1e-4,
            $"{MoreDataLongIterations} iterations CE loss ({lossLong:F6}) > " +
            $"{MoreDataShortIterations} iterations CE loss ({lossShort:F6}). " +
            "Optimizer may be diverging with more training.");
    }

    private static double ComputeCrossEntropy(Tensor<double> logits, Tensor<double> target)
    {
        int classCount = target.Shape[^1];
        if (classCount <= 0 || logits.Length != target.Length || target.Length % classCount != 0)
            throw new ArgumentException("Logits and target must have matching final class dimensions.");

        int samples = target.Length / classCount;
        double total = 0.0;
        for (int sample = 0; sample < samples; sample++)
        {
            int offset = sample * classCount;
            double maxLogit = logits[offset];
            for (int cls = 1; cls < classCount; cls++)
                maxLogit = Math.Max(maxLogit, logits[offset + cls]);

            double sumExp = 0.0;
            for (int cls = 0; cls < classCount; cls++)
                sumExp += Math.Exp(logits[offset + cls] - maxLogit);

            double logSumExp = Math.Log(sumExp) + maxLogit;
            for (int cls = 0; cls < classCount; cls++)
            {
                double targetValue = target[offset + cls];
                if (targetValue != 0.0)
                    total -= targetValue * (logits[offset + cls] - logSumExp);
            }
        }

        return total / samples;
    }
}
