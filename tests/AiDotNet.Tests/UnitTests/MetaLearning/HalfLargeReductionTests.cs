using System;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.MetaLearning;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.Models;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.Helpers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Counts and means over more than 2048 elements under <see cref="Half"/>: past 2048 a sum of a 0/1 mask in Half is
/// no longer exact, and past 65504 a sum overflows, so the meta-learning reductions count with an integer op and
/// take means as sums of value / count.
/// </summary>
public class HalfLargeReductionTests
{
    private sealed class AccuracyProbe : MetaLearnerBase<Half, Tensor<Half>, Tensor<Half>>
    {
        public AccuracyProbe()
            : base(new MockFullModel<Half>(), new MeanSquaredErrorLoss<Half>(), new MetaLearnerOptionsBase<Half>())
        {
        }

        public override MetaLearningAlgorithmType AlgorithmType => MetaLearningAlgorithmType.MAML;

        public override Half MetaTrain(TaskBatch<Half, Tensor<Half>, Tensor<Half>> taskBatch) =>
            throw new NotSupportedException();

        public override IModel<Tensor<Half>, Tensor<Half>, ModelMetadata<Half>> Adapt(
            IMetaLearningTask<Half, Tensor<Half>, Tensor<Half>> task) => throw new NotSupportedException();

        public Half Accuracy(Tensor<Half> predictions, Tensor<Half> labels) => ComputeAccuracy(predictions, labels);
    }

    private static Tensor<Half> Filled(int length, double value, int changedFrom = -1, double changed = 0.0)
    {
        var values = new Half[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = (Half)(changedFrom >= 0 && i >= changedFrom ? changed : value);
        }

        return new Tensor<Half>(values, new[] { length });
    }

    [Fact]
    public void ComputeAccuracy_CountsMatchesExactlyPastTheHalfIntegerLimit()
    {
        // 4097 of 4099 class indices match. In Half, 4097 rounds to 4096 and 4099 to 4100, so a count kept in T
        // gives 4096 / 4100 = 0.99902; the exact accuracy 4097 / 4099 rounds to 0.99951 (1 - 2^-11).
        const int length = 4099;
        var labels = Filled(length, 2.0);
        var predictions = Filled(length, 2.0, changedFrom: 4097, changed: 3.0);

        Half accuracy = new AccuracyProbe().Accuracy(predictions, labels);

        Assert.Equal((Half)(4097.0 / 4099.0), accuracy);
    }

    [Fact]
    public void MeanClampedRatio_DoesNotOverflowHalfForLargeFeatureVectors()
    {
        // 40000 entries whose ratio is 2: the ratios sum to 80000, beyond Half's 65504, so a sum-then-divide
        // returns infinity. The mean is 2. The tolerance is Half's own rounding: 1/40000 is a Half subnormal
        // (about 0.1% off), and Half's spacing near 2 is 2^-10.
        const int length = 40000;
        var raw = Filled(length, 2.0).ToVector();
        var adapted = Filled(length, 4.0).ToVector();

        var factor = ParameterModulation.MeanClampedRatio(AiDotNetEngine.Current, raw, adapted, 0.1, 3.0, 1e-3);

        Assert.NotNull(factor);
        if (factor is not null)
        {
            double mean = (double)factor[0];
            Assert.False(double.IsInfinity(mean) || double.IsNaN(mean), $"The mean is {mean}.");
            Assert.Equal(2.0, mean, 2);
        }
    }

    [Fact]
    public void MeanClampedRatio_AveragesOnlyThePassingEntriesPastTheHalfIntegerLimit()
    {
        // 3001 entries pass the threshold (ratio 3 for the first 1001, ratio 1 for the next 2000) and 2000 entries
        // have raw = 0 and are skipped: the mean is (3 * 1001 + 2000) / 3001 = 1.66711.
        const int passing = 3001;
        const int length = passing + 2000;
        var rawValues = new Half[length];
        var adaptedValues = new Half[length];
        for (int i = 0; i < length; i++)
        {
            rawValues[i] = (Half)(i < passing ? 1.0 : 0.0);
            adaptedValues[i] = (Half)(i < 1001 ? 3.0 : 1.0);
        }

        var factor = ParameterModulation.MeanClampedRatio(AiDotNetEngine.Current,
            new Vector<Half>(rawValues), new Vector<Half>(adaptedValues), 0.1, 5.0, 1e-3);

        Assert.NotNull(factor);
        if (factor is not null)
        {
            Assert.Equal((3.0 * 1001 + 2000) / passing, (double)factor[0], 2);
        }
    }
}
