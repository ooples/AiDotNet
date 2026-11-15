using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.FederatedLearning.ServerOptimizers;
using AiDotNet.FederatedLearning.Trainers;
using AiDotNet.FederatedLearning.Unlearning;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

/// <summary>
/// Half-precision ranges of the federated aggregation, server optimizer, influence solve and compression, and the
/// unlearning noise source. Half's largest finite value is 65504 and its smallest positive one about 6e-8, so squares,
/// small weights and rounded thresholds formed in Half leave the range (or move) where the double reference does not.
/// </summary>
public class FederatedHalfRangeReviewTests
{
    private static readonly INumericOperations<Half> HalfOps = MathHelper.GetNumericOperations<Half>();

    private static Half H(double value) => HalfOps.FromDouble(value);

    private static double D(Half value) => HalfOps.ToDouble(value);

    [Fact]
    public void FedYogi_InHalf_FollowsTheDoubleReferenceWhenTheSquaredDeltaOverflows()
    {
        // lr 1, b1 0.9, b2 0.99, tau 1e-3, x0 = 0 and target 300: g^2 = 90000 is past Half's 65504. Held in Half,
        // v became infinite and every step was 0. Double reference: x1 = 30 / (sqrt(900.000001) + 1e-3) = 0.9999667,
        // which Half stores as 1; then g = 299, m = 56.9, v = 1794.01 and x2 = 1 + 56.9 / (sqrt(v) + 1e-3) = 2.343351.
        var optimizer = new FedYogiServerOptimizer<Half>(learningRate: 1.0, beta1: 0.9, beta2: 0.99, epsilon: 1e-3);
        var target = new Vector<Half>(new[] { H(300) });

        var first = optimizer.Step(new Vector<Half>(new[] { H(0) }), target);
        Assert.Equal(0.9999667, D(first[0]), 3);

        var second = optimizer.Step(first, target);
        Assert.True(Math.Abs(D(second[0]) - 2.343351) < 2e-3, $"Expected 2.343351, got {D(second[0])}.");
    }

    [Fact]
    public void AsyncFedED_InHalf_KeepsAClientWhoseDecayedWeightUnderflows()
    {
        // Both clients last took part in round 0 and are aggregated in round 10 with a staleness decay of 0.01, so each
        // weight is ln 2 * 1e-20: zero in Half. Weighting in Half summed zeros and divided by a zero total (NaN);
        // the weights normalized in double are 1/2 each, so the mean is 5.
        var trainer = new AsyncFedEDTrainer<Half>(stalenessDecay: 0.01);
        trainer.UpdateClientEntropy(0, new[] { 1.0, 1.0 }, currentRound: 0);
        trainer.UpdateClientEntropy(1, new[] { 1.0, 1.0 }, currentRound: 0);
        var updates = new Dictionary<int, Dictionary<string, Half[]>>
        {
            [0] = new Dictionary<string, Half[]> { ["w"] = new[] { H(4), H(-2) } },
            [1] = new Dictionary<string, Half[]> { ["w"] = new[] { H(6), H(2) } },
        };

        var merged = trainer.AggregateWithEntropyWeights(updates, currentRound: 10);

        Assert.Equal(5.0, D(merged["w"][0]), 6);
        Assert.Equal(0.0, D(merged["w"][1]), 6);
    }

    [Fact]
    public void InfluenceSolve_InHalf_KeepsTheInnerProductsInRange()
    {
        // H = 100 I and v = [300, 400]: r.r = 250000 and p.Hp = 2.5e7 overflow Half, and alpha = inf / inf was NaN.
        // With the inner products in double, alpha = 0.01 and one step solves H x = v: x = [3, 4].
        var engine = AiDotNetEngine.Current;
        Tensor<Half> Hessian(Tensor<Half> p) => engine.TensorMultiplyScalar(p, H(100));

        var x = InfluenceFunctionUnlearner<Half>.SolveDamped(Hessian, new Tensor<Half>(new[] { H(300), H(400) }, new[] { 2 }), 0.0, 10, 1e-3);

        Assert.True(Math.Abs(D(x[0]) - 3.0) < 1e-2, $"Expected 3, got {D(x[0])}.");
        Assert.True(Math.Abs(D(x[1]) - 4.0) < 1e-2, $"Expected 4, got {D(x[1])}.");
    }

    [Fact]
    public void ThresholdSparsify_InHalf_ComparesAgainstTheExactThreshold()
    {
        // Half holds 0.2 as 0.199951171875, below the threshold 0.2; against the threshold rounded to Half it was kept.
        // (0.3 would not do: Half rounds it up, to 0.300048828125.)
        Assert.Equal(0.199951171875, D(H(0.2)));
        var delta = new Vector<Half>(new[] { H(0.2), H(0.5), H(-0.1), H(-0.75) });

        var sparse = InMemoryFederatedTrainer<Half, Matrix<Half>, Vector<Half>>.ThresholdSparsify(delta, 0.2, out int kept);

        Assert.Equal(2, kept);
        Assert.Equal(new[] { 0.0, 0.5, 0.0, -0.75 }, sparse.ToArray().Select(D).ToArray());
    }

    [Fact]
    public void StandardNormalsFromBytes_IsTheBoxMullerTransformOfTheWords()
    {
        // Words 0 and 0 give u1 = u2 = (2^31 + 1/2) / 2^32, just above 1/2: sqrt(-2 ln u1) cos(2 pi u2) = -sqrt(2 ln 2).
        var normals = UnlearningEngineOps.StandardNormalsFromBytes<double>(new byte[8], 1);

        Assert.Equal(-Math.Sqrt(2.0 * Math.Log(2.0)), normals[0], 6);
    }

    [Fact]
    public void SecureStandardNormals_HaveUnitMomentsAndDoNotRepeat()
    {
        const int count = 20000;
        var first = UnlearningEngineOps.SecureStandardNormals<double>(count).ToArray();
        var second = UnlearningEngineOps.SecureStandardNormals<double>(count).ToArray();

        double mean = first.Average();
        double variance = first.Select(z => (z - mean) * (z - mean)).Average();

        // Standard errors at n = 20000: 0.007 for the mean and 0.01 for the variance.
        Assert.InRange(mean, -0.05, 0.05);
        Assert.InRange(variance, 0.95, 1.05);
        Assert.False(first.SequenceEqual(second));
    }
}
