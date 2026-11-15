using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.FederatedLearning.ServerOptimizers;
using AiDotNet.FederatedLearning.Trainers;
using AiDotNet.FederatedLearning.Unlearning;
using AiDotNet.FederatedLearning.Verification;
using AiDotNet.FederatedLearning.Vertical;
using AiDotNet.Models.Options;
using Xunit;

namespace AiDotNet.Tests.FederatedLearning;

public class FederatedUnresolvedReviewTests
{
    private static Half H(double x) => MathHelper.GetNumericOperations<Half>().FromDouble(x);
    private static double D(Half x) => MathHelper.GetNumericOperations<Half>().ToDouble(x);
    private static Tensor<Half> Flat(params double[] x) => new Tensor<Half>(x.Select(H).ToArray(), new[] { x.Length });
    private static Dictionary<string, Half[]> Model(double x) => new() { ["w"] = new[] { H(x) } };
    private static Dictionary<int, Dictionary<string, Half[]>> Clients(int count) =>
        Enumerable.Range(0, count).ToDictionary(i => i, _ => Model(1));

    [Fact]
    public void Yogi_AddsAnOutOfRangeStepBeforeNarrowing()
    {
        var optimizer = new FedYogiServerOptimizer<Half>(100000, 0, 0, 1e-8);
        var result = optimizer.Step(new Vector<Half>(new[] { H(-60000) }), new Vector<Half>(new[] { H(0) }));
        Assert.Equal(40000, D(result[0]));
    }

    [Theory]
    [InlineData(OneShotAggregationMode.WeightedAverage)]
    [InlineData(OneShotAggregationMode.UniformAverage)]
    public void OneShot_ManyIdenticalHalfClients_HaveUnitMean(OneShotAggregationMode mode)
    {
        var clients = Clients(4096);
        var result = new OneShotFederatedTrainer<Half>(aggregationMode: mode)
            .Aggregate(clients, clients.Keys.ToDictionary(i => i, _ => 1));
        Assert.Equal(1, D(result["w"][0]));
    }

    [Fact]
    public void AsyncFedED_ManyIdenticalHalfClients_HaveUnitMean()
    {
        var clients = Clients(4096);
        var trainer = new AsyncFedEDTrainer<Half>();
        foreach (int id in clients.Keys) trainer.UpdateClientEntropy(id, new[] { 1.0, 1.0 }, 0);
        var result = trainer.AggregateWithEntropyWeights(clients, 0);
        Assert.Equal(1, D(result["w"][0]));
    }

    [Fact]
    public void OneShot_DiversitySubtractsInDouble()
    {
        var trainer = new OneShotFederatedTrainer<Half>(aggregationMode: OneShotAggregationMode.EnsembleDistillation);
        var result = trainer.Aggregate(new() { [0] = Model(-49152), [1] = Model(49152) }, new() { [0] = 1, [1] = 3 });
        Assert.Equal(24576, D(result["w"][0]));
        Assert.Equal(3019898880.0, trainer.LastEnsembleDiversity);
    }

    [Fact]
    public void SemiAsync_PreservesARepresentableDiscountedUpdate()
    {
        var trainer = new SemiAsyncFederatedTrainer<Half>();
        for (int i = 0; i < 200; i++) trainer.AdvanceRound();
        trainer.ReceiveUpdate(0, Model(60000), 0);
        var result = trainer.ApplyAsyncUpdates(Model(0));
        double expected = D(H(60000 * 0.1 * Math.Pow(0.9, 200)));
        Assert.True(expected > 0);
        Assert.Equal(expected, D(result["w"][0]));
    }

    [Fact]
    public void SemiAsync_BarrierKeepsUnitMeanForManyHalfClients()
    {
        var clients = Clients(4095);
        var result = new SemiAsyncFederatedTrainer<Half>().SynchronizationBarrier(
            Model(1), clients, clients.Keys.ToDictionary(i => i, _ => 1));
        Assert.Equal(1, D(result["w"][0]));
    }

    [Fact]
    public void ClusterMean_StreamsWideContributions()
    {
        var members = Enumerable.Range(0, 4096).Select(_ => (new Vector<Half>(new[] { H(1) }), 1.0)).ToList();
        var mean = InMemoryFederatedTrainer<Half, Matrix<Half>, Vector<Half>>.WeightedMean(members, members.Count);
        Assert.Equal(1, D(mean[0]));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(-1.0)]
    public void PFedMe_RejectsInvalidRatesAndMuBeforeCallingTheGradient(double value)
    {
        for (int property = 0; property < 3; property++)
        {
            var options = new FederatedPersonalizationOptions();
            if (property == 0) options.PFedMeMu = value;
            else if (property == 1) options.PFedMePersonalLearningRate = value;
            else options.PFedMeLearningRate = value;
            bool called = false;
            Assert.Throws<ArgumentOutOfRangeException>(() => InMemoryFederatedTrainer<double, Matrix<double>, Vector<double>>.PFedMeLocalUpdate(
                p => { called = true; return p; }, new Vector<double>(new[] { 1.0 }), options, 1, out _));
            Assert.False(called);
        }
    }

    [Fact]
    public void UnlearningMeanAndInfluence_DoNotOverflowHalf()
    {
        Assert.Equal(40000, D(UnlearningEngineOps.MeanFitted(new[] { Flat(40000), Flat(40000) }, 1)[0]));
        Assert.Equal(1, D(UnlearningEngineOps.InfluenceScores(new[] { Flat(1000), Flat(2000) }, 1)[0]));
    }

    [Fact]
    public void NativeFisher_KeepsSmallInverseCorrection()
    {
        var unlearner = new InfluenceFunctionUnlearner<Half>(new FederatedUnlearningOptions());
        var (model, _) = unlearner.Unlearn(0, Flat(0), new() { [0] = new() { Flat(1) }, [1] = new() { Flat(1000) } });
        Assert.Equal(D(H(-0.5 / 1000000.01)), D(model[0]));
        Assert.True(D(model[0]) < 0);
    }

    [Theory]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void InfluenceSolve_RejectsInvalidCurvature(double curvature)
    {
        var unlearner = new InfluenceFunctionUnlearner<double>(new FederatedUnlearningOptions(),
            (_, v) => AiDotNetEngine.Current.TensorMultiplyScalar(v, curvature));
        Assert.Throws<InvalidOperationException>(() => unlearner.Unlearn(0,
            new Tensor<double>(new[] { 0.0 }, new[] { 1 }),
            new() { [0] = new() { new Tensor<double>(new[] { 1.0 }, new[] { 1 }) } }));
    }

    [Fact]
    public void HashAggregation_UsesTheExactFloatOperandsInDouble()
    {
        var scheme = new HashCommitmentScheme<float>();
        Tensor<float> F(float x) => new Tensor<float>(new[] { x }, new[] { 1 });
        var parts = new[] { scheme.Commit(F(16777216)), scheme.Commit(F(1)), scheme.Commit(F(-16777216)) };
        Assert.True(scheme.VerifyAggregation(parts, scheme.Commit(F(1))));
        Assert.False(scheme.VerifyAggregation(parts, scheme.Commit(F(0))));
    }

    [Fact]
    public void ProxyVerifier_DoesNotRoundTheDoubleElementLimitUp()
    {
        var verifier = new ProxyZKPVerifier<float>(maxElementMagnitude: 0.99999999);
        var update = new Dictionary<string, float[]> { ["w"] = new[] { 1f } };
        var nonce = new byte[32];
        Assert.False(verifier.Verify(update, verifier.ComputeCommitment(update, nonce), nonce).IsValid);
    }

    [Fact]
    public void LabelClipping_UsesTheViewAndPreservesTinyScaleProducts()
    {
        var dp = new LabelDifferentialPrivacy<float>(epsilon: 1e12, seed: 1);
        var view = AiDotNetEngine.Current.TensorNarrow(new Tensor<float>(new[] { 0f, 10f }, new[] { 2, 1 }), 0, 1, 1);
        Assert.InRange(dp.ProtectGradients(view)[0, 0], 0.99999f, 1.00001f);
        var half = new LabelDifferentialPrivacy<Half>(epsilon: 1e12, maxGradientNorm: 0.001, seed: 1);
        Assert.Equal(D(H(0.001)), D(half.ProtectGradients(Flat(65504))[0]));
    }

    [Fact]
    public void MissingFeatureMean_IsFiniteWhenTheBatchSumExceedsHalf()
    {
        var handler = new MissingFeatureHandler<Half>(new MissingFeatureOptions { Strategy = MissingFeatureStrategy.Mean });
        handler.UpdateStatistics("p", new Tensor<Half>(new[] { H(40000), H(40000) }, new[] { 2, 1 }));
        Assert.Equal(4000, D(handler.ImputeEmbeddings("p", 1, 1)[0, 0]));
    }

    [Fact]
    public void VerticalInfluenceAndDivergence_SubtractBeforeNarrowing()
    {
        var unlearner = new VerticalFederatedUnlearner<Half>();
        Assert.Equal(80000, unlearner.ComputeInfluence(Flat(1), Flat(0), Flat(40000)));
        Assert.Equal(80000, unlearner.VerifyUnlearning(Flat(-40000), Flat(40000)).DivergenceScore);
    }

    [Fact]
    public void VerticalMse_KeepsWideDifferenceAndLoss()
    {
        var holder = new VerticalPartyLabelHolder<Half>("p", new Tensor<Half>(new[] { H(1) }, new[] { 1, 1 }), Flat(0), new[] { "a" }, embeddingDimension: 1);
        var (loss, gradient) = holder.ComputeLoss(Flat(40000, 40000, 40000, 40000), Flat(-40000, -40000, -40000, -40000));
        Assert.Equal(6400000000, loss);
        Assert.All(gradient.ToArray(), x => Assert.Equal(40000, D(x)));
    }

    [Fact]
    public void VerticalUpdate_AnnouncesItsMutationToTensorConsumers()
    {
        var parameters = new Tensor<double>(new[] { 4.0, 8.0 }, new[] { 2 });
        long before = parameters.Version;
        VerticalEngineOps.UpdateInPlace(parameters, new Tensor<double>(new[] { 2.0, 4.0 }, new[] { 2 }), 0.5);
        Assert.Equal(new[] { 3.0, 6.0 }, parameters.ToArray());
        Assert.True(parameters.Version > before);
    }

    [Theory]
    [InlineData(1e-100)]
    [InlineData(1e100)]
    public void PFedMe_RejectsRatesOutsideTheModelNumericRange(double rate)
    {
        var options = new FederatedPersonalizationOptions { PFedMePersonalLearningRate = rate };
        Assert.Throws<ArgumentOutOfRangeException>(() => InMemoryFederatedTrainer<Half, Matrix<Half>, Vector<Half>>.PFedMeLocalUpdate(
            p => p, new Vector<Half>(new[] { H(1) }), options, 1, out _));
    }
}
