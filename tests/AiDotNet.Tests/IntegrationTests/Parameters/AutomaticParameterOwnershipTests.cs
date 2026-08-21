using AiDotNet.Enums;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.CausalInference;
using AiDotNet.Models.Parameters;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.SelfSupervisedLearning;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Parameters;

/// <summary>End-to-end coverage for generator-owned parameter surfaces migrated from manual overrides.</summary>
public class AutomaticParameterOwnershipTests
{
    [Fact]
    public async Task FitSizedCausalTensors_RestoreIntoFreshInstancesWithoutManualOverrides()
    {
        await Task.Yield();
        var features = new Matrix<double>(8, 2);
        var treatment = new Vector<double>(8);
        var outcome = new Vector<double>(8);
        for (int row = 0; row < features.Rows; row++)
        {
            features[row, 0] = row * 0.2;
            features[row, 1] = 1.0 - row * 0.1;
            treatment[row] = row % 2;
            outcome[row] = 0.4 * features[row, 0] - 0.2 * features[row, 1] + 1.5 * treatment[row];
        }

        var sLearner = new SLearner<double>(maxIterations: 3);
        sLearner.Fit(features, treatment, outcome);
        AssertFreshRestoreRoundTrip(sLearner, new SLearner<double>(maxIterations: 3));

        var xLearner = new XLearner<double>(maxIterations: 3);
        xLearner.Fit(features, treatment, outcome);
        AssertFreshRestoreRoundTrip(xLearner, new XLearner<double>(maxIterations: 3));
    }

    [Fact]
    public async Task SelfSupervisedAliasesAndOwnershipRoles_AreRepresentedExactlyOnce()
    {
        await Task.Yield();
        var onlineEncoder = BuildEncoder();
        var targetEncoderNetwork = BuildEncoder();
        Materialize(onlineEncoder);
        Materialize(targetEncoderNetwork);

        var onlineProjector = new SymmetricProjector<double>(
            inputDim: 4, hiddenDim: 6, projectionDim: 3, predictorHiddenDim: 2, seed: 7);
        var targetProjector = new SymmetricProjector<double>(
            inputDim: 4, hiddenDim: 6, projectionDim: 3, predictorHiddenDim: 0, seed: 11);
        var byol = new BYOL<double>(
            onlineEncoder,
            new MomentumEncoder<double>(targetEncoderNetwork),
            onlineProjector,
            targetProjector);

        long expected = onlineEncoder.GetParameters().Length
            + targetEncoderNetwork.GetParameters().Length
            + onlineProjector.ParameterCount
            + targetProjector.ParameterCount;
        Assert.Equal(expected, byol.ParameterCount);
        Assert.Equal(expected, byol.GetParameters().Length);
        // BY ROLE IN SCALARS, PLUS "EXACTLY ONCE" AS DISTINCT SLOT IDENTITIES -- not a slot count.
        // Measured: BYOL reports 5 slots, not 4. The online encoder is an owned NeuralNetwork, so
        // the manifest walks INTO it and emits one slot per layer (_encoder/layers/00000000 = 36,
        // _encoder/layers/00000001 = 28), while the target encoder sits behind IMomentumEncoder and
        // stays opaque as a single _targetEncoder slot (64). A slot count therefore measures how
        // deeply each branch happens to be walked, which is not what this test is about; it only
        // read 2 while both branches were opaque.
        //
        // What the test IS about is its own name: every aliased parameter appears exactly once, and
        // ownership roles land on the right branch. _onlineProjector carries
        // [ParameterAlias(nameof(_projector))] and does correctly collapse to a single _projector
        // slot -- now verified by distinct identity rather than inferred from a total.
        var byolSlots = byol.ParameterLayout.Slots;
        Assert.Equal(byolSlots.Count, byolSlots.Select(slot => slot.StableId).Distinct().Count());

        long byolTrainableScalars = byolSlots
            .Where(slot => slot.Role == ParameterSlotRole.Trainable)
            .Sum(slot => slot.ParameterCount ?? 0L);
        long byolFrozenScalars = byolSlots
            .Where(slot => slot.Role == ParameterSlotRole.Frozen)
            .Sum(slot => slot.ParameterCount ?? 0L);

        Assert.Equal(
            onlineEncoder.GetParameters().Length + onlineProjector.ParameterCount,
            byolTrainableScalars);
        Assert.Equal(
            targetEncoderNetwork.GetParameters().Length + targetProjector.ParameterCount,
            byolFrozenScalars);

        var simSiamProjector = new SymmetricProjector<double>(
            inputDim: 4, hiddenDim: 6, projectionDim: 3, predictorHiddenDim: 2, seed: 13);
        var simSiam = new SimSiam<double>(onlineEncoder, simSiamProjector);
        long simSiamExpected = onlineEncoder.GetParameters().Length + simSiamProjector.ParameterCount;
        Assert.Equal(simSiamExpected, simSiam.ParameterCount);
        Assert.Equal(simSiamExpected, simSiam.GetParameters().Length);

        // Same reason as BYOL above: SimSiam's encoder is the same owned two-layer NeuralNetwork, so
        // it contributes one slot per layer rather than one slot in total. Assert on identity
        // uniqueness and total scalars, both of which hold regardless of walk depth.
        var simSiamSlots = simSiam.ParameterLayout.Slots;
        Assert.Equal(simSiamSlots.Count, simSiamSlots.Select(slot => slot.StableId).Distinct().Count());
        Assert.Equal(simSiamExpected, simSiamSlots.Sum(slot => slot.ParameterCount ?? 0L));
    }

    private static void AssertFreshRestoreRoundTrip<TModel>(TModel trained, TModel fresh)
        where TModel : IParameterSource<double>
    {
        var parameters = trained.GetParameters();
        Assert.Equal(trained.ParameterCount, parameters.Length);

        fresh.SetParameters(parameters);

        Assert.Equal(parameters.Length, fresh.ParameterCount);
        Assert.Equal(parameters.ToArray(), fresh.GetParameters().ToArray());
    }

    private static NeuralNetwork<double> BuildEncoder()
    {
        var layers = new List<ILayer<double>>
        {
            new DenseLayer<double>(6, activationFunction: new ReLUActivation<double>()),
            new DenseLayer<double>(4, activationFunction: (IActivationFunction<double>?)null),
        };
        return new NeuralNetwork<double>(new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 5,
            outputSize: 4,
            layers: layers));
    }

    private static void Materialize(NeuralNetwork<double> encoder)
    {
        _ = encoder.Predict(new Tensor<double>(new[] { 1, 5 }));
    }
}
