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

        // EVERY SLOT MUST REPORT A COUNT. `?? 0L` would turn a deferred or malformed slot into a
        // silent zero, letting it vanish from the totals below while the sums still balanced.
        Assert.All(byolSlots, slot => Assert.True(
            slot.ParameterCount.HasValue,
            $"Slot '{slot.StableId}' reports no ParameterCount; a deferred or malformed slot must " +
            "fail this test, not be counted as zero."));

        // MEMBERSHIP AND ROLE, not just uniqueness. Distinct identities prove nothing was emitted
        // twice; they do not prove the right things were emitted at all. An omission, a
        // substitution, or a role swap between two equal-sized slots would still balance the scalar
        // totals asserted below, so each branch is named and its role pinned here.
        //
        // The online encoder is an owned NeuralNetwork the manifest walks into, so it contributes
        // one slot PER LAYER; the target encoder is behind IMomentumEncoder and stays a single
        // opaque slot. Hence "one or more" for the online encoder and "exactly one" for the rest.
        var trainableIds = byolSlots.Where(slot => slot.Role == ParameterSlotRole.Trainable)
                                    .Select(slot => slot.StableId).ToArray();
        var frozenIds = byolSlots.Where(slot => slot.Role == ParameterSlotRole.Frozen)
                                 .Select(slot => slot.StableId).ToArray();

        Assert.Contains(trainableIds, id => id.Contains("::_encoder", StringComparison.Ordinal));
        Assert.Single(trainableIds.Where(id => id.EndsWith("::_projector", StringComparison.Ordinal)));
        Assert.Single(frozenIds.Where(id => id.EndsWith("::_targetEncoder", StringComparison.Ordinal)));
        Assert.Single(frozenIds.Where(id => id.EndsWith("::_targetProjector", StringComparison.Ordinal)));

        // The alias collapsed: _onlineProjector carries [ParameterAlias(nameof(_projector))] and must
        // NOT surface under its own name anywhere, in either role.
        Assert.DoesNotContain(byolSlots,
            slot => slot.StableId.Contains("_onlineProjector", StringComparison.Ordinal));

        long byolTrainableScalars = byolSlots
            .Where(slot => slot.Role == ParameterSlotRole.Trainable)
            .Sum(slot => slot.ParameterCount.Value);
        long byolFrozenScalars = byolSlots
            .Where(slot => slot.Role == ParameterSlotRole.Frozen)
            .Sum(slot => slot.ParameterCount.Value);

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

        Assert.All(simSiamSlots, slot => Assert.True(
            slot.ParameterCount.HasValue,
            $"Slot '{slot.StableId}' reports no ParameterCount; a deferred or malformed slot must " +
            "fail this test, not be counted as zero."));

        // SimSiam has no target branch, so every slot is trainable, the encoder is present, and the
        // projector appears exactly once under the aliased name.
        Assert.All(simSiamSlots, slot => Assert.Equal(ParameterSlotRole.Trainable, slot.Role));
        Assert.Contains(simSiamSlots,
            slot => slot.StableId.Contains("::_encoder", StringComparison.Ordinal));
        Assert.Single(simSiamSlots.Where(
            slot => slot.StableId.EndsWith("::_projector", StringComparison.Ordinal)));

        Assert.Equal(simSiamExpected, simSiamSlots.Sum(slot => slot.ParameterCount.Value));
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
