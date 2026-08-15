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
        Assert.Equal(2, byol.ParameterLayout.Slots.Count(slot => slot.Role == ParameterSlotRole.Trainable));
        Assert.Equal(2, byol.ParameterLayout.Slots.Count(slot => slot.Role == ParameterSlotRole.Frozen));

        var simSiamProjector = new SymmetricProjector<double>(
            inputDim: 4, hiddenDim: 6, projectionDim: 3, predictorHiddenDim: 2, seed: 13);
        var simSiam = new SimSiam<double>(onlineEncoder, simSiamProjector);
        long simSiamExpected = onlineEncoder.GetParameters().Length + simSiamProjector.ParameterCount;
        Assert.Equal(simSiamExpected, simSiam.ParameterCount);
        Assert.Equal(simSiamExpected, simSiam.GetParameters().Length);
        Assert.Equal(2, simSiam.ParameterLayout.Slots.Count);
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
