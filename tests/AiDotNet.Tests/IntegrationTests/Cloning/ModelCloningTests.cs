using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Cloning;

/// <summary>
/// Proves a model is rebuilt through its own constructor and carries its learned weights.
/// </summary>
/// <remarks>
/// These assert the two properties that make a clone worth having, and that the previous
/// scalar-count backstop could not establish: the copy predicts identically to the original
/// (so its parameters landed in the right slots), and mutating the copy does not reach the
/// original (so nothing is shared by reference).
/// </remarks>
public class ModelCloningTests
{
    private const int InDim = 4;

    private static FeedForwardNeuralNetwork<float> BuildModel()
    {
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(InDim),
            new DenseLayer<float>(6, activationFunction: new ReLUActivation<float>()),
            new DenseLayer<float>(1, activationFunction: new IdentityActivation<float>()),
        };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: InDim,
            outputSize: 1,
            layers: layers);

        return new FeedForwardNeuralNetwork<float>(
            architecture, lossFunction: new MeanSquaredErrorLoss<float>());
    }

    private static Tensor<float> SampleInput()
    {
        var x = new Tensor<float>(new[] { 1, InDim });
        for (int i = 0; i < InDim; i++) x[0, i] = 0.25f * (i + 1);
        return x;
    }

    [Fact]
    public void Clone_ProducesAnIndependentModelWithTheSamePredictions()
    {
        var source = BuildModel();
        var input = SampleInput();
        var expected = source.Predict(input);

        var clone = (NeuralNetworkBase<float>)source.Clone();

        Assert.NotNull(clone);
        Assert.NotSame(source, clone);
        Assert.IsType<FeedForwardNeuralNetwork<float>>(clone);

        var actual = clone.Predict(input);
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], 5);
        }
    }

    [Fact]
    public void Clone_DoesNotShareParameterStorageWithTheOriginal()
    {
        var source = BuildModel();
        var clone = (NeuralNetworkBase<float>)source.Clone();

        var before = source.GetParameters();
        var mutated = new Vector<float>(before.Length);
        for (int i = 0; i < before.Length; i++) mutated[i] = before[i] + 1.0f;

        clone.UpdateParameters(mutated);

        // The original must be untouched. A clone that aliases storage passes a predictions
        // check and then corrupts the model it was copied from the first time either is trained.
        var after = source.GetParameters();
        Assert.Equal(before.Length, after.Length);
        for (int i = 0; i < before.Length; i++)
        {
            Assert.Equal(before[i], after[i], 6);
        }
    }

    [Fact]
    public void DeepCopy_ProducesAnIndependentModelWithTheSamePredictions()
    {
        var source = BuildModel();
        var input = SampleInput();
        var expected = source.Predict(input);

        var copy = (NeuralNetworkBase<float>)source.DeepCopy();

        Assert.NotNull(copy);
        Assert.NotSame(source, copy);
        var actual = copy.Predict(input);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i], actual[i], 5);
        }
    }
}
