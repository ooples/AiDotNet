using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

/// <summary>Shared behavioral probes for safe eager construction and the later virtual initialization hook.</summary>
public abstract class ConstructorInitializationTestBase
{
    protected interface IInitializationProbe
    {
        int InitializationCalls { get; }
        void ReinitializeLayers();
    }

    protected static void AssertSafeInitialization<TModel>(
        Func<NeuralNetworkArchitecture<double>, TModel> create)
        where TModel : NeuralNetworkBase<double>, IInitializationProbe
    {
        var layer = new DenseLayer<double>(4);
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 4,
            layers: [layer])
        {
            RandomSeed = 271828
        };
        using var model = create(architecture);

        // The constructor must eagerly initialize its own layers without calling a derived override.
        Assert.Equal(0, model.InitializationCalls);
        Assert.Same(layer, Assert.Single(model.Layers));
        Assert.True(model.GetParameters().Length > 0);

        var input = new Tensor<double>([4], new Vector<double>(new[] { 0.1, 0.2, 0.3, 0.4 }));
        var first = model.Predict(input);
        Assert.Equal(new[] { 4 }, first.Shape);
        Assert.All(first.ToArray(), value => Assert.True(!double.IsNaN(value) && !double.IsInfinity(value)));
        Assert.Same(layer, Assert.Single(model.Layers));

        // Extensibility remains available after derived construction is complete.
        var callsBefore = model.InitializationCalls;
        model.ReinitializeLayers();
        Assert.Equal(callsBefore + 1, model.InitializationCalls);
        Assert.Same(layer, Assert.Single(model.Layers));
        var second = model.Predict(input);
        Assert.Equal(first.Shape, second.Shape);
        Assert.Equal(first.ToArray(), second.ToArray());
        Assert.Same(layer, Assert.Single(model.Layers));
    }
}
