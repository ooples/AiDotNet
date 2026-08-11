using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Parameters;
using AiDotNet.Regression;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Parameters;

public class ParameterManifestTests
{
    [Fact]
    public async Task Registry_OrdersStorageByStableId()
    {
        await Task.Yield();
        var registry = new ParameterComponentRegistry<double>();
        var z = new Vector<double>(new[] { 2d });
        var a = new Vector<double>(new[] { 1d });
        registry.Register("z", new VectorFieldWriteThroughSource<double>(() => z));
        registry.Register("a", new VectorFieldWriteThroughSource<double>(() => a));

        var parameters = registry.GetParameters();

        Assert.Equal(new[] { 1d, 2d }, parameters.ToArray());
        Assert.Equal(new[] { "a", "z" }, registry.ParameterLayout.Slots.Select(slot => slot.StableId));
    }

    [Fact]
    public async Task Registry_RejectsDuplicateStableIdentity()
    {
        await Task.Yield();
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("owner", new VectorFieldWriteThroughSource<double>(() => new Vector<double>(1)));

        Assert.Throws<InvalidOperationException>(() =>
            registry.Register("owner", new VectorFieldWriteThroughSource<double>(() => new Vector<double>(1))));
    }

    [Fact]
    public async Task NullNumericField_IsDeferredRatherThanParameterFree()
    {
        await Task.Yield();
        Matrix<double>? matrix = null;
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("matrix", new MatrixFieldParameterSource<double>(() => matrix));

        Assert.Equal(ParameterReadiness.ShapeDeferred, registry.ParameterLayout.Readiness);
        Assert.Null(registry.ParameterLayout.ParameterCount);
        Assert.Throws<ParameterLayoutNotReadyException>(() => _ = registry.ParameterCount);
        Assert.Throws<ParameterLayoutNotReadyException>(() => registry.GetParameters());
        Assert.Throws<ParameterLayoutNotReadyException>(() => registry.SetParameters(new Vector<double>(0)));
    }

    [Fact]
    public async Task KeyedCollections_UseCanonicalKeyOrder()
    {
        await Task.Yield();
        var values = new Dictionary<string, Vector<double>>
        {
            ["z key"] = new(new[] { 2d }),
            ["a/key"] = new(new[] { 1d })
        };
        var source = new KeyedVectorCollectionParameterSource<double, string>(() => values);

        Assert.Equal(new[] { 1d, 2d }, source.GetParameters().ToArray());
        Assert.Equal(new[] { "key=a%2Fkey", "key=z%20key" },
            source.GetParameterLayout().Select(slot => slot.StableId));
    }

    [Fact]
    public async Task GeneratedAndManualRegistration_ComposeIntoOneSurface()
    {
        await Task.Yield();
        var model = new GeneratedAndManualParameterModel<double>();

        var parameters = model.GetParameters();

        Assert.Equal(3, model.ParameterCount);
        Assert.Equal(new[] { 1d, 2d, 3d }, parameters.ToArray());
        Assert.Equal(2, model.ParameterLayout.Slots.Count);
        Assert.All(model.ParameterLayout.Slots, slot => Assert.NotNull(slot.ParameterCount));
    }

    [Fact]
    public async Task NeuralNetworkManifest_DescribesTheCompletePublicSurface()
    {
        await Task.Yield();
        var network = new AiDotNet.NeuralNetworks.NeuralNetwork<double>();

        var layout = network.ParameterLayout;

        Assert.Equal(ParameterReadiness.Materialized, layout.Readiness);
        Assert.Equal(network.ParameterCount, layout.ParameterCount);
        Assert.Equal(network.ParameterCount, network.GetParameters().Length);
    }

    [Fact]
    public async Task GeneratedPolicyRegistration_HasOneCanonicalNetworkOwner()
    {
        await Task.Yield();
        var policy = new AiDotNet.ReinforcementLearning.Policies.BetaPolicy<double>();

        var parameters = policy.GetParameters();

        Assert.Equal(policy.ParameterCount, parameters.Length);
        Assert.Equal(policy.ParameterCount, policy.ParameterLayout.ParameterCount);
        Assert.Single(policy.ParameterLayout.Slots);
    }

    [Fact]
    public async Task SweepWorker_IsolatesAndMeasuresOneModel()
    {
        await Task.Yield();
#if NET10_0_OR_GREATER
        var measurement = await ParameterSweepProcess.MeasureAsync(
            typeof(HistGradientBoostingRegression<double>),
            includeChunks: false,
            maximum: 1_000_000,
            timeout: TimeSpan.FromSeconds(15));

        Assert.Equal("ok", measurement.Status);
        Assert.Equal(1, measurement.Declared);
        Assert.Equal(1, measurement.Flat);
#endif
    }
}

internal partial class GeneratedAndManualParameterModel<T> : ModelBase<T, Vector<T>, Vector<T>>
{
    private readonly Matrix<T> _generated = new(1, 2);

    [ParameterAlias("manual-component")]
    private readonly Vector<T> _manual = new(1);

    public GeneratedAndManualParameterModel()
    {
        _generated[0, 0] = NumOps.One;
        _generated[0, 1] = NumOps.FromDouble(2);
        _manual[0] = NumOps.FromDouble(3);
    }

    public override ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    public override Vector<T> Predict(Vector<T> input) => input;

    public override void Train(Vector<T> input, Vector<T> expectedOutput)
    {
    }

    protected override void RegisterComponents()
    {
        base.RegisterComponents();
        RegisterParameterComponent(
            "manual-component",
            new VectorFieldWriteThroughSource<T>(() => _manual));
    }

    public override IFullModel<T, Vector<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var copy = new GeneratedAndManualParameterModel<T>();
        copy.SetParameters(parameters);
        return copy;
    }

    public override IFullModel<T, Vector<T>, Vector<T>> DeepCopy() => WithParameters(GetParameters());
}
