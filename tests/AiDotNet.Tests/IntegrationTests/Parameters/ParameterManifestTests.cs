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
    public async Task Registry_RequiresCanonicalNumericPathSegments()
    {
        await Task.Yield();
        var registry = new ParameterComponentRegistry<double>();

        Assert.Throws<ArgumentException>(() =>
            registry.Register("layers/2", new ContractProbeSource(1, new[] { 1d })));

        registry.Register("layers/00000010", new ContractProbeSource(1, new[] { 10d }));
        registry.Register("layers/00000002", new ContractProbeSource(1, new[] { 2d }));
        Assert.Equal(new[] { 2d, 10d }, registry.GetParameters().ToArray());
    }

    [Fact]
    public async Task LegacyIdentity_DoesNotDependOnGlobalRegistrationOrder()
    {
        await Task.Yield();
        var forward = new ParameterComponentRegistry<double>();
        forward.RegisterLegacy("Example.Model", "RegisterComponents", "_encoder",
            new ContractProbeSource(1, new[] { 1d }));
        forward.RegisterLegacy("Example.Model", "RegisterComponents", "_decoder",
            new ContractProbeSource(1, new[] { 2d }));

        var reverse = new ParameterComponentRegistry<double>();
        reverse.RegisterLegacy("Example.Model", "RegisterComponents", "_decoder",
            new ContractProbeSource(1, new[] { 2d }));
        reverse.RegisterLegacy("Example.Model", "RegisterComponents", "_encoder",
            new ContractProbeSource(1, new[] { 1d }));

        Assert.Equal(
            forward.ParameterLayout.Slots.Select(slot => slot.StableId),
            reverse.ParameterLayout.Slots.Select(slot => slot.StableId));
        Assert.Equal(forward.ParameterLayout.Fingerprint, reverse.ParameterLayout.Fingerprint);
        Assert.Equal(forward.GetParameters().ToArray(), reverse.GetParameters().ToArray());
    }

    [Fact]
    public async Task LegacyIdentity_UsesOnlyALocalPaddedIndexForRepeatedExpressions()
    {
        await Task.Yield();
        var registry = new ParameterComponentRegistry<double>();
        var first = new ContractProbeSource(1, new[] { 1d });
        registry.RegisterLegacy("Example.Model", "RegisterComponents", "layer", first);
        registry.RegisterLegacy("Example.Model", "RegisterComponents", "layer",
            new ContractProbeSource(1, new[] { 2d }));
        registry.RegisterLegacy("Example.Model", "RegisterComponents", "other",
            new ContractProbeSource(1, new[] { 3d }));
        registry.RegisterLegacy("Example.Model", "AnotherMember", "layer", first);

        var identityGroups = registry.ParameterLayout.Slots
            .Select(slot => slot.StableId)
            .GroupBy(id => id.Substring(0, id.LastIndexOf('/')))
            .OrderByDescending(group => group.Count())
            .ToArray();

        Assert.Equal(3, registry.ParameterLayout.Slots.Count);
        Assert.Equal(2, identityGroups[0].Count());
        Assert.Contains(identityGroups[0], id => id.EndsWith("/00000000", StringComparison.Ordinal));
        Assert.Contains(identityGroups[0], id => id.EndsWith("/00000001", StringComparison.Ordinal));
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
    public async Task Restore_UsesTheCapturedManifestCountRatherThanASecondSourceCount()
    {
        await Task.Yield();
        var first = new ContractProbeSource(2, new[] { 1d, 2d }, reportedParameterCount: 1);
        var second = new ContractProbeSource(1, new[] { 3d }, reportedParameterCount: 99);
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("first", first);
        registry.Register("second", second);

        registry.SetParameters(new Vector<double>(new[] { 10d, 20d, 30d }));

        Assert.Equal(new[] { 10d, 20d }, first.LastRestored);
        Assert.Equal(new[] { 30d }, second.LastRestored);
        Assert.Equal(0, first.ParameterCountReads);
        Assert.Equal(0, second.ParameterCountReads);
    }

    [Fact]
    public async Task Read_FailsBeforeReturningAFlatVectorWhoseLengthDisagreesWithTheManifest()
    {
        await Task.Yield();
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("drifted", new ContractProbeSource(2, new[] { 1d }));

        var error = Assert.Throws<global::AiDotNet.Models.Parameters.ParameterContractViolationException>(
            () => registry.GetParameters());

        Assert.Equal("drifted", error.StableId);
        Assert.Equal(2, error.ExpectedCount);
        Assert.Equal(1, error.ActualCount);
    }

    [Fact]
    public async Task Restore_GivesTheVariableTailWhateverFixedComponentsLeave()
    {
        await Task.Yield();
        var fixedSource = new ContractProbeSource(2, new[] { 1d, 2d });
        double[]? restoredTail = null;
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("fixed", fixedSource);
        registry.Register("tail", new VariableLengthParameterSource<double>(
            () => restoredTail?.Length ?? 0,
            () => new Vector<double>(restoredTail ?? Array.Empty<double>()),
            values => restoredTail = values.ToArray()));

        registry.SetParameters(new Vector<double>(new[] { 10d, 20d, 30d, 40d, 50d }));

        Assert.Equal(new[] { 10d, 20d }, fixedSource.LastRestored);
        Assert.Equal(new[] { 30d, 40d, 50d }, restoredTail);
        Assert.Equal(5, registry.ParameterCount);
    }

    [Fact]
    public async Task Restore_EmptyVariableTail_DoesNotMaterializeAnEmptyLazyPrefix()
    {
        await Task.Yield();
        var lazyNetwork = new AiDotNet.NeuralNetworks.NeuralNetwork<double>(
            new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
                AiDotNet.Enums.InputType.OneDimensional,
                AiDotNet.Enums.NeuralNetworkTaskType.Regression,
                AiDotNet.Enums.NetworkComplexity.Medium,
                inputSize: 3,
                outputSize: 2,
                layers: [new AiDotNet.NeuralNetworks.Layers.DenseLayer<double>(2)]));
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("fixed", lazyNetwork);
        registry.Register("tail", new VariableLengthParameterSource<double>(
            () => 0, () => new Vector<double>(0), values => Assert.Empty(values)));

        registry.SetParameters(new Vector<double>(0));

        Assert.Empty(lazyNetwork.GetParameters());
        Assert.Equal(0, registry.ParameterCount);
    }

    [Fact]
    public async Task Restore_RejectsAVariableComponentThatIsNotLastInStableOrder()
    {
        await Task.Yield();
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("a-variable", new VariableLengthParameterSource<double>(
            () => 0, () => new Vector<double>(0), _ => { }));
        registry.Register("z-fixed", new ContractProbeSource(1, new[] { 1d }));

        var error = Assert.Throws<InvalidOperationException>(() =>
            registry.SetParameters(new Vector<double>(new[] { 1d })));

        Assert.Contains("must be last", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Restore_RejectsMoreThanOneVariableComponent()
    {
        await Task.Yield();
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("first", new VariableLengthParameterSource<double>(
            () => 0, () => new Vector<double>(0), _ => { }));
        registry.Register("second", new VariableLengthParameterSource<double>(
            () => 0, () => new Vector<double>(0), _ => { }));

        var error = Assert.Throws<InvalidOperationException>(() =>
            registry.SetParameters(new Vector<double>(0)));

        Assert.Contains("at most one", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task LayoutSnapshot_CannotBeMutatedThroughItsPublicSlotCollection()
    {
        await Task.Yield();
        var source = new List<ParameterSlotDescriptor>
        {
            new("weight", ParameterSlotRole.Trainable, ParameterReadiness.Materialized, 1)
        };
        var snapshot = new ParameterLayoutSnapshot(source);

        source[0] = new ParameterSlotDescriptor(
            "replacement", ParameterSlotRole.Trainable, ParameterReadiness.Materialized, 1);

        Assert.Equal("weight", snapshot.Slots[0].StableId);
    }

    [Fact]
    public async Task LayoutFingerprint_ChangesWhenCheckpointOwnershipOrCountChanges()
    {
        await Task.Yield();
        var original = new ParameterLayoutSnapshot(new[]
        {
            new ParameterSlotDescriptor(
                "weight", ParameterSlotRole.Trainable, ParameterReadiness.Materialized, 2)
        });
        var renamed = new ParameterLayoutSnapshot(new[]
        {
            new ParameterSlotDescriptor(
                "renamed", ParameterSlotRole.Trainable, ParameterReadiness.Materialized, 2)
        });
        var resized = new ParameterLayoutSnapshot(new[]
        {
            new ParameterSlotDescriptor(
                "weight", ParameterSlotRole.Trainable, ParameterReadiness.Materialized, 3)
        });

        Assert.Equal(ParameterLayoutSnapshot.CurrentSchemaVersion, original.SchemaVersion);
        Assert.Matches("^[a-f0-9]{64}$", original.Fingerprint);
        Assert.NotEqual(original.Fingerprint, renamed.Fingerprint);
        Assert.NotEqual(original.Fingerprint, resized.Fingerprint);
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
    public async Task KeyedScalarCollections_RoundTripIndependentlyOfInsertionOrder()
    {
        await Task.Yield();
        var forward = new Dictionary<string, double>
        {
            ["state-z"] = 2d,
            ["state-a"] = 1d
        };
        var reverse = new Dictionary<string, double>
        {
            ["state-a"] = 10d,
            ["state-z"] = 20d
        };
        var first = new KeyedScalarCollectionParameterSource<double, string>(() => forward);
        var second = new KeyedScalarCollectionParameterSource<double, string>(() => reverse);

        Assert.Equal(new[] { 1d, 2d }, first.GetParameters().ToArray());
        Assert.Equal(
            first.GetParameterLayout().Select(slot => slot.StableId),
            second.GetParameterLayout().Select(slot => slot.StableId));

        second.SetParameters(first.GetParameters());
        Assert.Equal(1d, reverse["state-a"]);
        Assert.Equal(2d, reverse["state-z"]);
    }

    [Fact]
    public async Task NestedKeyedScalarCollections_PreserveSparseStateActionOwnership()
    {
        await Task.Yield();
        var table = new Dictionary<string, Dictionary<int, double>>
        {
            ["state-z"] = new() { [2] = 22d },
            ["state-a"] = new() { [10] = 110d, [1] = 11d }
        };
        var source = new NestedKeyedScalarCollectionParameterSource<double, string, int>(() => table);

        Assert.Equal(new[] { 11d, 110d, 22d }, source.GetParameters().ToArray());
        Assert.Equal(
            new[] { "key=state-a/key=1", "key=state-a/key=10", "key=state-z/key=2" },
            source.GetParameterLayout().Select(slot => slot.StableId));

        source.SetParameters(new Vector<double>(new[] { 1d, 10d, 2d }));
        Assert.Equal(1d, table["state-a"][1]);
        Assert.Equal(10d, table["state-a"][10]);
        Assert.Equal(2d, table["state-z"][2]);
        Assert.Throws<ArgumentException>(() =>
            source.SetParameters(new Vector<double>(new[] { 1d, 2d })));
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
    public async Task LegacyRegistration_DeduplicatesGeneratedAndManualAccessorsToTheSameComponent()
    {
        await Task.Yield();
        var component = new ContractProbeSource(1, new[] { 1d });
        var registry = new ParameterComponentRegistry<double>();
        registry.RegisterLegacy("Example.Model", "generated", "component",
            new ComponentAccessorParameterSource<double>(() => component));
        registry.RegisterLegacy("Example.Model", "manual", "component",
            new ComponentAccessorParameterSource<double>(() => component));

        Assert.Single(registry.Components);
        Assert.Single(registry.ParameterLayout.Slots);
        Assert.Equal(1, registry.ParameterCount);
    }

    [Fact]
    public async Task ExplicitRegistration_DeduplicatesPropertyAndFieldAccessorsToTheSameComponent()
    {
        await Task.Yield();
        var component = new ContractProbeSource(1, new[] { 1d });
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("model/component-property",
            new ComponentAccessorParameterSource<double>(() => component));
        registry.Register("model/component-field",
            new ComponentAccessorParameterSource<double>(() => component));

        Assert.Single(registry.Components);
        Assert.Single(registry.ParameterLayout.Slots);
        Assert.Equal(1, registry.ParameterCount);
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
    public async Task DuelingCombinationLayer_RestoreDoesNotDuplicateItsConstructionSizedTensors()
    {
        await Task.Yield();
        var source = new AiDotNet.NeuralNetworks.Layers.DuelingCombinationLayer<double>(4, 2, seed: 7);
        var target = new AiDotNet.NeuralNetworks.Layers.DuelingCombinationLayer<double>(4, 2, seed: 11);
        var parameters = source.GetParameters();

        target.SetParameters(parameters);

        Assert.Equal(parameters.Length, target.ParameterCount);
        Assert.Equal(parameters.ToArray(), target.GetParameters().ToArray());
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

internal sealed class ContractProbeSource : IParameterSource<double>, IParameterLayoutSource
{
    private readonly long _declaredCount;
    private readonly long _reportedParameterCount;
    private readonly double[] _values;

    public ContractProbeSource(long declaredCount, double[] values, long? reportedParameterCount = null)
    {
        _declaredCount = declaredCount;
        _reportedParameterCount = reportedParameterCount ?? declaredCount;
        _values = values;
    }

    public int ParameterCountReads { get; private set; }
    public double[]? LastRestored { get; private set; }

    public long ParameterCount
    {
        get
        {
            ParameterCountReads++;
            return _reportedParameterCount;
        }
    }

    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout() =>
        new[]
        {
            new ParameterSlotDescriptor(
                "$", ParameterSlotRole.Trainable, ParameterReadiness.Materialized, _declaredCount)
        };

    public Vector<double> GetParameters() => new(_values);

    public void SetParameters(Vector<double> parameters)
    {
        LastRestored = parameters.ToArray();
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
