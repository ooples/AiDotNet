using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Parameters;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Parameters;

public class ParameterManifestTests
{
    private sealed class SerializedTreeState
    {
        public int Feature { get; set; }
        public double Threshold { get; set; }
        public List<SerializedTreeState>? Children { get; set; }
    }

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
    public async Task Restore_GivesAnUnresolvedDeclaredVectorTailTheUnambiguousRemainder()
    {
        await Task.Yield();
        Vector<double>? tail = null;
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("fixed", new ContractProbeSource(2, new[] { 1d, 2d }));
        registry.Register(
            "tail",
            new VectorFieldParameterSource<double>(() => tail, replacement => tail = replacement),
            ParameterSlotRole.LearnedState,
            ParameterAvailability.Fit);

        Assert.Null(registry.ParameterLayout.ParameterCount);

        registry.SetParameters(new Vector<double>(new[] { 10d, 20d, 30d, 40d }));

        Assert.Equal(new[] { 30d, 40d }, tail!.ToArray());
        Assert.Equal(4, registry.ParameterCount);
    }

    [Fact]
    public async Task Restore_ReplaceableVectorLearnsItsWidthOnceThenEnforcesItExactly()
    {
        await Task.Yield();
        var value = Vector<double>.Empty();
        var registry = new ParameterComponentRegistry<double>();
        registry.Register(
            "deferred",
            new VectorFieldParameterSource<double>(() => value, replacement => value = replacement),
            ParameterSlotRole.LearnedState,
            ParameterAvailability.Fit);

        registry.SetParameters(new Vector<double>(new[] { 1d, 2d, 3d }));

        Assert.Equal(new[] { 1d, 2d, 3d }, value.ToArray());
        Assert.Equal(3, registry.ParameterCount);
        Assert.Throws<ArgumentException>(() =>
            registry.SetParameters(new Vector<double>(new[] { 4d, 5d })));
    }

    [Fact]
    public async Task SemanticRolesRatherThanNumericCountControlOptimizerEligibility()
    {
        await Task.Yield();
        var learnedStateOnly = new ParameterComponentRegistry<double>();
        learnedStateOnly.Register(
            "threshold",
            new ContractProbeSource(1, new[] { 0.5d }),
            ParameterSlotRole.LearnedState,
            ParameterAvailability.Fit);

        Assert.True(learnedStateOnly.HasPrimaryParameterComponents);
        Assert.False(learnedStateOnly.HasOptimizerUpdatableComponents);
        Assert.False(learnedStateOnly.CanInitializeOptimizerParameters);

        learnedStateOnly.Register(
            "weights",
            new ContractProbeSource(2, new[] { 1d, 2d }),
            ParameterSlotRole.Trainable);

        Assert.True(learnedStateOnly.HasOptimizerUpdatableComponents);
        Assert.True(learnedStateOnly.CanInitializeOptimizerParameters);
    }

    [Fact]
    public async Task OptimizerEligibility_RequiresResolvedTrainableShape()
    {
        await Task.Yield();
        Vector<double>? deferred = null;
        var registry = new ParameterComponentRegistry<double>();
        registry.Register(
            "weights",
            new VectorFieldParameterSource<double>(
                () => deferred,
                replacement => deferred = replacement),
            ParameterSlotRole.Trainable,
            ParameterAvailability.ShapeResolution);

        Assert.True(registry.HasOptimizerUpdatableComponents);
        Assert.False(registry.CanInitializeOptimizerParameters);

        deferred = new Vector<double>(3);
        Assert.True(registry.CanInitializeOptimizerParameters);
    }

    [Fact]
    public async Task Restore_EmptyVariableTail_RejectsAMissingShapeResolvedPrefix()
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

        var error = Assert.Throws<ArgumentException>(() =>
            registry.SetParameters(new Vector<double>(0)));

        Assert.Contains("Expected at least 8 parameters", error.Message, StringComparison.Ordinal);
        Assert.Equal(8, lazyNetwork.GetParameters().Length);
    }

    [Fact]
    public async Task Restore_CanonicallyPlacesVariableComponentAfterFixedStableIds()
    {
        await Task.Yield();
        double[] restoredTail = Array.Empty<double>();
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("a-variable", new VariableLengthParameterSource<double>(
            () => restoredTail.Length,
            () => new Vector<double>(restoredTail),
            values => restoredTail = values.ToArray()));
        registry.Register("z-fixed", new ContractProbeSource(1, new[] { 1d }));

        Assert.Equal(new[] { "z-fixed", "a-variable" },
            registry.ParameterLayout.Slots.Select(slot => slot.StableId));

        registry.SetParameters(new Vector<double>(new[] { 1d, 2d, 3d }));
        Assert.Equal(new[] { 2d, 3d }, restoredTail);
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

        Assert.Contains("at most one resizable", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task MatchingRestore_UsesStableIdsForMultipleVariableComponents()
    {
        await Task.Yield();
        double[] first = Array.Empty<double>();
        double[] second = Array.Empty<double>();
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("first", new VariableLengthParameterSource<double>(
            () => first.Length,
            () => new Vector<double>(first),
            values => first = values.ToArray()));
        registry.Register("second", new VariableLengthParameterSource<double>(
            () => second.Length,
            () => new Vector<double>(second),
            values => second = values.ToArray()));
        var checkpointLayout = new ParameterLayoutSnapshot(new[]
        {
            new ParameterSlotDescriptor(
                "first", ParameterSlotRole.LearnedState, ParameterReadiness.Materialized,
                parameterCount: 2, offset: 0, shape: new[] { 2 }),
            new ParameterSlotDescriptor(
                "second", ParameterSlotRole.LearnedState, ParameterReadiness.Materialized,
                parameterCount: 1, offset: 2, shape: new[] { 1 })
        });

        registry.SetMatchingParameters(
            new Vector<double>(new[] { 10d, 11d, 20d }), checkpointLayout);

        Assert.Equal(new[] { 10d, 11d }, first);
        Assert.Equal(new[] { 20d }, second);
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
    public async Task SerializedFittedGraph_RoundTripsThroughOneGeneratedParameterSource()
    {
        await Task.Yield();
        SerializedTreeState? state = new()
        {
            Feature = 3,
            Threshold = 1.25,
            Children = [new SerializedTreeState { Feature = 7, Threshold = -0.5 }]
        };
        var source = new SerializedObjectParameterSource<double>(
            () => state,
            value => state = (SerializedTreeState?)value,
            typeof(SerializedTreeState));

        var parameters = source.GetParameters();
        Assert.True(parameters.Length > 0);
        Assert.Equal(parameters.Length, source.ParameterCount);

        state = null;
        Assert.True(source.CanResizeOnRestore);
        source.SetParameters(parameters);

        Assert.NotNull(state);
        Assert.Equal(3, state.Feature);
        Assert.Equal(1.25, state.Threshold);
        var child = Assert.Single(state.Children!);
        Assert.Equal(7, child.Feature);
        Assert.Equal(-0.5, child.Threshold);
    }

    [Fact]
    public async Task LayoutSnapshot_PreservesKnownSubtotalWhileAnotherSlotIsDeferred()
    {
        await Task.Yield();
        var snapshot = new ParameterLayoutSnapshot(new[]
        {
            new ParameterSlotDescriptor(
                "resolved", ParameterSlotRole.Trainable,
                ParameterReadiness.ShapeResolvedUnmaterialized, 12),
            new ParameterSlotDescriptor(
                "deferred", ParameterSlotRole.Trainable,
                ParameterReadiness.ShapeDeferred, null)
        });

        Assert.Null(snapshot.ParameterCount);
        Assert.Equal(12, snapshot.KnownParameterCount);
    }

    [Fact]
    public void LayoutSnapshot_RestorableCountAddsKnownAndLiveDeferredSlots()
    {
        var snapshot = new ParameterLayoutSnapshot(new[]
        {
            new ParameterSlotDescriptor(
                "known-lazy", ParameterSlotRole.Trainable,
                ParameterReadiness.ShapeResolvedUnmaterialized, 12),
            new ParameterSlotDescriptor(
                "live-deferred", ParameterSlotRole.Trainable,
                ParameterReadiness.ShapeDeferred, null,
                materializedParameterCount: 3)
        });

        Assert.Null(snapshot.DeclaredParameterCount);
        Assert.Equal(12, snapshot.KnownParameterCount);
        Assert.Equal(3, snapshot.MaterializedParameterCount);
        Assert.Equal(15, snapshot.RestorableParameterCount);
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
    public async Task LayoutFingerprint_DistinguishesReadinessWithIdenticalIdentityAndCount()
    {
        await Task.Yield();
        var unmaterialized = new ParameterLayoutSnapshot(new[]
        {
            new ParameterSlotDescriptor(
                "weight", ParameterSlotRole.Trainable,
                ParameterReadiness.ShapeResolvedUnmaterialized, 12)
        });
        var materialized = new ParameterLayoutSnapshot(new[]
        {
            new ParameterSlotDescriptor(
                "weight", ParameterSlotRole.Trainable,
                ParameterReadiness.Materialized, 12)
        });

        Assert.NotEqual(unmaterialized.Fingerprint, materialized.Fingerprint);
        Assert.Equal(unmaterialized.DeclaredLayoutFingerprint,
            materialized.DeclaredLayoutFingerprint);
    }

    [Fact]
    public void DeclaredLayoutFingerprint_StillRejectsSemanticOrShapeChanges()
    {
        static ParameterLayoutSnapshot Snapshot(
            ParameterSlotRole role,
            int[] shape,
            ParameterOwnership ownership = ParameterOwnership.Owned) => new(new[]
        {
            new ParameterSlotDescriptor(
                "weight", role, ParameterReadiness.Materialized, 12,
                shape: shape, elementType: "System.Single", ownership: ownership)
        });

        var baseline = Snapshot(ParameterSlotRole.Trainable, new[] { 2, 6 });

        Assert.NotEqual(baseline.DeclaredLayoutFingerprint,
            Snapshot(ParameterSlotRole.Buffer, new[] { 2, 6 }).DeclaredLayoutFingerprint);
        Assert.NotEqual(baseline.DeclaredLayoutFingerprint,
            Snapshot(ParameterSlotRole.Trainable, new[] { 3, 4 }).DeclaredLayoutFingerprint);
        Assert.NotEqual(baseline.DeclaredLayoutFingerprint,
            Snapshot(ParameterSlotRole.Trainable, new[] { 2, 6 }, ParameterOwnership.Alias)
                .DeclaredLayoutFingerprint);
    }

    [Fact]
    public void LayoutSnapshot_SeparatesDeclaredCapacityFromMaterializedStorage()
    {
        var snapshot = new ParameterLayoutSnapshot(new[]
        {
            new ParameterSlotDescriptor(
                "live", ParameterSlotRole.Trainable, ParameterReadiness.Materialized, 3),
            new ParameterSlotDescriptor(
                "lazy", ParameterSlotRole.Trainable,
                ParameterReadiness.ShapeResolvedUnmaterialized, 12)
        });

        Assert.Equal(15, snapshot.DeclaredParameterCount);
        Assert.Equal(3, snapshot.MaterializedParameterCount);
        Assert.Equal(ParameterReadiness.ShapeResolvedUnmaterialized, snapshot.Readiness);
    }

    [Fact]
    public void LayoutSlot_RejectsMaterializedStorageBeyondDeclaredCapacity()
    {
        Assert.Throws<ArgumentException>(() => new ParameterSlotDescriptor(
            "weight", ParameterSlotRole.Trainable, ParameterReadiness.Materialized, 3,
            materializedParameterCount: 4));
    }

    [Fact]
    public async Task LayoutSnapshot_RejectsDuplicateStableIdentity()
    {
        await Task.Yield();
        var error = Assert.Throws<ArgumentException>(() => new ParameterLayoutSnapshot(new[]
        {
            new ParameterSlotDescriptor(
                "weight", ParameterSlotRole.Trainable, ParameterReadiness.Materialized, 2),
            new ParameterSlotDescriptor(
                "weight", ParameterSlotRole.Buffer, ParameterReadiness.Materialized, 2)
        }));

        Assert.Contains("duplicate stable identity", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task LayoutFingerprint_DistinguishesEqualCountDifferentShapeAndElementType()
    {
        await Task.Yield();
        var twoBySix = new ParameterLayoutSnapshot(new[]
        {
            new ParameterSlotDescriptor(
                "weight", ParameterSlotRole.Trainable, ParameterReadiness.Materialized, 12,
                shape: new[] { 2, 6 }, elementType: "System.Double")
        });
        var threeByFour = new ParameterLayoutSnapshot(new[]
        {
            new ParameterSlotDescriptor(
                "weight", ParameterSlotRole.Trainable, ParameterReadiness.Materialized, 12,
                shape: new[] { 3, 4 }, elementType: "System.Double")
        });
        var floats = new ParameterLayoutSnapshot(new[]
        {
            new ParameterSlotDescriptor(
                "weight", ParameterSlotRole.Trainable, ParameterReadiness.Materialized, 12,
                shape: new[] { 2, 6 }, elementType: "System.Single")
        });

        Assert.NotEqual(twoBySix.Fingerprint, threeByFour.Fingerprint);
        Assert.NotEqual(twoBySix.Fingerprint, floats.Fingerprint);
    }

    [Fact]
    public async Task LayoutFingerprint_DistinguishesOrthogonalStateSemantics()
    {
        await Task.Yield();
        static ParameterLayoutSnapshot Snapshot(
            ParameterUpdatePolicy update,
            ParameterPersistence persistence,
            ParameterOwnership ownership,
            ParameterAvailability availability) => new(new[]
        {
            new ParameterSlotDescriptor(
                "state", ParameterSlotRole.Trainable, ParameterReadiness.Materialized, 4,
                shape: new[] { 2, 2 }, elementType: "System.Double",
                updatePolicy: update, persistence: persistence, ownership: ownership,
                availability: availability)
        });

        var baseline = Snapshot(ParameterUpdatePolicy.Optimizer,
            ParameterPersistence.Persistent, ParameterOwnership.Owned,
            ParameterAvailability.Construction);

        Assert.NotEqual(baseline.Fingerprint, Snapshot(ParameterUpdatePolicy.Fit,
            ParameterPersistence.Persistent, ParameterOwnership.Owned,
            ParameterAvailability.Construction).Fingerprint);
        Assert.NotEqual(baseline.Fingerprint, Snapshot(ParameterUpdatePolicy.Optimizer,
            ParameterPersistence.Transient, ParameterOwnership.Owned,
            ParameterAvailability.Construction).Fingerprint);
        Assert.NotEqual(baseline.Fingerprint, Snapshot(ParameterUpdatePolicy.Optimizer,
            ParameterPersistence.Persistent, ParameterOwnership.Alias,
            ParameterAvailability.Construction).Fingerprint);
        Assert.NotEqual(baseline.Fingerprint, Snapshot(ParameterUpdatePolicy.Optimizer,
            ParameterPersistence.Persistent, ParameterOwnership.Owned,
            ParameterAvailability.ShapeResolution).Fingerprint);
    }

    [Fact]
    public async Task Registry_PreservesChildShapeTypeAndDeclaredAvailability()
    {
        await Task.Yield();
        Tensor<double>? tensor = new(new[] { 2, 3 });
        var registry = new ParameterComponentRegistry<double>();
        registry.Register(
            "model/weight",
            new TensorFieldParameterSource<double>(() => tensor),
            ParameterSlotRole.Trainable,
            ParameterAvailability.ShapeResolution);

        var slot = Assert.Single(registry.ParameterLayout.Slots);
        Assert.Equal(new[] { 2, 3 }, slot.Shape);
        Assert.Equal(typeof(double).FullName, slot.ElementType);
        Assert.Equal(ParameterAvailability.ShapeResolution, slot.Availability);
        Assert.Equal(ParameterUpdatePolicy.Optimizer, slot.UpdatePolicy);
        Assert.Equal(ParameterPersistence.Persistent, slot.Persistence);
        Assert.Equal(ParameterOwnership.Owned, slot.Ownership);
    }

    [Fact]
    public async Task Lifecycle_SeparatesShapeFitConditionalAndAbsentBufferState()
    {
        await Task.Yield();
        Tensor<double>? shapeValue = null;
        Tensor<double>? fitValue = null;
        Tensor<double>? optionalValue = null;
        Tensor<double>? bufferValue = null;

        var shape = new ParameterComponentRegistry<double>();
        shape.Register("shape", new TensorFieldParameterSource<double>(() => shapeValue),
            ParameterSlotRole.Trainable, ParameterAvailability.ShapeResolution);
        Assert.Equal(ParameterReadiness.ShapeDeferred, shape.ParameterLayout.Readiness);
        Assert.Throws<ParameterLayoutNotReadyException>(() => _ = shape.ParameterCount);

        var fitted = new ParameterComponentRegistry<double>();
        fitted.Register("fit", new TensorFieldParameterSource<double>(() => fitValue),
            ParameterSlotRole.LearnedState, ParameterAvailability.Fit);
        Assert.Equal(ParameterReadiness.FitDeferred, fitted.ParameterLayout.Readiness);
        var fitError = Assert.Throws<ParameterLayoutNotReadyException>(() => fitted.GetParameters());
        Assert.Contains("fit", fitError.Message, StringComparison.Ordinal);

        var optional = new ParameterComponentRegistry<double>();
        optional.Register("optional", new TensorFieldParameterSource<double>(() => optionalValue),
            ParameterSlotRole.Trainable, ParameterAvailability.Conditional);
        Assert.Equal(ParameterReadiness.ConditionalAbsent, optional.ParameterLayout.Readiness);
        Assert.Equal(0, optional.ParameterCount);
        Assert.Empty(optional.GetParameters());
        optional.SetParameters(new Vector<double>(0));

        var buffer = new ParameterComponentRegistry<double>();
        buffer.Register("training-data", new TensorFieldParameterSource<double>(() => bufferValue),
            ParameterSlotRole.Buffer, ParameterAvailability.Fit);
        Assert.Equal(ParameterReadiness.FitDeferred, buffer.ParameterLayout.Readiness);
        Assert.Null(buffer.ParameterLayout.ParameterCount);
        Assert.Throws<ParameterLayoutNotReadyException>(() => _ = buffer.ParameterCount);
        Assert.Throws<ParameterLayoutNotReadyException>(() => buffer.GetParameters());
    }

    [Fact]
    public async Task TensorFieldRestore_RejectsShortLongAndNullDestinations()
    {
        await Task.Yield();
        Tensor<double>? tensor = new(new[] { 2, 2 });
        var source = new TensorFieldParameterSource<double>(() => tensor);

        Assert.Throws<ArgumentException>(() => source.SetParameters(new Vector<double>(3)));
        Assert.Throws<ArgumentException>(() => source.SetParameters(new Vector<double>(5)));

        tensor = null;
        Assert.Throws<ParameterLayoutNotReadyException>(() =>
            source.SetParameters(new Vector<double>(4)));
    }

    [Fact]
    public async Task ResizableTensorFieldRestore_ResolvesOneDeferredAxisThenBecomesStrict()
    {
        await Task.Yield();
        Tensor<double>? tensor = new(new[] { 5, 0 });
        var source = new ResizableTensorFieldParameterSource<double>(
            () => tensor, value => tensor = value);
        var restored = new Vector<double>(Enumerable.Range(1, 15).Select(value => (double)value).ToArray());

        Assert.True(source.CanResizeOnRestore);
        Assert.Equal(ParameterReadiness.ShapeDeferred, Assert.Single(source.GetParameterLayout()).Readiness);

        source.SetParameters(restored);

        Assert.NotNull(tensor);
        Assert.Equal(new[] { 5, 3 }, tensor.Shape.ToArray());
        Assert.Equal(restored.ToArray(), source.GetParameters().ToArray());
        Assert.False(source.CanResizeOnRestore);
        Assert.Equal(ParameterReadiness.Materialized, Assert.Single(source.GetParameterLayout()).Readiness);
        Assert.Throws<ArgumentException>(() => source.SetParameters(new Vector<double>(10)));
    }

    [Fact]
    public async Task MatchingRestore_UsesStableIdsWhenAnEarlierSlotChangesShape()
    {
        await Task.Yield();
        Tensor<double>? changed = new(new[] { 2 });
        Tensor<double>? unchanged = new(new[] { 2 });
        changed[0] = -1;
        changed[1] = -2;
        unchanged[0] = -3;
        unchanged[1] = -4;

        var registry = new ParameterComponentRegistry<double>();
        registry.Register("changed", new TensorFieldParameterSource<double>(() => changed));
        registry.Register("unchanged", new TensorFieldParameterSource<double>(() => unchanged));

        var checkpointLayout = new ParameterLayoutSnapshot(new[]
        {
            new ParameterSlotDescriptor(
                "changed", ParameterSlotRole.Trainable, ParameterReadiness.Materialized,
                parameterCount: 3, offset: 0, shape: new[] { 3 }),
            new ParameterSlotDescriptor(
                "unchanged", ParameterSlotRole.Trainable, ParameterReadiness.Materialized,
                parameterCount: 2, offset: 3, shape: new[] { 2 })
        });
        var checkpoint = new Vector<double>(new[] { 10d, 11d, 12d, 20d, 21d });

        registry.SetMatchingParameters(checkpoint, checkpointLayout);

        Assert.Equal(new[] { -1d, -2d }, changed.ToArray());
        Assert.Equal(new[] { 20d, 21d }, unchanged.ToArray());
    }

    [Fact]
    public async Task MatchingRestore_RejectsInvalidGeometryBeforeMutatingAnySource()
    {
        await Task.Yield();
        Tensor<double>? first = new(new[] { 2 });
        Tensor<double>? second = new(new[] { 2 });
        first[0] = -1;
        first[1] = -2;
        second[0] = -3;
        second[1] = -4;

        var registry = new ParameterComponentRegistry<double>();
        registry.Register("first", new TensorFieldParameterSource<double>(() => first));
        registry.Register("second", new TensorFieldParameterSource<double>(() => second));
        var malformed = new ParameterLayoutSnapshot(new[]
        {
            new ParameterSlotDescriptor(
                "first", ParameterSlotRole.Trainable, ParameterReadiness.Materialized,
                parameterCount: 2, offset: 0, shape: new[] { 2 }),
            new ParameterSlotDescriptor(
                "second", ParameterSlotRole.Trainable, ParameterReadiness.Materialized,
                parameterCount: 2, offset: 3, shape: new[] { 2 })
        });

        var error = Assert.Throws<ArgumentException>(() =>
            registry.SetMatchingParameters(new Vector<double>(new[] { 10d, 11d, 12d, 13d }), malformed));

        Assert.Contains("second", error.Message, StringComparison.Ordinal);
        Assert.Equal(new[] { -1d, -2d }, first.ToArray());
        Assert.Equal(new[] { -3d, -4d }, second.ToArray());
    }

    [Fact]
    public async Task ResizableTensorFieldRestore_RejectsAmbiguousOrNonDivisibleShapes()
    {
        await Task.Yield();
        Tensor<double>? ambiguous = new(new[] { 0, 0 });
        var ambiguousSource = new ResizableTensorFieldParameterSource<double>(
            () => ambiguous, value => ambiguous = value);
        Assert.Throws<ParameterLayoutNotReadyException>(() =>
            ambiguousSource.SetParameters(new Vector<double>(6)));

        Tensor<double>? nonDivisible = new(new[] { 5, 0 });
        var nonDivisibleSource = new ResizableTensorFieldParameterSource<double>(
            () => nonDivisible, value => nonDivisible = value);
        Assert.Throws<ArgumentException>(() =>
            nonDivisibleSource.SetParameters(new Vector<double>(12)));
    }

    [Fact]
    public async Task MatrixAndVectorFieldRestore_RequireExactLengths()
    {
        await Task.Yield();
        Matrix<double>? matrix = new(2, 3);
        Vector<double>? vector = new(4);
        var matrixSource = new MatrixFieldParameterSource<double>(() => matrix);
        var vectorSource = new VectorFieldWriteThroughSource<double>(() => vector);

        Assert.Throws<ArgumentException>(() => matrixSource.SetParameters(new Vector<double>(5)));
        Assert.Throws<ArgumentException>(() => matrixSource.SetParameters(new Vector<double>(7)));
        Assert.Throws<ArgumentException>(() => vectorSource.SetParameters(new Vector<double>(3)));
        Assert.Throws<ArgumentException>(() => vectorSource.SetParameters(new Vector<double>(5)));

        matrix = null;
        vector = null;
        Assert.Throws<ParameterLayoutNotReadyException>(() =>
            matrixSource.SetParameters(new Vector<double>(6)));
        Assert.Throws<ParameterLayoutNotReadyException>(() =>
            vectorSource.SetParameters(new Vector<double>(4)));
    }

    [Fact]
    public async Task ComponentCollectionRestore_ValidatesTotalBeforeMutatingAnyMember()
    {
        await Task.Yield();
        var first = new ContractProbeSource(2, new[] { 1d, 2d });
        var second = new ContractProbeSource(1, new[] { 3d });
        var members = new IParameterSource<double>[] { first, second };
        var source = new ComponentCollectionParameterSource<double>(() => members);

        Assert.Throws<ArgumentException>(() => source.SetParameters(new Vector<double>(2)));

        Assert.Null(first.LastRestored);
        Assert.Null(second.LastRestored);
    }

    [Fact]
    public async Task ComponentCollection_PreservesDeclaredMemberLayoutAcrossReadAndRestore()
    {
        await Task.Yield();
        var member = new ContractProbeSource(
            declaredCount: 3,
            values: new[] { 1d, 2d, 3d },
            reportedParameterCount: 2);
        var source = new ComponentCollectionParameterSource<double>(() => new[] { member });

        var slot = Assert.Single(source.GetParameterLayout());
        Assert.Equal("index=00000000", slot.StableId);
        Assert.Equal(3, slot.ParameterCount);
        Assert.Equal(3, source.ParameterCount);
        Assert.Equal(new[] { 1d, 2d, 3d }, source.GetParameters().ToArray());

        source.SetParameters(new Vector<double>(new[] { 4d, 5d, 6d }));
        Assert.Equal(new[] { 4d, 5d, 6d }, member.LastRestored);
    }

    [Fact]
    public async Task LayerManifest_DoesNotDuplicateRegisteredSubLayerParameters()
    {
        await Task.Yield();
        var layer = new BidirectionalLayer<double>(
            new RecurrentLayer<double>(8),
            activationFunction: (IActivationFunction<double>?)null);
        var input = Tensor<double>.CreateRandom(2, 3, 4);

        layer.Forward(input);

        Assert.Equal(layer.GetParameters().Length, layer.ParameterCount);
        Assert.Equal(
            layer.GetParameters().Length,
            layer.GetParameterLayout().Sum(slot => slot.ParameterCount ?? 0));
    }

    [Fact]
    public async Task LayerManifest_ResolvesDeclaredCompositeChildShapesBeforeValueRead()
    {
        await Task.Yield();
        using var layer = new TransformerEncoderBlock<double>(
            hiddenSize: 8,
            numHeads: 2,
            ffnDim: 16);

        var layout = layer.GetParameterLayout();

        Assert.DoesNotContain(layout, slot =>
            slot.Readiness == ParameterReadiness.ShapeDeferred || !slot.ParameterCount.HasValue);
        Assert.Contains(layout, slot =>
            slot.Readiness == ParameterReadiness.ShapeResolvedUnmaterialized);
        Assert.Equal(
            layout.Sum(slot => slot.ParameterCount!.Value),
            layer.GetParameters().Length);
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
    public async Task GeneratedRegistration_ComposesInheritedAndDerivedModelFields()
    {
        await Task.Yield();
        var model = new VARMAModel<double>();

        var stableIds = model.ParameterLayout.Slots.Select(slot => slot.StableId).ToArray();

        Assert.Contains(stableIds, id => id.Contains(
            "VectorAutoRegressionModel<T>::_coefficients", StringComparison.Ordinal));
        Assert.Contains(stableIds, id => id.Contains(
            "VARMAModel<T>::_maCoefficients", StringComparison.Ordinal));
        Assert.Equal(6, stableIds.Length);
        Assert.Equal(model.ParameterCount, model.GetParameters().Length);
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
    public async Task Describe_PreparesEveryComponentBeforeCapturingOneTransactionalLayout()
    {
        await Task.Yield();
        var deferred = new AlwaysDeferredLifecycleSource();
        var resolved = new LifecycleProbeSource();
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("a-deferred", deferred);
        registry.Register("z-resolved", resolved);

        var error = Assert.Throws<ParameterLayoutNotReadyException>(() => _ = registry.ParameterCount);

        Assert.Equal(1, deferred.DescribeCount);
        Assert.Equal(1, resolved.DescribeCount);
        Assert.Equal(0, resolved.AllocationCount);
        Assert.Equal(
            ParameterReadiness.ShapeDeferred,
            Assert.Single(error.Layout.Slots, slot => slot.StableId == "a-deferred").Readiness);
        Assert.All(
            error.Layout.Slots.Where(slot => slot.StableId.StartsWith("z-resolved/", StringComparison.Ordinal)),
            slot =>
            {
                Assert.Equal(ParameterReadiness.ShapeResolvedUnmaterialized, slot.Readiness);
                Assert.NotNull(slot.ParameterCount);
            });
    }

    [Fact]
    public async Task GeneratedComponentLifecycle_DescribeIsAllocationFreeAndReadIsIdempotent()
    {
        await Task.Yield();
        var component = new LifecycleProbeSource();
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("owner", new ComponentAccessorParameterSource<double>(() => component));

        var described = registry.ParameterLayout;
        Assert.Equal(3, registry.ParameterCount);
        Assert.Equal(ParameterReadiness.ShapeResolvedUnmaterialized, described.Readiness);
        Assert.Equal(0, component.AllocationCount);
        Assert.Equal(new[] { "owner/running", "owner/weight" },
            described.Slots.Select(slot => slot.StableId).OrderBy(id => id, StringComparer.Ordinal));
        Assert.Equal(ParameterSlotRole.Buffer,
            Assert.Single(described.Slots, slot => slot.StableId == "owner/running").Role);
        Assert.Equal(ParameterSlotRole.Trainable,
            Assert.Single(described.Slots, slot => slot.StableId == "owner/weight").Role);

        var first = registry.GetParameters();
        var second = registry.GetParameters();

        Assert.Equal(new[] { 1d, 2d, 3d }, first.ToArray());
        Assert.Equal(first.ToArray(), second.ToArray());
        Assert.Equal(1, component.AllocationCount);
        Assert.Equal(3, registry.ParameterCount);
    }

    [Fact]
    public async Task GeneratedComponentLifecycle_RestorePreparesDestinationBeforeApplyingValues()
    {
        await Task.Yield();
        var component = new LifecycleProbeSource();
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("owner", new ComponentAccessorParameterSource<double>(() => component));

        registry.SetParameters(new Vector<double>(new[] { 4d, 5d, 6d }));

        Assert.Equal(1, component.AllocationCount);
        Assert.Equal(1, component.RestoreCount);
        Assert.Equal(new[] { 4d, 5d, 6d }, registry.GetParameters().ToArray());
    }

    [Fact]
    public async Task GeneratedComponentLifecycle_ConcurrentReadsMaterializeOnceAndStayStable()
    {
        await Task.Yield();
        var component = new LifecycleProbeSource();
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("owner", new ComponentAccessorParameterSource<double>(() => component));

        var reads = await Task.WhenAll(Enumerable.Range(0, 12)
            .Select(_ => Task.Run(() => registry.GetParameters().ToArray())));

        Assert.Equal(1, component.AllocationCount);
        Assert.All(reads, values => Assert.Equal(new[] { 1d, 2d, 3d }, values));
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
    public void GeneratedFixedParameterView_IsStableAndAllocationFreeAfterWarmup()
    {
        using var layer = new AiDotNet.NeuralNetworks.Layers.DenseLayer<double>(3);
        _ = layer.Forward(new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(new[] { 1, 4 }));

        var first = layer.GetTrainableParameters();
        var second = layer.GetTrainableParameters();
        Assert.Same(first, second);

#if NET5_0_OR_GREATER
        long before = GC.GetAllocatedBytesForCurrentThread();
#endif
        for (int i = 0; i < 1_024; i++)
            _ = layer.GetTrainableParameters();
#if NET5_0_OR_GREATER
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.True(allocated <= 128,
            $"Warm generated parameter views allocated {allocated:N0} bytes.");
#endif
    }

    [Fact]
    public void NeuralNetworkManifest_WarmReadsReuseSnapshotWithoutAllocating()
    {
        using var layer = new AiDotNet.NeuralNetworks.Layers.DenseLayer<double>(3);
        var architecture = new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
            AiDotNet.Enums.InputType.OneDimensional,
            AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 3,
            layers: new List<AiDotNet.Interfaces.ILayer<double>> { layer });
        using var network = new AiDotNet.NeuralNetworks.NeuralNetwork<double>(architecture);
        _ = network.Predict(new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(new[] { 1, 4 }));

        var snapshot = network.ParameterLayout;
        _ = network.ParameterCount;
#if NET5_0_OR_GREATER
        long before = GC.GetAllocatedBytesForCurrentThread();
#endif
        for (int i = 0; i < 1_024; i++)
        {
            _ = network.ParameterLayout;
            _ = network.ParameterCount;
        }
#if NET5_0_OR_GREATER
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
#endif

        Assert.Same(snapshot, network.ParameterLayout);
#if NET5_0_OR_GREATER
        Assert.True(allocated <= 128,
            $"Warm layout/count reads allocated {allocated:N0} bytes.");
#endif
    }

    [Fact]
    public void NeuralNetworkManifest_ParameterReplacementInvalidatesSnapshot()
    {
        using var layer = new AiDotNet.NeuralNetworks.Layers.DenseLayer<double>(3);
        var architecture = new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
            AiDotNet.Enums.InputType.OneDimensional,
            AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 3,
            layers: new List<AiDotNet.Interfaces.ILayer<double>> { layer });
        using var network = new AiDotNet.NeuralNetworks.NeuralNetwork<double>(architecture);
        _ = network.Predict(new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(new[] { 1, 4 }));
        var before = network.ParameterLayout;
        Assert.Equal(15, before.MaterializedParameterCount);

        _ = layer.Forward(new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(new[] { 1, 2 }));
        var after = network.ParameterLayout;

        Assert.NotSame(before, after);
        Assert.Equal(9, after.MaterializedParameterCount);
        Assert.Equal(9, network.ParameterCount);
        Assert.Equal(9, network.GetParameters().Length);
    }

    [Fact]
    public void NeuralNetworkManifest_ConcurrentWarmReadersSeeOneSnapshot()
    {
        using var layer = new AiDotNet.NeuralNetworks.Layers.DenseLayer<double>(3);
        var architecture = new AiDotNet.NeuralNetworks.NeuralNetworkArchitecture<double>(
            AiDotNet.Enums.InputType.OneDimensional,
            AiDotNet.Enums.NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 3,
            layers: new List<AiDotNet.Interfaces.ILayer<double>> { layer });
        using var network = new AiDotNet.NeuralNetworks.NeuralNetwork<double>(architecture);
        _ = network.Predict(new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(new[] { 1, 4 }));
        var expected = network.ParameterLayout;
        var observed = new ParameterLayoutSnapshot[64];

        Parallel.For(0, observed.Length, i =>
        {
            observed[i] = network.ParameterLayout;
            Assert.Equal(expected.MaterializedParameterCount, network.ParameterCount);
        });

        Assert.All(observed, snapshot => Assert.Same(expected, snapshot));
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
        var layout = policy.ParameterLayout;

        Assert.Equal(policy.ParameterCount, parameters.Length);
        Assert.Equal(policy.ParameterCount, layout.ParameterCount);
        Assert.Single(policy.GetNetworks());
        Assert.True(layout.Slots.Count > 1);
        Assert.Equal(layout.Slots.Count,
            layout.Slots.Select(slot => slot.StableId).Distinct(StringComparer.Ordinal).Count());
        Assert.Equal(policy.ParameterCount,
            layout.Slots.Sum(slot => slot.ParameterCount.GetValueOrDefault()));
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

    [Fact]
    public async Task BidirectionalLazyLstm_CloneBeforeMaterialization_KeepsIndependentManifests()
    {
        await Task.Yield();

        // Regression: LayerBase.Clone used to MemberwiseClone, so a clone taken while the inner
        // LSTM was still lazy shared the source's mutable parameter registry and manifest. When the
        // two directions later initialized, one registration pass overwrote the other's manifest and
        // a subsequent clone handed back fresh-random gate weights instead of the trained ones.
        var source = new BidirectionalLayer<double>(
            new LSTMLayer<double>(8),
            activationFunction: (IActivationFunction<double>?)null);

        // Clone BEFORE either direction is materialized.
        var earlyClone = (BidirectionalLayer<double>)source.Clone();

        var input = Tensor<double>.CreateRandom(2, 3, 4);
        source.Forward(input);
        earlyClone.Forward(input);

        // Neither manifest may have been clobbered by the other's registration pass.
        Assert.Equal(source.GetParameters().Length, source.ParameterCount);
        Assert.Equal(earlyClone.GetParameters().Length, earlyClone.ParameterCount);
        Assert.Equal(
            source.GetParameters().Length,
            source.GetParameterLayout().Sum(slot => slot.ParameterCount ?? 0));

        // Guard against a vacuous pass: every assertion below is trivially true on an empty vector.
        Assert.NotEmpty(source.GetParameters().ToArray());
        Assert.Equal(source.ParameterCount, earlyClone.ParameterCount);

        // A clone taken after materialization must carry the source's weights, not fresh-random ones.
        var trained = source.GetParameters();
        for (int i = 0; i < trained.Length; i++)
        {
            trained[i] = 0.25 + (i * 0.001);
        }
        source.SetParameters(trained);

        var lateClone = (BidirectionalLayer<double>)source.Clone();
        Assert.Equal(trained.ToArray(), lateClone.GetParameters().ToArray());

        // And the clone must be independent: mutating it must not write through to the source.
        var mutated = lateClone.GetParameters();
        mutated[0] = -99.0;
        lateClone.SetParameters(mutated);
        Assert.Equal(trained[0], source.GetParameters()[0]);
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

internal sealed class AlwaysDeferredLifecycleSource :
    IParameterSource<double>, IParameterLayoutSource, IParameterSurfaceLifecycle
{
    public int DescribeCount { get; private set; }

    public long ParameterCount => 0;

    public void PrepareParameterSurface(ParameterSurfaceIntent intent)
    {
        if (intent == ParameterSurfaceIntent.Describe) DescribeCount++;
    }

    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout() =>
        new[]
        {
            new ParameterSlotDescriptor(
                "$", ParameterSlotRole.Trainable, ParameterReadiness.ShapeDeferred, null)
        };

    public Vector<double> GetParameters() =>
        throw new InvalidOperationException("A genuinely deferred source cannot be read.");

    public void SetParameters(Vector<double> parameters) =>
        throw new InvalidOperationException("A genuinely deferred source cannot be restored.");
}

internal sealed class LifecycleProbeSource :
    IParameterSource<double>, IParameterLayoutSource, IParameterSurfaceLifecycle
{
    private readonly object _gate = new();
    private bool _structureReady;
    private bool _materialized;
    private double[] _values = new[] { 1d, 2d, 3d };

    public int DescribeCount { get; private set; }
    public int AllocationCount { get; private set; }
    public int RestoreCount { get; private set; }
    public long ParameterCount => _structureReady ? 3 : 0;

    public void PrepareParameterSurface(ParameterSurfaceIntent intent)
    {
        lock (_gate)
        {
            _structureReady = true;
            if (intent == ParameterSurfaceIntent.Describe)
            {
                DescribeCount++;
                return;
            }

            if (_materialized) return;
            _materialized = true;
            AllocationCount++;
        }
    }

    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout()
    {
        lock (_gate)
        {
            if (!_structureReady)
            {
                return new[]
                {
                    new ParameterSlotDescriptor(
                        "$", ParameterSlotRole.Trainable, ParameterReadiness.ShapeDeferred, null)
                };
            }

            var readiness = _materialized
                ? ParameterReadiness.Materialized
                : ParameterReadiness.ShapeResolvedUnmaterialized;
            return new[]
            {
                new ParameterSlotDescriptor(
                    "weight", ParameterSlotRole.Trainable, readiness, 2,
                    shape: new[] { 2 }, elementType: typeof(double).FullName),
                new ParameterSlotDescriptor(
                    "running", ParameterSlotRole.Buffer, readiness, 1,
                    shape: new[] { 1 }, elementType: typeof(double).FullName)
            };
        }
    }

    public Vector<double> GetParameters()
    {
        lock (_gate)
        {
            if (!_materialized)
                throw new InvalidOperationException("Read must prepare the component first.");
            return new Vector<double>(_values);
        }
    }

    public void SetParameters(Vector<double> parameters)
    {
        lock (_gate)
        {
            if (!_materialized)
                throw new InvalidOperationException("Restore must prepare the component first.");
            if (parameters.Length != 3)
                throw new ArgumentException("Expected three parameters.", nameof(parameters));
            _values = parameters.ToArray();
            RestoreCount++;
        }
    }
}

internal partial class GeneratedAndManualParameterModel<T> : ModelBase<T, Vector<T>, Vector<T>>
{
    [TrainableParameter]
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
