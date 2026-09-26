using AiDotNet.ComputerVision;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;
using AiDotNet.Interfaces;
using AiDotNet.Models.Parameters;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>Exercises the real shared adapters through the same accessors used by the model generator.</summary>
public sealed class CvAdapterLiveParameterTests
{
    public enum AdapterKind { Convolution, Dense, SelfAttention, BatchNormalization, TransposedConvolution }
    public enum RegistrationKind { Accessor, Collection }

    public CvAdapterLiveParameterTests() => TestModuleInitializer.EnsureInitialized();

    public static IEnumerable<object[]> RegisteredAdapters()
    {
        foreach (AdapterKind adapter in Enum.GetValues(typeof(AdapterKind)))
            foreach (RegistrationKind registration in Enum.GetValues(typeof(RegistrationKind)))
                yield return new object[] { adapter, registration };
    }

    [Theory]
    [MemberData(nameof(RegisteredAdapters))]
    public void Registry_RetainsActualLayerChunksAndFlatState(AdapterKind kind, RegistrationKind registration)
    {
        var fixture = Create(kind);
        fixture.Forward(fixture.Input);
        var direct = Assert.IsAssignableFrom<IParameterChunkSource<double>>(fixture.Source).GetParameterStateChunks().ToArray();
        var registry = Register(fixture.Source, registration);
        var actual = registry.GetParameterStateChunks().ToArray();

        Assert.NotEmpty(direct);
        Assert.Equal(direct.Length, actual.Length);
        for (int index = 0; index < direct.Length; index++)
        {
            Assert.Same(direct[index].Tensor, actual[index].Tensor);
            Assert.Equal(direct[index].Role, actual[index].Role);
            Assert.True(actual[index].IsWritableInPlace, actual[index].StableId);
        }
        Assert.Equal(fixture.Source.GetParameters().ToArray(), registry.GetParameters().ToArray());
        Assert.Equal(registry.GetParameters().ToArray(), actual.SelectMany(chunk => chunk.Tensor.ToArray()).ToArray());
        Assert.Equal(registry.ParameterLayout.ParameterCount, actual.Sum(chunk => (long)chunk.Tensor.Length));
    }

    [Theory]
    [MemberData(nameof(RegisteredAdapters))]
    public void Registry_ExposesTapeWeightsWhoseMutationChangesTheActualForward(AdapterKind kind, RegistrationKind registration)
    {
        var fixture = Create(kind);
        fixture.Forward(fixture.Input);
        var registry = Register(fixture.Source, registration);
        var weights = registry.GetParameterStateChunks()
            .Where(chunk => chunk.Role == ParameterSlotRole.Trainable).Select(chunk => chunk.Tensor).ToArray();
        Assert.NotEmpty(weights);
        foreach (var weight in weights) weight.Fill(0);

        using var tape = new GradientTape<double>();
        var before = fixture.Forward(fixture.Input);
        var loss = AiDotNetEngine.Current.ReduceSum(before, null);
        var gradients = tape.ComputeGradients(loss, weights);
        var liveGradient = weights.Select(weight => (Weight: weight, Gradient: gradients.TryGetValue(weight, out var gradient) ? gradient : null))
            .FirstOrDefault(item => item.Gradient is not null && item.Gradient.ToArray().Any(value => Math.Abs(value) > 0));
        var selectedWeight = Assert.IsType<Tensor<double>>(liveGradient.Weight);
        var selectedGradient = Assert.IsType<Tensor<double>>(liveGradient.Gradient);
        Assert.All(selectedGradient.ToArray(), value => Assert.True(!double.IsNaN(value) && !double.IsInfinity(value)));
        var baseline = before.ToArray();
        using (new NoGradScope<double>())
        {
            for (int index = 0; index < selectedGradient.Length; index++)
                selectedWeight[index] -= 0.01 * selectedGradient[index];
            var after = fixture.Forward(fixture.Input).ToArray();
            Assert.True(baseline.Where((value, index) => value != after[index]).Any(), "Updating the registered tape tensor did not change the actual forward.");
            Assert.True(after.Sum() < baseline.Sum(), "The gradient step did not reduce the independently defined sum objective.");
        }
    }

    [Theory]
    [InlineData(AdapterKind.Convolution)]
    [InlineData(AdapterKind.Dense)]
    [InlineData(AdapterKind.SelfAttention)]
    [InlineData(AdapterKind.BatchNormalization)]
    [InlineData(AdapterKind.TransposedConvolution)]
    public void LayoutQuery_DoesNotInitializeLazyAdapterValues(AdapterKind kind)
    {
        var fixture = Create(kind);
        var before = fixture.Source.GetParameters().ToArray();
        var layout = Assert.IsAssignableFrom<IParameterLayoutSource>(fixture.Source).GetParameterLayout();
        Assert.NotEmpty(layout);
        Assert.Equal(before, fixture.Source.GetParameters().ToArray());
        if (kind is AdapterKind.Convolution or AdapterKind.Dense or AdapterKind.TransposedConvolution)
        {
            Assert.Empty(before);
            Assert.Contains(layout, slot => slot.Readiness == ParameterReadiness.ShapeDeferred);
        }
        else if (kind == AdapterKind.SelfAttention)
        {
            Assert.Empty(before);
            Assert.Equal(68L, Assert.Single(layout).ParameterCount);
            Assert.Equal(ParameterReadiness.ShapeResolvedUnmaterialized, layout[0].Readiness);
        }
        fixture.Forward(fixture.Input);
        var resolved = Assert.IsAssignableFrom<IParameterLayoutSource>(fixture.Source).GetParameterLayout();
        Assert.DoesNotContain(resolved, slot => slot.Readiness == ParameterReadiness.ShapeDeferred);
        Assert.Equal(fixture.Source.ParameterCount, resolved.Sum(slot => slot.ParameterCount.GetValueOrDefault()));
    }

    [Theory]
    [InlineData(RegistrationKind.Accessor)]
    [InlineData(RegistrationKind.Collection)]
    public void SharedModule_PreservesOwnThenChildOrderAndLiveStorage(RegistrationKind registration)
    {
        var own = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 2.0, 3.0 }));
        var child = Create(AdapterKind.Convolution);
        child.Forward(child.Input);
        var module = new DelegatingCvParameterModule<double>(() => new IParameterSource<double>?[] { null, child.Source }, () => new[] { own });
        var registry = Register(module, registration);
        var chunks = registry.GetParameterStateChunks().ToArray();
        var direct = module.GetParameterStateChunks().ToArray();
        Assert.Equal(direct.Length, chunks.Length);
        Assert.Same(own, chunks[0].Tensor);
        for (int index = 0; index < direct.Length; index++) Assert.Same(direct[index].Tensor, chunks[index].Tensor);
        Assert.Equal(module.GetParameters().ToArray(), registry.GetParameters().ToArray());
        Assert.Equal(registry.GetParameters().ToArray(), chunks.SelectMany(chunk => chunk.Tensor.ToArray()).ToArray());
        var layout = Assert.IsAssignableFrom<IParameterLayoutSource>(module).GetParameterLayout();
        Assert.Equal("w0", layout[0].StableId);
        Assert.Equal(2L, layout[0].ParameterCount);
        Assert.StartsWith(ParameterStableId.IndexSegment(0), layout[1].StableId);
        AssertNormalizedOffsets(registry, module.GetParameters().ToArray());
    }

    [Fact]
    public void AttentionMetadataQuery_DoesNotAllocateConstructionSizedWeights()
    {
        var layer = new MultiHeadAttentionLayer<double>(2, 2, (IActivationFunction<double>?)null, new RejectMetadataInitialization());
        Assert.All(layer.GetTrainableParametersWithoutMaterialization(), tensor => Assert.Equal(0, tensor.Length));
        var layout = layer.GetParameterLayout();
        Assert.All(layer.GetTrainableParametersWithoutMaterialization(), tensor => Assert.Equal(0, tensor.Length));
        Assert.Equal(ParameterReadiness.ShapeResolvedUnmaterialized, Assert.Single(layout).Readiness);
        Assert.Equal(68L, layout[0].ParameterCount);
        Assert.Equal(0L, layout[0].MaterializedParameterCount);
    }

    [Theory]
    [InlineData(RegistrationKind.Accessor)]
    [InlineData(RegistrationKind.Collection)]
    public void SharedModule_ZeroSizedOwnSlotAndNullChildrenPreserveNormalizedOffsets(RegistrationKind registration)
    {
        var zero = new Tensor<double>(new[] { 0 });
        var own = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 2.0, 3.0 }));
        var first = Create(AdapterKind.Convolution);
        var second = Create(AdapterKind.Dense);
        first.Forward(first.Input);
        second.Forward(second.Input);
        var module = new DelegatingCvParameterModule<double>(
            () => new IParameterSource<double>?[] { null, first.Source, null, second.Source, null },
            () => new[] { zero, own });
        var registry = Register(module, registration);
        var slots = module.GetParameterLayout();
        Assert.Equal("w0", slots[0].StableId);
        Assert.Equal(ParameterReadiness.ParameterFree, slots[0].Readiness);
        Assert.Equal("w1", slots[1].StableId);
        Assert.StartsWith(ParameterStableId.IndexSegment(0), slots[2].StableId);
        Assert.StartsWith(ParameterStableId.IndexSegment(1), slots[3].StableId);
        Assert.All(slots, slot => Assert.Null(slot.Offset));
        Assert.Equal(module.GetParameters().ToArray(), registry.GetParameterStateChunks().SelectMany(chunk => chunk.Tensor.ToArray()).ToArray());
        AssertNormalizedOffsets(registry, module.GetParameters().ToArray());
    }

    [Fact]
    public void SharedModule_ReportsDeferredChildWithoutMaterializingOrDroppingOwnState()
    {
        var own = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 2.0 }));
        var child = Create(AdapterKind.Convolution);
        var module = new DelegatingCvParameterModule<double>(() => new IParameterSource<double>?[] { null, child.Source }, () => new[] { own });
        var slots = Assert.IsAssignableFrom<IParameterLayoutSource>(module).GetParameterLayout();
        Assert.Equal("w0", slots[0].StableId);
        Assert.Equal(ParameterReadiness.Materialized, slots[0].Readiness);
        Assert.Equal(1L, slots[0].ParameterCount);
        Assert.Contains(slots, slot => slot.Readiness == ParameterReadiness.ShapeDeferred);
        Assert.Equal(new[] { 2.0 }, module.GetParameters().ToArray());
        Assert.Empty(child.Source.GetParameters().ToArray());
    }

    [Fact]
    public void SharedModule_EmptyAndZeroLengthOwnStateRemainParameterFree()
    {
        var empty = new DelegatingCvParameterModule<double>(() => Array.Empty<IParameterSource<double>?>());
        var emptyLayout = Assert.IsAssignableFrom<IParameterLayoutSource>(empty).GetParameterLayout();
        Assert.Empty(emptyLayout);
        Assert.Empty(empty.GetParameterStateChunks());
        Assert.Empty(empty.GetParameters().ToArray());
        Assert.Equal(0L, Register(empty, RegistrationKind.Accessor).ParameterLayout.ParameterCount);

        var zero = new Tensor<double>(new[] { 0 });
        var zeroOwner = new DelegatingCvParameterModule<double>(() => Array.Empty<IParameterSource<double>?>(), () => new[] { zero });
        var slot = Assert.Single(Assert.IsAssignableFrom<IParameterLayoutSource>(zeroOwner).GetParameterLayout());
        Assert.Equal("w0", slot.StableId);
        Assert.Equal(ParameterReadiness.ParameterFree, slot.Readiness);
        Assert.Equal(0L, slot.ParameterCount);
        Assert.Empty(zeroOwner.GetParameterStateChunks());
        Assert.Equal(0L, Register(zeroOwner, RegistrationKind.Collection).ParameterLayout.ParameterCount);
    }

    [Theory]
    [InlineData(AdapterKind.Convolution)]
    [InlineData(AdapterKind.Dense)]
    [InlineData(AdapterKind.SelfAttention)]
    [InlineData(AdapterKind.BatchNormalization)]
    [InlineData(AdapterKind.TransposedConvolution)]
    public void SharedModule_PrefixesChildIdentityWithoutChangingItsMetadata(AdapterKind kind)
    {
        var fixture = Create(kind);
        fixture.Forward(fixture.Input);
        var sourceLayout = Assert.IsAssignableFrom<IParameterLayoutSource>(fixture.Source).GetParameterLayout();
        var module = new DelegatingCvParameterModule<double>(() => new IParameterSource<double>?[] { null, fixture.Source });
        var slots = Assert.IsAssignableFrom<IParameterLayoutSource>(module).GetParameterLayout();
        Assert.Equal(sourceLayout.Count, slots.Count);
        for (int index = 0; index < slots.Count; index++)
        {
            var expected = sourceLayout[index];
            var actual = slots[index];
            string prefix = ParameterStableId.IndexSegment(0);
            Assert.Equal(expected.StableId == "$" ? prefix : prefix + "/" + expected.StableId, actual.StableId);
            Assert.Equal(expected.Role, actual.Role);
            Assert.Equal(expected.Readiness, actual.Readiness);
            Assert.Equal(expected.ParameterCount, actual.ParameterCount);
            Assert.Equal(expected.MaterializedParameterCount, actual.MaterializedParameterCount);
            Assert.Equal(expected.Shape, actual.Shape);
            Assert.Equal(expected.ElementType, actual.ElementType);
            Assert.Equal(expected.UpdatePolicy, actual.UpdatePolicy);
            Assert.Equal(expected.Persistence, actual.Persistence);
            Assert.Equal(expected.Ownership, actual.Ownership);
            Assert.Equal(expected.Availability, actual.Availability);
        }
    }

    [Theory]
    [InlineData(RegistrationKind.Accessor)]
    [InlineData(RegistrationKind.Collection)]
    public void RegionProposalNetwork_ExposesItsRealSharedHeadChunks(RegistrationKind registration)
    {
        var rpn = new RPN<double>(2, 2);
        rpn.Forward(new Tensor<double>(new[] { 1, 2, 4, 4 }));
        var registry = Register(rpn, registration);
        var direct = ((IParameterChunkSource<double>)rpn).GetParameterStateChunks().ToArray();
        var actual = registry.GetParameterStateChunks().ToArray();
        Assert.NotEmpty(direct);
        Assert.Equal(direct.Length, actual.Length);
        for (int index = 0; index < direct.Length; index++) Assert.Same(direct[index].Tensor, actual[index].Tensor);
        Assert.All(actual, chunk => Assert.True(chunk.IsWritableInPlace));
        Assert.Equal(registry.GetParameters().ToArray(), actual.SelectMany(chunk => chunk.Tensor.ToArray()).ToArray());
    }

    private static ParameterComponentRegistry<double> Register(IParameterSource<double> source, RegistrationKind registration)
    {
        IParameterSource<double> adapter = registration switch
        {
            RegistrationKind.Accessor => new ComponentAccessorParameterSource<double>(() => source),
            RegistrationKind.Collection => new ComponentCollectionParameterSource<double>(() => new[] { source }),
            _ => throw new ArgumentOutOfRangeException(nameof(registration))
        };
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("fixture", adapter);
        return registry;
    }

    private static void AssertNormalizedOffsets(ParameterComponentRegistry<double> registry, double[] expectedFlat)
    {
        long offset = 0;
        var actualFlat = registry.GetParameters().ToArray();
        Assert.Equal(expectedFlat, actualFlat);
        foreach (var slot in registry.ParameterLayout.Slots)
        {
            Assert.Equal(offset, slot.Offset);
            Assert.True(slot.ParameterCount.HasValue);
            long count = slot.ParameterCount.GetValueOrDefault();
            Assert.Equal(expectedFlat.Skip(checked((int)offset)).Take(checked((int)count)),
                actualFlat.Skip(checked((int)slot.Offset.GetValueOrDefault())).Take(checked((int)count)));
            offset += count;
        }
        Assert.Equal(expectedFlat.LongLength, offset);
    }

    private sealed class Fixture
    {
        public Fixture(IParameterSource<double> source, Func<Tensor<double>, Tensor<double>> forward, Tensor<double> input)
        {
            Source = source;
            Forward = forward;
            Input = input;
        }

        public IParameterSource<double> Source { get; }
        public Func<Tensor<double>, Tensor<double>> Forward { get; }
        public Tensor<double> Input { get; }
    }

    private sealed class RejectMetadataInitialization : AiDotNet.Initialization.IInitializationStrategy<double>
    {
        public bool IsLazy => false;
        public bool LoadFromExternal => false;
        public void InitializeWeights(Tensor<double> weights, int inputSize, int outputSize)
            => throw new InvalidOperationException("A metadata-only query initialized weights.");
        public void InitializeBiases(Tensor<double> biases)
            => throw new InvalidOperationException("A metadata-only query initialized biases.");
    }

    private static Fixture Create(AdapterKind kind)
    {
        switch (kind)
        {
            case AdapterKind.Convolution:
                var convolution = new Conv2D<double>(2, 2, 1);
                return new(convolution, convolution.Forward, new Tensor<double>(new[] { 1, 2, 4, 4 }));
            case AdapterKind.Dense:
                var dense = new Dense<double>(4, 2);
                return new(dense, dense.Forward, new Tensor<double>(new[] { 2, 4 }));
            case AdapterKind.SelfAttention:
                var attention = new MultiHeadSelfAttention<double>(4, 2);
                return new(attention, attention.Forward, new Tensor<double>(new[] { 1, 3, 4 }));
            case AdapterKind.BatchNormalization:
                var normalization = new BatchNorm2D<double>(2);
                normalization.SetTrainingMode(false);
                return new(normalization, normalization.Forward, new Tensor<double>(new[] { 1, 2, 4, 4 }));
            case AdapterKind.TransposedConvolution:
                var transposed = new ConvTranspose2D<double>(2, 2, 2, 2);
                return new(transposed, transposed.Forward, new Tensor<double>(new[] { 1, 2, 2, 2 }));
            default:
                throw new ArgumentOutOfRangeException(nameof(kind));
        }
    }
}
