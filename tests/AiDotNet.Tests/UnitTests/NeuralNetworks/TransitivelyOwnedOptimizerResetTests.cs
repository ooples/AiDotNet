using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// <c>ResetBaseTrainOptimizerState</c> must reset every optimizer the model owns TRANSITIVELY, not
/// only the ones held in a directly optimizer-typed field.
/// </summary>
/// <remarks>
/// <para>
/// Discovery matched a field only when its declared type was assignable to
/// <c>IGradientBasedOptimizer&lt;T, Tensor&lt;T&gt;, Tensor&lt;T&gt;&gt;</c>. A
/// <c>List&lt;IGradientBasedOptimizer&lt;...&gt;&gt;</c> is not, so it was skipped outright:
/// <c>DomainDecompositionPINN</c> keeps its per-subdomain optimizers in exactly such a list, and every
/// one of them survived a reset with its momentum intact, so the next trajectory continued the
/// previous one. The same held one level down for a model that delegates training to sub-networks.
/// </para>
/// <para>
/// Each route is asserted separately, so a regression names which one broke rather than only that
/// something did.
/// </para>
/// </remarks>
public class TransitivelyOwnedOptimizerResetTests
{
    /// <summary>A real optimizer that records how many times it was reset.</summary>
    private sealed class ResetCountingAdam : AdamOptimizer<double, Tensor<double>, Tensor<double>>
    {
        internal int ResetCount { get; private set; }

        internal ResetCountingAdam() : base(null) { }

        public override void Reset()
        {
            ResetCount++;
            base.Reset();
        }
    }

    /// <summary>Minimal concrete model; the fields under test are declared on the subclasses.</summary>
    /// <remarks>
    /// The layout attributes are required by ADNSHAPE007, which makes every concrete model publish the
    /// caller-facing ranks it supports. These probes never run a forward pass -- they exist only to
    /// carry fields for reset discovery to walk -- so [Batch, Features] is simply the smallest honest
    /// declaration, inherited by each subclass below.
    /// </remarks>
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
    [TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
    private abstract class ProbeModel : NeuralNetworkBase<double>
    {
        protected ProbeModel()
            : base(new NeuralNetworkArchitecture<double>(inputFeatures: 2, outputSize: 1),
                   new AiDotNet.LossFunctions.MeanSquaredErrorLoss<double>()) { }

        public override bool SupportsTraining => true;
        protected override void InitializeLayers() { }
        protected override Tensor<double> PredictCore(Tensor<double> input) => input;
        public override ModelMetadata<double> GetModelMetadata() => new();
        protected override void SerializeNetworkSpecificData(System.IO.BinaryWriter writer) { }
        protected override void DeserializeNetworkSpecificData(System.IO.BinaryReader reader) { }
        protected override IFullModel<double, Tensor<double>, Tensor<double>> CreateNewInstance() =>
            throw new System.NotSupportedException("probe");
    }

    private sealed class DirectFieldModel : ProbeModel
    {
        internal readonly ResetCountingAdam Owned = new();
    }

    private sealed class OptimizerListModel : ProbeModel
    {
        internal readonly List<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>> Owned =
            new() { new ResetCountingAdam(), new ResetCountingAdam() };
    }

    private sealed class OptimizerArrayModel : ProbeModel
    {
        internal readonly ResetCountingAdam[] Owned = { new(), new() };
    }

    private sealed class NestedModelHolder : ProbeModel
    {
        internal readonly DirectFieldModel Child = new();
    }

    private sealed class NestedModelListHolder : ProbeModel
    {
        internal readonly List<DirectFieldModel> Children = new() { new(), new() };
    }

    /// <summary>A shared optimizer reached by two routes must still be reset exactly once.</summary>
    private sealed class AliasedOptimizerModel : ProbeModel
    {
        internal readonly ResetCountingAdam Owned = new();
        internal readonly List<IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>> Alias;

        internal AliasedOptimizerModel() => Alias = new() { Owned };
    }

    /// <summary>A cycle must terminate rather than recurse forever.</summary>
    private sealed class CyclicModel : ProbeModel
    {
        internal readonly ResetCountingAdam Owned = new();
        internal CyclicModel? Back;
    }

    /// <summary>
    /// A child model held in an INTERFACE-typed field. NeuralNetworkBase implements
    /// INeuralNetworkModel rather than deriving from it, so a one-directional assignability test
    /// misses this shape entirely.
    /// </summary>
    private sealed class InterfaceTypedChildModel : ProbeModel
    {
        internal readonly INeuralNetworkModel<double> Child = new DirectFieldModel();
    }

    /// <summary>Optimizers held as dictionary VALUES, which enumerate as KeyValuePair.</summary>
    private sealed class OptimizerDictionaryModel : ProbeModel
    {
        internal readonly Dictionary<string, IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>> Owned =
            new() { ["a"] = new ResetCountingAdam(), ["b"] = new ResetCountingAdam() };
    }

    /// <summary>Child models held as dictionary values.</summary>
    private sealed class ModelDictionaryModel : ProbeModel
    {
        internal readonly Dictionary<string, DirectFieldModel> Children =
            new() { ["x"] = new DirectFieldModel(), ["y"] = new DirectFieldModel() };
    }

    /// <summary>
    /// A read-only dictionary that deliberately implements only the generic contracts. This proves
    /// ownership traversal does not depend on the legacy non-generic IDictionary interface.
    /// </summary>
    private sealed class GenericReadOnlyDictionary<TKey, TValue> : IReadOnlyDictionary<TKey, TValue>
        where TKey : notnull
    {
        private readonly IReadOnlyDictionary<TKey, TValue> _items;

        internal GenericReadOnlyDictionary(IReadOnlyDictionary<TKey, TValue> items) => _items = items;

        public TValue this[TKey key] => _items[key];
        public IEnumerable<TKey> Keys => _items.Keys;
        public IEnumerable<TValue> Values => _items.Values;
        public int Count => _items.Count;
        public bool ContainsKey(TKey key) => _items.ContainsKey(key);
        public bool TryGetValue(TKey key, out TValue value) => _items.TryGetValue(key, out value!);
        public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator() => _items.GetEnumerator();
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
    }

    private sealed class ReadOnlyOptimizerDictionaryModel : ProbeModel
    {
        internal readonly GenericReadOnlyDictionary<string,
            IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>> Owned = new(
                new Dictionary<string, IGradientBasedOptimizer<double, Tensor<double>, Tensor<double>>>
                {
                    ["a"] = new ResetCountingAdam(),
                    ["b"] = new ResetCountingAdam(),
                });
    }

    private sealed class ReadOnlyModelDictionaryModel : ProbeModel
    {
        internal readonly GenericReadOnlyDictionary<string, DirectFieldModel> Children = new(
            new Dictionary<string, DirectFieldModel>
            {
                ["x"] = new DirectFieldModel(),
                ["y"] = new DirectFieldModel(),
            });
    }

    [Fact]
    public void DirectlyTypedField_IsReset()
    {
        var model = new DirectFieldModel();
        model.ResetBaseTrainOptimizerState();
        Assert.Equal(1, model.Owned.ResetCount);
    }

    [Fact]
    public void OptimizerHeldInAList_IsReset()
    {
        var model = new OptimizerListModel();
        model.ResetBaseTrainOptimizerState();
        foreach (var optimizer in model.Owned)
            Assert.Equal(1, ((ResetCountingAdam)optimizer).ResetCount);
    }

    [Fact]
    public void OptimizerHeldInAnArray_IsReset()
    {
        var model = new OptimizerArrayModel();
        model.ResetBaseTrainOptimizerState();
        foreach (var optimizer in model.Owned)
            Assert.Equal(1, optimizer.ResetCount);
    }

    [Fact]
    public void OptimizerOwnedByANestedModel_IsReset()
    {
        var model = new NestedModelHolder();
        model.ResetBaseTrainOptimizerState();
        Assert.Equal(1, model.Child.Owned.ResetCount);
    }

    [Fact]
    public void OptimizerOwnedByNestedModelsInAList_IsReset()
    {
        var model = new NestedModelListHolder();
        model.ResetBaseTrainOptimizerState();
        foreach (var child in model.Children)
            Assert.Equal(1, child.Owned.ResetCount);
    }

    [Fact]
    public void AnOptimizerReachedTwice_IsResetOnce()
    {
        var model = new AliasedOptimizerModel();
        model.ResetBaseTrainOptimizerState();
        Assert.Equal(1, model.Owned.ResetCount);
    }

    [Fact]
    public void OptimizerOwnedByAnInterfaceTypedChildModel_IsReset()
    {
        var model = new InterfaceTypedChildModel();
        model.ResetBaseTrainOptimizerState();
        Assert.Equal(1, ((DirectFieldModel)model.Child).Owned.ResetCount);
    }

    [Fact]
    public void OptimizerHeldAsADictionaryValue_IsReset()
    {
        var model = new OptimizerDictionaryModel();
        model.ResetBaseTrainOptimizerState();
        foreach (var optimizer in model.Owned.Values)
            Assert.Equal(1, ((ResetCountingAdam)optimizer).ResetCount);
    }

    [Fact]
    public void OptimizerOwnedByModelsHeldAsDictionaryValues_IsReset()
    {
        var model = new ModelDictionaryModel();
        model.ResetBaseTrainOptimizerState();
        foreach (var child in model.Children.Values)
            Assert.Equal(1, child.Owned.ResetCount);
    }

    [Fact]
    public void OptimizerHeldAsAReadOnlyDictionaryValue_IsReset()
    {
        var model = new ReadOnlyOptimizerDictionaryModel();
        model.ResetBaseTrainOptimizerState();
        foreach (var optimizer in model.Owned.Values)
            Assert.Equal(1, ((ResetCountingAdam)optimizer).ResetCount);
    }

    [Fact]
    public void OptimizerOwnedByModelsHeldAsReadOnlyDictionaryValues_IsReset()
    {
        var model = new ReadOnlyModelDictionaryModel();
        model.ResetBaseTrainOptimizerState();
        foreach (var child in model.Children.Values)
            Assert.Equal(1, child.Owned.ResetCount);
    }

    /// <summary>
    /// A model that never records a loss must not look like one that recorded zero. T is unconstrained,
    /// so for double the T? annotation is only a nullable-reference annotation and a null test on the
    /// value can never fire - the flag is the only thing that can answer this.
    /// </summary>
    [Fact]
    public void AModelThatNeverTrained_ReportsNoRecordedLoss()
    {
        var untrained = new DirectFieldModel();
        Assert.False(untrained.HasRecordedLoss,
            "a model that has never trained reported a recorded loss; for a value-type T the null "
            + "test this used to rely on is always false, so every model claimed to have one.");
        Assert.Equal(0.0, untrained.GetLastLoss());
    }

    [Fact]
    public void ACycleBetweenModels_Terminates()
    {
        var first = new CyclicModel();
        var second = new CyclicModel();
        first.Back = second;
        second.Back = first;

        first.ResetBaseTrainOptimizerState();

        Assert.Equal(1, first.Owned.ResetCount);
        Assert.Equal(1, second.Owned.ResetCount);
    }
}
