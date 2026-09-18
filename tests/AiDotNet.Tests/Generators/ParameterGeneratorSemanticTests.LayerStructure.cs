using System;
using System.Collections.Generic;
using Xunit;

namespace AiDotNet.Tests.Generators;

public partial class ParameterGeneratorSemanticTests
{
    public ParameterGeneratorSemanticTests() => TestModuleInitializer.EnsureInitialized();

    public enum StructureInitializerCase
    {
        DisjointWeightHelper,
        ExternalValueStrategy,
        NullableLazyChild,
        IndirectChildHelper,
        RefChildEscape,
        OwnerEscape,
        OwnerAlias,
        DelegateEscape,
        VirtualOwnerCall,
        UnresolvedOwnerCall,
        InitiallyEmptyCollection,
        InheritedInitializer,
        IncrementSetter,
        DeconstructionSetter,
        FieldOwnerAlias,
        ConvertedOwnerAlias,
        CachedOwnerStaticHelper,
        UnknownExternalReceiver,
        ConstructorOwnerEscape,
        ReassignedNullCallback,
        CoalescedNullCallback,
        UnknownHelperConstructor,
        ReadOnlyNullCallback,
        RefNullCallback,
        BinaryOperator,
        UnaryOperator,
        ConversionOperator,
        IncrementOperator,
        CompoundAssignmentOperator,
        DynamicOperator,
        PrimitiveOperators,
        FrameworkException,
        ExpressionBodiedValueProperty,
        ExpressionBodiedStructuralProperty,
        RuntimeTypeName,
        RuntimeTypeEquality,
        UnknownTypeName,
        UnknownTypeEquality,
        UnknownEnumerator,
        StaticEventSetter,
        UnknownInterpolation,
        DisposableInterfaceReceiver,
        EnumerableInterfaceReceiver,
        DisposableInterfaceInterpolation
    }

    public static IEnumerable<object[]> StructureInitializerCases()
    {
        foreach (StructureInitializerCase kind in Enum.GetValues(typeof(StructureInitializerCase)))
            yield return new object[] { kind };
    }

    [Theory]
    [MemberData(nameof(StructureInitializerCases))]
    public void LayerGenerator_OnlySkipsProvenChildIndependentInitializers(StructureInitializerCase kind)
    {
        string field = kind == StructureInitializerCase.InitiallyEmptyCollection
            ? "private readonly System.Collections.Generic.List<AiDotNet.Interfaces.ILayer<T>> _children = new();"
            : "private AiDotNet.Interfaces.ILayer<T>? _child;";
        string body = kind switch
        {
            StructureInitializerCase.DisjointWeightHelper => "AllocateWeights(); base.EnsureInitialized();",
            StructureInitializerCase.ExternalValueStrategy => "_strategy.InitializeWeights(_weights, 2, 2);",
            StructureInitializerCase.NullableLazyChild => "_child ??= new Child<T>();",
            StructureInitializerCase.IndirectChildHelper => "InitializeChild();",
            StructureInitializerCase.RefChildEscape => "AssignChild(ref _child);",
            StructureInitializerCase.OwnerEscape => "Observe(this);",
            StructureInitializerCase.OwnerAlias => "var owner = this; owner.InitializeChild();",
            StructureInitializerCase.DelegateEscape => "System.Action callback = InitializeChild; callback();",
            StructureInitializerCase.VirtualOwnerCall => "OnInitialize();",
            StructureInitializerCase.UnresolvedOwnerCall => "InitializeUnknown();",
            StructureInitializerCase.InitiallyEmptyCollection => "_children.Add(new Child<T>());",
            StructureInitializerCase.InheritedInitializer => "base.EnsureInitialized();",
            StructureInitializerCase.IncrementSetter => "Builder++;",
            StructureInitializerCase.DeconstructionSetter => "(Builder, _counter) = (1, 2);",
            StructureInitializerCase.FieldOwnerAlias => "_owner?.InitializeChild();",
            StructureInitializerCase.ConvertedOwnerAlias => "(_opaqueOwner as ConfiguredChildLayer<T>)?.InitializeChild();",
            StructureInitializerCase.CachedOwnerStaticHelper => "InitializeCachedOwner();",
            StructureInitializerCase.UnknownExternalReceiver => "_external.Build();",
            StructureInitializerCase.ConstructorOwnerEscape => "AllocateWeights();",
            StructureInitializerCase.ReassignedNullCallback => "RunCallback();",
            StructureInitializerCase.CoalescedNullCallback => "RunCallback();",
            StructureInitializerCase.UnknownHelperConstructor => "_ = new ExternalBuilder();",
            StructureInitializerCase.ReadOnlyNullCallback => "RunCallback();",
            StructureInitializerCase.RefNullCallback => "RunCallback();",
            StructureInitializerCase.BinaryOperator => "_ = _operand + _operand;",
            StructureInitializerCase.UnaryOperator => "_ = -_operand;",
            StructureInitializerCase.ConversionOperator => "_counter = (int)_operand;",
            StructureInitializerCase.IncrementOperator => "_operand++;",
            StructureInitializerCase.CompoundAssignmentOperator => "_operand += _operand;",
            StructureInitializerCase.DynamicOperator => "dynamic operand = _operand; _ = -operand;",
            StructureInitializerCase.PrimitiveOperators => "_counter += 1; _counter = (int)(-(decimal)_counter + 2m);",
            StructureInitializerCase.FrameworkException => "throw new System.InvalidOperationException(\"Invalid shape.\");",
            StructureInitializerCase.ExpressionBodiedValueProperty => "_counter = ScalarValue;",
            StructureInitializerCase.ExpressionBodiedStructuralProperty => "_counter = StructuralValueProperty;",
            StructureInitializerCase.RuntimeTypeName => "_counter = GetType().Name.Length;",
            StructureInitializerCase.RuntimeTypeEquality => "_counter = typeof(T) == typeof(double) ? 1 : 0;",
            StructureInitializerCase.UnknownTypeName => "_counter = _type.Name.Length;",
            StructureInitializerCase.UnknownTypeEquality => "_counter = _type == typeof(double) ? 1 : 0;",
            StructureInitializerCase.UnknownEnumerator => "foreach (int value in _enumerable) { _counter = value; }",
            StructureInitializerCase.StaticEventSetter => "ExternalBuilder.Changed += null;",
            StructureInitializerCase.UnknownInterpolation => "_ = $\"{_external}\";",
            StructureInitializerCase.DisposableInterfaceReceiver => "_disposable.Dispose();",
            StructureInitializerCase.EnumerableInterfaceReceiver => "_ = _legacyEnumerable.GetEnumerator().MoveNext();",
            StructureInitializerCase.DisposableInterfaceInterpolation => "_ = $\"{_disposable}\";",
            _ => throw new ArgumentOutOfRangeException(nameof(kind))
        };
        string baseType = kind == StructureInitializerCase.InheritedInitializer
            ? "Parent<T>" : "AiDotNet.NeuralNetworks.Layers.LayerBase<T>";
        string source = $$"""
            #nullable enable
            using AiDotNet.Attributes;
            using AiDotNet.Tensors.LinearAlgebra;
            public sealed class Child<T> : AiDotNet.Interfaces.ILayer<T> { }
            namespace AiDotNet.Initialization
            {
                public interface IInitializationStrategy<T> { void InitializeWeights(Tensor<T> tensor, int inputSize, int outputSize); }
            }
            public interface IExternalBuilder { void Build(); }
            public abstract class Parent<T> : AiDotNet.NeuralNetworks.Layers.LayerBase<T>
            {
                protected override void EnsureInitialized() { }
            }
            public partial class ConfiguredChildLayer<T> : {{baseType}}
            {
                [TrainableParameter] private Tensor<T> _weights = new();
                {{field}}
                private AiDotNet.Initialization.IInitializationStrategy<T> _strategy = new ValueStrategy<T>();
                [Scratch] private ConfiguredChildLayer<T>? _owner;
                [Scratch] private object? _opaqueOwner;
                private static ConfiguredChildLayer<T>? _cachedOwner;
                private IExternalBuilder _external = new ExternalBuilder();
                private int _counter;
                private StructuralValue _operand;
                private System.Type _type = typeof(int);
                private System.Collections.Generic.IEnumerable<int> _enumerable = System.Array.Empty<int>();
                private System.Collections.IEnumerable _legacyEnumerable = System.Array.Empty<int>();
                private System.IDisposable _disposable = new ExternalBuilder();
                private System.Action? _configuredCallback;
                private int Builder { get => 0; set { InitializeChild(); } }
                private int ScalarValue => 2;
                private int StructuralValueProperty => BuildReturningCounter();
                {{(kind == StructureInitializerCase.ConstructorOwnerEscape ? "public ConfiguredChildLayer() { Observe(this); }" : "")}}
                protected override void EnsureInitialized() { {{body}} }
                private void AllocateWeights() { _weights = new(); }
                private int BuildReturningCounter() { InitializeChild(); return 0; }
                private void InitializeChild() { {{(kind == StructureInitializerCase.InitiallyEmptyCollection ? "_children.Add(new Child<T>());" : "_child = new Child<T>();")}} }
                private static void AssignChild(ref AiDotNet.Interfaces.ILayer<T>? child) { child = new Child<T>(); }
                private static void Observe(object owner) { }
                private static void InitializeCachedOwner() { _cachedOwner?.InitializeChild(); }
                private void RunCallback(System.Action? callback = null)
                {
                    {{(kind == StructureInitializerCase.ReassignedNullCallback ? "callback = _configuredCallback;" : kind == StructureInitializerCase.CoalescedNullCallback ? "callback ??= _configuredCallback;" : "")}}
                    {{(kind == StructureInitializerCase.RefNullCallback ? "ReplaceCallback(ref callback);" : "")}}
                    callback?.Invoke();
                }
                private void ReplaceCallback(ref System.Action? callback) { callback = _configuredCallback; }
                protected virtual void OnInitialize() { }
                partial void InitializeUnknown();
                public void Configure() { InitializeChild(); }
                public void ConfigureCallback() { _configuredCallback = InitializeChild; }
                public void ConfigureType(System.Type type) { _type = type; }
                public void ConfigureEnumerable(System.Collections.Generic.IEnumerable<int> values) { _enumerable = values; }
                public void ConfigureExternalCallbacks()
                {
                    StructuralValue.Callback = InitializeChild;
                    ExternalBuilder.Callback = InitializeChild;
                }
            }
            public sealed class ValueStrategy<T> : AiDotNet.Initialization.IInitializationStrategy<T>
            {
                public void InitializeWeights(Tensor<T> tensor, int inputSize, int outputSize) { }
            }
            public sealed class ExternalBuilder : IExternalBuilder, System.IDisposable
            {
                public static System.Action? Callback;
                public ExternalBuilder() { Callback?.Invoke(); }
                public void Build() { Callback?.Invoke(); }
                public void Dispose() { Callback?.Invoke(); }
                public static event System.Action? Changed { add { Callback?.Invoke(); } remove { Callback?.Invoke(); } }
                public override string ToString() { Callback?.Invoke(); return "value"; }
            }
            public struct StructuralValue
            {
                public static System.Action? Callback;
                public static StructuralValue operator +(StructuralValue left, StructuralValue right) { Callback?.Invoke(); return left; }
                public static StructuralValue operator -(StructuralValue value) { Callback?.Invoke(); return value; }
                public static explicit operator int(StructuralValue value) { Callback?.Invoke(); return 0; }
                public static StructuralValue operator ++(StructuralValue value) { Callback?.Invoke(); return value; }
            }
            """;
        string generated = Run(new AiDotNet.Generators.TrainableParameterGenerator(), source);
        Assert.Contains("HasDeclaredSubLayerStructure => true", generated, StringComparison.Ordinal);
        bool independent = kind is StructureInitializerCase.DisjointWeightHelper or StructureInitializerCase.ExternalValueStrategy
            or StructureInitializerCase.ReadOnlyNullCallback or StructureInitializerCase.PrimitiveOperators
            or StructureInitializerCase.FrameworkException or StructureInitializerCase.ExpressionBodiedValueProperty
            or StructureInitializerCase.RuntimeTypeName or StructureInitializerCase.RuntimeTypeEquality;
        const string declaration = "protected override bool NeedsDeclaredSubLayerInitialization";
        if (independent)
        {
            Assert.Contains(declaration + " => GetType() != typeof(global::ConfiguredChildLayer<T>);", generated, StringComparison.Ordinal);
        }
        else
        {
            Assert.DoesNotContain(declaration, generated, StringComparison.Ordinal);
        }
    }
}
