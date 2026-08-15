using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Computes output shapes from declared axis roles (<see cref="TensorLayoutAttribute"/>) plus declared
/// axis relations (<see cref="IShapeContract"/>), without running a forward pass.
/// </summary>
/// <remarks>
/// <para>
/// The join between the two halves of the shape system. The layout attribute names a tensor's axes so a
/// raw <c>int[]</c> becomes <c>{Batch: 1, Channels: 3, Height: 32, Width: 32}</c>; the contract then says
/// how each OUTPUT axis is sized from those. Neither half is sufficient alone — roles without relations
/// cannot tell a stride-1 convolution from a stride-2 one, and relations without roles have no way to
/// say which input axis a relation refers to.
/// </para>
/// <para>
/// Everything here declines rather than guesses. A type that is unannotated, whose declared layout does
/// not accept the given rank, or whose contract contains an <c>Unknown</c> axis yields no shape at all.
/// A shape system that returns a plausible answer when it does not know is worse than one that returns
/// nothing, because callers cannot tell the two apart.
/// </para>
/// </remarks>
public static class ShapeInference
{
    /// <summary>
    /// Reports whether a type's effective <see cref="IShapeContract.OutputAxesFor"/> implementation
    /// declares a probeable symbolic law rather than an explicit or conditional unavailable marker.
    /// </summary>
    /// <param name="type">A closed concrete type to inspect.</param>
    /// <returns>
    /// <c>true</c> for a shape-contract implementation that should be instantiated and verified;
    /// otherwise <c>false</c>.
    /// </returns>
    /// <remarks>
    /// The interface map resolves overrides and explicit interface implementations correctly. Attribute
    /// inheritance is deliberately disabled: when a derived type overrides an unavailable base method,
    /// that override is a new contract and must be verified unless it explicitly opts out itself.
    /// Conditional base contracts can also require a virtual metadata property to be overridden; that
    /// requirement is checked statically so an inherited sentinel value never forces model construction.
    /// </remarks>
    public static bool HasDeclaredOutputShapeContract(Type type)
    {
        if (type is null) throw new ArgumentNullException(nameof(type));
        if (type.ContainsGenericParameters || !typeof(IShapeContract).IsAssignableFrom(type)) return false;

        MethodInfo? contractMethod = typeof(IShapeContract).GetMethod(
            nameof(IShapeContract.OutputAxesFor),
            BindingFlags.Public | BindingFlags.Instance,
            binder: null,
            types: new[] { typeof(int) },
            modifiers: null);
        if (contractMethod is null) return false;

        InterfaceMapping map = type.GetInterfaceMap(typeof(IShapeContract));
        for (int i = 0; i < map.InterfaceMethods.Length; i++)
        {
            if (map.InterfaceMethods[i] != contractMethod) continue;

            MethodInfo target = map.TargetMethods[i];
            if (target.IsDefined(typeof(ShapeContractUnavailableAttribute), inherit: false)) return false;

            foreach (ShapeContractRequiresPropertyOverrideAttribute requirement in target.GetCustomAttributes(
                         typeof(ShapeContractRequiresPropertyOverrideAttribute), inherit: false))
            {
                if (!OverridesRequiredProperty(type, target.DeclaringType, requirement.PropertyName))
                    return false;
            }

            return true;
        }

        return false;
    }

    private static bool OverridesRequiredProperty(Type concreteType, Type? contractDeclaringType,
        string propertyName)
    {
        if (contractDeclaringType is null) return false;

        const BindingFlags declaredInstance = BindingFlags.Instance | BindingFlags.Public
            | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
        PropertyInfo? declaredProperty = contractDeclaringType.GetProperty(propertyName, declaredInstance);
        MethodInfo? declaredGetter = declaredProperty?.GetGetMethod(nonPublic: true);
        if (declaredGetter is null || !declaredGetter.IsVirtual) return false;

        PropertyInfo? effectiveProperty = null;
        for (Type? cursor = concreteType; cursor is not null; cursor = cursor.BaseType)
        {
            effectiveProperty = cursor.GetProperty(propertyName, declaredInstance);
            if (effectiveProperty is not null) break;
        }

        MethodInfo? effectiveGetter = effectiveProperty?.GetGetMethod(nonPublic: true);
        return effectiveGetter is not null
            && effectiveGetter.DeclaringType != declaredGetter.DeclaringType
            && effectiveGetter.GetBaseDefinition() == declaredGetter.GetBaseDefinition();
    }

    /// <summary>
    /// Names the axes of a concrete shape using a type's declared INPUT layout.
    /// </summary>
    /// <param name="type">The annotated layer or model type.</param>
    /// <param name="shape">A concrete shape.</param>
    /// <returns>Axis role to size, or <c>null</c> when no declared input layout accepts this rank.</returns>
    /// <remarks>
    /// Ambiguity is refused, not resolved. When several declared layouts accept the same rank — the
    /// batch-optional <c>[C,H,W]</c> against a genuine 3-axis form, say — the axis names differ between
    /// them and picking one arbitrarily would silently mislabel every axis downstream.
    /// </remarks>
    public static IReadOnlyDictionary<TensorAxis, int>? NameAxes(Type type, IReadOnlyList<int> shape)
    {
        if (type is null) throw new ArgumentNullException(nameof(type));
        if (shape is null) throw new ArgumentNullException(nameof(shape));

        var candidates = InputLayouts(type)
            .Select(layout => layout.AxesForRank(shape.Count))
            .Where(axes => axes is not null)
            .Select(axes => axes!)
            .ToList();

        if (candidates.Count == 0) return null;

        // Distinct by the axis sequence: two declarations that resolve to the SAME axes at this rank are
        // not an ambiguity, they are the same answer written twice.
        var distinct = candidates
            .GroupBy(a => string.Join(",", a))
            .Select(g => g.First())
            .ToList();

        if (distinct.Count != 1) return null;

        var chosen = distinct[0];
        var named = new Dictionary<TensorAxis, int>();
        for (int i = 0; i < chosen.Length && i < shape.Count; i++)
        {
            // A repeated role (two Other axes, say) cannot be addressed unambiguously by name, so the
            // whole naming is refused rather than silently keeping the last one.
            if (named.ContainsKey(chosen[i])) return null;
            named[chosen[i]] = shape[i];
        }

        return named;
    }

    /// <summary>
    /// Computes the output shape a type produces for a given input shape, from its declarations alone.
    /// </summary>
    /// <param name="instance">The layer or model. Must implement <see cref="IShapeContract"/>.</param>
    /// <param name="inputShape">A concrete input shape.</param>
    /// <returns>The inferred output shape, or <c>null</c> if it cannot be determined.</returns>
    /// <remarks>
    /// Reads the INSTANCE, not the type, so the relations reflect the configuration this object was
    /// actually built with — the stride, kernel and scale factor it was handed, not defaults.
    /// </remarks>
    public static int[]? InferOutputShape(object instance, IReadOnlyList<int> inputShape)
        => InferOutputShape(instance, inputShape, isBatched: true);

    /// <summary>
    /// Computes the output shape a type produces, saying whether <paramref name="inputShape"/>
    /// INCLUDES a batch axis.
    /// </summary>
    /// <param name="instance">The layer or model. Must implement <see cref="IShapeContract"/>.</param>
    /// <param name="inputShape">A concrete input shape.</param>
    /// <param name="isBatched">
    /// <c>true</c> when the leading axis is a batch - a real tensor; <c>false</c> for a PER-SAMPLE
    /// shape, which is what chain resolution propagates.
    /// </param>
    /// <returns>The inferred output shape, or <c>null</c> if it cannot be determined.</returns>
    /// <remarks>
    /// The single-argument overload defaults to <c>true</c>, preserving what every existing caller
    /// gets. Chain resolution must pass <c>false</c>: it propagates per-sample shapes, and a layer that
    /// treats the leading axis as a batch would otherwise collapse the wrong axes.
    /// </remarks>
    public static int[]? InferOutputShape(object instance, IReadOnlyList<int> inputShape, bool isBatched)
    {
        if (instance is null) throw new ArgumentNullException(nameof(instance));
        if (inputShape is null) throw new ArgumentNullException(nameof(inputShape));

        if (instance is not IShapeContract contract) return null;

        var named = NameAxes(instance.GetType(), inputShape);
        if (named is null) return null;

        var axes = contract.OutputAxesFor(inputShape.Count, isBatched);
        if (axes is null || axes.Count == 0) return null;

        var result = new int[axes.Count];
        for (int i = 0; i < axes.Count; i++)
        {
            if (!axes[i].Relation.TryResolve(named, out int size)) return null;
            result[i] = size;
        }

        return result;
    }

    /// <summary>
    /// Computes the output shape a MULTI-INPUT type produces for the given input shapes.
    /// </summary>
    /// <param name="instance">The layer or model. Must implement <see cref="IShapeContract"/>.</param>
    /// <param name="inputShapes">One concrete shape per input port, in port order.</param>
    /// <returns>The inferred output shape, or <c>null</c> if it cannot be determined.</returns>
    /// <remarks>
    /// <para>
    /// Axis NAMING still comes from the type's declared input layout, and every port is named with that
    /// same layout - which is correct for the layers this exists for, since a join requires its inputs
    /// to agree on every axis except the joined one. A port whose rank the layout does not accept makes
    /// the whole inference decline rather than resolving from the ports that did name.
    /// </para>
    /// <para>
    /// The single-shape overload remains the path for the overwhelming majority of layers; this is only
    /// consulted when a type genuinely has several inputs.
    /// </para>
    /// </remarks>
    public static int[]? InferOutputShape(object instance, IReadOnlyList<IReadOnlyList<int>> inputShapes)
    {
        if (instance is null) throw new ArgumentNullException(nameof(instance));
        if (inputShapes is null) throw new ArgumentNullException(nameof(inputShapes));
        if (inputShapes.Count == 0) return null;

        if (instance is not IShapeContract contract) return null;

        var named = new IReadOnlyDictionary<TensorAxis, int>[inputShapes.Count];
        var ranks = new int[inputShapes.Count];
        for (int i = 0; i < inputShapes.Count; i++)
        {
            var shape = inputShapes[i];
            if (shape is null) return null;
            var axes = NameAxes(instance.GetType(), shape);
            if (axes is null) return null;
            named[i] = axes;
            ranks[i] = shape.Count;
        }

        var outputAxes = contract.OutputAxesForPorts(ranks);
        if (outputAxes is null || outputAxes.Count == 0) return null;

        var result = new int[outputAxes.Count];
        for (int i = 0; i < outputAxes.Count; i++)
        {
            if (!outputAxes[i].Relation.TryResolve(named, out int size)) return null;
            result[i] = size;
        }

        return result;
    }

    /// <summary>
    /// Checks that a type's <see cref="IShapeContract"/> agrees with its declared output layout.
    /// </summary>
    /// <param name="instance">The layer or model.</param>
    /// <param name="mismatch">Describes the disagreement when there is one.</param>
    /// <returns><c>false</c> when the two declarations contradict each other.</returns>
    /// <remarks>
    /// Two claims by the same type about the same tensor. If the contract sizes
    /// <c>[Batch, Channels, Height, Width]</c> while the layout declares
    /// <c>[Batch, Height, Width, Channels]</c>, at least one is wrong, and the inferred shape would be a
    /// correct set of sizes in the wrong order — which is far harder to spot than an outright failure.
    /// A type with no output layout declared is skipped, not failed: annotating incrementally is allowed.
    /// </remarks>
    public static bool ContractMatchesLayout(object instance, int inputRank, out string? mismatch)
    {
        mismatch = null;
        if (instance is null) throw new ArgumentNullException(nameof(instance));
        if (instance is not IShapeContract contract) return true;

        var axes = contract.OutputAxesFor(inputRank);
        if (axes is null) return true;   // rank not accepted; nothing is claimed, so nothing contradicts

        var layouts = OutputLayouts(instance.GetType()).ToList();
        if (layouts.Count == 0) return true;

        var contractAxes = axes.Select(a => a.Axis).ToArray();

        foreach (var layout in layouts)
        {
            var declared = layout.AxesForRank(contractAxes.Length);
            if (declared is not null && declared.SequenceEqual(contractAxes)) return true;
        }

        mismatch =
            $"{instance.GetType().Name}: at input rank {inputRank} the shape contract sizes "
            + $"[{string.Join(", ", contractAxes)}], which matches none of its declared output layouts ["
            + string.Join(" | ", layouts.Select(l => l.ToString()))
            + "]. Both are claims by this type about the same tensor, so one of them is wrong.";
        return false;
    }

    /// <summary>
    /// Derives the coarse <see cref="Layers.ShapeRelationKind"/> implied by a symbolic contract.
    /// </summary>
    /// <param name="contract">The type's declared output axes.</param>
    /// <returns>The category the relations add up to.</returns>
    /// <remarks>
    /// <para>
    /// <see cref="Layers.ShapeRelationKind"/> already existed as a per-layer-type CATEGORY, and it
    /// carries no kernel, stride or padding — it can say "spatial axes follow the convolution formula"
    /// but nothing can evaluate that, which is precisely the gap <see cref="IShapeContract"/> fills.
    /// The two must not become independent claims that drift apart, so the coarse kind is DERIVED here
    /// and a conformance test holds each layer's hand-declared <c>OutputShapeRelation</c> to it. One
    /// source of truth, with the older, coarser view computed from it rather than maintained beside it.
    /// </para>
    /// <para>
    /// Anything that does not fall into a named category is <c>Unknown</c>, which is the honest answer
    /// and also the existing default — a layer whose relations are genuinely richer than these four
    /// buckets loses nothing, because the symbolic contract remains available and is strictly more
    /// informative.
    /// </para>
    /// </remarks>
    public static Layers.ShapeRelationKind DeriveRelationKind(IReadOnlyList<OutputAxisContract> contract)
    {
        if (contract is null) throw new ArgumentNullException(nameof(contract));
        if (contract.Count == 0) return Layers.ShapeRelationKind.Unknown;

        bool IsSame(int i) => contract[i].Relation.Kind == AxisRelation.Form.Same;
        bool IsFixed(int i) => contract[i].Relation.Kind == AxisRelation.Form.Fixed;
        bool IsWindow(int i) => contract[i].Relation.Kind == AxisRelation.Form.Window;

        if (Enumerable.Range(0, contract.Count).All(IsSame)) return Layers.ShapeRelationKind.Identity;

        // The layer-determined axis is the one it FIXES. Which end it sits at is what separates the
        // channel-first world from the feature-last one, and reading a feature-last layer with a
        // channel-first rule inverts which axis is a real claim.
        int fixedCount = Enumerable.Range(0, contract.Count).Count(IsFixed);
        if (fixedCount == 1)
        {
            // Batch is a passthrough in every one of these categories, so it never decides the shape.
            var spatial = Enumerable.Range(0, contract.Count)
                .Where(i => contract[i].Axis != TensorAxis.Batch)
                .ToList();
            if (spatial.Count > 0)
            {
                int first = spatial[0];
                int last = spatial[spatial.Count - 1];
                var rest = spatial.Where(i => i != first).ToList();
                var allButLast = spatial.Where(i => i != last).ToList();

                if (IsFixed(first) && rest.All(IsWindow)) return Layers.ShapeRelationKind.Convolutional;
                if (IsFixed(first) && rest.All(IsSame)) return Layers.ShapeRelationKind.ChannelOnly;
                if (IsFixed(last) && allButLast.All(IsSame)) return Layers.ShapeRelationKind.FeatureOnly;
            }
        }

        return Layers.ShapeRelationKind.Unknown;
    }

    /// <summary>Declared input layouts, read through the open generic definition.</summary>
    /// <remarks>
    /// Through the OPEN generic definition deliberately: attributes are declared on
    /// <c>ConvolutionalLayer&lt;T&gt;</c>, and reflecting over the closed
    /// <c>ConvolutionalLayer&lt;double&gt;</c> finds them only because the runtime forwards the lookup.
    /// Going straight to the definition keeps that independent of how the instance was closed.
    /// </remarks>
    public static IEnumerable<TensorLayoutAttribute> InputLayouts(Type type)
        => LayoutsFor(type, TensorLayoutDirection.Input);

    /// <summary>Declared output layouts, read through the open generic definition.</summary>
    public static IEnumerable<TensorLayoutAttribute> OutputLayouts(Type type)
        => LayoutsFor(type, TensorLayoutDirection.Output);

    private static IEnumerable<TensorLayoutAttribute> LayoutsFor(Type type, TensorLayoutDirection direction)
    {
        if (type is null) throw new ArgumentNullException(nameof(type));
        var definition = type.IsGenericType && !type.IsGenericTypeDefinition
            ? type.GetGenericTypeDefinition()
            : type;

        return definition
            .GetCustomAttributes(typeof(TensorLayoutAttribute), inherit: true)
            .Cast<TensorLayoutAttribute>()
            .Where(a => a.Direction == direction);
    }
}
