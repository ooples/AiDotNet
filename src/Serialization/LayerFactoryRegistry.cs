using System.Collections.Concurrent;
using System.Reflection;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Serialization;

/// <summary>
/// Rebuilds a layer that the generated factory table does not know about.
/// </summary>
/// <typeparam name="T">The layer's numeric type.</typeparam>
/// <remarks>
/// <para>
/// <c>GeneratedLayerFactories</c> is compiled from AiDotNet's own source, so it can only ever name
/// layers AiDotNet ships. A layer defined in a consumer's assembly has no entry in it and never
/// will: the generator does not run in their compilation, and even if it did, their generated class
/// would live in their assembly where this one cannot name it. Cloning a user-defined layer failed
/// for that reason, with an error telling the author to add <c>[LayerState]</c> -- advice that
/// cannot work from outside this assembly.
/// </para>
/// <para>
/// So there are two ways in besides the generated table. <see cref="Register"/> takes a factory
/// from any assembly, which is what generated code in a consumer's project will call. Failing that,
/// a layer is rebuilt by reading its constructor's parameters out of the saved state by name --
/// slower and unverified at compile time, but it means a hand-written layer works without its
/// author registering anything, which is the promise the options half of this already keeps.
/// </para>
/// </remarks>
public static class LayerFactoryRegistry<T>
{
    private static readonly ConcurrentDictionary<Type, Func<LayerStateBag, object?, object?, object?>> Registered =
        new();

    /// <summary>Registers a factory for a layer this assembly cannot name.</summary>
    /// <param name="genericDefinition">The layer's open generic type, or the type itself if not generic.</param>
    /// <param name="factory">Builds the layer from its saved state and restored activations.</param>
    public static void Register(
        Type genericDefinition,
        Func<LayerStateBag, object?, object?, object?> factory)
    {
        if (genericDefinition is null) throw new ArgumentNullException(nameof(genericDefinition));
        if (factory is null) throw new ArgumentNullException(nameof(factory));

        Registered[genericDefinition] = factory;
    }

    /// <summary>Whether a factory has been registered for the given open generic type.</summary>
    /// <param name="genericDefinition">The layer's open generic type.</param>
    /// <returns><c>true</c> when a registered factory exists.</returns>
    public static bool IsRegistered(Type genericDefinition) => Registered.ContainsKey(genericDefinition);

    /// <summary>Rebuilds a layer from a registered factory, or by reading its constructor.</summary>
    /// <param name="closedType">The layer's closed type.</param>
    /// <param name="genericDefinition">The layer's open generic type.</param>
    /// <param name="state">The layer's saved construction state.</param>
    /// <param name="scalarActivation">The restored scalar activation, if any.</param>
    /// <param name="vectorActivation">The restored vector activation, if any.</param>
    /// <param name="layer">The rebuilt layer.</param>
    /// <returns><c>true</c> when the layer could be rebuilt.</returns>
    public static bool TryCreate(
        Type closedType,
        Type genericDefinition,
        LayerStateBag state,
        object? scalarActivation,
        object? vectorActivation,
        out object layer)
    {
        if (Registered.TryGetValue(genericDefinition, out var factory)
            && factory(state, scalarActivation, vectorActivation) is { } registered)
        {
            layer = registered;
            return true;
        }

        return TryReflect(closedType, state, scalarActivation, vectorActivation, out layer);
    }

    /// <summary>
    /// Rebuilds by matching constructor parameters to saved values by name.
    /// </summary>
    /// <remarks>
    /// The metadata keys ARE the parameter names, which is what makes this possible at all. The
    /// widest constructor every one of whose parameters can be supplied is preferred, so a layer
    /// with both a full and a convenience constructor is rebuilt through the one carrying the most
    /// state rather than the one that discards it.
    /// </remarks>
    private static bool TryReflect(
        Type closedType,
        LayerStateBag state,
        object? scalarActivation,
        object? vectorActivation,
        out object layer)
    {
        layer = null!;

        foreach (var ctor in closedType.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                     .OrderByDescending(c => c.GetParameters().Count(p => state.Has(p.Name ?? string.Empty))))
        {
            var formal = ctor.GetParameters();
            var args = new object?[formal.Length];
            var usable = true;

            for (var i = 0; i < formal.Length && usable; i++)
            {
                var p = formal[i];
                var name = p.Name ?? string.Empty;

                if (IsActivation(p.ParameterType))
                {
                    args[i] = p.ParameterType.IsInstanceOfType(vectorActivation) ? vectorActivation
                        : p.ParameterType.IsInstanceOfType(scalarActivation) ? scalarActivation
                        : p.HasDefaultValue ? p.DefaultValue
                        : null;
                    continue;
                }

                if (state.Has(name) && TryRead(state, name, p.ParameterType, out var value))
                {
                    args[i] = value;
                    continue;
                }

                // An absent optional argument takes the default its signature declares, never
                // default(T) -- restoring `useBias = true` as false is the silent loss this exists
                // to prevent.
                if (p.HasDefaultValue) { args[i] = p.DefaultValue; continue; }

                usable = false;
            }

            if (!usable) continue;

            try
            {
                layer = ctor.Invoke(args);
                return true;
            }
            catch (TargetInvocationException)
            {
                // A constructor that rejects these values is the wrong one; try a narrower.
            }
        }

        return false;
    }

    private static bool IsActivation(Type type)
        => type.IsInterface
           && type.IsGenericType
           && type.Name is "IActivationFunction`1" or "IVectorActivationFunction`1";

    private static bool TryRead(LayerStateBag state, string key, Type target, out object? value)
    {
        var type = Nullable.GetUnderlyingType(target) ?? target;
        value = null;

        try
        {
            if (type == typeof(int)) { value = state.Int32(key); return true; }
            if (type == typeof(long)) { value = state.Int64(key); return true; }
            if (type == typeof(double)) { value = state.Double(key); return true; }
            if (type == typeof(float)) { value = state.Single(key); return true; }
            if (type == typeof(bool)) { value = state.Boolean(key); return true; }
            if (type == typeof(string)) { value = state.String(key); return true; }
            if (type == typeof(int[])) { value = state.Int32Array(key); return true; }
            if (type == typeof(double[])) { value = state.DoubleArray(key); return true; }
            if (type == typeof(bool[])) { value = state.BooleanArray(key); return true; }
            if (type == typeof(string[])) { value = state.StringArray(key); return true; }
            if (type == typeof(int[][])) { value = state.Int32Jagged(key); return true; }
            if (type.IsEnum) { value = Enum.Parse(type, state.String(key), ignoreCase: true); return true; }
        }
        catch (Exception)
        {
            // An unparseable value means this constructor cannot be satisfied from what was saved.
        }

        return false;
    }
}
