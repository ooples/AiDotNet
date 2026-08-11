using System;
using System.Collections.Generic;
using AiDotNet.Models;
using AiDotNet.Models.Parameters;
using AiDotNet.Serialization;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Cloning for layers, built on the construction state layers already record for serialization.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> <c>layer.Clone()</c> gives you a separate copy of a layer, including what
/// it has learned. You do not write anything to make this work on a layer of your own: mark the
/// constructor arguments the layer needs with <c>[LayerState]</c> — which it already needs for
/// saving and loading — and cloning follows.
/// </para>
/// <para>
/// This deliberately reuses <c>WriteConstructionState</c> and <c>GeneratedLayerFactories.TryCreate</c>
/// rather than introducing a second reconstruction mechanism. Those already call the layer's real
/// constructor with the values it was originally given, and the build already fails for a layer
/// whose required state cannot be sourced. A separate clone-only path would be a second thing to
/// keep correct, and the two would be free to disagree — which is the entire failure this work
/// exists to remove. Sharing one path means a layer that saves and loads correctly also clones
/// correctly, by construction.
/// </para>
/// <para>
/// The learned parameters travel separately, through <c>GetParameters</c> and
/// <c>UpdateParameters</c>. That is the contract training exercises on every step, so a clone
/// cannot disagree with training about what the parameters are.
/// </para>
/// </remarks>
public static class LayerCloning
{
    /// <summary>
    /// Creates an independent copy of a layer.
    /// </summary>
    /// <typeparam name="T">The layer's numeric type.</typeparam>
    /// <param name="source">The layer to copy.</param>
    /// <param name="options">What the copy carries; defaults to <see cref="CloneOptions.Full"/>.</param>
    /// <returns>A new layer of the same type and configuration.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="source"/> is null.</exception>
    /// <exception cref="NotSupportedException">
    /// Thrown when no generated factory exists for the layer's type.
    /// </exception>
    /// <remarks>
    /// <para>
    /// The clone is rebuilt by calling the layer's constructor with its recorded state, so
    /// everything the constructor derives — weight buffers sized from the output width, sub-layers,
    /// initialization strategy — is re-derived rather than copied. A stale derived value in the
    /// original therefore cannot reach the copy, which is the advantage reconstruction has over
    /// field-copying.
    /// </para>
    /// <para>
    /// Learned parameters are then written in, when <see cref="CloneOptions.IncludeParameters"/>
    /// says so. With it off the result is the same architecture, freshly initialized — the
    /// equivalent of scikit-learn's <c>clone()</c>, which returns an unfitted estimator carrying
    /// the same hyperparameters.
    /// </para>
    /// </remarks>
    public static ILayer<T> Clone<T>(this LayerBase<T> source, CloneOptions? options = null)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));

        var settings = options ?? CloneOptions.Full;
        var clone = Reconstruct(source);

        if (settings.IncludeParameters)
        {
            var parameters = source.GetParameters();
            if (parameters.Length > 0)
            {
                // Rebuilding from the same construction state must produce the same parameter
                // shape. A mismatch means a value the constructor needs is not recorded, and the
                // copy would silently differ from the original -- so it is reported rather than
                // written through. This is the backstop for what the build cannot prove: a
                // constructor that reads state from somewhere other than its arguments.
                // Compare LAYOUTS where both sides publish one, and only fall back to scalar
                // counts where they do not. A rebuilt layer has not run a forward pass, so a
                // deferred slot makes ParameterCount throw ParameterLayoutNotReadyException rather
                // than return a different number -- the scalar comparison cannot even be evaluated
                // there, let alone trusted. Slot-wise comparison also catches a same-total
                // reordering, which the scalar test passes and which would restore each
                // component's values into its neighbour.
                if (clone is IParameterManifestProvider cloneLayout
                    && source is IParameterManifestProvider sourceLayout)
                {
                    var expected = sourceLayout.ParameterLayout;
                    var actual = cloneLayout.ParameterLayout;
                    if (!actual.DescribesSameLayoutAs(expected))
                    {
                        throw new InvalidOperationException(
                            $"{source.GetType().Name} rebuilt with a different parameter layout: "
                            + $"{actual.DescribeDifferenceFrom(expected)}. A constructor argument "
                            + "that determines size is not marked [LayerState], so the copy is a "
                            + "different shape from the original.");
                    }
                }
                else if (clone.ParameterCount != source.ParameterCount)
                {
                    throw new InvalidOperationException(
                        $"{source.GetType().Name} rebuilt with {clone.ParameterCount} parameters but "
                        + $"the original has {source.ParameterCount}. A constructor argument that "
                        + "determines size is not marked [LayerState], so the copy is a different "
                        + "shape from the original.");
                }

                clone.UpdateParameters(parameters);
            }
        }

        return clone;
    }

    /// <summary>
    /// Rebuilds a layer from the construction state it records for serialization.
    /// </summary>
    /// <typeparam name="T">The layer's numeric type.</typeparam>
    /// <param name="source">The layer to rebuild.</param>
    /// <returns>A new, freshly constructed layer of the same type and configuration.</returns>
    /// <exception cref="NotSupportedException">Thrown when the type has no generated factory.</exception>
    private static LayerBase<T> Reconstruct<T>(LayerBase<T> source)
    {
        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
        source.WriteConstructionState(metadata);

        var values = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var pair in metadata) values[pair.Key] = pair.Value;

        var type = source.GetType();
        var bag = new LayerStateBag(values, type.Name);

        var definition = type.IsGenericType ? type.GetGenericTypeDefinition() : type;

        // The generated table first -- it is compile-checked and names the constructor directly.
        // Then the registry, which is the only way a layer defined in ANOTHER assembly can take
        // part: its generated class lives there and this one cannot name it. Then reflection, so a
        // hand-written layer works without its author registering anything at all.
        if (!GeneratedLayerFactories<T>.TryCreate(
                definition, bag, source.ScalarActivation, source.VectorActivation, out var rebuilt)
            && !LayerFactoryRegistry<T>.TryCreate(
                type, definition, bag, source.ScalarActivation, source.VectorActivation, out rebuilt))
        {
            throw new NotSupportedException(
                $"{type.Name} cannot be rebuilt: no generated factory, no registered factory, and its "
                + "constructor could not be satisfied from the saved state. If this layer lives "
                + "outside AiDotNet, register a factory with "
                + $"LayerFactoryRegistry<{typeof(T).Name}>.Register, or make sure each constructor "
                + "argument is stored in a field of the same name so it is written at save time.");
        }

        if (rebuilt is not LayerBase<T> layer)
        {
            throw new NotSupportedException(
                $"{type.Name} was rebuilt as {rebuilt?.GetType().Name ?? "null"}, which is not a layer.");
        }

        return layer;
    }
}
