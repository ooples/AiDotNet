using System;
using System.Linq;
using System.Collections.Generic;
using AiDotNet.Models;
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

        // The clone derives its own stream unless asked to reuse the original's. A layer with no
        // seed set has opted out of reproducibility, and copying null keeps it opted out.
        if (settings.ShareRandomState) clone.RandomSeed = source.RandomSeed;

        if (settings.IncludeParameters)
        {
            // INSTALL TENSORS, NOT A FLAT VECTOR. A tensor carries its own shape, so installing one
            // resolves a clone whose input width is lazy; a flat Vector<T> carries no shape, and
            // pushing 16 values into a DenseLayer rebuilt from `outputSize` alone threw "Expected 0
            // parameters, but got 16". That is why cloning a layer which had been USED failed while
            // cloning a fresh one appeared to work: both sides were unresolved and agreed at zero.
            // RESOLVE THE CLONE BEFORE INSTALLING. A lazy layer materializes its weights on its
            // first forward, and that initialization overwrites anything installed beforehand: the
            // clone came back structurally right but carrying fresh random weights, and the
            // DenseLayer round trip read "original 0, clone 0.36892061820885858". Resolving here
            // means the install writes into tensors that already exist, so the first forward has
            // nothing left to initialize. Only meaningful when the SOURCE is resolved -- cloning an
            // untouched layer should stay untouched.
            if (source.IsShapeResolved && !clone.IsShapeResolved)
            {
                var resolved = source.GetInputShape();
                var batched = new int[resolved.Length + 1];
                batched[0] = 1;
                Array.Copy(resolved, 0, batched, 1, resolved.Length);

                // Two shape conventions, same reason the sweep probes both: GetInputShape describes
                // one sample for most layers and the full input for others.
                foreach (var candidate in new[] { resolved, batched })
                {
                    try { clone.ResolveFromShape(candidate); break; }
                    catch (ArgumentException) { /* try the other; install as-is if neither fits */ }
                    catch (InvalidOperationException) { }
                }
            }


            var tensors = source.GetTrainableParameters();
            if (tensors.Count > 0)
            {
                var installed = new Tensor<T>[tensors.Count];
                for (var i = 0; i < tensors.Count; i++)
                {
                    // Shared hands over the ORIGINAL tensors, so both handles are one set of
                    // weights and training either trains both.
                    //
                    // Deep and CopyOnWrite both take CloneShared views. They are observationally
                    // identical by construction -- the first write on either side splits them -- so
                    // a copy-on-write view IS a deep copy, reached without materialising a second
                    // set of weights. This is what NeuralNetworkBase.DeepCopy already relies on.
                    installed[i] = settings.Mode == CloneMode.Shared
                        ? tensors[i]
                        : (Tensor<T>)tensors[i].CloneShared();
                }

                clone.SetTrainableParameters(installed);

                // AFTER the install, not before. Checking first measured an empty clone against a
                // resolved original and reported every lazy layer as broken.
                if (clone.ParameterCount != source.ParameterCount)
                {
                    throw new InvalidOperationException(
                        $"{source.GetType().Name} rebuilt with {clone.ParameterCount} parameters but "
                        + $"the original has {source.ParameterCount}. A constructor argument that "
                        + "determines size is not recorded, so the copy is a different shape from "
                        + "the original.");
                }
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
