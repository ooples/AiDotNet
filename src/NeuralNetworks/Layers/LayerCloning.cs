using System;
using System.Linq;
using System.Collections.Generic;
using System.Reflection;
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
    private const string CloneRandomSeedKey = "__aidotnet_clone_random_seed";

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
        int? cloneSeed = settings.ShareRandomState
            ? source.RandomSeed
            : DeriveCloneSeed(source.RandomSeed);

        // An unseeded architecture clone still needs a genuinely fresh initialization. Several
        // legacy layers expose `seed = 42` on their constructor without reflecting it into
        // LayerBase.RandomSeed; replaying that literal would make every configuration-only clone
        // start from identical weights. The reserved factory value affects construction only. The
        // public RandomSeed remains null, preserving the caller's deliberate unseeded contract.
        int? constructionSeed = cloneSeed;
        if (!settings.IncludeParameters && !constructionSeed.HasValue)
        {
            constructionSeed = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom().Next();
        }

        var clone = Reconstruct(source, constructionSeed);

        // A shared stream must restart from the same deterministic seed. The default derives a
        // different, reproducible stream so two independently-trained clones do not receive the
        // same stochastic masks forever. An unseeded source remains intentionally unseeded.
        clone.RandomSeed = cloneSeed;

        if (settings.IncludeParameters || settings.IncludeBuffers || settings.IncludeOptimizerState)
        {
            InstallInto(source, clone, settings);

            // AFTER the install, not before. Checking first measured an empty clone against a
            // resolved original and reported every lazy layer as broken.
            if (settings.IncludeParameters && clone.ParameterCount != source.ParameterCount)
            {
                throw new InvalidOperationException(
                    $"{source.GetType().Name} rebuilt with {clone.ParameterCount} parameters but "
                    + $"the original has {source.ParameterCount}. A constructor argument that "
                    + "determines size is not recorded, so the copy is a different shape from "
                    + "the original.");
            }
        }


        // LAST: reconstruction and any shape-resolution probe may consume stochastic counters or
        // Random instances. Apply the requested stream semantics only after that work is finished.
        CopyRandomState(source, clone, settings.ShareRandomState);

        return clone;
    }

    private static void CopyRandomState<T>(
        LayerBase<T> source,
        LayerBase<T> clone,
        bool shareRandomState)
    {
        source.CopyBaseRandomStateTo(clone, shareRandomState);

        int randomFieldIndex = 0;
        for (Type? type = source.GetType();
             type is not null && type != typeof(LayerBase<T>);
             type = type.BaseType)
        {
            foreach (var field in type.GetFields(
                         BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public
                         | BindingFlags.DeclaredOnly))
            {
                if (field.FieldType == typeof(Random))
                {
                    var sourceRandom = (Random?)field.GetValue(source);
                    if (sourceRandom is null) continue;

                    Random replacement;
                    if (shareRandomState)
                    {
                        replacement = CloneRandom(sourceRandom);
                    }
                    else if (clone.RandomSeed.HasValue)
                    {
                        int fieldSeed = DeriveFieldSeed(
                            clone.RandomSeed.Value, field.Name, randomFieldIndex++);
                        replacement = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(fieldSeed);
                    }
                    else
                    {
                        // The constructor already supplied an independent secure stream.
                        continue;
                    }

                    field.SetValue(clone, replacement);
                    continue;
                }

                if (!IsStochasticCounter(field)) continue;
                field.SetValue(clone, shareRandomState
                    ? field.GetValue(source)
                    : Activator.CreateInstance(field.FieldType));
            }
        }
    }

    /// <summary>
    /// Preserves the future initialization state of a copy-on-write layer whose parameter tensors
    /// are still wholly deferred.
    /// </summary>
    internal static void CopyDeferredRandomState<T>(LayerBase<T> source, LayerBase<T> clone)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (clone is null) throw new ArgumentNullException(nameof(clone));
        if (source.GetType() != clone.GetType())
            throw new ArgumentException("Deferred random state requires matching layer types.", nameof(clone));

        clone.RandomSeed = source.RandomSeed;
        CopyRandomState(source, clone, shareRandomState: true);
    }

    private static bool IsStochasticCounter(FieldInfo field)
    {
        if (field.IsInitOnly || field.IsStatic
            || !field.Name.EndsWith("Counter", StringComparison.OrdinalIgnoreCase))
            return false;

        bool stochasticName = field.Name.IndexOf("seed", StringComparison.OrdinalIgnoreCase) >= 0
                              || field.Name.IndexOf("dropPath", StringComparison.OrdinalIgnoreCase) >= 0
                              || field.Name.IndexOf("init", StringComparison.OrdinalIgnoreCase) >= 0;
        if (!stochasticName) return false;

        Type type = field.FieldType;
        return type == typeof(int) || type == typeof(uint)
            || type == typeof(long) || type == typeof(ulong);
    }

    private static int DeriveFieldSeed(int seed, string fieldName, int index)
    {
        uint hash = unchecked((uint)seed) ^ unchecked((uint)index * 0x9E3779B9u);
        for (int i = 0; i < fieldName.Length; i++)
            hash = unchecked((hash ^ fieldName[i]) * 16777619u);
        hash ^= hash >> 16;
        hash *= 0x85EBCA6Bu;
        hash ^= hash >> 13;
        return unchecked((int)hash);
    }

    private static readonly MethodInfo MemberwiseCloneMethod = typeof(object).GetMethod(
        "MemberwiseClone", BindingFlags.Instance | BindingFlags.NonPublic)
        ?? throw new InvalidOperationException("Object.MemberwiseClone is unavailable.");

    private static Random CloneRandom(Random source)
    {
        var clone = (Random)MemberwiseCloneMethod.Invoke(source, null)!;

        // .NET Framework's Random stores its mutable seed table in an int[]; modern runtimes use
        // primitive state fields or a private implementation object. Deep-copy both representations
        // so source and clone advance identically but never consume one shared mutable stream.
        for (Type? type = source.GetType(); type is not null; type = type.BaseType)
        {
            foreach (var field in type.GetFields(
                         BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public
                         | BindingFlags.DeclaredOnly))
            {
                object? value = field.GetValue(source);
                if (value is Array array)
                    field.SetValue(clone, array.Clone());
                else if (value is not null
                         && !field.FieldType.IsValueType
                         && field.FieldType != typeof(string))
                    field.SetValue(clone, MemberwiseCloneMethod.Invoke(value, null));
            }
        }

        return clone;
    }

    private static int? DeriveCloneSeed(int? sourceSeed)
    {
        if (!sourceSeed.HasValue) return null;

        // SplitMix's integer avalanche gives a stable child seed without consuming or coupling the
        // source stream. Keep this local and deterministic across target frameworks.
        uint value = unchecked((uint)sourceSeed.Value + 0x9E3779B9u);
        value = (value ^ (value >> 16)) * 0x85EBCA6Bu;
        value = (value ^ (value >> 13)) * 0xC2B2AE35u;
        return unchecked((int)(value ^ (value >> 16)));
    }

    /// <summary>
    /// Copies one layer's learned tensors into another, then does the same for its sub-layers.
    /// </summary>
    /// <remarks>
    /// Recursive because <see cref="LayerBase{T}.GetTrainableParameters"/> reports only a layer's
    /// OWN tensors. A composite therefore installed nothing into its children, which kept whatever
    /// the constructor gave them — <c>SwinTransformerBlockLayer</c> rebuilt 98 parameters against
    /// the original's 130, its six registered children never filled.
    /// </remarks>
    private static void InstallInto<T>(LayerBase<T> source, LayerBase<T> clone, CloneOptions settings)
    {
        // RESOLVE THE CLONE BEFORE INSTALLING. A lazy layer materializes its weights on its first
        // forward, and that initialization overwrites anything installed beforehand: the clone came
        // back structurally right but carrying fresh random weights, and the DenseLayer round trip
        // read "original 0, clone 0.36892061820885858". Resolving here means the install writes into
        // tensors that already exist, so the first forward has nothing left to initialize. Only
        // meaningful when the SOURCE is resolved -- cloning an untouched layer should stay untouched.
        int[]? declared = null;
        try
        {
            declared = source.GetInputShape();
        }
        catch (Exception)
        {
            // A layer that will not describe its input cannot be probed; the install below still
            // runs and the count assertion still reports any shortfall.
        }

        // EXACT means every axis is concrete. Only then may the clone be RESOLVED from it, because
        // ResolveFromShape pins the axes it is given and pinning one to a guess would contradict
        // whatever length the layer is actually used at later.
        var exact = declared is not null && Array.TrueForAll(declared, d => d > 0) ? declared : null;

        // COPY BEFORE RESOLVING. Installing tensors already resolves a lazily-shaped clone --
        // a tensor carries its shape -- so resolving FIRST only forced every lazy child to
        // allocate and randomly initialize weights that the copy below immediately overwrote.
        // For a deep VAE that cost minutes per clone in InitializeWeights alone; copying first
        // lets ConvolutionalLayer.EnsureInitialized hit its idempotent "kernels already have the
        // expected shape" path and skip both the allocation and the RNG fill.
        CopyOwnTensors(source, clone, settings);

        if (exact is not null && !clone.IsShapeResolved)
        {
            // Two shape conventions, same reason the sweep probes both: GetInputShape describes
            // one sample for most layers and the full input for others.
            foreach (var candidate in new[] { exact, WithBatchAxis(exact) })
            {
                try { clone.ResolveFromShape(candidate); break; }
                catch (ArgumentException) { /* try the other; install as-is if neither fits */ }
                catch (InvalidOperationException) { }
            }
        }

        CopyChildren(source, clone, settings);

        if (clone.ParameterCount == source.ParameterCount || declared is null) return;

        // A SECOND PASS BEHIND A FORWARD, because there are two different ways a composite arrives
        // under-filled and neither mechanism covers the other.
        //
        // SwinTransformerBlockLayer registers its six children in the constructor, so they exist on
        // both sides and CopyChildren pairs them off; a forward probe cannot help it at all, since
        // GetInputShape reports [dim] while the block actually consumes a spatial input and every
        // candidate shape throws. CitrinetBlockLayer, ContextNetBlockLayer, HiFiGANResBlockLayer and
        // WaveNetResidualBlockLayer are the mirror image: the generated EnsureSubLayersRegistered()
        // runs during shape resolution, so a clone that never resolved has NO children for
        // CopyChildren to pair with and came back holding 0 parameters against the original's 401.
        // A forward is what brings those into existence.
        //
        // Those four also explain why this cannot wait for IsShapeResolved. They declare
        // [channels, -1] and the -1 is a genuinely free axis, so the flag reads false even on a
        // layer that HAS been forwarded and has materialized all nine children -- gating on it
        // skipped precisely the layers that needed the probe. A free axis therefore has to be
        // filled with a guess to forward at all, and that is safe here for the same reason it is
        // free: the layer does not pin it (IsShapeResolved is still false afterwards) and it
        // contributes no parameters. Should either assumption fail, the guessed shape produces the
        // wrong count and the assertion in Clone reports it rather than returning a quiet mis-copy.
        //
        // The probe runs only once the cheap paths have been tried and the counts still disagree --
        // a forward has side effects, and ResetState clears what it leaves behind before the retry
        // writes the real weights over the fresh random ones the probe just initialized.
        foreach (var candidate in ProbeShapes(declared))
        {
            try
            {
                clone.Forward(new Tensor<T>(candidate));
                clone.ResetState();
                break;
            }
            catch (Exception)
            {
                // A layer that refuses this probe keeps whatever it managed to resolve; the count
                // assertion after the install still reports the shortfall.
            }
        }

        CopyOwnTensors(source, clone, settings);
        CopyChildren(source, clone, settings);
    }

    /// <summary>Prepends a size-1 batch axis to a shape.</summary>
    internal static int[] WithBatchAxis(int[] shape)
    {
        var batched = new int[shape.Length + 1];
        batched[0] = 1;
        Array.Copy(shape, 0, batched, 1, shape.Length);

        return batched;
    }

    /// <summary>
    /// Concrete shapes to try forwarding through a clone, derived from a declared input shape.
    /// </summary>
    /// <remarks>
    /// Free axes come back as <c>-1</c> and are filled with a concrete length; both the with-batch
    /// and without-batch conventions are offered because <c>GetInputShape</c> describes one sample
    /// for some layers and the full input for others. Two fill sizes rather than one: a strided
    /// block consumes length, so <c>CitrinetBlockLayer</c> (kernel 3, stride 2) has nothing left to
    /// convolve at length 4 and only the longer probe survives.
    /// </remarks>
    internal static IEnumerable<int[]> ProbeShapes(int[] declared)
    {
        // 16 before 4, and NOT 1. Trying a length-1 fill first was measured and rejected: it made
        // DeepCopy neutral-to-worse (CanaryQwen 6,341 -> 6,953 ms, F5TTS 1,783 -> 2,413 ms) even
        // though it succeeded on the first candidate every time, so the extra attempt was not the
        // cost. What that rules out is the assumption behind it -- the probe's expense is not the
        // sequence-length arithmetic, it is materializing the layer's weights, and that is sized by
        // the layer rather than by the probe. Shortening the free axis cannot make it cheaper, which
        // means the probe cost is work the clone has to do anyway rather than overhead to remove.
        foreach (var fill in new[] { 16, 4 })
        {
            var concrete = new int[declared.Length];
            for (var i = 0; i < declared.Length; i++) concrete[i] = declared[i] > 0 ? declared[i] : fill;

            yield return WithBatchAxis(concrete);
            yield return concrete;

            // A shape that was already concrete does not vary with the fill, so the second pass
            // over it would repeat four throwing probes for nothing.
            if (Array.TrueForAll(declared, d => d > 0)) yield break;
        }
    }

    /// <summary>Writes a layer's own learned tensors into another layer of the same type.</summary>
    private static void CopyOwnTensors<T>(LayerBase<T> source, LayerBase<T> clone, CloneOptions settings)
    {
        // INSTALL TENSORS, NOT A FLAT VECTOR. A tensor carries its own shape, so installing one
        // resolves a clone whose input width is lazy; a flat Vector<T> carries no shape, and pushing
        // 16 values into a DenseLayer rebuilt from `outputSize` alone threw "Expected 0 parameters,
        // but got 16". That is why cloning a layer which had been USED failed while cloning a fresh
        // one appeared to work: both sides were unresolved and agreed at zero.
        CopyRegisteredBuffers(source, clone, settings);

        if (!settings.IncludeParameters) return;

        var tensors = source.GetTrainableParameters();
        if (tensors.Count == 0) return;

        var installed = new Tensor<T>[tensors.Count];
        for (var i = 0; i < tensors.Count; i++)
        {
            // Shared hands over the ORIGINAL tensors, so both handles are one set of weights and
            // training either trains both.
            //
            installed[i] = CloneTensorForMode(tensors[i], settings.Mode);
        }

        clone.SetTrainableParameters(installed);
    }

    /// <summary>Copies registered buffer values, which the trainable install cannot reach.</summary>
    /// <remarks>
    /// <c>GetTrainableParameters</c> is the OPTIMIZER view and omits every buffer, so the only
    /// reason a clone ever came back holding one was that buffers rode the flat parameter vector.
    /// An input-sized buffer cannot ride it -- its width is the caller's data rather than the
    /// architecture -- so the values are copied here instead.
    ///
    /// By NAME, for the reason the serialized block is keyed by name: registration order can differ
    /// between a used instance and a freshly reconstructed one, and pairing by position would load
    /// an adjacency matrix into an edge-feature slot. A clone that already holds a same-width tensor
    /// is written through, matching the restore path; otherwise the generated field map installs it.
    /// </remarks>
    private static void CopyRegisteredBuffers<T>(LayerBase<T> source, LayerBase<T> clone, CloneOptions settings)
    {
        var sourceBuffers = source.GetRegisteredBufferState();
        if (sourceBuffers is null || sourceBuffers.Count == 0) return;

        var target = new Dictionary<string, Tensor<T>>(StringComparer.Ordinal);
        var cloneBuffers = clone.GetRegisteredBuffers();
        if (cloneBuffers is not null)
        {
            foreach (var (name, tensor) in cloneBuffers)
            {
                if (tensor is not null) target[name] = tensor;
            }
        }

        foreach (var (name, tensor, persistenceRole, _) in sourceBuffers)
        {
            if (tensor is null || string.IsNullOrEmpty(name)) continue;

            bool optimizerState = persistenceRole == PersistentTensorRole.OptimizerState;
            if (optimizerState ? !settings.IncludeOptimizerState : !settings.IncludeBuffers)
                continue;

            // Shared means aliasing storage, including when the constructor allocated a same-sized
            // destination buffer. The generated field map performs the rebind and retains the
            // declaration's original roles.
            if (settings.Mode == CloneMode.Shared
                && clone.InstallRestoredBuffer(name, tensor))
                continue;

            if (target.TryGetValue(name, out var existing) && existing.Length == tensor.Length)
            {
                if (ReferenceEquals(existing, tensor)) continue;
                // BULK COPY, not a per-element indexer loop. Tensor's indexer is a generic
                // virtual call, so copying a 512x512x128 activation buffer one element at a
                // time ran 33.5M dispatches for a single buffer -- the dominant cost in a
                // 231-second VAEDecoder clone. GetDataArray is the write-intent accessor and
                // un-shares copy-on-write storage before the write, which is what we want here.
                System.Array.Copy(tensor.GetReadOnlyDataArray(), 0, existing.GetDataArray(), 0, tensor.Length);
                continue;
            }

            var installed = CloneTensorForMode(tensor, settings.Mode);
            clone.InstallRestoredBuffer(name, installed);
        }
    }

    /// <summary>Clones one persistent tensor without densifying sparse state.</summary>
    private static Tensor<T> CloneTensorForMode<T>(Tensor<T> tensor, CloneMode mode)
    {
        if (mode == CloneMode.Shared) return tensor;

        // Tensor.Clone/CloneShared intentionally reject SparseTensor because a dense storage clone
        // would discard its COO topology. Preserve row/column indices and only duplicate the
        // non-zero payload. Copy-on-write currently has no sparse storage primitive, so an eager
        // independent sparse copy is the correct conservative implementation for that mode.
        if (tensor is SparseTensor<T> sparse)
        {
            if (sparse.Shape.Length != 2)
                throw new InvalidOperationException(
                    $"Sparse clone requires rank 2, got rank {sparse.Shape.Length}.");

            return new SparseTensor<T>(
                sparse.Shape[0],
                sparse.Shape[1],
                sparse.RowIndices.ToArray(),
                sparse.ColumnIndices.ToArray(),
                sparse.DataVector.ToArray());
        }

        return mode == CloneMode.CopyOnWrite
            ? (Tensor<T>)tensor.CloneShared()
            : tensor.Clone();
    }

    /// <summary>Copies each registered sub-layer's parameters into the matching sub-layer.</summary>
    /// <remarks>
    /// Pairwise by index: both sides were built by the same constructor in the same order, which is
    /// the pairing <c>GetTrainableParameters</c> and <c>ParameterCount</c> already rely on when they
    /// walk this list. Needs no shape at all — a tensor carries its own — so it reaches composites a
    /// forward probe cannot. Recursion carries the mode with it, so a Shared clone shares its
    /// children's weights too rather than quietly deep-copying them.
    /// </remarks>
    private static void CopyChildren<T>(LayerBase<T> source, LayerBase<T> clone, CloneOptions settings)
    {
        var sourceChildren = source.GetSubLayers();
        var cloneChildren = clone.GetSubLayers();

        if (sourceChildren is null || cloneChildren is null) return;
        if (sourceChildren.Count != cloneChildren.Count) return;

        for (var i = 0; i < sourceChildren.Count; i++)
        {
            if (sourceChildren[i] is LayerBase<T> childSource
                && cloneChildren[i] is LayerBase<T> childClone)
            {
                InstallInto(childSource, childClone, settings);
            }
        }
    }

    /// <summary>
    /// Rebuilds a layer from the construction state it records for serialization.
    /// </summary>
    /// <typeparam name="T">The layer's numeric type.</typeparam>
    /// <param name="source">The layer to rebuild.</param>
    /// <returns>A new, freshly constructed layer of the same type and configuration.</returns>
    /// <exception cref="NotSupportedException">Thrown when the type has no generated factory.</exception>
    private static LayerBase<T> Reconstruct<T>(LayerBase<T> source, int? constructionSeed)
    {
        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
        source.CaptureConstructionState(metadata);

        var values = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var pair in metadata) values[pair.Key] = pair.Value;
        source.CaptureConstructionObjects(values);
        if (constructionSeed.HasValue) values[CloneRandomSeedKey] = constructionSeed.Value;

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
