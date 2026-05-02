using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// Provides PyTorch-style training step using tape-based automatic differentiation,
/// with two-level caching for parameter collection that outperforms PyTorch's
/// <c>model.parameters()</c> which rebuilds the full list on every call.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para><b>Performance advantages over PyTorch:</b></para>
/// <list type="bullet">
/// <item><b>Level 1 — Parameter array cache:</b> The recursive walk and flat array are
/// computed once and reused across training steps. PyTorch's <c>parameters()</c> is a
/// generator that walks the module tree on every call. Our cache turns O(N) recursive
/// walk per step into O(1) array return.</item>
/// <item><b>Level 2 — ParameterBuffer cache:</b> When a ParameterBuffer is attached,
/// the contiguous buffer layout is preserved across steps. PyTorch's FSDP periodically
/// rebuilds the flat parameter buffer. Ours stays valid until layers change.</item>
/// </list>
/// <para>Both caches are invalidated via a version counter that increments when
/// the layer structure changes (layers added/removed).</para>
/// </remarks>
public static class TapeTrainingStep<T>
{
    // Level 1 cache: flattened parameter array from recursive walk.
    //
    // Cache validity is keyed on a TOPOLOGY FINGERPRINT instead of just
    // (firstLayer, count, version). The previous tuple key collided when
    // two distinct graphs shared the same first top-level layer instance
    // and the same count — e.g. CycleGAN's 2 generators + 2 discriminators
    // that all start with a Conv2D and have similar layer counts. A cache
    // hit on a wrong-graph entry could return stale tensors that ZeroGrad
    // / optimizer updates would then operate on, silently corrupting
    // training. The fingerprint is a deterministic FNV-1a hash over each
    // layer's runtime type + ParameterCount, walked top-level only (the
    // recursive walk happens on miss). One-time cost: ~O(layers); collision
    // probability over a 64-bit FNV is negligible for any realistic graph.
    [ThreadStatic]
    private static List<Tensor<T>>? _cachedParameters;
    [ThreadStatic]
    private static int _cachedVersion;
    [ThreadStatic]
    private static int _cachedLayerCount;
    [ThreadStatic]
    private static ILayer<T>? _cachedFirstLayer;
    [ThreadStatic]
    private static ulong _cachedTopologyFingerprint;

    // Level 2 cache: trainable layer references for ZeroGrad
    [ThreadStatic]
    private static ITrainableLayer<T>[]? _cachedTrainableLayers;
    [ThreadStatic]
    private static int _cachedTrainableVersion;
    [ThreadStatic]
    private static int _cachedTrainableLayerCount;
    [ThreadStatic]
    private static ILayer<T>? _cachedTrainableFirstLayer;
    [ThreadStatic]
    private static ulong _cachedTrainableTopologyFingerprint;

    /// <summary>
    /// Computes a topology fingerprint over the layer list using FNV-1a 64-bit.
    /// Two distinct graphs with the same first layer and same count will have
    /// different fingerprints whenever any layer's runtime type or
    /// ParameterCount differs at any position — which catches the
    /// collision case the prior (firstLayer, count) key missed.
    /// </summary>
    private static ulong ComputeTopologyFingerprint(IList<ILayer<T>> layers)
    {
        const ulong FnvOffsetBasis = 14695981039346656037UL;
        const ulong FnvPrime = 1099511628211UL;
        ulong hash = FnvOffsetBasis;
        for (int i = 0; i < layers.Count; i++)
        {
            var layer = layers[i];
            if (layer is null)
            {
                hash ^= 0xDEADBEEFUL;
                hash *= FnvPrime;
                continue;
            }
            // Hash the runtime type + parameter count + index. Type captures
            // the layer kind; ParameterCount captures shape variation within
            // a kind (e.g. Dense(64) vs Dense(128) at the same index); index
            // ensures order matters.
            hash ^= (ulong)i;
            hash *= FnvPrime;
            hash ^= (ulong)layer.GetType().GetHashCode();
            hash *= FnvPrime;
            try
            {
                hash ^= (ulong)layer.ParameterCount;
                hash *= FnvPrime;
            }
            catch
            {
                // Some lazy layers may throw on ParameterCount before shape
                // resolution (e.g. when InputShape is unresolved). Treat
                // that as a stable contributing value so the cache still
                // distinguishes them.
                hash ^= 0xC0FFEEUL;
                hash *= FnvPrime;
            }
        }
        return hash;
    }

    /// <summary>
    /// Executes a single training step using tape-based autodiff.
    /// </summary>
    public static T Step(
        IReadOnlyList<ITrainableLayer<T>> layers,
        Tensor<T> input,
        Tensor<T> target,
        T learningRate,
        Func<Tensor<T>, Tensor<T>> forward,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> computeLoss)
    {
        var numOps = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;

        // 1. Zero gradients (PyTorch: optimizer.zero_grad())
        foreach (var layer in layers)
        {
            layer.ZeroGrad();
        }

        // 2. Collect all trainable parameters
        var allParams = new List<Tensor<T>>();
        foreach (var layer in layers)
        {
            allParams.AddRange(layer.GetTrainableParameters());
        }
        var paramArray = allParams.ToArray();

        // 3. Forward pass + loss computation under tape recording
        Tensor<T> loss;
        Dictionary<Tensor<T>, Tensor<T>> grads;

        using (var tape = new GradientTape<T>())
        {
            var predicted = forward(input);
            loss = computeLoss(predicted, target);

            // 4. Compute gradients (PyTorch: loss.backward())
            grads = tape.ComputeGradients(loss, paramArray);
        }

        // 5. Update parameters with SGD (PyTorch: optimizer.step())
        foreach (var param in paramArray)
        {
            if (grads.TryGetValue(param, out var grad))
            {
                var update = engine.TensorMultiplyScalar(grad, learningRate);
                engine.TensorSubtractInPlace(param, update);
            }
        }

        return loss.Length > 0 ? loss[0] : numOps.Zero;
    }

    /// <summary>
    /// Collects all trainable parameters by recursively walking layers and their sub-layers.
    /// Uses Level 1 caching: the recursive walk runs once, subsequent calls return the cached array.
    /// </summary>
    /// <param name="layers">Top-level layers to collect parameters from.</param>
    /// <param name="structureVersion">Optional version counter. When this changes, the cache
    /// is invalidated and the recursive walk runs again. Pass -1 to force refresh.</param>
    /// <returns>Read-only list of all trainable parameter tensors, deduplicated by reference identity. Zero allocation on cache hit.</returns>
    public static IReadOnlyList<Tensor<T>> CollectParameters(IEnumerable<ILayer<T>> layers, int structureVersion = 0)
    {
        var layerList = layers as IList<ILayer<T>> ?? layers.ToList();
        ILayer<T>? firstLayer = layerList.Count > 0 ? layerList[0] : null;
        ulong fingerprint = ComputeTopologyFingerprint(layerList);

        // Cache hit ALSO requires topology fingerprint match. Without this,
        // two sibling networks (e.g. CycleGAN's two generators) that happen
        // to share the same first-layer instance and layer count would
        // alias to the same cache entry — ZeroGrad / optimizer.step would
        // then operate on the wrong network's parameter list. Fingerprint
        // catches the collision class.
        if (_cachedParameters is not null
            && _cachedVersion == structureVersion
            && _cachedLayerCount == layerList.Count
            && ReferenceEquals(_cachedFirstLayer, firstLayer)
            && _cachedTopologyFingerprint == fingerprint
            && structureVersion >= 0)
        {
            return _cachedParameters;
        }

        // Cache miss: recursive walk
        var seen = new HashSet<Tensor<T>>(TensorReferenceComparer<Tensor<T>>.Instance);
        var parameters = new List<Tensor<T>>();

        CollectRecursive(layerList, parameters, seen);

        // Update cache — List<T> implements IReadOnlyList<T>, zero-allocation return
        _cachedParameters = parameters;
        _cachedVersion = structureVersion;
        _cachedLayerCount = layerList.Count;
        _cachedFirstLayer = firstLayer;
        _cachedTopologyFingerprint = fingerprint;

        return _cachedParameters;
    }

    /// <summary>
    /// Recursively walks layers and sub-layers to collect all ITrainableLayer parameters.
    /// Deduplicates by tensor reference identity to handle shared parameters.
    /// </summary>
    private static void CollectRecursive(
        IEnumerable<ILayer<T>> layers,
        List<Tensor<T>> parameters,
        HashSet<Tensor<T>> seen)
    {
        foreach (var layer in layers)
        {
            // Collect this layer's own parameters if it's trainable
            if (layer is ITrainableLayer<T> trainable)
            {
                foreach (var param in trainable.GetTrainableParameters())
                {
                    // Deduplicate: shared parameters (e.g., tied weights) appear only once
                    if (seen.Add(param))
                    {
                        parameters.Add(param);
                    }
                }
            }

            // Recurse into sub-layers (composite layers expose children via GetSubLayers)
            var subLayers = layer.GetSubLayers();
            if (subLayers.Count > 0)
            {
                CollectRecursive(subLayers, parameters, seen);
            }
        }
    }

    /// <summary>
    /// Collects all ITrainableLayer instances for ZeroGrad. Cached separately from parameters.
    /// </summary>
    public static ITrainableLayer<T>[] CollectTrainableLayers(IEnumerable<ILayer<T>> layers, int structureVersion = 0)
    {
        var layerList = layers as IList<ILayer<T>> ?? layers.ToList();
        ILayer<T>? firstLayer = layerList.Count > 0 ? layerList[0] : null;
        ulong fingerprint = ComputeTopologyFingerprint(layerList);

        // Same topology-fingerprint guard as CollectParameters: prevents
        // sibling-network cache collisions on (firstLayer, count) alone.
        if (_cachedTrainableLayers is not null
            && _cachedTrainableVersion == structureVersion
            && _cachedTrainableLayerCount == layerList.Count
            && ReferenceEquals(_cachedTrainableFirstLayer, firstLayer)
            && _cachedTrainableTopologyFingerprint == fingerprint
            && structureVersion >= 0)
        {
            return _cachedTrainableLayers;
        }

        var trainableLayers = new List<ITrainableLayer<T>>();
        var seen = new HashSet<ILayer<T>>(TensorReferenceComparer<ILayer<T>>.Instance);
        CollectTrainableRecursive(layerList, trainableLayers, seen);

        var result = trainableLayers.ToArray();
        _cachedTrainableLayers = result;
        _cachedTrainableVersion = structureVersion;
        _cachedTrainableLayerCount = layerList.Count;
        _cachedTrainableFirstLayer = firstLayer;
        _cachedTrainableTopologyFingerprint = fingerprint;
        return result;
    }

    private static void CollectTrainableRecursive(
        IEnumerable<ILayer<T>> layers,
        List<ITrainableLayer<T>> result,
        HashSet<ILayer<T>> seen)
    {
        foreach (var layer in layers)
        {
            if (!seen.Add(layer)) continue; // Skip already-visited (handles diamond patterns)

            if (layer is ITrainableLayer<T> trainable)
                result.Add(trainable);

            var subLayers = layer.GetSubLayers();
            if (subLayers.Count > 0)
                CollectTrainableRecursive(subLayers, result, seen);
        }
    }

    /// <summary>
    /// Zeros gradients for all trainable layers, including those nested in composite layers.
    /// Uses cached trainable layer list for O(1) access after first call.
    /// </summary>
    public static void ZeroGradAll(IEnumerable<ILayer<T>> layers, int structureVersion = 0)
    {
        var trainableLayers = CollectTrainableLayers(layers, structureVersion);
        foreach (var trainable in trainableLayers)
        {
            trainable.ZeroGrad();
        }
    }

    /// <summary>
    /// Invalidates all caches. Call when layer structure changes (layers added/removed).
    /// </summary>
    public static void InvalidateCache()
    {
        _cachedParameters = null;
        _cachedLayerCount = 0;
        _cachedFirstLayer = null;
        _cachedTrainableLayers = null;
        _cachedTrainableLayerCount = 0;
        _cachedTrainableFirstLayer = null;
    }
}
