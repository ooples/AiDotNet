using System.Runtime.CompilerServices;
using AiDotNet.Helpers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Manages hidden state caching across autoregressive inference steps for SSM models.
/// </summary>
/// <remarks>
/// <para>
/// During autoregressive generation (e.g., language model text generation), each new token
/// depends on the hidden states produced by all previous tokens. Rather than reprocessing
/// the entire sequence for each new token, the state cache stores the SSM hidden states
/// so only the new token needs to be processed. This is the SSM equivalent of the KV-cache
/// used in Transformer models.
/// </para>
/// <para>
/// The cache stores two types of state per layer:
/// 1. SSM hidden state: the state space model's recurrent state [batch, innerDim, stateDim]
/// 2. Conv buffer: the convolution buffer for Conv1D layers [batch, innerDim, convKernelSize-1]
///
/// This matches the state needed by Mamba's selective scan and Conv1D operations.
/// </para>
/// <para><b>For Beginners:</b> When a language model generates text one word at a time, it needs
/// to "remember" what it processed before. Instead of re-reading the entire text for each new word,
/// the state cache saves the model's memory from previous words.
///
/// Think of it like taking notes while reading:
/// - Without cache: re-read the entire book for each new sentence (slow!)
/// - With cache: just read your notes and the new sentence (fast!)
///
/// For SSM models, this cache stores:
/// - The "hidden state" (the model's current memory/summary)
/// - The "conv buffer" (recent context needed by the convolution layer)
/// </para>
/// <para>
/// <b>Usage example:</b>
/// <code>
/// var cache = new SSMStateCache&lt;float&gt;();
/// // Generate tokens one at a time
/// for each new token:
///     var prevState = cache.GetSSMState(layerIndex);
///     var newState = mambaBlock.StepForward(token, prevState);
///     cache.CacheSSMState(layerIndex, newState);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
public class SSMStateCache<T>
{
    private static IEngine Engine => AiDotNetEngine.Current;
    private static INumericOperations<T> NumOps => MathHelper.GetNumericOperations<T>();

    private readonly Dictionary<int, Tensor<T>> _ssmStates;
    private readonly Dictionary<int, Tensor<T>> _convBuffers;
    private readonly bool _enableCompression;
    private readonly int _compressionBitWidth;

    /// <summary>
    /// Gets the number of layers that have cached SSM states.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Shows how many layers have saved their memory so far.
    /// This grows as you process tokens through more layers of the model.</para>
    /// </remarks>
    public int CachedLayerCount => _ssmStates.Count;

    /// <summary>
    /// Gets whether state precision reduction is enabled.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, cached states are quantized and dequantized on storage,
    /// reducing their effective precision (e.g., to 8-bit resolution). This simulates lower-precision
    /// storage for accuracy-aware deployment testing. Note: the current implementation still stores
    /// values as Tensor&lt;T&gt;, so actual memory usage is unchanged. Future versions may use packed
    /// byte storage for true memory savings.</para>
    /// </remarks>
    public bool CompressionEnabled => _enableCompression;

    /// <summary>
    /// Creates a new SSM state cache.
    /// </summary>
    /// <param name="enableCompression">
    /// Whether to quantize cached states to reduced precision on storage. Default: false.
    /// <para><b>For Beginners:</b> Enable this to simulate lower-precision cached states (for example,
    /// 8-bit resolution). This helps test how the model behaves with reduced precision, but the current
    /// implementation still stores values in Tensor&lt;T&gt;, so actual memory usage is unchanged.</para>
    /// </param>
    /// <param name="compressionBitWidth">
    /// Bit width used when quantizing cached states. Default: 8.
    /// <para><b>For Beginners:</b> Lower values use fewer distinct levels and can increase quantization
    /// error; 8-bit is usually a good balance. This controls precision only, not the in-memory size of
    /// the underlying Tensor&lt;T&gt;.</para>
    /// </param>
    public SSMStateCache(bool enableCompression = false, int compressionBitWidth = 8)
    {
        if (compressionBitWidth <= 0 || compressionBitWidth > 32)
            throw new ArgumentException(
                $"Compression bit width ({compressionBitWidth}) must be between 1 and 32.",
                nameof(compressionBitWidth));

        _ssmStates = new Dictionary<int, Tensor<T>>();
        _convBuffers = new Dictionary<int, Tensor<T>>();
        _enableCompression = enableCompression;
        _compressionBitWidth = compressionBitWidth;
    }

    /// <summary>
    /// Caches the SSM hidden state for a specific layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> After processing a token through a Mamba layer, this stores
    /// the layer's "memory" so it can be used when processing the next token.</para>
    /// </remarks>
    /// <param name="layerIndex">Index of the layer in the model.</param>
    /// <param name="hiddenState">The SSM hidden state tensor to cache.</param>
    public void CacheSSMState(int layerIndex, Tensor<T> hiddenState)
    {
        if (hiddenState == null)
            throw new ArgumentNullException(nameof(hiddenState));

        if (_enableCompression)
        {
            _ssmStates[layerIndex] = CompressState(hiddenState);
        }
        else
        {
            _ssmStates[layerIndex] = CloneTensor(hiddenState);
        }
    }

    /// <summary>
    /// Retrieves the cached SSM hidden state for a specific layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Retrieves the layer's saved "memory" from the last token
    /// it processed. Returns null for the first token (no previous state exists).</para>
    /// </remarks>
    /// <param name="layerIndex">Index of the layer in the model.</param>
    /// <returns>The cached hidden state, or null if no state has been cached for this layer.</returns>
    public Tensor<T>? GetSSMState(int layerIndex)
    {
        if (_ssmStates.TryGetValue(layerIndex, out var state))
        {
            if (_enableCompression)
            {
                return DecompressState(state);
            }
            return CloneTensor(state);
        }
        return null;
    }

    /// <summary>
    /// Caches the Conv1D buffer state for a specific layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Mamba blocks include a causal Conv1D that needs the last (kernelSize - 1) inputs as context.
    /// This buffer stores those recent inputs so they don't need to be recomputed.
    /// </para>
    /// <para><b>For Beginners:</b> The convolution layer needs to see a few previous tokens to work.
    /// This stores those recent tokens so they're available for the next step.</para>
    /// </remarks>
    /// <param name="layerIndex">Index of the layer in the model.</param>
    /// <param name="convBuffer">The convolution buffer tensor to cache.</param>
    public void CacheConvBuffer(int layerIndex, Tensor<T> convBuffer)
    {
        if (convBuffer == null)
            throw new ArgumentNullException(nameof(convBuffer));

        _convBuffers[layerIndex] = CloneTensor(convBuffer);
    }

    /// <summary>
    /// Retrieves the cached Conv1D buffer for a specific layer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Retrieves the saved convolution context from the last token.
    /// Returns null for the first token (no previous context exists).</para>
    /// </remarks>
    /// <param name="layerIndex">Index of the layer in the model.</param>
    /// <returns>The cached conv buffer, or null if not cached.</returns>
    public Tensor<T>? GetConvBuffer(int layerIndex)
    {
        if (_convBuffers.TryGetValue(layerIndex, out var buffer))
        {
            return CloneTensor(buffer);
        }
        return null;
    }

    /// <summary>
    /// Clears all cached states, preparing for a new sequence.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this when starting to generate a completely new sequence.
    /// It clears all saved memories so the model starts fresh.</para>
    /// </remarks>
    public void Reset()
    {
        _ssmStates.Clear();
        _convBuffers.Clear();
    }

    /// <summary>
    /// Creates a deep copy of this cache, useful for beam search where multiple hypotheses
    /// need independent state copies.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> During beam search (trying multiple possible continuations),
    /// each "beam" needs its own copy of the state. This creates an independent copy so that
    /// updating one beam's state doesn't affect the others.</para>
    /// </remarks>
    /// <returns>A new SSMStateCache with cloned states.</returns>
    public SSMStateCache<T> Clone()
    {
        var clone = new SSMStateCache<T>(_enableCompression, _compressionBitWidth);

        foreach (var kvp in _ssmStates)
        {
            clone._ssmStates[kvp.Key] = CloneTensor(kvp.Value);
        }

        foreach (var kvp in _convBuffers)
        {
            clone._convBuffers[kvp.Key] = CloneTensor(kvp.Value);
        }

        return clone;
    }

    /// <summary>
    /// Gets the approximate memory usage of the cache in bytes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This reports actual in-memory usage based on element size of T. Note that when compression
    /// is enabled, states are stored with reduced precision but still as Tensor&lt;T&gt;, so the
    /// actual memory footprint is the same as uncompressed. The precision reduction benefits
    /// accuracy-aware deployment, not memory reduction. Future implementations may use packed
    /// byte storage for true memory compression.
    /// </para>
    /// <para><b>For Beginners:</b> Reports how much memory the cache is using. Useful for monitoring
    /// memory during long sequence generation to decide when to trim or compress.</para>
    /// </remarks>
    /// <returns>Estimated memory usage in bytes.</returns>
    public long GetMemoryUsageBytes()
    {
        long bytes = 0;
        int elementSize = Unsafe.SizeOf<T>();

        foreach (var state in _ssmStates.Values)
        {
            bytes += state.Length * elementSize;
        }

        foreach (var buffer in _convBuffers.Values)
        {
            bytes += buffer.Length * elementSize;
        }

        return bytes;
    }

    /// <summary>
    /// Checks whether a specific layer has a cached SSM state.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Checks if a particular layer has already saved its memory.
    /// Returns false for the first token in a new sequence.</para>
    /// </remarks>
    /// <param name="layerIndex">Index of the layer to check.</param>
    /// <returns>True if a state is cached for this layer.</returns>
    public bool HasSSMState(int layerIndex) => _ssmStates.ContainsKey(layerIndex);

    /// <summary>
    /// Checks whether a specific layer has a cached conv buffer.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Checks if a particular layer has saved its convolution context.
    /// The conv buffer stores recent token information needed by the causal convolution.</para>
    /// </remarks>
    /// <param name="layerIndex">Index of the layer to check.</param>
    /// <returns>True if a conv buffer is cached for this layer.</returns>
    public bool HasConvBuffer(int layerIndex) => _convBuffers.ContainsKey(layerIndex);

    /// <summary>
    /// Gets all layer indices that have cached SSM states.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns a list of which layers currently have saved hidden states.
    /// Useful when you need to iterate over all cached states, for example when migrating states
    /// to a compressed cache.</para>
    /// </remarks>
    /// <returns>An enumerable of layer indices with cached SSM states.</returns>
    public IEnumerable<int> GetSSMStateLayerIndices() => _ssmStates.Keys;

    /// <summary>
    /// Gets all layer indices that have cached conv buffers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Returns a list of which layers currently have saved convolution buffers.
    /// Useful when iterating over all cached conv states.</para>
    /// </remarks>
    /// <returns>An enumerable of layer indices with cached conv buffers.</returns>
    public IEnumerable<int> GetConvBufferLayerIndices() => _convBuffers.Keys;

    private Tensor<T> CompressState(Tensor<T> state)
    {
        // Quantize-then-dequantize: reduces effective precision while keeping Tensor<T> storage
        T minVal = state[0];
        T maxVal = state[0];

        for (int i = 1; i < state.Length; i++)
        {
            if (NumOps.LessThan(state[i], minVal)) minVal = state[i];
            if (NumOps.GreaterThan(state[i], maxVal)) maxVal = state[i];
        }

        T range = NumOps.Subtract(maxVal, minVal);
        if (NumOps.Equals(range, NumOps.Zero))
        {
            // All values are the same, just clone
            return CloneTensor(state);
        }

        // 32-bit is a no-op: the quantize-dequantize round-trip would not meaningfully change values
        if (_compressionBitWidth >= 32)
        {
            return CloneTensor(state);
        }

        // Store quantized values (still as T, but with reduced effective precision)
        long levels = (1L << _compressionBitWidth) - 1;
        T levelsT = NumOps.FromDouble(levels);
        var compressed = new Tensor<T>(state.Shape);

        for (int i = 0; i < state.Length; i++)
        {
            T normalized = NumOps.Divide(NumOps.Subtract(state[i], minVal), range);
            T quantized = NumOps.FromDouble(Math.Round(
                NumOps.ToDouble(NumOps.Multiply(normalized, levelsT))));
            // Store: quantized/levels * range + min (dequantize immediately for simplicity)
            compressed[i] = NumOps.Add(
                NumOps.Multiply(NumOps.Divide(quantized, levelsT), range),
                minVal);
        }

        return compressed;
    }

    private Tensor<T> DecompressState(Tensor<T> state)
    {
        // When compression is enabled, we store the dequantized values directly,
        // so decompression is just a clone.
        return CloneTensor(state);
    }

    private static Tensor<T> CloneTensor(Tensor<T> source)
    {
        var clone = new Tensor<T>(source.Shape);
        for (int i = 0; i < source.Length; i++)
        {
            clone[i] = source[i];
        }
        return clone;
    }
}
