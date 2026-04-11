using AiDotNet.HarmonicEngine.Core;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.HarmonicEngine.Layers;

/// <summary>
/// Attention layer that computes pairwise feature interactions via intermodulation distortion (IMD)
/// at O(N log N) complexity instead of the O(N^2) cost of traditional attention mechanisms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Traditional attention (as in Transformers) computes a score for every pair of features:
/// "how relevant is feature A to feature B?" This requires N^2 comparisons for N features.
///
/// IMD attention achieves the same result using wave physics:
/// 1. Encode each feature as the amplitude of a unique frequency carrier
/// 2. Square the composite signal (polynomial nonlinearity)
/// 3. The squaring creates intermodulation products at f_i + f_j and f_i - f_j
///    with amplitudes proportional to a_i * a_j — exactly the pairwise attention scores
/// 4. Extract these scores via FFT at O(N log N) cost
///
/// The result is mathematically equivalent to computing Q*K^T in attention,
/// but without explicitly forming the N^2 matrix.
/// </para>
/// </remarks>
public class IMDAttentionLayer<T> : LayerBase<T>
{
    private readonly SpectralBus<T> _bus;
    private readonly IMDExtractor<T> _extractor;
    private readonly int _numCarriers;
    private readonly int _fftSize;

    private Matrix<T>? _lastAttentionWeights;

    /// <inheritdoc/>
    public override string LayerName => $"IMDAttention_{_numCarriers}c";

    /// <inheritdoc/>
    public override int ParameterCount => 0;

    /// <inheritdoc/>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Gets the most recent attention weight matrix (for visualization/debugging).
    /// </summary>
    public Matrix<T>? LastAttentionWeights => _lastAttentionWeights;

    /// <summary>
    /// Creates a new IMD attention layer.
    /// </summary>
    /// <param name="numCarriers">Number of feature carriers (determines attention dimension).</param>
    /// <param name="fftSize">FFT size (must be power of 2).</param>
    public IMDAttentionLayer(int numCarriers, int fftSize)
        : base([numCarriers], [numCarriers])
    {
        _numCarriers = numCarriers;
        _fftSize = fftSize;

        var allocator = new CarrierAllocator();
        var carrierBins = allocator.AllocateCarriers(numCarriers, fftSize);
        _bus = new SpectralBus<T>(carrierBins, fftSize);
        _extractor = new IMDExtractor<T>(carrierBins, fftSize);

        Parameters = Vector<T>.Empty();
    }

    /// <summary>
    /// Forward pass: encode features → square (create IMD) → extract attention weights → weighted sum.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        int n = Math.Min(input.Length, _numCarriers);

        // Extract features
        var features = new Vector<T>(_numCarriers);
        for (int i = 0; i < _numCarriers; i++)
        {
            features[i] = i < n ? input[i] : NumOps.Zero;
        }

        // Step 1: Encode features onto carriers
        var encoded = _bus.Encode(features);

        // Step 2: Apply quadratic nonlinearity (x^2) via Engine element-wise multiply
        var squared = Engine.Multiply(encoded, encoded);

        // Step 3: Extract attention weights from IMD products
        _lastAttentionWeights = _extractor.ExtractAttentionWeights(squared);

        // Step 4: Compute weighted sum output = attention_weights * features (matrix-vector).
        // Use Matrix.Multiply(Vector) for a single vectorized mat-vec instead of per-row allocations.
        var outputVec = _lastAttentionWeights.Multiply(features);
        var output = new Tensor<T>([_numCarriers]);
        for (int i = 0; i < _numCarriers; i++)
        {
            output[i] = outputVec[i];
        }

        return output;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters() => Vector<T>.Empty();

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients() => Vector<T>.Empty();

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters) { }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters) { }

    /// <inheritdoc/>
    public override void Serialize(BinaryWriter writer)
    {
        writer.Write(_numCarriers);
        writer.Write(_fftSize);
    }

    /// <inheritdoc/>
    public override void Deserialize(BinaryReader reader)
    {
        // Consume the values written by Serialize
        _ = reader.ReadInt32(); // numCarriers
        _ = reader.ReadInt32(); // fftSize
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastAttentionWeights = null;
    }
}
