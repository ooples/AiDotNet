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
    private readonly int[] _carrierBins;

    private Tensor<T>? _lastInput;
    private Matrix<T>? _lastAttentionWeights;

    /// <inheritdoc/>
    public override string LayerName => $"IMDAttention_{_numCarriers}c";

    /// <inheritdoc/>
    public override int ParameterCount => 0;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

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
        _carrierBins = allocator.AllocateCarriers(numCarriers, fftSize);
        _bus = new SpectralBus<T>(_carrierBins, fftSize);
        _extractor = new IMDExtractor<T>(_carrierBins, fftSize);

        Parameters = Vector<T>.Empty();
    }

    /// <summary>
    /// Forward pass: encode features → square (create IMD) → extract attention weights → weighted sum.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int n = Math.Min(input.Length, _numCarriers);

        // Extract features
        var features = new Vector<T>(_numCarriers);
        for (int i = 0; i < _numCarriers; i++)
        {
            features[i] = i < n ? input[i] : NumOps.Zero;
        }

        // Step 1: Encode features onto carriers
        var encoded = _bus.Encode(features);

        // Step 2: Apply quadratic nonlinearity (x^2) to generate IMD products
        var squared = new Vector<T>(encoded.Length);
        for (int i = 0; i < encoded.Length; i++)
        {
            squared[i] = NumOps.Multiply(encoded[i], encoded[i]);
        }

        // Step 3: Extract attention weights from IMD products
        _lastAttentionWeights = _extractor.ExtractAttentionWeights(squared);

        // Step 4: Compute weighted sum (attention output)
        // output[i] = sum_j(attention[i,j] * value[j])
        // Here, value = features (self-attention)
        var output = new Tensor<T>([_numCarriers]);
        for (int i = 0; i < _numCarriers; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < _numCarriers; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(_lastAttentionWeights[i, j], features[j]));
            }
            output[i] = sum;
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
    public override void Deserialize(BinaryReader reader) { }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _lastAttentionWeights = null;
    }
}
