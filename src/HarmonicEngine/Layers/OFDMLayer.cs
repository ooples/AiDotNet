using AiDotNet.HarmonicEngine.Activations;
using AiDotNet.HarmonicEngine.Core;
using AiDotNet.HarmonicEngine.Enums;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.HarmonicEngine.Layers;

/// <summary>
/// OFDM (Orthogonal Frequency Division Multiplexing) layer that replaces the traditional dense layer's
/// matrix multiplication with a spectral broadcast-and-interfere operation at O(N log N) complexity.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A traditional dense layer computes output = W * input + bias, which requires
/// an N x M weight matrix and costs O(N*M) operations. The OFDM layer does something fundamentally different:
///
/// 1. Encode: Each input feature is placed on a unique frequency carrier (like radio stations)
/// 2. Combine: All carriers are mixed into one time-domain signal via inverse FFT
/// 3. Activate: A nonlinearity is applied, creating intermodulation products (cross-feature interactions)
/// 4. Decode: FFT extracts the features at carrier frequencies + IMD interaction features
///
/// Total cost: O(N log N) instead of O(N^2), with cross-feature interactions emerging naturally
/// from the physics of wave interference rather than from explicit weight connections.
/// </para>
/// </remarks>
public class OFDMLayer<T> : LayerBase<T>
{
    private readonly SpectralBus<T> _bus;
    private readonly IActivationFunction<T> _activation;
    private readonly int _numCarriers;
    private readonly int _fftSize;
    private readonly IReadOnlyList<int> _carrierBins;

    /// <inheritdoc/>
    public override string LayerName => $"OFDM_{_numCarriers}c_{_fftSize}fft";

    /// <inheritdoc/>
    public override int ParameterCount => 0; // No learnable weights — carrier positions are fixed

    /// <inheritdoc/>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Gets the carrier frequency bin indices used by this layer.
    /// </summary>
    public IReadOnlyList<int> CarrierBins => _carrierBins;

    /// <summary>
    /// Creates a new OFDM layer.
    /// </summary>
    /// <param name="inputSize">Number of input features (must match carrier count).</param>
    /// <param name="numCarriers">Number of orthogonal frequency carriers.</param>
    /// <param name="fftSize">FFT size (must be power of 2, large enough for carriers).</param>
    /// <param name="nonlinearity">Type of spectral nonlinearity to apply.</param>
    public OFDMLayer(int inputSize, int numCarriers, int fftSize,
        NonlinearityType nonlinearity = NonlinearityType.SpectralGating)
        : base([inputSize], [numCarriers])
    {
        _numCarriers = numCarriers;
        _fftSize = fftSize;

        var allocator = new CarrierAllocator();
        _carrierBins = allocator.AllocateCarriers(numCarriers, fftSize);
        _bus = new SpectralBus<T>(_carrierBins, fftSize);

        _activation = nonlinearity switch
        {
            NonlinearityType.ModReLU => new ModReLUActivation<T>(bias: -0.001),
            NonlinearityType.SpectralGating => new SpectralGatingActivation<T>(),
            NonlinearityType.InstantaneousFreq => new InstantaneousFreqActivation<T>(),
            _ => new SpectralGatingActivation<T>()
        };

        Parameters = Vector<T>.Empty();
    }

    /// <summary>
    /// Forward pass: encode → activate → decode.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        int featureCount = input.Length;

        // Extract features from tensor to vector
        var features = new Vector<T>(featureCount);
        for (int i = 0; i < featureCount; i++)
        {
            features[i] = input[i];
        }

        // Pad or truncate to carrier count
        var carrierFeatures = new Vector<T>(_numCarriers);
        for (int i = 0; i < _numCarriers; i++)
        {
            carrierFeatures[i] = i < featureCount ? features[i] : NumOps.Zero;
        }

        // Step 1: Encode features onto carriers (IFFT to time domain)
        var encoded = _bus.Encode(carrierFeatures);

        // Step 2: Apply nonlinearity in time domain (creates IMD products)
        var activated = _activation.Activate(encoded);

        // Step 3: Decode features from carrier frequencies (FFT)
        var decoded = _bus.Decode(activated);

        // Convert output vector to tensor
        var output = new Tensor<T>([_numCarriers]);
        for (int i = 0; i < _numCarriers; i++)
        {
            output[i] = decoded[i];
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
        foreach (int bin in _carrierBins)
        {
            writer.Write(bin);
        }
    }

    /// <inheritdoc/>
    public override void Deserialize(BinaryReader reader)
    {
        // Consume the values written by Serialize
        int numCarriers = reader.ReadInt32();
        _ = reader.ReadInt32(); // fftSize
        for (int i = 0; i < numCarriers; i++)
        {
            _ = reader.ReadInt32(); // carrier bin
        }
    }

    /// <inheritdoc/>
    public override void ResetState() { }
}
