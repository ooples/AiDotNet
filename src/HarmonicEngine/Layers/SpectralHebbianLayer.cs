using AiDotNet.HarmonicEngine.Learning;
using AiDotNet.HarmonicEngine.Transforms;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.HarmonicEngine.Layers;

/// <summary>
/// Layer that learns a spectral filter via Hebbian learning instead of backpropagation.
/// The filter is updated during the forward pass (when in training mode) based on
/// input-output phase coherence at each frequency.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> In traditional neural networks, a layer learns by backpropagation:
/// run forward, compute error, propagate error backward, update weights. This is slow and
/// requires many epochs over the data.
///
/// The Spectral Hebbian Layer learns differently:
/// 1. Transform input to frequency domain via FFT
/// 2. Apply a learned spectral filter H(k) at each frequency
/// 3. Transform back to time domain via IFFT
/// 4. During training, update H(k) using the Hebbian rule:
///    if input and target are in phase at frequency k, strengthen H(k)
///    if they are out of phase, weaken H(k)
///
/// This converges to the Wiener optimal filter in a single pass over the data —
/// no backpropagation, no optimizer, no epochs needed.
///
/// The anti-Hebbian component prevents all frequencies from converging to the same
/// representation, forcing the filter to capture diverse features.
/// </para>
/// </remarks>
public class SpectralHebbianLayer<T> : LayerBase<T>
{
    private readonly FastFourierTransform<T> _fft;
    private readonly SpectralHebbianRule<T> _rule;
    private readonly int _signalLength;

    // The learned spectral filter H(k)
    private Vector<Complex<T>> _filter;
    private Tensor<T>? _lastInput;

    /// <inheritdoc/>
    public override string LayerName => $"SpectralHebbian_{_signalLength}";

    /// <inheritdoc/>
    public override int ParameterCount => _signalLength * 2; // Real + imaginary per frequency bin

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the current spectral filter for inspection/visualization.
    /// </summary>
    public Vector<Complex<T>> Filter => _filter;

    /// <summary>
    /// Creates a new Spectral Hebbian Layer.
    /// </summary>
    /// <param name="signalLength">Length of the input/output signal (must be power of 2).</param>
    /// <param name="learningRate">Hebbian learning rate (eta).</param>
    /// <param name="antiHebbianAlpha">Strength of anti-Hebbian decorrelation.</param>
    public SpectralHebbianLayer(int signalLength, double learningRate = 0.01, double antiHebbianAlpha = 0.1)
        : base([signalLength], [signalLength])
    {
        _signalLength = signalLength;
        _fft = new FastFourierTransform<T>();
        _rule = new SpectralHebbianRule<T>(learningRate, antiHebbianAlpha);

        // Initialize filter to unity (pass-through)
        _filter = new Vector<Complex<T>>(signalLength);
        var one = new Complex<T>(NumOps.One, NumOps.Zero);
        for (int i = 0; i < signalLength; i++)
        {
            _filter[i] = one;
        }

        // Flatten filter into Parameters vector (real, imag, real, imag, ...)
        Parameters = new Vector<T>(signalLength * 2);
        SyncFilterToParameters();
    }

    /// <summary>
    /// Forward pass: FFT → multiply by filter → IFFT.
    /// When in training mode, also updates the filter using the Hebbian rule.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;

        // Extract signal
        var signal = new Vector<T>(_signalLength);
        for (int i = 0; i < _signalLength && i < input.Length; i++)
        {
            signal[i] = input[i];
        }

        // Forward FFT
        var spectrum = _fft.Forward(signal);
        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();

        // Apply spectral filter: Y(k) = H(k) * X(k)
        var filteredSpectrum = new Vector<Complex<T>>(_signalLength);
        for (int k = 0; k < _signalLength; k++)
        {
            filteredSpectrum[k] = complexOps.Multiply(_filter[k], spectrum[k]);
        }

        // Inverse FFT
        var filtered = _fft.Inverse(filteredSpectrum);

        // Convert to output tensor
        var output = new Tensor<T>([_signalLength]);
        for (int i = 0; i < _signalLength; i++)
        {
            output[i] = filtered[i];
        }

        return output;
    }

    /// <summary>
    /// Updates the spectral filter using the Hebbian rule given a target signal.
    /// Call this during training after Forward() to update the filter.
    /// </summary>
    /// <param name="target">The desired output signal.</param>
    public void HebbianUpdate(Vector<T> target)
    {
        if (!IsTrainingMode || _lastInput is null) return;

        // Get input signal
        var input = new Vector<T>(_signalLength);
        for (int i = 0; i < _signalLength && i < _lastInput.Length; i++)
        {
            input[i] = _lastInput[i];
        }

        // Compute spectra
        var inputSpectrum = _fft.Forward(input);
        var targetSpectrum = _fft.Forward(target);

        // Apply Hebbian update
        _rule.Update(_filter, inputSpectrum, targetSpectrum);

        // Sync to Parameters vector
        SyncFilterToParameters();
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        SyncFilterToParameters();
        return Parameters;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        // Gradients are not used — Hebbian learning is non-gradient-based
        return new Vector<T>(_signalLength * 2);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate) { }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        SetParameters(parameters);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        for (int k = 0; k < _signalLength; k++)
        {
            _filter[k] = new Complex<T>(parameters[k * 2], parameters[k * 2 + 1]);
        }
        Parameters = parameters;
    }

    /// <inheritdoc/>
    public override void Serialize(BinaryWriter writer)
    {
        writer.Write(_signalLength);
        SyncFilterToParameters();
        for (int i = 0; i < Parameters.Length; i++)
        {
            writer.Write(NumOps.ToDouble(Parameters[i]));
        }
    }

    /// <inheritdoc/>
    public override void Deserialize(BinaryReader reader)
    {
        int length = reader.ReadInt32();
        for (int i = 0; i < length * 2; i++)
        {
            Parameters[i] = NumOps.FromDouble(reader.ReadDouble());
        }
        SyncParametersToFilter();
    }

    private void SyncFilterToParameters()
    {
        for (int k = 0; k < _signalLength; k++)
        {
            Parameters[k * 2] = _filter[k].Real;
            Parameters[k * 2 + 1] = _filter[k].Imaginary;
        }
    }

    private void SyncParametersToFilter()
    {
        for (int k = 0; k < _signalLength; k++)
        {
            _filter[k] = new Complex<T>(Parameters[k * 2], Parameters[k * 2 + 1]);
        }
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
    }
}
