using AiDotNet.HarmonicEngine.Transforms;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.HarmonicEngine.Layers;

/// <summary>
/// Layer that computes a scale-and-shift-invariant fingerprint of the input signal
/// using the Mellin-Fourier transform. The output is identical for the same pattern
/// regardless of how it is scaled or shifted in time.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Imagine you see a pattern in stock prices — say a "double bottom."
/// This same pattern might appear:
/// - At different price levels (scaled by 2x or 0.5x)
/// - At different times (shifted left or right on the chart)
///
/// A human recognizes it's the same pattern regardless. Traditional neural networks struggle with this
/// because the raw numbers look completely different. The Mellin-Fourier layer solves this by computing
/// a "fingerprint" that is mathematically guaranteed to be identical for any scaled or shifted version
/// of the same underlying pattern.
///
/// This layer is typically used as the first stage of the HRE pipeline to normalize inputs
/// before they enter the OFDM spectral bus.
/// </para>
/// </remarks>
public class MellinFourierLayer<T> : LayerBase<T>
{
    private readonly MellinTransform<T> _mellin;
    private readonly int _outputSize;

    /// <inheritdoc/>
    public override string LayerName => $"MellinFourier_{_outputSize}";

    /// <inheritdoc/>
    public override int ParameterCount => 0; // No learnable parameters

    /// <inheritdoc/>
    public override bool SupportsTraining => false;

    /// <summary>
    /// Creates a new Mellin-Fourier invariance layer.
    /// </summary>
    /// <param name="inputSize">Length of the input signal (must be power of 2).</param>
    /// <param name="outputSize">
    /// Length of the output fingerprint. If less than inputSize, the fingerprint is truncated
    /// to the first outputSize components (which contain the most information).
    /// </param>
    public MellinFourierLayer(int inputSize, int outputSize)
        : base([inputSize], [outputSize])
    {
        _mellin = new MellinTransform<T>();
        _outputSize = outputSize;
    }

    /// <summary>
    /// Forward pass: compute the scale-and-shift-invariant fingerprint.
    /// </summary>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Extract signal from tensor
        var signal = new Vector<T>(input.Length);
        for (int i = 0; i < input.Length; i++)
        {
            signal[i] = input[i];
        }

        // Compute scale+shift invariant fingerprint
        var fingerprint = _mellin.ScaleShiftInvariantFingerprint(signal);

        // Truncate or pad to output size
        var output = new Tensor<T>([_outputSize]);
        for (int i = 0; i < _outputSize; i++)
        {
            output[i] = i < fingerprint.Length ? fingerprint[i] : NumOps.Zero;
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
        writer.Write(_outputSize);
    }

    /// <inheritdoc/>
    public override void Deserialize(BinaryReader reader)
    {
        // Consume the values written by Serialize
        _ = reader.ReadInt32(); // outputSize
    }

    /// <inheritdoc/>
    public override void ResetState() { }
}
