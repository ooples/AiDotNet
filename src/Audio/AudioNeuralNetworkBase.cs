using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;

namespace AiDotNet.Audio;

/// <summary>
/// Base class for audio-focused neural networks that can operate in both ONNX inference and native training modes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This class extends <see cref="NeuralNetworkBase{T}"/> to provide audio-specific functionality
/// while maintaining full integration with the AiDotNet neural network infrastructure.
/// </para>
/// <para>
/// <b>For Beginners:</b> Audio neural networks process sound data (like speech or music).
/// This base class provides:
///
/// - Support for pre-trained ONNX models (fast inference with existing models)
/// - Full training capability from scratch (like other neural networks)
/// - Audio preprocessing utilities (mel spectrograms, etc.)
/// - Sample rate handling
///
/// You can use this class in two ways:
/// 1. Load a pre-trained ONNX model for quick inference
/// 2. Build and train a new model from scratch
/// </para>
/// </remarks>
public abstract class AudioNeuralNetworkBase<T> : NeuralNetworkBase<T>
{
    /// <summary>
    /// Gets the sample rate expected by this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common values: 16000 Hz (speech), 22050 Hz (music), 44100 Hz (high quality).
    /// Input audio should be resampled to match this rate.
    /// </para>
    /// </remarks>
    public int SampleRate { get; protected set; } = 16000;

    /// <summary>
    /// Gets the number of mel spectrogram channels used by this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Mel spectrograms divide the frequency range into perceptual bands.
    /// Common values: 64, 80, or 128 mel bins.
    /// </para>
    /// </remarks>
    public int NumMels { get; protected set; } = 80;

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, the model uses pre-trained ONNX weights for inference.
    /// When false, the model uses native layers and can be trained.
    /// </para>
    /// </remarks>
    public bool IsOnnxMode => OnnxEncoder is not null || OnnxDecoder is not null || OnnxModel is not null;

    /// <summary>
    /// Gets or sets the ONNX encoder model (for encoder-decoder architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxEncoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX decoder model (for encoder-decoder architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxDecoder { get; set; }

    /// <summary>
    /// Gets or sets the ONNX model (for single-model architectures).
    /// </summary>
    protected OnnxModel<T>? OnnxModel { get; set; }

    /// <summary>
    /// Gets the mel spectrogram extractor for preprocessing.
    /// </summary>
    protected MelSpectrogram<T>? MelSpec { get; set; }

    /// <summary>
    /// Initializes a new instance of the AudioNeuralNetworkBase class with the specified architecture.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="lossFunction">The loss function to use. If null, a default MSE loss is used.</param>
    /// <param name="maxGradNorm">Maximum gradient norm for gradient clipping.</param>
    protected AudioNeuralNetworkBase(
        NeuralNetworkArchitecture<T> architecture,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>(), maxGradNorm)
    {
        Options = new AudioNeuralNetworkOptions();
    }

    /// <summary>
    /// Gets whether this network supports training.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In ONNX mode, training is not supported - the model is inference-only.
    /// In native mode, training is fully supported.
    /// </para>
    /// </remarks>
    public override bool SupportsTraining => !IsOnnxMode;

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    /// <param name="rawAudio">Raw audio waveform tensor [samples] or [batch, samples].</param>
    /// <returns>Preprocessed audio features suitable for model input.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Raw audio is just a series of numbers representing sound pressure.
    /// Neural networks often work better with transformed representations like mel spectrograms.
    /// This method converts raw audio into the format the model expects.
    /// </para>
    /// </remarks>
    protected abstract Tensor<T> PreprocessAudio(Tensor<T> rawAudio);

    /// <summary>
    /// Postprocesses model output into the final result format.
    /// </summary>
    /// <param name="modelOutput">Raw output from the model.</param>
    /// <returns>Postprocessed output in the expected format.</returns>
    protected abstract Tensor<T> PostprocessOutput(Tensor<T> modelOutput);

    /// <summary>
    /// Runs inference using ONNX model(s).
    /// </summary>
    /// <param name="input">Preprocessed input tensor.</param>
    /// <returns>Model output tensor.</returns>
    /// <remarks>
    /// <para>
    /// Override this method to implement ONNX-specific inference logic
    /// for models with complex encoder-decoder or multi-model architectures.
    /// </para>
    /// </remarks>
    protected virtual Tensor<T> RunOnnxInference(Tensor<T> input)
    {
        if (OnnxModel is not null)
        {
            return OnnxModel.Run(input);
        }

        if (OnnxEncoder is not null)
        {
            var encoded = OnnxEncoder.Run(input);
            if (OnnxDecoder is not null)
            {
                return OnnxDecoder.Run(encoded);
            }
            return encoded;
        }

        throw new InvalidOperationException("No ONNX model is loaded.");
    }

    /// <summary>
    /// Performs a forward pass through the native neural network layers.
    /// </summary>
    /// <param name="input">Preprocessed input tensor.</param>
    /// <returns>Model output tensor.</returns>
    protected virtual Tensor<T> Forward(Tensor<T> input)
    {
        Tensor<T> output = input;
        foreach (var layer in Layers)
        {
            output = layer.Forward(output);
        }
        return output;
    }

    /// <summary>
    /// Gets the default loss function for this model.
    /// </summary>
    public override ILossFunction<T> DefaultLossFunction => LossFunction;

    /// <summary>
    /// Disposes of resources used by this model.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            OnnxEncoder?.Dispose();
            OnnxDecoder?.Dispose();
            OnnxModel?.Dispose();
        }
        base.Dispose(disposing);
    }

    /// <summary>
    /// Creates a mel spectrogram extractor with the model's settings.
    /// </summary>
    /// <param name="sampleRate">Sample rate of input audio.</param>
    /// <param name="nMels">Number of mel bands.</param>
    /// <param name="nFft">FFT window size.</param>
    /// <param name="hopLength">Hop length between frames.</param>
    /// <returns>A configured mel spectrogram extractor.</returns>
    protected MelSpectrogram<T> CreateMelSpectrogram(
        int sampleRate = 16000,
        int nMels = 80,
        int nFft = 1024,
        int hopLength = 256)
    {
        return new MelSpectrogram<T>(
            sampleRate: sampleRate,
            nMels: nMels,
            nFft: nFft,
            hopLength: hopLength);
    }
}
