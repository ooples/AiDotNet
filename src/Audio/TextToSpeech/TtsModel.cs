using System.Diagnostics;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.TextToSpeech;

/// <summary>
/// Text-to-speech model for synthesizing speech from text.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This TTS model uses a two-stage pipeline:
/// 1. Acoustic Model (FastSpeech2): Converts text/phonemes to mel spectrogram
/// 2. Vocoder (HiFi-GAN or Griffin-Lim): Converts mel spectrogram to audio waveform
/// </para>
/// <para><b>For Beginners:</b> Text-to-Speech works like this:
/// 1. Your text is converted to phonemes (speech sounds)
/// 2. The acoustic model predicts what the speech should "look like" (mel spectrogram)
/// 3. The vocoder makes it actually sound like speech
///
/// Usage:
/// <code>
/// var tts = await TtsModel&lt;float&gt;.CreateAsync(new TtsOptions
/// {
///     SpeakingRate = 1.0,
///     PitchShift = 0.0
/// });
///
/// var result = tts.Synthesize("Hello, world!");
/// SaveAudio(result, 22050);  // Your audio saving code
///
/// tts.Dispose();
/// </code>
/// </para>
/// </remarks>
public class TtsModel<T> : AudioNeuralNetworkBase<T>, ITextToSpeech<T>
{
    private readonly TtsOptions _options;
    private readonly OnnxModel<T>? _acousticModel;
    private readonly OnnxModel<T>? _vocoder;
    private readonly GriffinLim<T>? _griffinLim;
    private readonly TtsPreprocessor _preprocessor;
    private bool _disposed;

    /// <summary>
    /// Gets the model options.
    /// </summary>
    public TtsOptions Options => _options;

    /// <summary>
    /// Gets whether the model is ready for synthesis.
    /// </summary>
    public bool IsReady => _acousticModel?.IsLoaded == true &&
        (_vocoder?.IsLoaded == true || _options.UseGriffinLimFallback);

    /// <summary>
    /// Gets the list of available built-in voices.
    /// </summary>
    public IReadOnlyList<VoiceInfo<T>> AvailableVoices { get; }

    /// <summary>
    /// Gets whether this model supports voice cloning from reference audio.
    /// </summary>
    public bool SupportsVoiceCloning => false;

    /// <summary>
    /// Gets whether this model supports emotional expression control.
    /// </summary>
    public bool SupportsEmotionControl => false;

    /// <summary>
    /// Gets whether this model supports streaming audio generation.
    /// </summary>
    public bool SupportsStreaming => false;

    /// <summary>
    /// Creates a new TtsModel instance.
    /// </summary>
    private TtsModel(
        TtsOptions options,
        OnnxModel<T>? acousticModel,
        OnnxModel<T>? vocoder,
        GriffinLim<T>? griffinLim)
        : base(CreateMinimalArchitecture(options))
    {
        _options = options;
        _acousticModel = acousticModel;
        _vocoder = vocoder;
        _griffinLim = griffinLim;
        _preprocessor = new TtsPreprocessor();

        // Store vocoder as OnnxModel for base class
        OnnxModel = vocoder;

        // Set audio properties
        SampleRate = options.SampleRate;
        NumMels = options.NumMels;

        // Initialize available voices
        AvailableVoices = GetDefaultVoices();
    }

    private static NeuralNetworkArchitecture<T> CreateMinimalArchitecture(TtsOptions options)
    {
        // Create minimal architecture for ONNX-only model
        return new NeuralNetworkArchitecture<T>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Generative,
            complexity: NetworkComplexity.Deep,
            inputSize: 256, // Max phoneme sequence length
            outputSize: options.SampleRate * 30 // Max 30 seconds of audio
        );
    }

    private static IReadOnlyList<VoiceInfo<T>> GetDefaultVoices()
    {
        return new[]
        {
            new VoiceInfo<T>
            {
                Id = "default",
                Name = "Default Voice",
                Language = "en",
                Gender = VoiceGender.Neutral,
                Style = "neutral"
            }
        };
    }

    /// <summary>
    /// Creates a TtsModel asynchronously, downloading models if needed.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="progress">Optional download progress reporter.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>The initialized TtsModel.</returns>
    public static async Task<TtsModel<T>> CreateAsync(
        TtsOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new TtsOptions();

        OnnxModel<T>? acousticModel = null;
        OnnxModel<T>? vocoder = null;
        GriffinLim<T>? griffinLim = null;

        try
        {
            // Load acoustic model
            if (options.AcousticModelPath is not null && options.AcousticModelPath.Length > 0)
            {
                acousticModel = new OnnxModel<T>(options.AcousticModelPath, options.OnnxOptions);
            }
            else
            {
                // Download from HuggingFace
                var downloader = new OnnxModelDownloader();
                var path = await downloader.DownloadAsync(
                    OnnxModelRepositories.Tts.FastSpeech2,
                    "model.onnx",
                    progress: new Progress<double>(p => progress?.Report(p * 0.5)),
                    cancellationToken);
                acousticModel = new OnnxModel<T>(path, options.OnnxOptions);
            }

            // Load vocoder
            if (options.VocoderModelPath is not null && options.VocoderModelPath.Length > 0)
            {
                vocoder = new OnnxModel<T>(options.VocoderModelPath, options.OnnxOptions);
            }
            else if (!options.UseGriffinLimFallback)
            {
                var downloader = new OnnxModelDownloader();
                var path = await downloader.DownloadAsync(
                    OnnxModelRepositories.Tts.HiFiGan,
                    "model.onnx",
                    progress: new Progress<double>(p => progress?.Report(0.5 + p * 0.5)),
                    cancellationToken);
                vocoder = new OnnxModel<T>(path, options.OnnxOptions);
            }

            // Create Griffin-Lim fallback
            if (options.UseGriffinLimFallback)
            {
                griffinLim = new GriffinLim<T>(
                    nFft: options.FftSize,
                    hopLength: options.HopLength,
                    iterations: options.GriffinLimIterations);
            }

            return new TtsModel<T>(options, acousticModel, vocoder, griffinLim);
        }
        catch
        {
            acousticModel?.Dispose();
            vocoder?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Creates a TtsModel from local model files.
    /// </summary>
    public static TtsModel<T> FromFiles(
        string acousticModelPath,
        string? vocoderPath = null,
        TtsOptions? options = null)
    {
        options ??= new TtsOptions();
        options.AcousticModelPath = acousticModelPath;
        options.VocoderModelPath = vocoderPath;

        var acousticModel = new OnnxModel<T>(acousticModelPath, options.OnnxOptions);
        OnnxModel<T>? vocoder = null;
        GriffinLim<T>? griffinLim = null;

        if (vocoderPath is not null && vocoderPath.Length > 0)
        {
            vocoder = new OnnxModel<T>(vocoderPath, options.OnnxOptions);
        }

        if (options.UseGriffinLimFallback || vocoder is null)
        {
            griffinLim = new GriffinLim<T>(
                nFft: options.FftSize,
                hopLength: options.HopLength,
                iterations: options.GriffinLimIterations);
        }

        return new TtsModel<T>(options, acousticModel, vocoder, griffinLim);
    }

    #region ITextToSpeech Implementation

    /// <summary>
    /// Synthesizes speech from text.
    /// </summary>
    public Tensor<T> Synthesize(
        string text,
        string? voiceId = null,
        double speakingRate = 1.0,
        double pitch = 0.0)
    {
        ThrowIfDisposed();

        var stopwatch = Stopwatch.StartNew();

        // Override options with parameters
        double effectiveRate = Math.Abs(speakingRate - 1.0) > 0.01 ? speakingRate : _options.SpeakingRate;

        // Preprocess text to phonemes
        var phonemes = _preprocessor.TextToPhonemes(text);

        // Generate mel spectrogram
        var melSpectrogram = GenerateMelSpectrogram(phonemes, voiceId);

        // Apply rate modifications
        if (Math.Abs(effectiveRate - 1.0) > 0.01)
        {
            melSpectrogram = ModifyDuration(melSpectrogram, 1.0 / effectiveRate);
        }

        // Generate audio waveform
        Tensor<T> audio;
        if (_vocoder is not null)
        {
            audio = VocoderSynthesize(melSpectrogram);
        }
        else if (_griffinLim is not null)
        {
            audio = GriffinLimSynthesize(melSpectrogram);
        }
        else
        {
            throw new InvalidOperationException("No vocoder available.");
        }

        // Apply energy/volume
        if (Math.Abs(_options.Energy - 1.0) > 0.01)
        {
            var result = new Tensor<T>(audio.Shape);
            for (int i = 0; i < audio.Length; i++)
            {
                result[i] = NumOps.Multiply(audio[i], NumOps.FromDouble(_options.Energy));
            }
            audio = result;
        }

        stopwatch.Stop();

        return audio;
    }

    /// <summary>
    /// Synthesizes speech from text asynchronously.
    /// </summary>
    public Task<Tensor<T>> SynthesizeAsync(
        string text,
        string? voiceId = null,
        double speakingRate = 1.0,
        double pitch = 0.0,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Synthesize(text, voiceId, speakingRate, pitch), cancellationToken);
    }

    /// <summary>
    /// Synthesizes speech using a cloned voice from reference audio.
    /// </summary>
    public Tensor<T> SynthesizeWithVoiceCloning(
        string text,
        Tensor<T> referenceAudio,
        double speakingRate = 1.0,
        double pitch = 0.0)
    {
        throw new NotSupportedException("Voice cloning is not supported by this TTS model.");
    }

    /// <summary>
    /// Synthesizes speech with emotional expression.
    /// </summary>
    public Tensor<T> SynthesizeWithEmotion(
        string text,
        string emotion,
        double emotionIntensity = 0.5,
        string? voiceId = null,
        double speakingRate = 1.0)
    {
        throw new NotSupportedException("Emotion control is not supported by this TTS model.");
    }

    /// <summary>
    /// Extracts speaker embedding from reference audio for voice cloning.
    /// </summary>
    public Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
    {
        throw new NotSupportedException("Speaker embedding extraction is not supported by this TTS model.");
    }

    /// <summary>
    /// Starts a streaming synthesis session.
    /// </summary>
    public IStreamingSynthesisSession<T> StartStreamingSession(string? voiceId = null, double speakingRate = 1.0)
    {
        throw new NotSupportedException("Streaming synthesis is not supported by this TTS model.");
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // For TTS, input is text not audio
        // This method is used for reference audio in voice cloning scenarios
        if (MelSpec is not null)
        {
            return MelSpec.Forward(rawAudio);
        }

        return rawAudio;
    }

    /// <summary>
    /// Postprocesses model output into the final result format.
    /// </summary>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // For TTS, postprocessing is handled in Synthesize method
        return modelOutput;
    }

    /// <summary>
    /// Initializes the neural network layers.
    /// </summary>
    protected override void InitializeLayers()
    {
        // ONNX-only model - no native layers to initialize
    }

    /// <summary>
    /// Makes a prediction using the model.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // For TTS, prediction means synthesis from phoneme input
        if (_acousticModel is null)
            throw new InvalidOperationException("Acoustic model not loaded.");

        return _acousticModel.Run(input);
    }

    /// <summary>
    /// Updates model parameters.
    /// </summary>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (IsOnnxMode)
        {
            throw new NotSupportedException("Cannot update parameters in ONNX inference mode.");
        }

        // Native training mode - update layer parameters from the parameter vector
        // This would be implemented when native training support is added
        throw new NotImplementedException("Native parameter updates for TTS are not yet implemented.");
    }

    /// <summary>
    /// Trains the model on input data.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode)
        {
            throw new NotSupportedException("Cannot train in ONNX inference mode. Load the model in training mode instead.");
        }

        throw new NotImplementedException("Native training for TTS is not yet implemented.");
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "TtsModel-FastSpeech2-HiFiGAN",
            Description = "Text-to-speech model using FastSpeech2 acoustic model and HiFi-GAN vocoder",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = 256,
            Complexity = 2
        };
        metadata.AdditionalInfo["InputFormat"] = "Text/Phonemes";
        metadata.AdditionalInfo["OutputFormat"] = $"Audio ({SampleRate}Hz)";
        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_options.SampleRate);
        writer.Write(_options.NumMels);
        writer.Write(_options.SpeakingRate);
        writer.Write(_options.Energy);
        writer.Write(_options.UseGriffinLimFallback);
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _options.SampleRate = reader.ReadInt32();
        _options.NumMels = reader.ReadInt32();
        _options.SpeakingRate = reader.ReadDouble();
        _options.Energy = reader.ReadDouble();
        _options.UseGriffinLimFallback = reader.ReadBoolean();
    }

    /// <summary>
    /// Creates a new instance of this model for cloning.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new TtsModel<T>(_options, null, null, null);
    }

    #endregion

    #region Private Methods

    private Tensor<T> GenerateMelSpectrogram(int[] phonemes, string? voiceId)
    {
        if (_acousticModel is null)
            throw new InvalidOperationException("Acoustic model not loaded.");

        // Create phoneme tensor
        var phonemeTensor = new Tensor<T>([1, phonemes.Length]);
        for (int i = 0; i < phonemes.Length; i++)
        {
            phonemeTensor[0, i] = NumOps.FromDouble(phonemes[i]);
        }

        // Add speaker ID if multi-speaker
        var inputs = new Dictionary<string, Tensor<T>>
        {
            ["phoneme_ids"] = phonemeTensor
        };

        if (_options.SpeakerId.HasValue)
        {
            var speakerTensor = new Tensor<T>([1]);
            speakerTensor[0] = NumOps.FromDouble(_options.SpeakerId.Value);
            inputs["speaker_id"] = speakerTensor;
        }

        var outputs = _acousticModel.Run(inputs);
        return outputs.Values.First();
    }

    private Tensor<T> ModifyDuration(Tensor<T> melSpectrogram, double factor)
    {
        int originalFrames = melSpectrogram.Shape[^2];
        int numMels = melSpectrogram.Shape[^1];
        int newFrames = (int)(originalFrames * factor);

        var modified = new Tensor<T>([1, newFrames, numMels]);

        for (int f = 0; f < newFrames; f++)
        {
            double srcFrame = f / factor;
            int srcIdx = Math.Min((int)srcFrame, originalFrames - 1);

            for (int m = 0; m < numMels; m++)
            {
                modified[0, f, m] = melSpectrogram.Rank == 3
                    ? melSpectrogram[0, srcIdx, m]
                    : melSpectrogram[srcIdx, m];
            }
        }

        return modified;
    }

    private Tensor<T> VocoderSynthesize(Tensor<T> melSpectrogram)
    {
        if (_vocoder is null)
            throw new InvalidOperationException("Vocoder not loaded.");

        return _vocoder.Run(melSpectrogram);
    }

    private Tensor<T> GriffinLimSynthesize(Tensor<T> melSpectrogram)
    {
        if (_griffinLim is null)
            throw new InvalidOperationException("Griffin-Lim not available.");

        // Griffin-Lim expects 2D mel spectrogram [frames, mels]
        Tensor<T> mel2D;
        if (melSpectrogram.Rank == 3)
        {
            int frames = melSpectrogram.Shape[1];
            int mels = melSpectrogram.Shape[2];
            mel2D = new Tensor<T>([frames, mels]);

            for (int f = 0; f < frames; f++)
            {
                for (int m = 0; m < mels; m++)
                {
                    mel2D[f, m] = melSpectrogram[0, f, m];
                }
            }
        }
        else
        {
            mel2D = melSpectrogram;
        }

        return _griffinLim.Reconstruct(mel2D);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName);
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes the model and releases resources.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            _acousticModel?.Dispose();
            // _vocoder is stored as OnnxModel and disposed by base class
        }

        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
