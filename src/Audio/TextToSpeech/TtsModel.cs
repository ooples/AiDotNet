using System.Diagnostics;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
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
/// var result = await tts.SynthesizeAsync("Hello, world!");
/// SaveAudio(result.Audio, result.SampleRate);  // Your audio saving code
///
/// tts.Dispose();
/// </code>
/// </para>
/// </remarks>
public class TtsModel<T> : IDisposable
{
    private readonly INumericOperations<T> _numOps;
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
    /// Creates a new TtsModel instance.
    /// </summary>
    private TtsModel(
        TtsOptions options,
        OnnxModel<T>? acousticModel,
        OnnxModel<T>? vocoder,
        GriffinLim<T>? griffinLim)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options;
        _acousticModel = acousticModel;
        _vocoder = vocoder;
        _griffinLim = griffinLim;
        _preprocessor = new TtsPreprocessor();
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
            if (!string.IsNullOrEmpty(options.AcousticModelPath))
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
            if (!string.IsNullOrEmpty(options.VocoderModelPath))
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

        if (!string.IsNullOrEmpty(vocoderPath))
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

    /// <summary>
    /// Synthesizes speech from text.
    /// </summary>
    /// <param name="text">The text to synthesize.</param>
    /// <returns>The synthesized audio.</returns>
    public TtsResult<T> Synthesize(string text)
    {
        ThrowIfDisposed();

        var stopwatch = Stopwatch.StartNew();

        // Preprocess text to phonemes
        var phonemes = _preprocessor.TextToPhonemes(text);

        // Generate mel spectrogram
        var melSpectrogram = GenerateMelSpectrogram(phonemes);

        // Apply rate/pitch modifications
        if (Math.Abs(_options.SpeakingRate - 1.0) > 0.01)
        {
            melSpectrogram = ModifyDuration(melSpectrogram, 1.0 / _options.SpeakingRate);
        }

        // Generate audio waveform
        T[] audio;
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
            for (int i = 0; i < audio.Length; i++)
            {
                audio[i] = _numOps.Multiply(audio[i], _numOps.FromDouble(_options.Energy));
            }
        }

        stopwatch.Stop();

        return new TtsResult<T>
        {
            Audio = audio,
            SampleRate = _options.SampleRate,
            Duration = (double)audio.Length / _options.SampleRate,
            ProcessingTimeMs = stopwatch.ElapsedMilliseconds
        };
    }

    /// <summary>
    /// Synthesizes speech from text asynchronously.
    /// </summary>
    public Task<TtsResult<T>> SynthesizeAsync(
        string text,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Synthesize(text), cancellationToken);
    }

    private Tensor<T> GenerateMelSpectrogram(int[] phonemes)
    {
        if (_acousticModel is null)
            throw new InvalidOperationException("Acoustic model not loaded.");

        // Create phoneme tensor
        var phonemeTensor = new Tensor<T>([1, phonemes.Length]);
        for (int i = 0; i < phonemes.Length; i++)
        {
            phonemeTensor[0, i] = _numOps.FromDouble(phonemes[i]);
        }

        // Add speaker ID if multi-speaker
        var inputs = new Dictionary<string, Tensor<T>>
        {
            ["phoneme_ids"] = phonemeTensor
        };

        if (_options.SpeakerId.HasValue)
        {
            var speakerTensor = new Tensor<T>([1]);
            speakerTensor[0] = _numOps.FromDouble(_options.SpeakerId.Value);
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

    private T[] VocoderSynthesize(Tensor<T> melSpectrogram)
    {
        if (_vocoder is null)
            throw new InvalidOperationException("Vocoder not loaded.");

        var output = _vocoder.Run(melSpectrogram);
        return output.ToArray();
    }

    private T[] GriffinLimSynthesize(Tensor<T> melSpectrogram)
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

        var audio = _griffinLim.Reconstruct(mel2D);
        return audio.ToArray();
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    /// <summary>
    /// Disposes the model and releases resources.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes managed and unmanaged resources.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            _acousticModel?.Dispose();
            _vocoder?.Dispose();
        }

        _disposed = true;
    }
}
