using AiDotNet.Diffusion.Audio;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;

namespace AiDotNet.Diffusion;

/// <summary>
/// Base class for audio diffusion models that generate sound and music.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This abstract base class provides common functionality for all audio diffusion models,
/// including text-to-audio generation, text-to-music, text-to-speech, and audio transformation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the foundation for audio generation models like AudioLDM.
/// It extends latent diffusion to work with audio by converting sound to spectrograms
/// (visual representations of sound) and back.
/// </para>
/// <para>
/// How audio diffusion works:
/// 1. Audio is converted to a mel spectrogram (frequency vs time image)
/// 2. The spectrogram is encoded to latent space (like images)
/// 3. Diffusion denoising happens in latent space
/// 4. The result is decoded to a spectrogram
/// 5. A vocoder converts the spectrogram back to audio
/// </para>
/// </remarks>
public abstract class AudioDiffusionModelBase<T> : LatentDiffusionModelBase<T>, IAudioDiffusionModel<T>
{
    /// <summary>
    /// Sample rate in Hz.
    /// </summary>
    private readonly int _sampleRate;

    /// <summary>
    /// Default audio duration in seconds.
    /// </summary>
    private readonly double _defaultDurationSeconds;

    /// <summary>
    /// Number of mel spectrogram channels.
    /// </summary>
    private readonly int _melChannels;

    /// <summary>
    /// GPU-accelerated mel spectrogram processor.
    /// </summary>
    private MelSpectrogram<T>? _melSpectrogramProcessor;

    /// <summary>
    /// GPU-accelerated Griffin-Lim audio reconstructor.
    /// </summary>
    private GriffinLim<T>? _griffinLimProcessor;

    /// <inheritdoc />
    public virtual int SampleRate => _sampleRate;

    /// <inheritdoc />
    public virtual double DefaultDurationSeconds => _defaultDurationSeconds;

    /// <inheritdoc />
    public abstract bool SupportsTextToAudio { get; }

    /// <inheritdoc />
    public abstract bool SupportsTextToMusic { get; }

    /// <inheritdoc />
    public abstract bool SupportsTextToSpeech { get; }

    /// <inheritdoc />
    public abstract bool SupportsAudioToAudio { get; }

    /// <inheritdoc />
    public virtual int MelChannels => _melChannels;

    /// <summary>
    /// Gets the hop length for spectrogram computation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Hop length is the number of audio samples between successive frames.
    /// Lower values = higher time resolution but more computation.
    /// Typical values: 256, 512, 1024.
    /// </para>
    /// </remarks>
    public virtual int HopLength { get; protected set; } = 512;

    /// <summary>
    /// Gets the FFT window size.
    /// </summary>
    public virtual int FFTSize { get; protected set; } = 2048;

    /// <summary>
    /// Gets the minimum frequency for mel filterbank.
    /// </summary>
    public virtual double MinFrequency { get; protected set; } = 0.0;

    /// <summary>
    /// Gets the maximum frequency for mel filterbank.
    /// </summary>
    public virtual double MaxFrequency { get; protected set; } = 8000.0;

    /// <summary>
    /// Initializes a new instance of the AudioDiffusionModelBase class.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model.</param>
    /// <param name="scheduler">Optional custom scheduler.</param>
    /// <param name="sampleRate">Audio sample rate in Hz.</param>
    /// <param name="defaultDurationSeconds">Default audio duration.</param>
    /// <param name="melChannels">Number of mel spectrogram channels.</param>
    protected AudioDiffusionModelBase(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        int sampleRate = 16000,
        double defaultDurationSeconds = 10.0,
        int melChannels = 64,
        NeuralNetworkArchitecture<T>? architecture = null)
        : base(options, scheduler, architecture)
    {
        _sampleRate = sampleRate;
        _defaultDurationSeconds = defaultDurationSeconds;
        _melChannels = melChannels;
    }

    #region IAudioDiffusionModel<T> Implementation

    /// <inheritdoc />
    public virtual Tensor<T> GenerateFromText(
        string prompt,
        string? negativePrompt = null,
        double? durationSeconds = null,
        int numInferenceSteps = 100,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        if (!SupportsTextToAudio)
            throw new NotSupportedException("This model does not support text-to-audio generation.");

        if (Conditioner == null)
            throw new InvalidOperationException("Text-to-audio generation requires a conditioning module.");

        var effectiveDuration = durationSeconds ?? DefaultDurationSeconds;
        var useCFG = guidanceScale > 1.0 && NoisePredictor.SupportsCFG;

        // Encode text prompts
        var promptTokens = Conditioner.Tokenize(prompt);
        var promptEmbedding = Conditioner.EncodeText(promptTokens);

        Tensor<T>? negativeEmbedding = null;
        if (useCFG)
        {
            if (!string.IsNullOrEmpty(negativePrompt))
            {
                var negTokens = Conditioner.Tokenize(negativePrompt ?? string.Empty);
                negativeEmbedding = Conditioner.EncodeText(negTokens);
            }
            else
            {
                negativeEmbedding = Conditioner.GetUnconditionalEmbedding(1);
            }
        }

        // Calculate spectrogram dimensions
        var numSamples = (int)(effectiveDuration * SampleRate);
        var numFrames = numSamples / HopLength + 1;
        var latentTimeFrames = numFrames / VAE.DownsampleFactor;
        var latentShape = new[] { 1, LatentChannels, MelChannels / VAE.DownsampleFactor, latentTimeFrames };

        // Generate initial noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = DiffusionNoiseHelper<T>.SampleGaussian(latentShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop
        foreach (var timestep in Scheduler.Timesteps)
        {
            Tensor<T> noisePrediction;

            if (useCFG && negativeEmbedding != null)
            {
                var condPred = NoisePredictor.PredictNoise(latents, timestep, promptEmbedding);
                var uncondPred = NoisePredictor.PredictNoise(latents, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidance(uncondPred, condPred, guidanceScale);
            }
            else
            {
                noisePrediction = NoisePredictor.PredictNoise(latents, timestep, promptEmbedding);
            }

            // Scheduler step
            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);
        }

        // Decode to mel spectrogram
        var melSpectrogram = DecodeFromLatent(latents);

        // Convert to waveform
        return MelSpectrogramToWaveform(melSpectrogram);
    }

    /// <inheritdoc />
    public virtual Tensor<T> GenerateMusic(
        string prompt,
        string? negativePrompt = null,
        double? durationSeconds = null,
        int numInferenceSteps = 100,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        if (!SupportsTextToMusic)
            throw new NotSupportedException("This model does not support text-to-music generation.");

        // Music generation uses the same pipeline as audio generation
        // Derived classes may override for specialized music models
        return GenerateFromText(prompt, negativePrompt, durationSeconds, numInferenceSteps, guidanceScale, seed);
    }

    /// <inheritdoc />
    public virtual Tensor<T> TextToSpeech(
        string text,
        Tensor<T>? speakerEmbedding = null,
        double speakingRate = 1.0,
        int numInferenceSteps = 50,
        int? seed = null)
    {
        if (!SupportsTextToSpeech)
            throw new NotSupportedException("This model does not support text-to-speech generation.");

        if (Conditioner == null)
            throw new InvalidOperationException("Text-to-speech requires a conditioning module.");

        // Encode text
        var textTokens = Conditioner.Tokenize(text);
        var textEmbedding = Conditioner.EncodeText(textTokens);

        // Combine with speaker embedding if provided
        if (speakerEmbedding != null)
        {
            textEmbedding = CombineTextAndSpeakerEmbeddings(textEmbedding, speakerEmbedding);
        }

        // Estimate duration based on text length and speaking rate
        var estimatedDuration = EstimateSpeechDuration(text, speakingRate);
        var numSamples = (int)(estimatedDuration * SampleRate);
        var numFrames = numSamples / HopLength + 1;
        var latentTimeFrames = numFrames / VAE.DownsampleFactor;
        var latentShape = new[] { 1, LatentChannels, MelChannels / VAE.DownsampleFactor, latentTimeFrames };

        // Generate initial noise
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var latents = DiffusionNoiseHelper<T>.SampleGaussian(latentShape, rng);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Denoising loop
        foreach (var timestep in Scheduler.Timesteps)
        {
            var noisePrediction = NoisePredictor.PredictNoise(latents, timestep, textEmbedding);

            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);
        }

        // Decode to mel spectrogram
        var melSpectrogram = DecodeFromLatent(latents);

        // Convert to waveform
        return MelSpectrogramToWaveform(melSpectrogram);
    }

    /// <inheritdoc />
    public virtual Tensor<T> AudioToAudio(
        Tensor<T> inputAudio,
        string prompt,
        string? negativePrompt = null,
        double strength = 0.5,
        int numInferenceSteps = 100,
        double guidanceScale = 3.0,
        int? seed = null)
    {
        if (!SupportsAudioToAudio)
            throw new NotSupportedException("This model does not support audio-to-audio transformation.");

        if (Conditioner == null)
            throw new InvalidOperationException("Audio-to-audio transformation requires a conditioning module.");

        strength = MathPolyfill.Clamp(strength, 0.0, 1.0);
        var useCFG = guidanceScale > 1.0 && NoisePredictor.SupportsCFG;

        // Convert input audio to mel spectrogram
        var inputMel = WaveformToMelSpectrogram(inputAudio);

        // Encode to latent space
        var latents = EncodeToLatent(inputMel, sampleMode: false);
        var latentShape = latents.Shape;

        // Encode text prompts
        var promptTokens = Conditioner.Tokenize(prompt);
        var promptEmbedding = Conditioner.EncodeText(promptTokens);

        Tensor<T>? negativeEmbedding = null;
        if (useCFG)
        {
            if (!string.IsNullOrEmpty(negativePrompt))
            {
                var negTokens = Conditioner.Tokenize(negativePrompt ?? string.Empty);
                negativeEmbedding = Conditioner.EncodeText(negTokens);
            }
            else
            {
                negativeEmbedding = Conditioner.GetUnconditionalEmbedding(1);
            }
        }

        // Calculate starting timestep based on strength
        Scheduler.SetTimesteps(numInferenceSteps);
        var startStep = (int)(numInferenceSteps * (1.0 - strength));
        var startTimestep = Scheduler.Timesteps.Skip(startStep).First();

        // Add noise to latents at starting timestep
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var noise = DiffusionNoiseHelper<T>.SampleGaussian(latentShape, rng);
        var noisyLatents = Scheduler.AddNoise(latents.ToVector(), noise.ToVector(), startTimestep);
        latents = new Tensor<T>(latentShape, noisyLatents);

        // Denoising loop (starting from startStep)
        foreach (var timestep in Scheduler.Timesteps.Skip(startStep))
        {
            Tensor<T> noisePrediction;

            if (useCFG && negativeEmbedding != null)
            {
                var condPred = NoisePredictor.PredictNoise(latents, timestep, promptEmbedding);
                var uncondPred = NoisePredictor.PredictNoise(latents, timestep, negativeEmbedding);
                noisePrediction = ApplyGuidance(uncondPred, condPred, guidanceScale);
            }
            else
            {
                noisePrediction = NoisePredictor.PredictNoise(latents, timestep, promptEmbedding);
            }

            var latentVector = latents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            latents = new Tensor<T>(latentShape, latentVector);
        }

        // Decode to mel spectrogram
        var melSpectrogram = DecodeFromLatent(latents);

        // Convert to waveform
        return MelSpectrogramToWaveform(melSpectrogram);
    }

    /// <inheritdoc />
    public virtual Tensor<T> ContinueAudio(
        Tensor<T> inputAudio,
        string? prompt = null,
        double extensionSeconds = 5.0,
        int numInferenceSteps = 100,
        int? seed = null)
    {
        // Convert input audio to mel spectrogram
        var inputMel = WaveformToMelSpectrogram(inputAudio);

        // Encode to latent
        var inputLatents = EncodeToLatent(inputMel, sampleMode: false);
        var inputShape = inputLatents.Shape;

        // Calculate extension latent dimensions
        var extensionSamples = (int)(extensionSeconds * SampleRate);
        var extensionFrames = extensionSamples / HopLength + 1;
        var extensionLatentFrames = extensionFrames / VAE.DownsampleFactor;

        var extensionShape = new[] { inputShape[0], inputShape[1], inputShape[2], extensionLatentFrames };

        // Generate noise for extension
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var extensionLatents = DiffusionNoiseHelper<T>.SampleGaussian(extensionShape, rng);

        // Get conditioning from end of input
        var contextLatents = ExtractLatentContext(inputLatents);

        // Set up scheduler
        Scheduler.SetTimesteps(numInferenceSteps);

        // Encode prompt if provided
        Tensor<T>? promptEmbedding = null;
        if (!string.IsNullOrEmpty(prompt) && Conditioner != null)
        {
            var promptTokens = Conditioner.Tokenize(prompt ?? string.Empty);
            promptEmbedding = Conditioner.EncodeText(promptTokens);
        }

        // Denoising loop for extension
        foreach (var timestep in Scheduler.Timesteps)
        {
            // Predict noise (using context-aware prediction)
            var noisePrediction = PredictNoiseWithContext(
                extensionLatents, timestep, contextLatents, promptEmbedding);

            var latentVector = extensionLatents.ToVector();
            var noiseVector = noisePrediction.ToVector();
            latentVector = Scheduler.Step(noiseVector, timestep, latentVector, NumOps.Zero);
            extensionLatents = new Tensor<T>(extensionShape, latentVector);
        }

        // Decode extension to mel spectrogram
        var extensionMel = DecodeFromLatent(extensionLatents);

        // Convert to waveform
        var extensionWaveform = MelSpectrogramToWaveform(extensionMel);

        // Concatenate with input audio
        return ConcatenateAudio(inputAudio, extensionWaveform);
    }

    /// <inheritdoc />
    public virtual Tensor<T> WaveformToMelSpectrogram(Tensor<T> waveform)
    {
        // Ensure the mel spectrogram processor is initialized
        EnsureMelSpectrogramProcessorInitialized();

        var waveformShape = waveform.Shape;
        var batchSize = waveformShape[0];
        var numSamples = waveformShape[^1];

        // Process each batch item using the GPU-accelerated mel spectrogram
        var numFrames = numSamples / HopLength + 1;
        var melShape = new[] { batchSize, 1, MelChannels, numFrames };
        var melSpectrogram = new Tensor<T>(melShape);
        var melSpan = melSpectrogram.AsWritableSpan();

        for (int b = 0; b < batchSize; b++)
        {
            // Extract single waveform from batch
            var singleWaveform = ExtractBatchWaveform(waveform, b, numSamples);

            // Compute mel spectrogram using GPU-accelerated processor
            var singleMel = _melSpectrogramProcessor!.Forward(singleWaveform);

            // Copy result to batch output
            CopyMelToBatch(melSpan, singleMel, b, MelChannels, numFrames);
        }

        return melSpectrogram;
    }

    /// <summary>
    /// Ensures the mel spectrogram processor is initialized.
    /// </summary>
    private void EnsureMelSpectrogramProcessorInitialized()
    {
        if (_melSpectrogramProcessor == null)
        {
            _melSpectrogramProcessor = new MelSpectrogram<T>(
                sampleRate: SampleRate,
                nFft: FFTSize,
                hopLength: HopLength,
                nMels: MelChannels,
                fMin: MinFrequency,
                fMax: MaxFrequency,
                logMel: true);
        }
    }

    /// <summary>
    /// Extracts a single waveform from a batch tensor.
    /// </summary>
    private Tensor<T> ExtractBatchWaveform(Tensor<T> waveform, int batchIndex, int numSamples)
    {
        var singleWaveform = new Tensor<T>(new[] { numSamples });
        var srcSpan = waveform.AsSpan();
        var dstSpan = singleWaveform.AsWritableSpan();
        var offset = batchIndex * numSamples;

        for (int i = 0; i < numSamples; i++)
        {
            dstSpan[i] = srcSpan[offset + i];
        }

        return singleWaveform;
    }

    /// <summary>
    /// Copies a mel spectrogram to a specific batch position.
    /// </summary>
    private void CopyMelToBatch(Span<T> batchSpan, Tensor<T> singleMel, int batchIndex, int melChannels, int numFrames)
    {
        var srcSpan = singleMel.AsSpan();
        var batchOffset = batchIndex * melChannels * numFrames;
        var srcLen = Math.Min(srcSpan.Length, melChannels * numFrames);

        for (int i = 0; i < srcLen; i++)
        {
            batchSpan[batchOffset + i] = srcSpan[i];
        }
    }

    /// <inheritdoc />
    public virtual Tensor<T> MelSpectrogramToWaveform(Tensor<T> melSpectrogram)
    {
        // Ensure processors are initialized
        EnsureMelSpectrogramProcessorInitialized();
        EnsureGriffinLimProcessorInitialized();

        var melShape = melSpectrogram.Shape;
        var batchSize = melShape[0];
        var numFrames = melShape[^1];
        var numSamples = (numFrames - 1) * HopLength;

        // Output shape: [batch, numSamples]
        var waveformShape = new[] { batchSize, numSamples };
        var waveform = new Tensor<T>(waveformShape);
        var waveSpan = waveform.AsWritableSpan();

        // Process each batch item using GPU-accelerated Griffin-Lim
        for (int b = 0; b < batchSize; b++)
        {
            // Extract single mel spectrogram from batch
            var singleMel = ExtractBatchMel(melSpectrogram, b, MelChannels, numFrames);

            // Invert mel spectrogram to linear magnitude using the filterbank
            var magnitude = _melSpectrogramProcessor!.InvertMelToMagnitude(singleMel);

            // Reconstruct audio using Griffin-Lim
            var singleAudio = _griffinLimProcessor!.Reconstruct(magnitude, numSamples);

            // Copy result to batch output
            CopyAudioToBatch(waveSpan, singleAudio, b, numSamples);
        }

        return waveform;
    }

    /// <summary>
    /// Ensures the Griffin-Lim processor is initialized.
    /// </summary>
    private void EnsureGriffinLimProcessorInitialized()
    {
        if (_griffinLimProcessor == null)
        {
            _griffinLimProcessor = new GriffinLim<T>(
                nFft: FFTSize,
                hopLength: HopLength,
                iterations: 32,
                momentum: 0.99);
        }
    }

    /// <summary>
    /// Extracts a single mel spectrogram from a batch tensor.
    /// </summary>
    private Tensor<T> ExtractBatchMel(Tensor<T> melSpectrogram, int batchIndex, int melChannels, int numFrames)
    {
        var singleMel = new Tensor<T>(new[] { melChannels, numFrames });
        var srcSpan = melSpectrogram.AsSpan();
        var dstSpan = singleMel.AsWritableSpan();
        var offset = batchIndex * melChannels * numFrames;

        for (int i = 0; i < melChannels * numFrames; i++)
        {
            dstSpan[i] = srcSpan[offset + i];
        }

        return singleMel;
    }

    /// <summary>
    /// Copies audio samples to a specific batch position.
    /// </summary>
    private void CopyAudioToBatch(Span<T> batchSpan, Tensor<T> singleAudio, int batchIndex, int numSamples)
    {
        var srcSpan = singleAudio.AsSpan();
        var batchOffset = batchIndex * numSamples;
        var srcLen = Math.Min(srcSpan.Length, numSamples);

        for (int i = 0; i < srcLen; i++)
        {
            batchSpan[batchOffset + i] = srcSpan[i];
        }
    }

    /// <inheritdoc />
    public virtual Tensor<T> ExtractSpeakerEmbedding(Tensor<T> referenceAudio)
    {
        // Ensure mel spectrogram processor is initialized
        EnsureMelSpectrogramProcessorInitialized();

        // Extract mel spectrogram features from the reference audio
        var audioShape = referenceAudio.Shape;
        var numSamples = audioShape[^1];

        // Get single channel waveform for processing
        Tensor<T> waveform;
        if (audioShape.Length > 1 && audioShape[0] > 1)
        {
            // Multi-channel: extract first channel
            waveform = ExtractBatchWaveform(referenceAudio, 0, numSamples);
        }
        else
        {
            waveform = referenceAudio;
        }

        // Compute mel spectrogram using GPU-accelerated processor
        var melSpectrogram = _melSpectrogramProcessor!.Forward(waveform);
        var melSpan = melSpectrogram.AsSpan();

        // Compute statistical embedding from mel spectrogram
        // Using mel-frequency cepstral statistics as speaker embedding features
        var embeddingDim = 256;
        var embedding = new Tensor<T>(new[] { 1, embeddingDim });
        var embSpan = embedding.AsWritableSpan();

        var melChannels = melSpectrogram.Shape[0];
        var melFrames = melSpectrogram.Shape[^1];

        // Compute per-channel statistics (mean, variance, delta mean, delta variance)
        var statsPerChannel = Math.Min(4, embeddingDim / melChannels);
        var featureIdx = 0;

        for (int mel = 0; mel < melChannels && featureIdx < embeddingDim; mel++)
        {
            // Compute mean for this mel channel
            double channelSum = 0;
            for (int frame = 0; frame < melFrames; frame++)
            {
                channelSum += NumOps.ToDouble(melSpan[mel * melFrames + frame]);
            }
            var channelMean = channelSum / melFrames;

            // Compute variance for this mel channel
            double channelVarSum = 0;
            for (int frame = 0; frame < melFrames; frame++)
            {
                var diff = NumOps.ToDouble(melSpan[mel * melFrames + frame]) - channelMean;
                channelVarSum += diff * diff;
            }
            var channelVar = channelVarSum / melFrames;

            // Compute delta statistics (temporal dynamics)
            double deltaSum = 0, deltaVarSum = 0;
            for (int frame = 1; frame < melFrames; frame++)
            {
                var delta = NumOps.ToDouble(melSpan[mel * melFrames + frame]) -
                            NumOps.ToDouble(melSpan[mel * melFrames + frame - 1]);
                deltaSum += delta;
                deltaVarSum += delta * delta;
            }
            var deltaMean = deltaSum / Math.Max(1, melFrames - 1);
            var deltaVar = deltaVarSum / Math.Max(1, melFrames - 1);

            // Store statistics as embedding features
            if (featureIdx < embeddingDim)
                embSpan[featureIdx++] = NumOps.FromDouble(channelMean);
            if (featureIdx < embeddingDim)
                embSpan[featureIdx++] = NumOps.FromDouble(Math.Sqrt(channelVar + 1e-6));
            if (statsPerChannel > 2 && featureIdx < embeddingDim)
                embSpan[featureIdx++] = NumOps.FromDouble(deltaMean);
            if (statsPerChannel > 3 && featureIdx < embeddingDim)
                embSpan[featureIdx++] = NumOps.FromDouble(Math.Sqrt(deltaVar + 1e-6));
        }

        // Fill remaining dimensions with zero padding
        while (featureIdx < embeddingDim)
        {
            embSpan[featureIdx++] = NumOps.Zero;
        }

        // L2 normalize the embedding
        double norm = 0;
        for (int i = 0; i < embeddingDim; i++)
        {
            var val = NumOps.ToDouble(embSpan[i]);
            norm += val * val;
        }
        norm = Math.Sqrt(norm + 1e-8);

        for (int i = 0; i < embeddingDim; i++)
        {
            embSpan[i] = NumOps.FromDouble(NumOps.ToDouble(embSpan[i]) / norm);
        }

        return embedding;
    }

    #endregion

    #region Protected Helper Methods

    /// <summary>
    /// Combines text embedding with speaker embedding for TTS.
    /// </summary>
    protected virtual Tensor<T> CombineTextAndSpeakerEmbeddings(Tensor<T> textEmbedding, Tensor<T> speakerEmbedding)
    {
        // Concatenate along feature dimension
        return Engine.TensorConcatenate<T>(new[] { textEmbedding, speakerEmbedding }, axis: -1);
    }

    /// <summary>
    /// Estimates speech duration based on text length and speaking rate.
    /// </summary>
    protected virtual double EstimateSpeechDuration(string text, double speakingRate)
    {
        // Rough estimate: ~150 words per minute at normal speed
        var wordCount = text.Split(new[] { ' ', '\t', '\n' }, StringSplitOptions.RemoveEmptyEntries).Length;
        var wordsPerSecond = 150.0 / 60.0 * speakingRate;
        return Math.Max(1.0, wordCount / wordsPerSecond);
    }

    /// <summary>
    /// Extracts context from the end of latent representation for continuation.
    /// </summary>
    protected virtual Tensor<T> ExtractLatentContext(Tensor<T> latents)
    {
        var shape = latents.Shape;
        var contextFrames = Math.Min(shape[^1], 16); // Use last 16 frames as context

        var contextShape = new int[shape.Length];
        Array.Copy(shape, contextShape, shape.Length);
        contextShape[^1] = contextFrames;

        var context = new Tensor<T>(contextShape);
        var contextSpan = context.AsWritableSpan();
        var latentSpan = latents.AsSpan();

        var startFrame = shape[^1] - contextFrames;
        var framesPerBatch = shape[1] * shape[2] * shape[3];

        for (int b = 0; b < shape[0]; b++)
        {
            for (int c = 0; c < shape[1]; c++)
            {
                for (int h = 0; h < shape[2]; h++)
                {
                    for (int f = 0; f < contextFrames; f++)
                    {
                        var srcIdx = b * framesPerBatch + c * shape[2] * shape[3] + h * shape[3] + startFrame + f;
                        var dstIdx = b * (shape[1] * shape[2] * contextFrames) + c * shape[2] * contextFrames + h * contextFrames + f;
                        contextSpan[dstIdx] = latentSpan[srcIdx];
                    }
                }
            }
        }

        return context;
    }

    /// <summary>
    /// Predicts noise with context from previous audio.
    /// </summary>
    protected virtual Tensor<T> PredictNoiseWithContext(
        Tensor<T> latents,
        int timestep,
        Tensor<T> context,
        Tensor<T>? promptEmbedding)
    {
        // Default: use noise predictor directly
        // Derived classes may implement cross-attention to context
        return NoisePredictor.PredictNoise(latents, timestep, promptEmbedding);
    }

    /// <summary>
    /// Concatenates two latent tensors along the time dimension.
    /// </summary>
    protected virtual Tensor<T> ConcatenateLatents(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorConcatenate<T>(new[] { a, b }, axis: -1);
    }

    /// <summary>
    /// Concatenates two audio waveforms.
    /// </summary>
    protected virtual Tensor<T> ConcatenateAudio(Tensor<T> a, Tensor<T> b)
    {
        return Engine.TensorConcatenate<T>(new[] { a, b }, axis: -1);
    }

    #endregion
}
