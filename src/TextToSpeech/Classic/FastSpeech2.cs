using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Audio.Pitch;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// FastSpeech 2: non-autoregressive TTS with a variance adaptor for duration, pitch and energy.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b> "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech" (Ren et al., 2021).</para>
/// <para>
/// The layer stack is the paper's (§2.2, App. A): phoneme embedding and sinusoidal positions, 4 feed-forward
/// Transformer blocks (hidden 256, 2 heads, 1D convolutions of kernel 9 and 1 with 1024 filters), the variance
/// adaptor (duration, pitch as a CWT spectrogram, energy; <see cref="VarianceAdaptorLayer{T}"/>), positions again,
/// 4 more FFT blocks and a linear projection to 80 mel channels. The model is trained on the sum of the mel MAE and
/// the MSE of each variance predictor, with ground-truth duration, pitch and energy driving the adaptor (§2.3).
/// </para>
/// <para>
/// Training data: durations come from an external forced aligner (MFA in the paper), so a plain
/// <c>Train(tokens, mel)</c> cannot train this model faithfully and throws; pass a <see cref="TtsTrainingSample{T}"/>
/// with <see cref="TtsTrainingSample{T}.Durations"/>. Pitch (WORLD DIO + StoneMask), energy (STFT frame norm)
/// and the mel target are derived from <see cref="TtsTrainingSample{T}.Audio"/> when not supplied, as the paper
/// derives them.
/// </para>
/// <para><b>For Beginners:</b> FastSpeech 2 reads phonemes, decides how long each one lasts and how high and loud
/// it is, stretches the phoneme features to that many frames, and turns them into a mel spectrogram all at once.
/// A vocoder (e.g. HiFi-GAN) turns the spectrogram into audio.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech",
    "https://arxiv.org/abs/2006.04558",
    Year = 2020,
    Authors = "Ren et al."
)]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-9,
                Schedule = LearningRateSchedulerType.Noam, WarmupSteps = 4000,
                ReferenceBatchSize = 48,
                Source = "Ren et al. 2021, Sec. 4.1: Adam with beta1 0.9, beta2 0.98, eps 1e-9 following the learning rate schedule of Vaswani et al. 2017, batch size 48 sentences, 160k steps. No constant rate is declared because that schedule states none.")]
public partial class FastSpeech2<T> : VarianceAdaptorTtsModelBase<T>, IAcousticModel<T>
{
    private readonly FastSpeech2Options _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;

    public FastSpeech2(NeuralNetworkArchitecture<T> architecture, string modelPath, FastSpeech2Options? options = null)
        : base(architecture)
    {
        _options = options ?? new FastSpeech2Options();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    public FastSpeech2(
        NeuralNetworkArchitecture<T> architecture,
        FastSpeech2Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new FastSpeech2Options();
        _useNativeMode = true;
        _optimizer = optimizer
            ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
            ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int MelChannels => _options.MelChannels;
    public new int HopSize => _options.HopSize;
    public int FftSize => _options.FftSize;

    /// <summary>Generates a mel spectrogram from text (FastSpeech 2 is an acoustic model; pair it with a vocoder).</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    /// <inheritdoc />
    /// <remarks>FastSpeech 2 optimizes the mel spectrogram with mean absolute error (Ren et al. 2021, §3.1).</remarks>
    protected override bool UsesL1MelLoss => true;

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            AddUnsplitLayers(Architecture.Layers);
            return;
        }

        AddEncoderDecoderLayers(
            LayerHelper<T>.CreateDefaultFastSpeechEncoderLayers(
                _options.VocabSize, _options.EncoderDim, _options.HiddenDim, _options.NumEncoderLayers, _options.NumHeads,
                _options.FftFilterSize, _options.FftKernelSizes[0], _options.FftKernelSizes[1], _options.DropoutRate,
                _options.MaxTextLength),
            LayerHelper<T>.CreateDefaultFastSpeech2DecoderLayers(
                _options.HiddenDim, _options.MelChannels, _options.NumDecoderLayers, _options.NumHeads,
                _options.FftFilterSize, _options.FftKernelSizes[0], _options.FftKernelSizes[1], _options.DropoutRate,
                _options.MaxMelLength, _options.VariancePredictorFilterSize, _options.VariancePredictorKernelSize,
                _options.VariancePredictorDropout, _options.NumPitchBins, _options.PitchMinHz, _options.PitchMaxHz,
                _options.NumEnergyBins, _options.FftSize, _options.UsePitchPredictor, _options.UseEnergyPredictor));
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc/>
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "FastSpeech2-Native" : "FastSpeech2-ONNX",
            Description =
                "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (Ren et al., 2020)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "FastSpeech2";
        m.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        m.AdditionalInfo["MelChannels"] = _options.MelChannels.ToString();
        return m;
    }
}
