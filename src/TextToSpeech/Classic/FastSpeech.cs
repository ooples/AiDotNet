using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// FastSpeech: non-autoregressive TTS with knowledge-distilled duration predictor for parallel generation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "FastSpeech: Fast, Robust and Controllable Text to Speech" (Ren et al., 2019)</item></list></para>
/// <para><b>For Beginners:</b> /// FastSpeech: non-autoregressive TTS with knowledge-distilled duration predictor for parallel generation.
///. This model converts text input into speech audio output.</para>
/// <example>
/// <code>
/// // Create a FastSpeech model for non-autoregressive parallel speech synthesis
/// // with knowledge-distilled duration predictor for fast generation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputSize: 200, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new FastSpeech&lt;double&gt;(architecture, "fastspeech.onnx");
///
/// // Training mode with native layers
/// var trainModel = new FastSpeech&lt;double&gt;(architecture, new FastSpeechOptions());
/// </code>
/// </example>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "FastSpeech: Fast, Robust and Controllable Text to Speech",
    "https://arxiv.org/abs/1905.09263",
    Year = 2019,
    Authors = "Ren et al."
)]
[PaperOptimizer(OptimizerKind.Adam, Beta1 = 0.9, Beta2 = 0.98, Epsilon = 1e-9,
                Schedule = LearningRateSchedulerType.Noam, WarmupSteps = 4000,
                ReferenceBatchSize = 64,
                Provenance = RecipeProvenance.DerivedFromCitedWork,
                Source = "Ren et al. 2019, Sec. 4.3: Adam with beta1 0.9, beta2 0.98 and epsilon 1e-9, "
                        + "following the learning rate schedule of its reference [25], Vaswani et al. "
                        + "2017 -- the inverse-square-root schedule, whose warmup is 4000 steps (Vaswani "
                        + "Sec. 5.3). The paper restates neither the peak rate nor the warmup length, so "
                        + "the provenance records that both come from the cited work. Batch 64 (Table 5).")]
public partial class FastSpeech<T> : VarianceAdaptorTtsModelBase<T>, IAcousticModel<T>
{
    private readonly FastSpeechOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;

    public FastSpeech(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        FastSpeechOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new FastSpeechOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    public FastSpeech(
        NeuralNetworkArchitecture<T> architecture,
        FastSpeechOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new FastSpeechOptions();
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

    /// <summary>Generates a mel spectrogram from text (FastSpeech is an acoustic model; pair it with a vocoder).</summary>
    public Tensor<T> TextToMel(string text) => Synthesize(text);

    /// <inheritdoc />
    /// <remarks>The paper states only the duration loss; the mel spectrogram is scored with squared error, as in its
    /// reference implementations (FastSpeech 2 is where mean absolute error is specified).</remarks>
    protected override bool UsesL1MelLoss => false;

    /// <inheritdoc />
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <summary>
    /// Builds the paper's layer stack (Ren et al. 2019, §3, §4.2): phoneme embedding and positions, 6 FFT blocks, the
    /// duration predictor with the length regulator, positions, 6 FFT blocks, and the linear projection to mel.
    /// </summary>
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
            // FastSpeech's length regulator is FastSpeech 2's variance adaptor without the pitch and energy branches.
            LayerHelper<T>.CreateDefaultFastSpeech2DecoderLayers(
                _options.HiddenDim, _options.MelChannels, _options.NumDecoderLayers, _options.NumHeads,
                _options.FftFilterSize, _options.FftKernelSizes[0], _options.FftKernelSizes[1], _options.DropoutRate,
                _options.MaxMelLength, _options.DurationPredictorFilterSize, _options.DurationPredictorKernelSize,
                _options.DurationPredictorDropout, usePitch: false, useEnergy: false));
        VarianceAdaptor.DurationScale = _options.DurationScale;
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "FastSpeech-Native" : "FastSpeech-ONNX",
            Description =
                "FastSpeech: Fast, Robust and Controllable Text to Speech (Ren et al., 2019)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "FastSpeech";
        return m;
    }
}
