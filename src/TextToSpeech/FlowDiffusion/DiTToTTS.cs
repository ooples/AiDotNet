using AiDotNet.LearningRateSchedulers;
using AiDotNet.Enums;
using System.Collections.Generic;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.FlowDiffusion;
/// <summary>DiTToTTS: DiTTo-TTS: Efficient and Scalable Zero-Shot TTS with Diffusion Transformer.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "DiTTo-TTS: Efficient and Scalable Zero-Shot TTS with Diffusion Transformer" (Lee et al., 2024)</item></list></para><para><b>For Beginners:</b> DiTToTTS: DiTTo-TTS: Efficient and Scalable Zero-Shot TTS with Diffusion Transformer.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create a DiTTo-TTS model for efficient zero-shot TTS
/// // using diffusion transformer architecture for scalable synthesis
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new DiTToTTS&lt;double&gt;(architecture, "ditto_tts.onnx");
///
/// // Training mode with native layers
/// var trainModel = new DiTToTTS&lt;double&gt;(architecture, new DiTToTTSOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("DiTTo-TTS: Diffusion Transformers for Scalable Text-to-Speech without Domain-Specific Factors", "https://arxiv.org/abs/2406.11427", Year = 2024, Authors = "Lee et al.")]
[PaperOptimizer(OptimizerKind.AdamW, LearningRate = 1e-4, Beta1 = 0.9, Beta2 = 0.999,
                WeightDecay = 0, WarmupSteps = 1000, MinLearningRate = 0,
                Schedule = LearningRateSchedulerType.CosineAnnealing,
                Source = "Lee et al. 2024, Sec. 4: the AdamW optimizer with a learning rate of 1e-4, "
                        + "betas (0.9, 0.999) and a weight decay of 0.0, under a cosine learning rate "
                        + "scheduler with a warmup of 1K steps. The zero weight decay is the paper value "
                        + "rather than an omission.")]
public partial class DiTToTTS<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly DiTToTTSOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public DiTToTTS(NeuralNetworkArchitecture<T> architecture, string modelPath, DiTToTTSOptions? options = null) : base(architecture) { _options = options ?? new DiTToTTSOptions(); Options = _options; _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayersCore(); }
    public DiTToTTS(NeuralNetworkArchitecture<T> architecture, DiTToTTSOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new DiTToTTSOptions(); Options = _options; _useNativeMode = true; _optimizer = optimizer
        ?? PaperOptimizerFactory.CreateFor<T, Tensor<T>, Tensor<T>>(this)
        ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; InitializeLayersCore(); }
    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public new int HiddenDim => _options.HiddenDim; public int NumFlowSteps => _options.NumFlowSteps;
    /// <summary>
    /// Synthesizes speech from text.
    /// Per Lee et al. (2024): DiT blocks with adaptive layer norm for iterative denoising.
    /// </summary>
    public override Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed(); var input = PreprocessText(text); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        // The synthesized audio is the network's own output. It used to run the layers, discard the result and return a
        // waveform built from sine waves and the character codes, so the trained model never reached the caller.
        return PostprocessAudio(Predict(input));
    }
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    protected override void InitializeLayers() => InitializeLayersCore();

    // Constructors initialize this model's layers without dispatching into an unfinished derived instance.
    private void InitializeLayersCore() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultFlowMatchingTTSLayers(_options.EncoderDim, _options.FlowDim, _options.DecoderDim, _options.NumEncoderLayers, _options.NumFlowLayers, _options.NumHeads, _options.DropoutRate, inputFeatures: _options.MelChannels, vocabSize: _options.VocabSize)); }
    protected override Tensor<T> PredictCore(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); SetTrainingMode(false); var c = input; foreach (var l in Layers) c = l.Forward(c); return c; }
    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "DiTToTTS-Native" : "DiTToTTS-ONNX", Description = "DiTToTTS TTS", FeatureCount = _options.HiddenDim, AdditionalInfo = new Dictionary<string, object> { ["ModelType"] = "DiTToTTS", ["Mode"] = _useNativeMode ? "Native" : "ONNX", ["HiddenDim"] = _options.HiddenDim, ["EncoderDim"] = _options.EncoderDim, ["FlowDim"] = _options.FlowDim, ["DecoderDim"] = _options.DecoderDim, ["NumEncoderLayers"] = _options.NumEncoderLayers, ["NumFlowLayers"] = _options.NumFlowLayers, ["NumHeads"] = _options.NumHeads, ["SampleRate"] = _options.SampleRate, ["MelChannels"] = _options.MelChannels } }; }


    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(DiTToTTS<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
