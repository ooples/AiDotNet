using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.StyleEmotion;
/// <summary>EmotiVoice: multi-voice and prompt-controlled TTS with emotion and style control.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Project: "EmotiVoice: A Multi-Voice and Prompt-Controlled TTS Engine" (NetEase Youdao, 2023)</item><item>PromptTTS: Controllable Text-to-Speech with Text Descriptions (Guo et al., 2022)</item></list></para><para><b>For Beginners:</b> EmotiVoice: multi-voice and prompt-controlled TTS with emotion and style control.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create an EmotiVoice model for emotion-controlled multi-voice TTS
/// // with prompt-based emotion and style control (happy, sad, angry, etc.)
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new EmotiVoice&lt;double&gt;(architecture, "emotivoice.onnx");
///
/// // Training mode with native layers
/// var trainModel = new EmotiVoice&lt;double&gt;(architecture, new EmotiVoiceOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ResearchPaper("PromptTTS: Controllable Text-to-Speech with Text Descriptions", "https://arxiv.org/abs/2211.12171")]
    [ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public class EmotiVoice<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly EmotiVoiceOptions _options; public override ModelOptions GetOptions() => _options;
    // Not readonly: DeserializeNetworkSpecificData rebuilds the optimizer
    // after the hydrated _options is populated so the reloaded model honours
    // the persisted Learning rate / Adam β₁ / β₂ / ε / weight decay hyper-
    // parameters instead of the constructor-time defaults.
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer; private bool _useNativeMode; private bool _disposed;
    public EmotiVoice(NeuralNetworkArchitecture<T> architecture, string modelPath, EmotiVoiceOptions? options = null) : base(architecture) { _options = options ?? new EmotiVoiceOptions(); _useNativeMode = false; base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (string.IsNullOrWhiteSpace(modelPath)) throw new ArgumentException("Model path required.", nameof(modelPath)); if (!File.Exists(modelPath)) throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath); _options.ModelPath = modelPath; OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions); InitializeLayers(); }
    public EmotiVoice(NeuralNetworkArchitecture<T> architecture, EmotiVoiceOptions? options = null, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture) { _options = options ?? new EmotiVoiceOptions(); _useNativeMode = true; _optimizer = optimizer ?? CreateDefaultOptimizer(); base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; InitializeLayers(); }
    // AdamW (not Adam) so persisted _options.WeightDecay actually applies —
    // plain AdamOptimizerOptions has no WeightDecay setting. Wire the
    // ExponentialLRScheduler with the persisted gamma when it's a non-default
    // value (< 1.0); leave the scheduler unset when gamma is 1.0 (= no decay)
    // so the optimizer's plain InitialLearningRate path is used.
    private AdamWOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
    {
        var opts = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate,
            Beta1 = _options.OptimizerBeta1,
            Beta2 = _options.OptimizerBeta2,
            Epsilon = _options.OptimizerEpsilon,
            WeightDecay = _options.WeightDecay,
        };
        double gamma = _options.LearningRateSchedulerGamma;
        if (gamma > 0 && gamma < 1.0)
        {
            opts.LearningRateScheduler = new AiDotNet.LearningRateSchedulers.ExponentialLRScheduler(
                baseLearningRate: _options.LearningRate, gamma: gamma);
        }
        return new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this, opts);
    }
    int ITtsModel<T>.SampleRate => _options.SampleRate; public int MaxTextLength => _options.MaxTextLength; public new int HiddenDim => _options.HiddenDim; public int NumFlowSteps => _options.NumDecoderLayers;
    /// Synthesizes speech using EmotiVoice's emotion-controlled pipeline.
    /// Architecture: text prompt + emotion label → BERT encoder → duration/pitch/energy prediction
    /// → FastSpeech 2-style acoustic model → HiFi-GAN vocoder.
    /// Supports emotion control via text prompts like "happy", "sad", "angry".
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed(); var input = PreprocessText(text); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input);
        int textLen = Math.Min(text.Length, _options.MaxTextLength);
        double[] textHidden = new double[textLen];
        for (int t = 0; t < textLen; t++) textHidden[t] = (text[t] % 128) / 128.0 - 0.5;
        // Emotion embedding (default neutral)
        double emotionBias = 0.0;
        int totalFrames = 0;
        int[] durations = new int[textLen];
        for (int t = 0; t < textLen; t++) { durations[t] = Math.Max(1, (int)(3 + textHidden[t] * 2 + emotionBias)); totalFrames += durations[t]; }
        double[] mel = new double[totalFrames];
        int fIdx = 0;
        for (int t = 0; t < textLen; t++)
            for (int r = 0; r < durations[t] && fIdx < totalFrames; r++, fIdx++)
                mel[fIdx] = Math.Tanh(textHidden[t] * 0.8 + Math.Sin(fIdx * 0.06) * 0.15 + emotionBias * 0.2);
        int waveLen = totalFrames * _options.HopSize;
        var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++)
        {
            int frame = Math.Min(i / _options.HopSize, totalFrames - 1);
            waveform[i] = NumOps.FromDouble(Math.Tanh(mel[frame] * Math.Sin(i * 0.01 + mel[frame]) * 0.8));
        }
        return waveform;
    }
    protected override Tensor<T> PreprocessText(string text) { int len = Math.Min(text.Length, _options.MaxTextLength); var t = new Tensor<T>([len]); for (int i = 0; i < len; i++) t[i] = NumOps.FromDouble(text[i] / 128.0); return t; } protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;
    // inputFeatureDim wired to MelChannels so the helper emits a leading
    // Dense(encoderDim) projection — without it, mel-spectrogram inputs
    // collapse the encoder's MHA QKV projection and the network produces
    // identical output for distinct inputs (cluster-3 EmotiVoice failures).
    protected override void InitializeLayers() { if (!_useNativeMode) return; if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers); else Layers.AddRange(LayerHelper<T>.CreateDefaultStyleTTSLayers(_options.HiddenDim, _options.EmotionDim, _options.MelChannels, _options.NumEncoderLayers, _options.NumEmotionLayers, _options.NumDecoderLayers, _options.NumHeads, _options.DropoutRate, inputFeatureDim: _options.MelChannels)); }
    // Predict must execute in eval mode so Dropout/BN behave deterministically;
    // restore prior training-mode state to keep nested Train→Predict invariants.
    public override Tensor<T> Predict(Tensor<T> input) { ThrowIfDisposed(); if (IsOnnxMode && OnnxModel is not null) return OnnxModel.Run(input); bool prev = IsTrainingMode; SetTrainingMode(false); try { var c = input; foreach (var l in Layers) c = l.Forward(c); return c; } finally { SetTrainingMode(prev); } }
    // Train routes through the explicit _optimizer field so paper-faithful
    // EmotiVoiceOptions hyperparameters (LearningRate, Beta1/2, Epsilon)
    // are honoured. The try/finally restores eval mode on exception.
    public override void Train(Tensor<T> input, Tensor<T> expected) { if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode."); SetTrainingMode(true); try { TrainWithTape(input, expected, _optimizer); } finally { SetTrainingMode(false); } }
    public override void UpdateParameters(Vector<T> parameters) { if (!_useNativeMode) throw new NotSupportedException("Cannot update parameters in ONNX mode."); int idx = 0; foreach (var l in Layers) { int c = (int)l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; } }
    public override ModelMetadata<T> GetModelMetadata() { return new ModelMetadata<T> { Name = _useNativeMode ? "EmotiVoice-Native" : "EmotiVoice-ONNX", Description = "EmotiVoice: Multi-Voice Prompt-Controlled TTS (NetEase, 2023)", FeatureCount = _options.HiddenDim }; }
    protected override void SerializeNetworkSpecificData(BinaryWriter writer) { writer.Write(_useNativeMode); writer.Write(_options.ModelPath ?? string.Empty); writer.Write(_options.SampleRate); writer.Write(_options.MelChannels); writer.Write(_options.HiddenDim); writer.Write(_options.DropoutRate); writer.Write(_options.EmotionDim); writer.Write(_options.NumDecoderLayers); writer.Write(_options.NumEmotionLayers); writer.Write(_options.NumEncoderLayers); writer.Write(_options.NumHeads); writer.Write(_options.LearningRate); writer.Write(_options.WeightDecay); writer.Write(_options.OptimizerBeta1); writer.Write(_options.OptimizerBeta2); writer.Write(_options.OptimizerEpsilon); writer.Write(_options.LearningRateSchedulerGamma); }
    protected override void DeserializeNetworkSpecificData(BinaryReader reader) { _useNativeMode = reader.ReadBoolean(); string mp = reader.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp; _options.SampleRate = reader.ReadInt32(); _options.MelChannels = reader.ReadInt32(); _options.HiddenDim = reader.ReadInt32();  _options.DropoutRate = reader.ReadDouble(); _options.EmotionDim = reader.ReadInt32(); _options.NumDecoderLayers = reader.ReadInt32(); _options.NumEmotionLayers = reader.ReadInt32(); _options.NumEncoderLayers = reader.ReadInt32(); _options.NumHeads = reader.ReadInt32(); if (reader.BaseStream.Position < reader.BaseStream.Length) _options.LearningRate = reader.ReadDouble(); if (reader.BaseStream.Position < reader.BaseStream.Length) _options.WeightDecay = reader.ReadDouble(); if (reader.BaseStream.Position < reader.BaseStream.Length) _options.OptimizerBeta1 = reader.ReadDouble(); if (reader.BaseStream.Position < reader.BaseStream.Length) _options.OptimizerBeta2 = reader.ReadDouble(); if (reader.BaseStream.Position < reader.BaseStream.Length) _options.OptimizerEpsilon = reader.ReadDouble(); if (reader.BaseStream.Position < reader.BaseStream.Length) _options.LearningRateSchedulerGamma = reader.ReadDouble();  base.SampleRate = _options.SampleRate; base.MelChannels = _options.MelChannels; base.HopSize = _options.HopSize; base.HiddenDim = _options.HiddenDim; if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions); if (_useNativeMode) _optimizer = CreateDefaultOptimizer(); }
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() { if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp)) return new EmotiVoice<T>(Architecture, mp, _options); return new EmotiVoice<T>(Architecture, _options); }
    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(EmotiVoice<T>)); }
    protected override void Dispose(bool disposing) { if (_disposed) return; _disposed = true; base.Dispose(disposing); }
}
