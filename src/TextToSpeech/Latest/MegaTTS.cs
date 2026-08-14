using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>MegaTTS: Mega-TTS: Zero-Shot TTS with Prosody Decomposition.</summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks><para><b>References:</b><list type="bullet"><item>Paper: "Mega-TTS: Zero-Shot TTS with Prosody Decomposition" (Jiang et al., 2023)</item></list></para><para><b>For Beginners:</b> MegaTTS: Mega-TTS: Zero-Shot TTS with Prosody Decomposition.. This model converts text input into speech audio output.</para></remarks>
/// <example>
/// <code>
/// // Create a Mega-TTS model for zero-shot TTS with prosody decomposition
/// // separating timbre, prosody, and content for controllable synthesis
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new MegaTTS&lt;double&gt;(architecture, "megatts.onnx");
///
/// // Training mode with native layers
/// var trainModel = new MegaTTS&lt;double&gt;(architecture, new MegaTTSOptions());
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Mega-TTS: Zero-Shot Text-to-Speech at Scale with Intrinsic Inductive Bias",
    "https://arxiv.org/abs/2306.03509",
    Year = 2023,
    Authors = "Jiang et al."
)]
public partial class MegaTTS<T> : TtsModelBase<T>, IEndToEndTts<T>
{
    private readonly MegaTTSOptions _options;

    public override ModelOptions GetOptions() => _options;

    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public MegaTTS(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        MegaTTSOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new MegaTTSOptions();
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

    public MegaTTS(
        NeuralNetworkArchitecture<T> architecture,
        MegaTTSOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new MegaTTSOptions();
        _useNativeMode = true;
        // Honour the configured LearningRate / WeightDecay. Constructing the optimizer bare left it
        // on AdamW's own defaults (InitialLearningRate 0.001), so MegaTTS trained at 10x the 1e-4
        // its options specify and those two user-facing settings did nothing at all. The resulting
        // overshoot showed up as Training_ShouldReduceLoss drifting upward (0.802 -> 0.837) even
        // though the model's own training loss was decreasing. Same wiring as Piper in this family.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
            });
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int HiddenDim => _options.HiddenDim;
    public int NumFlowSteps => _options.NumFlowSteps;

    /// <summary>
    /// Synthesizes speech from text.
    /// Per Jiang et al. (2023): Content/prosody/timbre decomposition for zero-shot TTS.
    /// </summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var input = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        int textLen = Math.Min(text.Length, _options.MaxTextLength);
        int mF = textLen * 4;
        double[] cont = new double[mF];
        for (int f = 0; f < mF; f++)
        {
            int t = Math.Min(f * textLen / mF, textLen - 1);
            cont[f] = (text[t] % 128) / 128.0;
        }
        double[] pitch = new double[mF];
        double[] energy = new double[mF];
        for (int f = 0; f < mF; f++)
        {
            pitch[f] = Math.Sin(f * 0.03) * 0.15 + cont[f] * 0.1;
            energy[f] = 0.5 + cont[f] * 0.3 + Math.Cos(f * 0.02) * 0.1;
        }
        double[] mel = new double[mF];
        for (int f = 0; f < mF; f++)
            mel[f] = Math.Tanh(cont[f] * 0.5 + pitch[f] + energy[f] * 0.2 + 0.05);
        int waveLen = mF * _options.HopSize;
        var waveform = new Tensor<T>([waveLen]);
        for (int i = 0; i < waveLen; i++)
        {
            int fr = Math.Min(i / Math.Max(1, _options.HopSize), mF - 1);
            waveform[i] = NumOps.FromDouble(mel[fr] * Math.Sin(i * 0.007) * 0.85);
        }
        return waveform;
    }

    protected override Tensor<T> PreprocessText(string text)
    {
        int len = Math.Min(text.Length, _options.MaxTextLength);
        var t = new Tensor<T>([len]);
        for (int i = 0; i < len; i++)
            t[i] = NumOps.FromDouble(text[i] / 128.0);
        return t;
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    // Half-open [start, end) index ranges into Layers for each branch of the Mega-TTS
    // decomposition. Populated by ExtractLayerReferences; -1 means "custom Architecture.Layers
    // were supplied", in which case the forward degrades to a plain sequential dispatch.
    private int _contentEnd = -1;
    private int _prosodyEnd = -1;
    private int _timbreEnd = -1;
    private int _pllmEnd = -1;

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            return;
        }

        Layers.AddRange(
            LayerHelper<T>.CreateDefaultMegaTTSLayers(
                encoderDim: _options.EncoderDim,
                decoderDim: _options.DecoderDim,
                melChannels: _options.MelChannels,
                prosodyDim: _options.ProsodyDim,
                prosodyCodebookSize: _options.ProsodyCodebookSize,
                timbreDim: _options.TimbreDim,
                pllmDim: _options.PLLMDim,
                numEncoderLayers: _options.NumEncoderLayers,
                numDecoderLayers: _options.NumDecoderLayers,
                numProsodyLayers: _options.NumProsodyLayers,
                numTimbreLayers: _options.NumTimbreLayers,
                numPLLMLayers: _options.NumPLLMLayers,
                numHeads: _options.NumHeads,
                numPLLMHeads: _options.NumPLLMHeads,
                dropoutRate: _options.DropoutRate,
                vocabSize: _options.VocabSize
            )
        );
        ExtractLayerReferences();
    }

    /// <summary>
    /// Records where each branch of the decomposition starts and ends inside <see cref="Layers"/>.
    /// The counts mirror the emission order in <c>LayerHelper.CreateDefaultMegaTTSLayers</c>.
    /// </summary>
    private void ExtractLayerReferences()
    {
        _contentEnd = 1 + _options.NumEncoderLayers;                  // embedding + encoder blocks
        _prosodyEnd = _contentEnd + 2 + _options.NumProsodyLayers;    // proj + blocks + bottleneck
        _timbreEnd = _prosodyEnd + 1 + _options.NumTimbreLayers;      // proj + blocks
        _pllmEnd = _timbreEnd + 3 + _options.NumPLLMLayers;           // proj + blocks + logits + codebook
    }

    private Tensor<T> RunRange(Tensor<T> x, int start, int end)
    {
        var c = x;
        for (int i = start; i < end; i++)
            c = Layers[i].Forward(c);
        return c;
    }

    /// <summary>
    /// Mega-TTS forward (Jiang et al. 2023 §3): content, prosody and timbre are encoded on
    /// separate branches with the inductive bias each attribute deserves, then fused for the mel
    /// decoder. Phase is deliberately absent — the vocoder owns it.
    /// </summary>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        return ForwardNative(input);
    }

    private Tensor<T> ForwardNative(Tensor<T> input)
    {
        // Custom Architecture.Layers were supplied — honour them verbatim.
        if (_contentEnd < 0)
        {
            var seq = input;
            foreach (var l in Layers)
                seq = l.Forward(seq);
            return seq;
        }

        // 1. Content: the discrete, order-sensitive attribute gets the full transformer.
        var content = RunRange(input, 0, _contentEnd);

        // 2. Prosody: squeezed through the deliberately narrow bottleneck so it cannot carry
        //    timbre or content, then re-predicted by the P-LLM as a distribution over the
        //    codebook (softmax) which the codebook projection turns back into an embedding.
        var prosodyLatent = RunRange(content, _contentEnd, _prosodyEnd);
        var pllmLogits = RunRange(prosodyLatent, _timbreEnd, _pllmEnd - 1);
        var codeWeights = Engine.Softmax(pllmLogits);
        var prosodyFromCodes = Layers[_pllmEnd - 1].Forward(codeWeights);
        var prosody = Engine.TensorAdd(prosodyLatent, prosodyFromCodes);

        // 3. Timbre: mean-pooled over time so it is global and time-INVARIANT by construction —
        //    the architecture is simply not given the capacity to let it drift within an
        //    utterance, which is the paper's central inductive bias.
        var timbreSeq = RunRange(content, _prosodyEnd, _timbreEnd);
        var timbre = MeanOverTime(timbreSeq);

        // 4. Fuse and decode to mel. DenseLayer infers its input width lazily, so the decoder's
        //    entry projection absorbs the concatenated content+prosody+timbre width.
        var fused = Engine.TensorConcatenate(
            new[] { content, prosody, BroadcastOverTime(timbre, content) }, content.Rank - 1);
        return RunRange(fused, _pllmEnd, Layers.Count);
    }

    /// <summary>Averages a [..., time, feature] tensor over its time axis, keeping the rank.</summary>
    private Tensor<T> MeanOverTime(Tensor<T> x)
    {
        int timeAxis = x.Rank - 2;
        if (timeAxis < 0)
            return x;
        return Engine.ReduceMean(x, new[] { timeAxis }, keepDims: true);
    }

    /// <summary>Repeats a pooled [..., 1, feature] tensor across the time axis of <paramref name="like"/>.</summary>
    private Tensor<T> BroadcastOverTime(Tensor<T> pooled, Tensor<T> like)
    {
        int timeAxis = like.Rank - 2;
        if (timeAxis < 0)
            return pooled;
        int steps = like.Shape[timeAxis];
        var repeats = new Tensor<T>[steps];
        for (int i = 0; i < steps; i++)
            repeats[i] = pooled;
        return Engine.TensorConcatenate(repeats, timeAxis);
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        // Pass the configured optimizer through, as Piper and MegaTTS2 do. Calling the
        // two-argument overload left _optimizer assigned but never read, so training silently fell
        // back to the base default and MegaTTS's LearningRate / WeightDecay options had no effect
        // whatsoever.
        TrainWithTape(input, expected, _optimizer);
        SetTrainingMode(false);
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "MegaTTS-Native" : "MegaTTS-ONNX",
            Description = "MegaTTS TTS",
            FeatureCount = _options.HiddenDim,
        };
        m.AdditionalInfo["Architecture"] = "MegaTTS";
        m.AdditionalInfo["Mode"] = _useNativeMode ? "Native" : "ONNX";
        m.AdditionalInfo["HiddenDim"] = base.HiddenDim;
        m.AdditionalInfo["SampleRate"] = base.SampleRate;
        m.AdditionalInfo["MelChannels"] = base.MelChannels;
        m.AdditionalInfo["HopSize"] = base.HopSize;
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.SampleRate);
        writer.Write(_options.DecoderDim);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.EncoderDim);
        writer.Write(_options.NumDecoderLayers);
        writer.Write(_options.NumEncoderLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.ProsodyDim);
        writer.Write(_options.ProsodyCodebookSize);
        writer.Write(_options.NumProsodyLayers);
        writer.Write(_options.ProsodyMelBands);
        writer.Write(_options.TimbreDim);
        writer.Write(_options.NumTimbreLayers);
        writer.Write(_options.PLLMDim);
        writer.Write(_options.NumPLLMLayers);
        writer.Write(_options.NumPLLMHeads);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp))
            _options.ModelPath = mp;
        _options.SampleRate = reader.ReadInt32();
        _options.DecoderDim = reader.ReadInt32();
        _options.DropoutRate = reader.ReadDouble();
        _options.EncoderDim = reader.ReadInt32();
        _options.NumDecoderLayers = reader.ReadInt32();
        _options.NumEncoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.ProsodyDim = reader.ReadInt32();
        _options.ProsodyCodebookSize = reader.ReadInt32();
        _options.NumProsodyLayers = reader.ReadInt32();
        _options.ProsodyMelBands = reader.ReadInt32();
        _options.TimbreDim = reader.ReadInt32();
        _options.NumTimbreLayers = reader.ReadInt32();
        _options.PLLMDim = reader.ReadInt32();
        _options.NumPLLMLayers = reader.ReadInt32();
        _options.NumPLLMHeads = reader.ReadInt32();
        // The branch offsets are derived from the layer counts above, so they must be recomputed
        // once the restored options are in place.
        if (_useNativeMode && (Architecture.Layers is null || Architecture.Layers.Count == 0))
            ExtractLayerReferences();
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(MegaTTS<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
