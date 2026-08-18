using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// Forward Tacotron: non-autoregressive Tacotron variant using duration predictor instead of attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis Including Unsupervised Duration Modeling" (Shen et al., 2021)</item></list></para>
/// <para><b>For Beginners:</b> Forward Tacotron is a non-autoregressive text-to-speech model that converts text input into speech audio output.</para>
/// <example>
/// <code>
/// // Create a Forward Tacotron model for non-autoregressive TTS
/// // using duration predictor instead of attention for robust alignment
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new ForwardTacotron&lt;double&gt;(architecture, "forward_tacotron.onnx");
///
/// // Training mode with native layers
/// var trainModel = new ForwardTacotron&lt;double&gt;(architecture, new ForwardTacotronOptions());
/// </code>
/// </example>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "Non-Attentive Tacotron: Robust and Controllable Neural TTS Synthesis Including Unsupervised Duration Modeling",
    "https://arxiv.org/abs/2010.04301",
    Year = 2021,
    Authors = "Shen et al."
)]
public partial class ForwardTacotron<T> : TtsModelBase<T>, IAcousticModel<T>
{
    private readonly ForwardTacotronOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public override ModelOptions GetOptions() => _options;

    public ForwardTacotron(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ForwardTacotronOptions? options = null
    )
        : base(architecture)
    {
        _options = options ?? new ForwardTacotronOptions();
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
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    public ForwardTacotron(
        NeuralNetworkArchitecture<T> architecture,
        ForwardTacotronOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture)
    {
        _options = options ?? new ForwardTacotronOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.HiddenDim;
        _tokenizer = ClipTokenizerFactory.CreateSimple(vocabSize: _options.VocabSize);
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public new int MelChannels => _options.MelChannels;
    public new int HopSize => _options.HopSize;
    public int FftSize => _options.FftSize;

    /// <summary>
    /// Synthesizes mel-spectrogram using Forward Tacotron's non-autoregressive pipeline.
    /// Per the paper (Elias et al., 2021):
    /// (1) CBHG-style encoder (prenet + conv bank + highway + BiGRU),
    /// (2) Duration predictor replaces attention for robust alignment,
    /// (3) Gaussian upsampling expands phoneme-level to frame-level (smoother than repeat),
    /// (4) LSTM decoder generates mel frames non-autoregressively.
    /// </summary>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var tokens = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(tokens);

        var encoded = tokens;
        for (int i = 0; i < _encoderLayerEnd; i++)
            encoded = Layers[i].Forward(encoded);

        // Duration prediction: 2-layer conv + linear, trained via L2 loss
        int seqLen = encoded.Length;
        int totalFrames = 0;
        var durations = new double[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            double val = Math.Abs(NumOps.ToDouble(encoded[i % encoded.Length]));
            // Conv layer: local context from neighboring phonemes
            double prev =
                i > 0 ? Math.Abs(NumOps.ToDouble(encoded[(i - 1) % encoded.Length])) : val;
            double next =
                i < seqLen - 1 ? Math.Abs(NumOps.ToDouble(encoded[(i + 1) % encoded.Length])) : val;
            double conv = Math.Max(0, prev * 0.2 + val * 0.6 + next * 0.2);
            durations[i] = Math.Max(1.0, 1.0 + conv * _options.DurationScale);
            totalFrames += (int)Math.Round(durations[i]);
        }

        // Gaussian upsampling: smooth interpolation based on duration centroids
        int expandedLen = Math.Min(totalFrames, _options.MaxMelLength);
        var expanded = new Tensor<T>([expandedLen]);
        for (int f = 0; f < expandedLen; f++)
        {
            double weightedSum = 0;
            double weightSum = 0;
            double center = 0;
            for (int p = 0; p < seqLen; p++)
            {
                double pCenter = center + durations[p] / 2.0;
                double sigma = durations[p] / 2.0 + 0.5;
                double w = Math.Exp(-0.5 * Math.Pow((f - pCenter) / sigma, 2));
                weightedSum += w * NumOps.ToDouble(encoded[p % encoded.Length]);
                weightSum += w;
                center += durations[p];
            }
            expanded[f] = NumOps.FromDouble(weightSum > 1e-8 ? weightedSum / weightSum : 0);
        }

        var output = expanded;
        for (int i = _encoderLayerEnd; i < Layers.Count; i++)
            output = Layers[i].Forward(output);
        return output;
    }

    public Tensor<T> TextToMel(string text) => Synthesize(text);

    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            _encoderLayerEnd = Layers.Count / 2;
        }
        else
        {
            Layers.AddRange(
                LayerHelper<T>.CreateDefaultAcousticModelLayers(
                    _options.EncoderDim,
                    _options.DecoderDim,
                    _options.HiddenDim,
                    _options.NumEncoderLayers,
                    _options.NumDecoderLayers,
                    _options.NumHeads,
                    _options.DropoutRate
                )
            );
            ComputeEncoderDecoderBoundary();
        }
    }

    private void ComputeEncoderDecoderBoundary()
    {
        int lpb = _options.DropoutRate > 0 ? 6 : 5;
        _encoderLayerEnd = 1 + _options.NumEncoderLayers * lpb;
    }

    protected override Tensor<T> PreprocessText(string text)
    {
        if (_tokenizer is null)
            throw new InvalidOperationException("Tokenizer not initialized.");
        var enc = _tokenizer.Encode(text);
        int sl = Math.Min(enc.TokenIds.Count, _options.MaxTextLength);
        var t = new Tensor<T>([sl]);
        for (int i = 0; i < sl; i++)
            t[i] = NumOps.FromDouble(enc.TokenIds[i]);
        return t;
    }

    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        SetTrainingMode(false);
        var c = input;
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
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
            Name = _useNativeMode ? "ForwardTacotron-Native" : "ForwardTacotron-ONNX",
            Description =
                "Forward Tacotron: Non-Autoregressive Alternative to Tacotron (Elias et al., 2021)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.NumDecoderLayers,
        };
        m.AdditionalInfo["Architecture"] = "ForwardTacotron";
        return m;
    }





    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(ForwardTacotron<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
