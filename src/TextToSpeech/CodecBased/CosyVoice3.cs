using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>
/// CosyVoice 3: Fun-CosyVoice 3 zero-shot multilingual TTS with enhanced semantic tokens.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "Fun-CosyVoice 3: Zero-Shot Multilingual TTS" (Alibaba DAMO, 2025)</item></list></para>
/// <para><b>For Beginners:</b> CosyVoice 3 is an advanced version of CosyVoice that supports
/// zero-shot multilingual speech synthesis. It uses enhanced supervised semantic tokens and a
/// multi-scale flow matching decoder to produce natural-sounding speech in multiple languages
/// without needing training data for each target speaker.</para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
public class CosyVoice3<T> : TtsModelBase<T>, ICodecTts<T>
{
    private readonly CosyVoice3Options _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Initializes a new instance of the <see cref="CosyVoice3{T}"/> class in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional model configuration options.</param>
    public CosyVoice3(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        CosyVoice3Options? options = null) : base(architecture)
    {
        _options = options ?? new CosyVoice3Options();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.LLMDim;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path required.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="CosyVoice3{T}"/> class in native training/inference mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="options">Optional model configuration options.</param>
    /// <param name="optimizer">Optional gradient-based optimizer for training.</param>
    public CosyVoice3(
        NeuralNetworkArchitecture<T> architecture,
        CosyVoice3Options? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null) : base(architecture)
    {
        _options = options ?? new CosyVoice3Options();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.LLMDim;
        InitializeLayers();
    }

    int ITtsModel<T>.SampleRate => _options.SampleRate;
    public int MaxTextLength => _options.MaxTextLength;
    public int NumCodebooks => _options.NumCodebooks;
    public int CodebookSize => _options.CodebookSize;
    public int CodecFrameRate => _options.CodecFrameRate;

    /// <summary>
    /// Synthesizes speech from text using CosyVoice 3's enhanced pipeline.
    /// </summary>
    /// <param name="text">The input text to synthesize.</param>
    /// <returns>A tensor containing the generated audio representation.</returns>
    /// <remarks>
    /// <para>Per Alibaba (2025): Uses enhanced supervised semantic tokens and a multi-scale
    /// flow matching decoder for improved zero-shot multilingual synthesis.</para>
    /// <para><b>For Beginners:</b> This method takes text and generates speech audio. The "zero-shot"
    /// capability means it can mimic a speaker's voice from just a short audio sample,
    /// without needing to be specifically trained on that voice.</para>
    /// </remarks>
    public Tensor<T> Synthesize(string text)
    {
        ThrowIfDisposed();
        var input = PreprocessText(text);
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        var output = Predict(input);
        return PostprocessAudio(output);
    }

    /// <summary>
    /// Encodes audio into discrete codec tokens.
    /// </summary>
    /// <param name="audio">The input audio tensor.</param>
    /// <returns>A tensor of discrete codec tokens.</returns>
    public Tensor<T> EncodeToTokens(Tensor<T> audio)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(audio);
        return Predict(audio);
    }

    /// <summary>
    /// Decodes discrete codec tokens back into audio.
    /// </summary>
    /// <param name="tokens">The codec tokens to decode.</param>
    /// <returns>A tensor containing the reconstructed audio.</returns>
    public Tensor<T> DecodeFromTokens(Tensor<T> tokens)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(tokens);
        return Predict(tokens);
    }

    /// <inheritdoc />
    protected override Tensor<T> PreprocessText(string text)
    {
        int len = Math.Min(text.Length, _options.MaxTextLength);
        var t = new Tensor<T>([len]);
        for (int i = 0; i < len; i++)
            t[i] = NumOps.FromDouble(text[i] / 128.0);
        return t;
    }

    /// <inheritdoc />
    protected override Tensor<T> PostprocessAudio(Tensor<T> output) => output;

    /// <inheritdoc />
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
            Layers.AddRange(Architecture.Layers);
        else
            Layers.AddRange(LayerHelper<T>.CreateDefaultCodecLMLayers(
                _options.TextEncoderDim, _options.LLMDim,
                _options.NumCodebooks * _options.CodebookSize,
                _options.NumEncoderLayers, _options.NumLLMLayers,
                _options.NumHeads, _options.DropoutRate));
    }

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxModel is not null)
            return OnnxModel.Run(input);
        var c = input;
        foreach (var l in Layers)
            c = l.Forward(c);
        return c;
    }

    /// <inheritdoc />
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            var o = Predict(input);
            var g = LossFunction.CalculateDerivative(o.ToVector(), expected.ToVector());
            var gt = Tensor<T>.FromVector(g);
            for (int i = Layers.Count - 1; i >= 0; i--)
                gt = Layers[i].Backward(gt);
            _optimizer?.UpdateParameters(Layers);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("Cannot update parameters in ONNX mode.");
        int idx = 0;
        foreach (var l in Layers)
        {
            int c = l.ParameterCount;
            l.UpdateParameters(parameters.Slice(idx, c));
            idx += c;
        }
    }

    /// <inheritdoc />
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = _useNativeMode ? "CosyVoice3-Native" : "CosyVoice3-ONNX",
            Description = "CosyVoice 3: Zero-Shot Multilingual TTS (Alibaba DAMO, 2025)",
            FeatureCount = _options.LLMDim
        };
    }

    /// <inheritdoc />
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(_options.ModelPath ?? string.Empty);
        writer.Write(_options.SampleRate);
        writer.Write(_options.CodebookSize);
        writer.Write(_options.DropoutRate);
        writer.Write(_options.LLMDim);
        writer.Write(_options.NumCodebooks);
        writer.Write(_options.NumEncoderLayers);
        writer.Write(_options.NumHeads);
        writer.Write(_options.NumLLMLayers);
        writer.Write(_options.TextEncoderDim);
    }

    /// <inheritdoc />
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        string mp = reader.ReadString();
        if (!string.IsNullOrEmpty(mp))
            _options.ModelPath = mp;
        _options.SampleRate = reader.ReadInt32();
        _options.CodebookSize = reader.ReadInt32();
        _options.DropoutRate = reader.ReadDouble();
        _options.LLMDim = reader.ReadInt32();
        _options.NumCodebooks = reader.ReadInt32();
        _options.NumEncoderLayers = reader.ReadInt32();
        _options.NumHeads = reader.ReadInt32();
        _options.NumLLMLayers = reader.ReadInt32();
        _options.TextEncoderDim = reader.ReadInt32();
        base.SampleRate = _options.SampleRate;
        base.MelChannels = _options.MelChannels;
        base.HopSize = _options.HopSize;
        base.HiddenDim = _options.LLMDim;
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    /// <inheritdoc />
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } mp && !string.IsNullOrEmpty(mp))
            return new CosyVoice3<T>(Architecture, mp, _options);
        return new CosyVoice3<T>(Architecture, _options);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(CosyVoice3<T>));
    }

    /// <inheritdoc />
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
