using AiDotNet.Attributes;
using AiDotNet.Diffusion.SuperResolution;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Video.Options;

namespace AiDotNet.Video.Enhancement;

/// <summary>
/// StableVideoSR: stable diffusion with temporal conditioning for video super-resolution.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// StableVideoSR (2024) adapts the Stable Diffusion architecture for video SR:
/// - Temporal conditioning modules: cross-attention layers inserted between spatial attention
///   in the U-Net attend to features from adjacent frames, maintaining temporal coherence
/// - Direct low-resolution conditioning: noised RGB frames are concatenated with the latent
///   input, matching the released seven-channel U-Net contract
/// - Classifier-free guidance: balances between faithful reconstruction and generative
///   enhancement during inference
/// - Noise schedule: adapted from image diffusion to preserve temporal structure
/// </para>
/// <para>
/// <b>For Beginners:</b> StableVideoSR extends the popular Stable Diffusion image AI to
/// handle video. It adds "temporal awareness" so the model considers what happened in
/// previous and next frames when upscaling each frame, preventing the flickering that
/// occurs when frames are processed independently.
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputHeight: 128, inputWidth: 128, inputDepth: 3);
/// var model = new StableVideoSR&lt;float&gt;(arch, "stablevideosr.onnx");
/// var hrFrames = model.Upscale(lrFrames);
/// </code>
/// </para>
/// <para>
/// <b>Reference:</b> "StableVideoSR: Video Super-Resolution via Stable Diffusion with
/// Temporal Conditioning" (2024)
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Video)]
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution",
    "https://arxiv.org/abs/2312.06640",
    Year = 2024,
    Authors = "Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, Chen Change Loy")]
public partial class StableVideoSR<T> : VideoSuperResolutionBase<T>
{
    #region Fields

    private readonly StableVideoSROptions _options;
    public override ModelOptions GetOptions() => _options;
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;
    private UpscaleAVideoModel<T>? _diffusionCore;
    private readonly bool _usesInjectedDiffusionCore;
    [ExternalState]
    private readonly IConditioningModule<T>? _conditioner;

    #endregion

    #region Constructors

    /// <summary>Creates a StableVideoSR model in ONNX inference mode.</summary>
    public StableVideoSR(NeuralNetworkArchitecture<T> architecture, string modelPath, StableVideoSROptions? options = null)
        : base(architecture)
    {
        if (string.IsNullOrEmpty(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        _options = options is null ? new StableVideoSROptions() : new StableVideoSROptions(options);
        _useNativeMode = false;
        ScaleFactor = _options.ScaleFactor;
        _options.ModelPath = modelPath;
        OnnxModel = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        InitializeLayers();
    }

    /// <summary>Creates a StableVideoSR model in native training mode.</summary>
    /// <param name="architecture">The wrapper input/output architecture.</param>
    /// <param name="options">StableVideoSR inference and training options.</param>
    /// <param name="optimizer">Optional optimizer for wrapper-owned layers.</param>
    /// <param name="conditioner">Optional text conditioner used by the default diffusion core.</param>
    /// <param name="diffusionCore">
    /// Optional architecture-compatible Upscale-A-Video core. When omitted, the released
    /// paper-scale core is created. This injection point supports constrained deployments and
    /// integration tests without changing any production defaults.
    /// </param>
    public StableVideoSR(NeuralNetworkArchitecture<T> architecture, StableVideoSROptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        IConditioningModule<T>? conditioner = null,
        UpscaleAVideoModel<T>? diffusionCore = null)
        : base(architecture)
    {
        _options = options is null ? new StableVideoSROptions() : new StableVideoSROptions(options);
        _options.ValidateNativePaperContract();
        _useNativeMode = true;
        _conditioner = conditioner;
        _diffusionCore = diffusionCore;
        _usesInjectedDiffusionCore = diffusionCore is not null;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        ScaleFactor = _options.ScaleFactor;
        InitializeLayers();
    }

    #endregion

    #region Video Super-Resolution

    /// <inheritdoc />
    public override Tensor<T> Upscale(Tensor<T> lowResFrames)
    {
        ThrowIfDisposed();
        var preprocessed = PreprocessFrames(lowResFrames);
        if (!IsOnnxMode) EnsureFlowContract();
        var output = IsOnnxMode
            ? RunOnnxInference(preprocessed)
            : _diffusionCore is not null
                ? UpscaleNative(preprocessed, null, null)
                : Forward(preprocessed);
        return PostprocessOutput(output);
    }

    /// <summary>
    /// Upscales with externally estimated RAFT-compatible bidirectional flows in
    /// [B,2,F-1,H,W] layout, enabling the paper's selected-step x0 propagation.
    /// </summary>
    public Tensor<T> UpscaleWithFlows(
        Tensor<T> lowResFrames,
        Tensor<T> forwardFlows,
        Tensor<T> backwardFlows)
    {
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Flow-guided native propagation is unavailable in ONNX mode.");
        if (!_options.EnableFlowGuidedPropagation)
            throw new InvalidOperationException(
                "Flow-guided propagation is disabled in StableVideoSROptions.");
        if (lowResFrames is null) throw new ArgumentNullException(nameof(lowResFrames));
        ValidateFlowTensor(lowResFrames, forwardFlows, nameof(forwardFlows));
        ValidateFlowTensor(lowResFrames, backwardFlows, nameof(backwardFlows));
        return PostprocessOutput(UpscaleNative(
            PreprocessFrames(lowResFrames), forwardFlows, backwardFlows));
    }

    private static void ValidateFlowTensor(
        Tensor<T> frames,
        Tensor<T>? flows,
        string parameterName)
    {
        if (flows is null) throw new ArgumentNullException(parameterName);
        if (frames.Rank != 5)
            throw new ArgumentException(
                "Flow-guided upscaling requires frames in [B,F,C,H,W] layout.", nameof(frames));
        if (flows.Rank != 5 || flows.Shape[1] != 2)
            throw new ArgumentException(
                "Flows require [B,2,F-1,H,W] layout.", parameterName);
        if (frames.Shape[1] < 2)
            throw new ArgumentException(
                "Flow-guided upscaling requires at least two frames.", nameof(frames));
        if (flows.Shape[0] != frames.Shape[0]
            || flows.Shape[2] != frames.Shape[1] - 1
            || flows.Shape[3] != frames.Shape[3]
            || flows.Shape[4] != frames.Shape[4])
        {
            throw new ArgumentException(
                "Flows must match the frame batch, F-1 interval count, height, and width.",
                parameterName);
        }
    }

    private Tensor<T> UpscaleNative(
        Tensor<T> input,
        Tensor<T>? forwardFlows,
        Tensor<T>? backwardFlows)
    {
        if (_diffusionCore is null)
            return Forward(input);
        return _diffusionCore.Upscale(
            input,
            _options.Prompt,
            _options.NumDenoisingSteps,
            _options.GuidanceScale,
            seed: Architecture.RandomSeed,
            noiseLevel: _options.NoiseLevel,
            temporalWindowSize: _options.TemporalWindowSize,
            temporalWindowOverlap: _options.TemporalWindowOverlap,
            forwardFlows: forwardFlows,
            backwardFlows: backwardFlows,
            propagationSteps: _options.EnableFlowGuidedPropagation
                ? _options.PropagationSteps
                : null,
            negativePrompt: _options.NegativePrompt);
    }

    #endregion

    #region NeuralNetworkBase

    /// <inheritdoc />
    /// <remarks>
    /// The native diffusion core owns the model's trainable surface. Registering it gives
    /// checkpointing, optimizers, cloning, and the performance census the same stable view of
    /// those parameters instead of reporting an empty outer wrapper.
    /// </remarks>
    protected override void RegisterComponents()
    {
        if (_diffusionCore is not null)
            RegisterParameterComponent("diffusion/core", _diffusionCore);
    }

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (_diffusionCore is not null) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else if (_diffusionCore is null)
        {
            _diffusionCore = new UpscaleAVideoModel<T>(
                conditioner: _conditioner, seed: Architecture.RandomSeed);
        }
    }

    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode) return RunOnnxInference(input);
        EnsureFlowContract();
        return _diffusionCore is not null
            ? UpscaleNative(input, null, null)
            : Forward(input);
    }

    private void EnsureFlowContract()
    {
        if (_options.EnableFlowGuidedPropagation && _options.PropagationSteps.Length > 0)
            throw new InvalidOperationException(
                "Configured propagation steps require bidirectional optical flow. " +
                "Call UpscaleWithFlows or clear PropagationSteps.");
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            var normalizedInput = PreprocessFrames(input);
            var normalizedExpected = PreprocessFrames(expected);
            if (_diffusionCore is not null)
                _diffusionCore.TrainConditioned(
                    normalizedInput, normalizedExpected, _options.Prompt, _options.NoiseLevel);
            else
                TrainWithTape(normalizedInput, normalizedExpected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <inheritdoc />
    public override Vector<T> GetParameterGradients()
    {
        if (_diffusionCore is not null)
            return _diffusionCore.GetLastTrainingParameterGradients();
        return base.GetParameterGradients();
    }
    protected override Tensor<T> PreprocessFrames(Tensor<T> rawFrames) => NormalizeFrames(rawFrames);

    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput) => DenormalizeFrames(modelOutput);

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "StableVideoSR-Native" : "StableVideoSR-ONNX",
            Description = $"StableVideoSR {_options.Variant} temporal diffusion VSR (2024)",
            Complexity = _options.NumTemporalModules
        };
        m.AdditionalInfo["Variant"] = _options.Variant.ToString();
        m.AdditionalInfo["NumFeatures"] = _options.NumFeatures.ToString();
        m.AdditionalInfo["NumDenoisingSteps"] = _options.NumDenoisingSteps.ToString();
        m.AdditionalInfo["NumTemporalModules"] = _options.NumTemporalModules.ToString();
        m.AdditionalInfo["GuidanceScale"] = _options.GuidanceScale.ToString();
        m.AdditionalInfo["ScaleFactor"] = _options.ScaleFactor.ToString();
        m.AdditionalInfo["LatentScaleFactor"] = _options.LatentScaleFactor.ToString();
        m.AdditionalInfo["MaximumNoiseLevel"] = _options.MaximumNoiseLevel.ToString();
        m.AdditionalInfo["TemporalWindow"] = $"{_options.TemporalWindowSize} (overlap {_options.TemporalWindowOverlap})";
        m.AdditionalInfo["FlowGuidedPropagation"] = _options.EnableFlowGuidedPropagation.ToString();
        m.AdditionalInfo["NoiseLevel"] = _options.NoiseLevel.ToString();
        m.AdditionalInfo["PropagationSteps"] = string.Join(",", _options.PropagationSteps);
        m.AdditionalInfo["TextConditioner"] = _conditioner is null
            ? "required for guidance > 1"
            : $"{_conditioner.GetType().Name} ({_conditioner.EmbeddingDimension}D)";
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode);
        w.Write(_options.ModelPath ?? string.Empty);
        w.Write((int)_options.Variant);
        w.Write(_options.NumFeatures);
        w.Write(_options.NumDenoisingSteps);
        w.Write(_options.NumTemporalModules);
        w.Write(_options.ScaleFactor);
        w.Write(_options.LatentDim);
        w.Write(_options.GuidanceScale);
        w.Write(_options.DropoutRate);
        w.Write(_options.LatentScaleFactor);
        w.Write(_options.MaximumNoiseLevel);
        w.Write(_options.TemporalWindowSize);
        w.Write(_options.TemporalWindowOverlap);
        w.Write(_options.EnableFlowGuidedPropagation);
        w.Write(_options.NoiseLevel);
        w.Write(_options.Prompt ?? string.Empty);
        w.Write(_options.PropagationSteps.Length);
        foreach (int step in _options.PropagationSteps) w.Write(step);

        w.Write(_diffusionCore is not null);
        if (_diffusionCore is not null)
        {
            var chunks = _diffusionCore.GetParameterChunks().ToList();
            w.Write(chunks.Count);
            foreach (var chunk in chunks)
                SerializationHelper<T>.SerializeTensor(w, chunk);
        }
        // Appended for backward compatibility: older payloads end immediately
        // after the optional diffusion-core chunks.
        w.Write(_options.NegativePrompt ?? string.Empty);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean();
        string mp = r.ReadString();
        if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.Variant = (VideoModelVariant)r.ReadInt32();
        _options.NumFeatures = r.ReadInt32();
        _options.NumDenoisingSteps = r.ReadInt32();
        _options.NumTemporalModules = r.ReadInt32();
        _options.ScaleFactor = r.ReadInt32();
        _options.LatentDim = r.ReadInt32();
        _options.GuidanceScale = r.ReadDouble();
        _options.DropoutRate = r.ReadDouble();
        _options.LatentScaleFactor = r.ReadDouble();
        _options.MaximumNoiseLevel = r.ReadInt32();
        _options.TemporalWindowSize = r.ReadInt32();
        _options.TemporalWindowOverlap = r.ReadInt32();
        _options.EnableFlowGuidedPropagation = r.ReadBoolean();
        _options.NoiseLevel = r.ReadInt32();
        _options.Prompt = r.ReadString();
        int propagationCount = r.ReadInt32();
        _options.PropagationSteps = new int[propagationCount];
        for (int i = 0; i < propagationCount; i++)
            _options.PropagationSteps[i] = r.ReadInt32();
        if (_useNativeMode) _options.ValidateNativePaperContract();

        bool hasDiffusionCore = r.ReadBoolean();
        if (hasDiffusionCore)
        {
            _diffusionCore ??= new UpscaleAVideoModel<T>(
                conditioner: _conditioner, seed: Architecture.RandomSeed);
            int chunkCount = r.ReadInt32();
            var chunks = new Tensor<T>[chunkCount];
            for (int i = 0; i < chunkCount; i++)
                chunks[i] = SerializationHelper<T>.DeserializeTensor(r);
            _diffusionCore.SetParameterChunks(chunks);
        }
        if (r.BaseStream.Position < r.BaseStream.Length)
            _options.NegativePrompt = r.ReadString();
        ScaleFactor = _options.ScaleFactor;
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
        {
            OnnxModel?.Dispose();
            OnnxModel = new OnnxModel<T>(p, _options.OnnxOptions);
        }
        // Native-mode layers (with their trained weights) are already reconstructed by
        // the base deserializer before this override runs; re-initializing here would
        // discard them and leave the model randomly initialized.
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p))
            return new StableVideoSR<T>(Architecture, p, new StableVideoSROptions(_options));
        var clonedCore = _usesInjectedDiffusionCore && _diffusionCore is not null
            ? (UpscaleAVideoModel<T>)_diffusionCore.Clone()
            : null;
        return new StableVideoSR<T>(
            Architecture,
            new StableVideoSROptions(_options),
            conditioner: _conditioner,
            diffusionCore: clonedCore);
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        if (!_usesInjectedDiffusionCore || _diffusionCore is null)
            return base.DeepCopy();

        // An injected core carries architecture that StableVideoSROptions intentionally
        // does not describe. Clone that core directly instead of serializing its chunks
        // into a newly-created paper-default graph with a different per-tensor layout.
        return new StableVideoSR<T>(
            Architecture,
            new StableVideoSROptions(_options),
            conditioner: _conditioner,
            diffusionCore: (UpscaleAVideoModel<T>)_diffusionCore.Clone());
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> Clone() => DeepCopy();
    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(StableVideoSR<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing)
        {
            OnnxModel?.Dispose();
            _diffusionCore?.Dispose();
        }
        base.Dispose(disposing);
    }

    #endregion
}
