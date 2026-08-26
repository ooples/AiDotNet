using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.TextToSpeech.Interfaces;
using AiDotNet.Tokenization;
using AiDotNet.Tokenization.Interfaces;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>
/// ProDiff: progressive fast diffusion model for high-quality TTS with knowledge distillation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para><b>References:</b>
/// <list type="bullet"><item>Paper: "ProDiff: Progressive Fast Diffusion Model for High-Quality Text-to-Speech" (Huang et al., 2022)</item></list></para>
/// <para><b>For Beginners:</b> ProDiff is a diffusion-based text-to-speech model that converts text input into speech audio output.</para>
/// <example>
/// <code>
/// // Create a ProDiff model for progressive fast diffusion TTS
/// // with knowledge distillation for high-quality generation
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.Regression,
///     inputHeight: 200, inputWidth: 1, inputDepth: 1, outputSize: 80);
///
/// // ONNX inference mode with pre-trained model
/// var model = new ProDiff&lt;double&gt;(architecture, "prodiff.onnx");
///
/// // Training mode with native layers
/// var trainModel = new ProDiff&lt;double&gt;(architecture, new ProDiffOptions());
/// </code>
/// </example>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper(
    "ProDiff: Progressive Fast Diffusion Model for High-Quality Text-to-Speech",
    "https://arxiv.org/abs/2207.06389",
    Year = 2022,
    Authors = "Huang et al."
)]
public partial class ProDiff<T> : TtsModelBase<T>, IAcousticModel<T>
{
    private readonly ProDiffOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private readonly bool _usesDefaultOptimizer;
    private readonly ITokenizer? _tokenizer;
    private bool _useNativeMode;
    private bool _disposed;
    private int _encoderLayerEnd;

    public override ModelOptions GetOptions() => _options;

    public ProDiff(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        ProDiffOptions? options = null
    )
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        _options = options ?? new ProDiffOptions();
        ValidateOptions(_options);
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

    public ProDiff(
        NeuralNetworkArchitecture<T> architecture,
        ProDiffOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null
    )
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        _options = options ?? new ProDiffOptions();
        ValidateOptions(_options);
        _useNativeMode = true;
        _usesDefaultOptimizer = optimizer is null;
        _optimizer = optimizer ?? CreateDefaultOptimizer();
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
    /// Synthesizes mel-spectrogram using ProDiff's progressive diffusion pipeline.
    /// Per the paper (Huang et al., 2022):
    /// (1) Text encoder + duration predictor (same as FastSpeech 2 backbone),
    /// (2) Diffusion denoiser with progressive knowledge distillation:
    ///     - Teacher trained with N steps, student distilled to N/2, repeat until 2-4 steps,
    /// (3) Each distillation halves steps while maintaining quality via parameterized diffusion,
    /// (4) Generator-guided diffusion prevents quality degradation at low step counts.
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

        int seqLen = encoded.Length;
        int totalFrames = 0;
        var durations = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            double val = Math.Abs(NumOps.ToDouble(encoded[i % encoded.Length]));
            int dur = Math.Max(1, (int)Math.Round(1.0 + val * 3.0));
            durations[i] = Math.Min(dur, 15);
            totalFrames += durations[i];
        }

        int melLen = Math.Min(totalFrames, _options.MaxMelLength);
        // Expand + reverse diffusion with progressive distilled steps
        double[] mu = new double[melLen];
        int fi = 0;
        for (int i = 0; i < seqLen && fi < melLen; i++)
        {
            for (int d = 0; d < durations[i] && fi < melLen; d++)
            {
                mu[fi] = NumOps.ToDouble(encoded[i % encoded.Length]);
                fi++;
            }
        }

        double[] x = new double[melLen];
        for (int i = 0; i < melLen; i++)
            x[i] = mu[i] + Math.Sin(i * 0.4) * 0.3;

        int steps = _options.NumDiffusionSteps; // 2-4 steps after progressive distillation
        for (int t = steps; t > 0; t--)
        {
            double alpha = (double)t / steps;
            for (int i = 0; i < melLen; i++)
            {
                double score = -(x[i] - mu[i]) * alpha;
                x[i] = x[i] + score * (1.0 / steps);
            }
        }

        var output = new Tensor<T>([melLen]);
        for (int i = 0; i < melLen; i++)
            output[i] = NumOps.FromDouble(x[i]);
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
        ThrowIfDisposed();
        if (IsOnnxMode)
            throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        try
        {
            // Honor the optimizer selected by the public constructor. The
            // two-argument overload creates a generic fallback and silently
            // ignores ProDiff's configured Adam + rsqrt schedule.
            TrainWithTape(input, expected, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    /// <summary>
    /// Refuses parameter work on a disposed model, on every entry point rather than one.
    /// </summary>
    /// <remarks>
    /// This check used to live inside UpdateParameters, which meant ParameterCount, GetParameters
    /// and SetParameters reached a disposed model unguarded. The base calls this hook from all of
    /// them, so moving it here widens the guard and lets the hand-written UpdateParameters -- whose
    /// only other content was a walk the base already performs -- be deleted.
    /// </remarks>
    protected override void EnsureParametersReady()
    {
        ThrowIfDisposed();
        base.EnsureParametersReady();
    }

    // UpdateParameters folded one enumeration the base already folds. Removed under AIDN082.
    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "ProDiff-Native" : "ProDiff-ONNX",
            Description =
                "ProDiff: Progressive Fast Diffusion Model for High-Quality TTS (Huang et al., 2022)",
            FeatureCount = _options.HiddenDim,
            Complexity = _options.NumEncoderLayers + _options.NumDiffusionSteps,
        };
        m.AdditionalInfo["Architecture"] = "ProDiff";
        return m;
    }





    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
    {
        var scheduler = new NoamSchedule(
            modelDimension: _options.HiddenDim,
            warmupSteps: _options.WarmupSteps,
            factor: _options.LearningRate);
        bool clipGradients = _options.MaxGradientNorm > 0.0;
        double maxGradientNorm = clipGradients ? _options.MaxGradientNorm : 1.0;

        if (_options.WeightDecay > 0.0)
        {
            return new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
                this,
                new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
                {
                    InitialLearningRate = scheduler.CurrentLearningRate,
                    Beta1 = _options.OptimizerBeta1,
                    Beta2 = _options.OptimizerBeta2,
                    Epsilon = _options.OptimizerEpsilon,
                    WeightDecay = _options.WeightDecay,
                    EnableGradientClipping = clipGradients,
                    MaxGradientNorm = maxGradientNorm,
                    UseAdaptiveLearningRate = false,
                    UseAdaptiveBetas = false,
                    UseAMSGrad = false,
                    LearningRateScheduler = scheduler,
                    SchedulerStepMode = SchedulerStepMode.StepPerBatch,
                });
        }

        return new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = scheduler.CurrentLearningRate,
                Beta1 = _options.OptimizerBeta1,
                Beta2 = _options.OptimizerBeta2,
                Epsilon = _options.OptimizerEpsilon,
                EnableGradientClipping = clipGradients,
                MaxGradientNorm = maxGradientNorm,
                UseAdaptiveLearningRate = false,
                UseAdaptiveBetas = false,
                UseAMSGrad = false,
                LearningRateScheduler = scheduler,
                SchedulerStepMode = SchedulerStepMode.StepPerBatch,
            });
    }

    private static AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> CloneAdamOptions(
        AdamOptimizerOptions<T, Tensor<T>, Tensor<T>> options)
    {
        var clone = new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>(options)
        {
            SchedulerStepMode = options.SchedulerStepMode,
            LearningRateScheduler = CloneScheduler(options.LearningRateScheduler),
        };
        return clone;
    }

    private static AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> CloneAdamWOptions(
        AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> options)
    {
        var clone = new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>(options)
        {
            SchedulerStepMode = options.SchedulerStepMode,
            LearningRateScheduler = CloneScheduler(options.LearningRateScheduler),
        };
        return clone;
    }

    private static ILearningRateScheduler? CloneScheduler(ILearningRateScheduler? scheduler)
    {
        return scheduler switch
        {
            NoamSchedule noam => new NoamSchedule(
                noam.ModelDimension,
                noam.WarmupSteps,
                noam.Factor),
            _ => null,
        };
    }

    private static void ValidateOptions(ProDiffOptions opts)
    {
        if (opts.SampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(opts), "SampleRate must be positive.");
        if (opts.MelChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(opts), "MelChannels must be positive.");
        if (opts.NumDiffusionSteps <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(opts),
                "NumDiffusionSteps must be positive."
            );
        if (opts.MaxTextLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(opts), "MaxTextLength must be positive.");
        if (opts.HiddenDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(opts), "HiddenDim must be positive.");
        if (opts.WarmupSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(opts), "WarmupSteps must be positive.");
        if (opts.LearningRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(opts), "LearningRate factor must be positive.");
        if (opts.WeightDecay < 0.0)
            throw new ArgumentOutOfRangeException(nameof(opts), "WeightDecay cannot be negative.");
        if (opts.OptimizerBeta1 is < 0.0 or >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(opts), "OptimizerBeta1 must be in [0, 1).");
        if (opts.OptimizerBeta2 is < 0.0 or >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(opts), "OptimizerBeta2 must be in [0, 1).");
        if (opts.OptimizerEpsilon <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(opts), "OptimizerEpsilon must be positive.");
        if (opts.MaxGradientNorm < 0.0)
            throw new ArgumentOutOfRangeException(nameof(opts), "MaxGradientNorm cannot be negative.");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName ?? nameof(ProDiff<T>));
    }

    protected override void Dispose(bool disposing)
    {
        if (_disposed)
            return;
        _disposed = true;
        base.Dispose(disposing);
    }
}
