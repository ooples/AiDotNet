using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Evaluation;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// High-level pipeline for SSL pretraining.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This pipeline provides a simple, high-level interface
/// for SSL pretraining. Just provide your encoder and data, and it handles the
/// rest: method selection, augmentation, training loop, and evaluation.</para>
///
/// <para><b>Example usage:</b></para>
/// <code>
/// var pipeline = new SSLPretrainingPipeline&lt;double&gt;(encoder)
///     .WithMethod(SSLMethodType.SimCLR)
///     .WithConfig(config => config.PretrainingEpochs = 100);
///
/// var result = pipeline.Train(dataLoader);
/// </code>
/// </remarks>
public class SSLPretrainingPipeline<T>
{
    private readonly INeuralNetwork<T> _encoder;
    private readonly int _encoderOutputDim;
    private SSLConfig _config;
    private ISSLMethod<T>? _method;
    private Func<INeuralNetwork<T>, INeuralNetwork<T>>? _createEncoderCopy;

    /// <summary>
    /// Event raised during training for progress updates.
    /// </summary>
    public event Action<int, int, T>? OnProgress;

    /// <summary>
    /// Initializes a new SSL pretraining pipeline.
    /// </summary>
    /// <param name="encoder">The encoder network to pretrain.</param>
    /// <param name="encoderOutputDim">Output dimension of the encoder.</param>
    public SSLPretrainingPipeline(INeuralNetwork<T> encoder, int encoderOutputDim)
    {
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _encoderOutputDim = encoderOutputDim;
        _config = new SSLConfig();
    }

    /// <summary>
    /// Sets the SSL method to use.
    /// </summary>
    public SSLPretrainingPipeline<T> WithMethod(SSLMethodType method)
    {
        _config.Method = method;
        return this;
    }

    /// <summary>
    /// Configures the training parameters.
    /// </summary>
    public SSLPretrainingPipeline<T> WithConfig(Action<SSLConfig> configure)
    {
        configure(_config);
        return this;
    }

    /// <summary>
    /// Sets the function to create encoder copies (for momentum methods).
    /// </summary>
    public SSLPretrainingPipeline<T> WithEncoderCopyFactory(
        Func<INeuralNetwork<T>, INeuralNetwork<T>> createCopy)
    {
        _createEncoderCopy = createCopy;
        return this;
    }

    /// <summary>
    /// Trains the encoder using SSL.
    /// </summary>
    /// <param name="dataLoader">Function that yields batches of unlabeled data.</param>
    /// <param name="validationData">Optional validation data for monitoring.</param>
    /// <param name="validationLabels">Optional validation labels for k-NN evaluation.</param>
    /// <returns>Training result with pretrained encoder.</returns>
    public SSLResult<T> Train(
        Func<IEnumerable<Tensor<T>>> dataLoader,
        Tensor<T>? validationData = null,
        int[]? validationLabels = null)
    {
        // Create SSL method
        _method = CreateMethod();

        // Create session
        var session = new SSLSession<T>(_method, _config);

        // Subscribe to events
        session.OnEpochEnd += (epoch, loss) =>
        {
            OnProgress?.Invoke(epoch, _config.PretrainingEpochs ?? 100, loss);
        };

        // Train
        var result = session.Train(dataLoader, validationData, validationLabels);

        // Run final evaluation if validation data provided
        if (result.IsSuccess && validationData is not null && validationLabels is not null)
        {
            result = RunFinalEvaluation(result, validationData, validationLabels);
        }

        return result;
    }

    private ISSLMethod<T> CreateMethod()
    {
        var method = _config.Method ?? SSLMethodType.SimCLR;

        return method switch
        {
            SSLMethodType.SimCLR => CreateSimCLR(),
            SSLMethodType.MoCo => CreateMoCo(),
            SSLMethodType.MoCoV2 => CreateMoCoV2(),
            SSLMethodType.MoCoV3 => CreateMoCoV3(),
            SSLMethodType.BYOL => CreateBYOL(),
            SSLMethodType.SimSiam => CreateSimSiam(),
            SSLMethodType.BarlowTwins => CreateBarlowTwins(),
            SSLMethodType.DINO => CreateDINO(),
            SSLMethodType.MAE => CreateMAE(),
            _ => CreateSimCLR()
        };
    }

    private ISSLMethod<T> CreateSimCLR()
    {
        var projDim = _config.ProjectorOutputDimension ?? 128;
        var hiddenDim = _config.ProjectorHiddenDimension ?? 2048;

        var projector = new MLPProjector<T>(_encoderOutputDim, hiddenDim, projDim);

        return new SimCLR<T>(_encoder, projector, _config);
    }

    private ISSLMethod<T> CreateMoCo()
    {
        var projDim = _config.ProjectorOutputDimension ?? 128;

        if (_createEncoderCopy is null)
            throw new InvalidOperationException("Encoder copy factory required for MoCo");

        var projector = new LinearProjector<T>(_encoderOutputDim, projDim);
        var momentumProjector = new LinearProjector<T>(_encoderOutputDim, projDim);
        momentumProjector.SetParameters(projector.GetParameters());

        var encoderCopy = _createEncoderCopy(_encoder);
        var momentumEncoder = new MomentumEncoder<T>(encoderCopy, 0.999);

        return new MoCo<T>(_encoder, momentumEncoder, projector, momentumProjector, projDim, _config);
    }

    private ISSLMethod<T> CreateMoCoV2()
    {
        var projDim = _config.ProjectorOutputDimension ?? 128;
        var hiddenDim = _config.ProjectorHiddenDimension ?? 2048;

        if (_createEncoderCopy is null)
            throw new InvalidOperationException("Encoder copy factory required for MoCoV2");

        var projector = new MLPProjector<T>(_encoderOutputDim, hiddenDim, projDim);
        var momentumProjector = new MLPProjector<T>(_encoderOutputDim, hiddenDim, projDim);
        momentumProjector.SetParameters(projector.GetParameters());

        var encoderCopy = _createEncoderCopy(_encoder);
        var momentumEncoder = new MomentumEncoder<T>(encoderCopy, 0.999);

        return new MoCoV2<T>(_encoder, momentumEncoder, projector, momentumProjector, projDim, _config);
    }

    private ISSLMethod<T> CreateMoCoV3()
    {
        var projDim = _config.ProjectorOutputDimension ?? 256;
        var hiddenDim = _config.ProjectorHiddenDimension ?? 4096;

        if (_createEncoderCopy is null)
            throw new InvalidOperationException("Encoder copy factory required for MoCoV3");

        var projector = new MLPProjector<T>(_encoderOutputDim, hiddenDim, projDim);
        var momentumProjector = new MLPProjector<T>(_encoderOutputDim, hiddenDim, projDim);
        momentumProjector.SetParameters(projector.GetParameters());

        var predictor = new MLPProjector<T>(projDim, hiddenDim / 4, projDim);

        var encoderCopy = _createEncoderCopy(_encoder);
        var momentumEncoder = new MomentumEncoder<T>(encoderCopy, 0.99);

        return new MoCoV3<T>(_encoder, momentumEncoder, projector, momentumProjector, predictor, _config);
    }

    private ISSLMethod<T> CreateBYOL()
    {
        var projDim = _config.ProjectorOutputDimension ?? 256;
        var hiddenDim = _config.ProjectorHiddenDimension ?? 4096;

        if (_createEncoderCopy is null)
            throw new InvalidOperationException("Encoder copy factory required for BYOL");

        var onlineProjector = new SymmetricProjector<T>(_encoderOutputDim, hiddenDim, projDim, hiddenDim);
        var targetProjector = new SymmetricProjector<T>(_encoderOutputDim, hiddenDim, projDim, 0);

        var encoderCopy = _createEncoderCopy(_encoder);
        var targetEncoder = new MomentumEncoder<T>(encoderCopy, 0.996);

        return new BYOL<T>(_encoder, targetEncoder, onlineProjector, targetProjector, _config);
    }

    private ISSLMethod<T> CreateSimSiam()
    {
        var projDim = _config.ProjectorOutputDimension ?? 2048;
        var hiddenDim = _config.ProjectorHiddenDimension ?? 2048;

        var projector = new SymmetricProjector<T>(_encoderOutputDim, hiddenDim, projDim, 512);

        return new SimSiam<T>(_encoder, projector, _config);
    }

    private ISSLMethod<T> CreateBarlowTwins()
    {
        var projDim = _config.ProjectorOutputDimension ?? 8192;
        var hiddenDim = _config.ProjectorHiddenDimension ?? 8192;

        var projector = new MLPProjector<T>(_encoderOutputDim, hiddenDim, projDim, useBatchNormOnOutput: true);

        return new BarlowTwins<T>(_encoder, projector, _config);
    }

    private ISSLMethod<T> CreateDINO()
    {
        var outputDim = 65536;
        var hiddenDim = _config.ProjectorHiddenDimension ?? 2048;

        if (_createEncoderCopy is null)
            throw new InvalidOperationException("Encoder copy factory required for DINO");

        var studentProjector = new MLPProjector<T>(_encoderOutputDim, hiddenDim, outputDim, useBatchNormOnOutput: true);
        var teacherProjector = new MLPProjector<T>(_encoderOutputDim, hiddenDim, outputDim, useBatchNormOnOutput: true);
        teacherProjector.SetParameters(studentProjector.GetParameters());

        var encoderCopy = _createEncoderCopy(_encoder);
        var teacherEncoder = new MomentumEncoder<T>(encoderCopy, 0.996);

        return new DINO<T>(_encoder, teacherEncoder, studentProjector, teacherProjector, outputDim, _config);
    }

    private ISSLMethod<T> CreateMAE()
    {
        var maeConfig = _config.MAE ?? new MAEConfig();
        var patchSize = maeConfig.PatchSize ?? 16;
        var maskRatio = maeConfig.MaskRatio ?? 0.75;

        return new MAE<T>(_encoder, null, patchSize, 224, maskRatio, _config);
    }

    private SSLResult<T> RunFinalEvaluation(
        SSLResult<T> result,
        Tensor<T> validationData,
        int[] validationLabels)
    {
        var numClasses = validationLabels.Max() + 1;

        // Linear evaluation
        var linearEval = new LinearEvaluator<T>(
            result.Encoder!, _encoderOutputDim, numClasses, epochs: 20);

        result.LinearEvaluation = linearEval.Train(
            validationData, validationLabels);

        // k-NN evaluation
        var knn = new KNNEvaluator<T>(result.Encoder!);
        knn.Fit(validationData, validationLabels);
        result.KNNAccuracy = knn.Evaluate(validationData, validationLabels);

        return result;
    }
}
