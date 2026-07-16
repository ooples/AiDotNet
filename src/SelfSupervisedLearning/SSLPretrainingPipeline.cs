using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Evaluation;
using AiDotNet.Validation;

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
/// // Leave the method unset to get SimCLR, the standard contrastive baseline.
/// var pipeline = new SSLPretrainingPipeline&lt;double&gt;(encoder, encoderOutputDim)
///     .WithConfig(config => config.PretrainingEpochs = 100);
///
/// // Or name one — including a method the library does not ship.
/// pipeline.WithMethod(MoCoV2&lt;double&gt;.Create(encoder, createEncoderCopy, encoderOutputDim));
///
/// var result = pipeline.Train(dataLoader);
/// </code>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Optimization)]
[ModelTask(ModelTask.Embedding)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("A Simple Framework for Contrastive Learning of Visual Representations", "https://arxiv.org/abs/2002.05709", Year = 2020, Authors = "Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton")]
public class SSLPretrainingPipeline<T>
{
    private readonly INeuralNetwork<T> _encoder;
    private readonly int _encoderOutputDim;
    private SSLConfig<T> _config;
    private ISSLMethod<T>? _method;

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
        Guard.NotNull(encoder);
        _encoder = encoder;
        _encoderOutputDim = encoderOutputDim;
        _config = new SSLConfig<T>();
    }

    /// <summary>
    /// Sets the SSL method to use.
    /// </summary>
    /// <param name="method">
    /// The SSL method. Every shipped method has a static <c>Create</c> factory that builds it from an
    /// encoder; any other <see cref="ISSLMethod{T}"/> works too. Leave this unset to get
    /// <see cref="SimCLR{T}"/>.
    /// </param>
    public SSLPretrainingPipeline<T> WithMethod(ISSLMethod<T> method)
    {
        Guard.NotNull(method);
        _config.Method = method;
        return this;
    }

    /// <summary>
    /// Configures the training parameters.
    /// </summary>
    public SSLPretrainingPipeline<T> WithConfig(Action<SSLConfig<T>> configure)
    {
        configure(_config);
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

    // Null means the standard method. SimCLR is the standard contrastive baseline and is what this
    // pipeline defaulted to before, so that is the default here.
    private ISSLMethod<T> CreateMethod() => _config.Method ?? CreateSimCLR();

    private ISSLMethod<T> CreateSimCLR()
    {
        var projDim = _config.ProjectorOutputDimension ?? 128;
        var hiddenDim = _config.ProjectorHiddenDimension ?? 2048;

        var projector = new MLPProjector<T>(_encoderOutputDim, hiddenDim, projDim);

        return new SimCLR<T>(_encoder, projector, _config);
    }

    private SSLResult<T> RunFinalEvaluation(
        SSLResult<T> result,
        Tensor<T> validationData,
        int[] validationLabels)
    {
        var numClasses = validationLabels.Max() + 1;

        var encoder = result.Encoder ?? throw new InvalidOperationException("Encoder has not been initialized in the result.");

        // Linear evaluation
        var linearEval = new LinearEvaluator<T>(encoder, _encoderOutputDim, numClasses, epochs: 20);

        result.LinearEvaluation = linearEval.Train(
            validationData, validationLabels);

        // k-NN evaluation
        var knn = new KNNEvaluator<T>(encoder);
        knn.Fit(validationData, validationLabels);
        result.KNNAccuracy = knn.Evaluate(validationData, validationLabels);

        return result;
    }
}
