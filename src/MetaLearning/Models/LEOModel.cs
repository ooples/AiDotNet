using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Models;

/// <summary>
/// LEO model for few-shot classification with latent space optimization.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// The adapted state of LEO for one task: its own copy of the feature encoder, the linear softmax classifier LEO
/// generated and adapted - one weight row per class - and the adapted latent codes. <see cref="Predict"/> returns
/// class probabilities per example, <c>[rows, NumClasses]</c>, for Tensor and Matrix outputs - a class the task did
/// not contain gets probability zero - and the most probable class of each example for a Vector output.
/// </para>
/// <para><b>For Beginners:</b> After LEO adapts to a new task by optimizing
/// in latent space, this model stores:
/// </para>
/// <list type="bullet">
/// <item>The feature encoder for extracting embeddings</item>
/// <item>The classifier parameters decoded from the adapted latent code</item>
/// <item>The adapted latent code itself (useful for further fine-tuning)</item>
/// </list>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Meta-Learning with Latent Embedding Optimization",
    "https://arxiv.org/abs/1807.05960",
    Year = 2019,
    Authors = "Rusu, A. A., Rao, D., Sygnowski, J., Vinyals, O., Pascanu, R., Osindero, S., & Hadsell, R.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class LEOModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IFullModel<T, TInput, TOutput> _featureEncoder;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _classifierParams;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _latentCode;
    private readonly LEOOptions<T, TInput, TOutput> _options;
    private readonly int[] _classSlots;

    /// <summary>
    /// Initializes a new instance of the LEOModel with one classifier row per class, class <c>c</c> in row <c>c</c>.
    /// </summary>
    /// <param name="featureEncoder">The feature encoder network; the model keeps its own copy.</param>
    /// <param name="classifierParams">
    /// The classifier weights, <c>NumClasses * EmbeddingDimension</c> values, one row per class.
    /// </param>
    /// <param name="latentCode">The optimized latent code.</param>
    /// <param name="options">The LEO options.</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    /// <exception cref="ArgumentException">The classifier weights do not hold one row per class.</exception>
    public LEOModel(
        IFullModel<T, TInput, TOutput> featureEncoder,
        Vector<T> classifierParams,
        Vector<T> latentCode,
        LEOOptions<T, TInput, TOutput> options)
        : this(featureEncoder, classifierParams, latentCode, options,
            Enumerable.Range(0, options?.NumClasses ?? 0).ToArray())
    {
    }

    /// <summary>Initializes the model with classifier rows for the given class labels, in order.</summary>
    internal LEOModel(
        IFullModel<T, TInput, TOutput> featureEncoder,
        Vector<T> classifierParams,
        Vector<T> latentCode,
        LEOOptions<T, TInput, TOutput> options,
        int[] classSlots)
    {
        Guard.NotNull(featureEncoder);
        Guard.NotNull(classifierParams);
        Guard.NotNull(latentCode);
        Guard.NotNull(options);
        if (classifierParams.Length != classSlots.Length * options.EmbeddingDimension)
        {
            throw new ArgumentException(
                $"{classifierParams.Length} classifier weights are not {classSlots.Length} rows of "
                + $"{options.EmbeddingDimension}.", nameof(classifierParams));
        }

        _featureEncoder = featureEncoder.DeepCopy();
        _classifierParams = classifierParams;
        _latentCode = latentCode;
        _options = options;
        _classSlots = classSlots;
    }

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>
    /// Gets the adapted classifier parameters, one row of EmbeddingDimension weights per class.
    /// </summary>
    public Vector<T> ClassifierParams => _classifierParams;

    /// <summary>
    /// Gets the optimized latent code, one LatentDimension block per class.
    /// </summary>
    public Vector<T> LatentCode => _latentCode;

    /// <summary>
    /// Gets the latent dimension.
    /// </summary>
    public int LatentDimension => _options.LatentDimension;

    /// <summary>
    /// Gets the number of classes.
    /// </summary>
    public int NumClasses => _options.NumClasses;

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var rows = ClassifierOutputs<T>.AsRows(_featureEncoder.Predict(input));
        if (rows.Shape[1] != _options.EmbeddingDimension)
        {
            throw new InvalidOperationException(
                $"The feature encoder emits {rows.Shape[1]}-wide embeddings but EmbeddingDimension is "
                + $"{_options.EmbeddingDimension}.");
        }

        int classes = _classSlots.Length;
        var weights = Tensor<T>.FromVector(_classifierParams).Reshape(classes, _options.EmbeddingDimension);
        var logits = engine.TensorMatMul(rows, engine.TensorTranspose(weights));
        var probabilities = engine.Softmax(logits, axis: 1);

        if (typeof(TOutput) == typeof(Vector<T>))
        {
            int n = probabilities.Shape[0];
            var predicted = new Vector<T>(n);
            for (int r = 0; r < n; r++)
            {
                int best = 0;
                for (int c = 1; c < classes; c++)
                {
                    if (NumOps.GreaterThan(probabilities[r * classes + c], probabilities[r * classes + best])) best = c;
                }

                predicted[r] = NumOps.FromDouble(_classSlots[best]);
            }

            return (TOutput)(object)predicted;
        }

        var columns = new Tensor<T>(new[] { classes, _options.NumClasses });
        for (int c = 0; c < classes; c++) columns[c * _options.NumClasses + _classSlots[c]] = NumOps.One;
        return ClassifierOutputs<T>.ToOutput<TOutput>(engine.TensorMatMul(probabilities, columns));
    }

    /// <inheritdoc/>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Use the LEO algorithm to train the model.");
    }

    /// <inheritdoc/>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("LEO model parameters are set during adaptation.");
    }

    /// <inheritdoc/>
    public Vector<T> GetParameters()
    {
        // Return combined feature encoder + classifier parameters + latent code
        var encoderParams = InterfaceGuard.Parameterizable(_featureEncoder).GetParameters();
        int totalSize = encoderParams.Length + _classifierParams.Length + _latentCode.Length;
        var combined = new Vector<T>(totalSize);

        int idx = 0;
        for (int i = 0; i < encoderParams.Length; i++)
        {
            combined[idx++] = encoderParams[i];
        }
        for (int i = 0; i < _classifierParams.Length; i++)
        {
            combined[idx++] = _classifierParams[i];
        }
        for (int i = 0; i < _latentCode.Length; i++)
        {
            combined[idx++] = _latentCode[i];
        }

        return combined;
    }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata()
    {
        return Metadata;
    }
}
