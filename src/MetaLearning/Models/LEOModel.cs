using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
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
/// This model stores the adapted state of LEO after latent space optimization.
/// It contains the feature encoder, adapted classifier parameters, and the
/// optimized latent code.
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
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Meta-Learning with Latent Embedding Optimization", "https://arxiv.org/abs/1807.05960", Year = 2019, Authors = "Andrei A. Rusu, Dushyant Rao, Jakub Sygnowski, Oriol Vinyals, Razvan Pascanu, Simon Osindero, Raia Hadsell")]
public class LEOModel<T, TInput, TOutput> : MetaLearningModelBase<T, TInput, TOutput>
{
    private Vector<T> _classifierParams;
    private Vector<T> _latentCode;
    private readonly LEOOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the LEOModel.
    /// </summary>
    /// <param name="featureEncoder">The feature encoder network.</param>
    /// <param name="classifierParams">The adapted classifier parameters.</param>
    /// <param name="latentCode">The optimized latent code.</param>
    /// <param name="options">The LEO options.</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    public LEOModel(
        IFullModel<T, TInput, TOutput> featureEncoder,
        Vector<T> classifierParams,
        Vector<T> latentCode,
        LEOOptions<T, TInput, TOutput> options)
        : base(featureEncoder)
    {
        Guard.NotNull(classifierParams);
        _classifierParams = classifierParams;
        Guard.NotNull(latentCode);
        _latentCode = latentCode;
        Guard.NotNull(options);
        _options = options;
    }

    /// <summary>
    /// Gets the adapted classifier parameters.
    /// </summary>
    public Vector<T> ClassifierParams => _classifierParams;

    /// <summary>
    /// Gets the optimized latent code.
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
    public override TOutput Predict(TInput input)
    {
        var embeddings = ExtractFeaturesFromBaseModel(input, _options.EmbeddingDimension);
        var logits = ComputeLogits(embeddings);
        return ConvertVectorToOutput(logits);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var encoderParams = BaseModel.GetParameters();
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
    public override void SetParameters(Vector<T> parameters)
    {
        Guard.NotNull(parameters);
        var encoderParams = BaseModel.GetParameters();
        int expectedLength = encoderParams.Length + _classifierParams.Length + _latentCode.Length;
        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException(
                $"Parameter count mismatch: expected {expectedLength}, got {parameters.Length}.",
                nameof(parameters));
        }

        int idx = 0;
        var newEncoderParams = new Vector<T>(encoderParams.Length);
        for (int i = 0; i < encoderParams.Length; i++)
        {
            newEncoderParams[i] = parameters[idx++];
        }
        BaseModel.SetParameters(newEncoderParams);

        for (int i = 0; i < _classifierParams.Length; i++)
        {
            _classifierParams[i] = parameters[idx++];
        }
        for (int i = 0; i < _latentCode.Length; i++)
        {
            _latentCode[i] = parameters[idx++];
        }
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        var model = DeepCopy();
        model.SetParameters(parameters);
        return model;
    }

    /// <inheritdoc/>
    public override IFullModel<T, TInput, TOutput> DeepCopy()
    {
        var clonedEncoder = BaseModel.DeepCopy();
        return new LEOModel<T, TInput, TOutput>(
            clonedEncoder, _classifierParams.Clone(), _latentCode.Clone(), _options);
    }

    private Vector<T> ComputeLogits(Vector<T> embeddings)
    {
        var logits = new Vector<T>(_options.NumClasses);
        int embDim = Math.Min(embeddings.Length, _options.EmbeddingDimension);

        for (int c = 0; c < _options.NumClasses; c++)
        {
            T sum = NumOps.Zero;
            for (int e = 0; e < embDim; e++)
            {
                int paramIdx = c * _options.EmbeddingDimension + e;
                if (paramIdx < _classifierParams.Length)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(embeddings[e], _classifierParams[paramIdx]));
                }
            }
            logits[c] = sum;
        }

        return logits;
    }
}
