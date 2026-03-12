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
/// ANIL model for few-shot classification with head-only adaptation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This model stores the adapted state of ANIL after inner-loop adaptation.
/// It contains the frozen feature extractor (body) and the adapted classification head.
/// </para>
/// <para><b>For Beginners:</b> After ANIL adapts to a new task by training only
/// the classification head on support examples, this model stores:
/// </para>
/// <list type="bullet">
/// <item>The frozen body (feature extractor) from meta-training</item>
/// <item>The adapted head weights specific to this task</item>
/// <item>The adapted head bias (if used)</item>
/// </list>
/// <para>
/// When making predictions, the model extracts features using the frozen body
/// and classifies using the adapted head.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML", "https://arxiv.org/abs/1909.09157", Year = 2020, Authors = "Aniruddh Raghu, Maithra Raghu, Samy Bengio, Oriol Vinyals")]
public class ANILModel<T, TInput, TOutput> : MetaLearningModelBase<T, TInput, TOutput>
{
    private Vector<T> _headWeights;
    private Vector<T>? _headBias;
    private readonly ANILOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the ANILModel.
    /// </summary>
    /// <param name="featureExtractor">The feature extractor (frozen body).</param>
    /// <param name="headWeights">The adapted head weight parameters.</param>
    /// <param name="headBias">The adapted head bias parameters (optional).</param>
    /// <param name="options">The ANIL options.</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    public ANILModel(
        IFullModel<T, TInput, TOutput> featureExtractor,
        Vector<T> headWeights,
        Vector<T>? headBias,
        ANILOptions<T, TInput, TOutput> options)
        : base(featureExtractor)
    {
        Guard.NotNull(headWeights);
        _headWeights = headWeights;
        _headBias = headBias;
        Guard.NotNull(options);
        _options = options;
    }

    /// <summary>
    /// Gets the adapted head weights.
    /// </summary>
    public Vector<T> HeadWeights => _headWeights;

    /// <summary>
    /// Gets the adapted head bias (may be null if not used).
    /// </summary>
    public Vector<T>? HeadBias => _headBias;

    /// <summary>
    /// Gets the number of classes this model is adapted for.
    /// </summary>
    public int NumClasses => _options.NumClasses;

    /// <summary>
    /// Gets the feature dimension expected by the head.
    /// </summary>
    public int FeatureDimension => _options.FeatureDimension;

    /// <inheritdoc/>
    public override TOutput Predict(TInput input)
    {
        var features = ExtractFeaturesFromBaseModel(input, _options.FeatureDimension);
        var logits = ComputeLogits(features);
        return ConvertVectorToOutput(logits);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var bodyParams = BaseModel.GetParameters();
        int totalSize = bodyParams.Length + _headWeights.Length + (_headBias?.Length ?? 0);
        var combined = new Vector<T>(totalSize);

        int idx = 0;
        for (int i = 0; i < bodyParams.Length; i++)
        {
            combined[idx++] = bodyParams[i];
        }
        for (int i = 0; i < _headWeights.Length; i++)
        {
            combined[idx++] = _headWeights[i];
        }
        if (_headBias is not null)
        {
            for (int i = 0; i < _headBias.Length; i++)
            {
                combined[idx++] = _headBias[i];
            }
        }

        return combined;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        Guard.NotNull(parameters);
        var bodyParams = BaseModel.GetParameters();
        int expectedLength = bodyParams.Length + _headWeights.Length + (_headBias?.Length ?? 0);
        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException(
                $"Parameter count mismatch: expected {expectedLength}, got {parameters.Length}.",
                nameof(parameters));
        }

        int idx = 0;
        var newBodyParams = new Vector<T>(bodyParams.Length);
        for (int i = 0; i < bodyParams.Length; i++)
        {
            newBodyParams[i] = parameters[idx++];
        }
        BaseModel.SetParameters(newBodyParams);

        for (int i = 0; i < _headWeights.Length; i++)
        {
            _headWeights[i] = parameters[idx++];
        }
        if (_headBias is not null)
        {
            for (int i = 0; i < _headBias.Length; i++)
            {
                _headBias[i] = parameters[idx++];
            }
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
        var clonedBody = BaseModel.DeepCopy();
        var clonedHeadWeights = _headWeights.Clone();
        var clonedHeadBias = _headBias?.Clone();
        return new ANILModel<T, TInput, TOutput>(clonedBody, clonedHeadWeights, clonedHeadBias, _options);
    }

    private Vector<T> ComputeLogits(Vector<T> features)
    {
        var logits = new Vector<T>(_options.NumClasses);
        int featureDim = Math.Min(features.Length, _options.FeatureDimension);

        for (int c = 0; c < _options.NumClasses; c++)
        {
            T sum = NumOps.Zero;
            for (int f = 0; f < featureDim; f++)
            {
                int weightIdx = c * _options.FeatureDimension + f;
                if (weightIdx < _headWeights.Length)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(features[f], _headWeights[weightIdx]));
                }
            }

            if (_headBias is not null && c < _headBias.Length)
            {
                sum = NumOps.Add(sum, _headBias[c]);
            }

            logits[c] = sum;
        }

        return logits;
    }
}
