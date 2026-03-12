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
/// BOIL model for few-shot classification with body-only adaptation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This model stores the adapted state of BOIL after inner-loop adaptation.
/// It contains the adapted feature extractor (body) and the frozen classification head.
/// </para>
/// <para><b>For Beginners:</b> After BOIL adapts to a new task by training only
/// the feature extractor (body) on support examples, this model stores:
/// </para>
/// <list type="bullet">
/// <item>The adapted body (feature extractor) specific to this task</item>
/// <item>The frozen head weights from meta-training</item>
/// <item>The frozen head bias (if used)</item>
/// </list>
/// <para>
/// When making predictions, the model extracts features using the adapted body
/// and classifies using the frozen head. This is the opposite of ANIL which
/// freezes the body and adapts the head.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("BOIL: Towards Representation Change for Few-shot Learning", "https://arxiv.org/abs/2008.08882", Year = 2021, Authors = "Jaehoon Oh, Hyungjun Yoo, ChangHwan Kim, Se-Young Yun")]
public class BOILModel<T, TInput, TOutput> : MetaLearningModelBase<T, TInput, TOutput>
{
    private Vector<T> _adaptedBodyParams;
    private Vector<T> _headWeights;
    private Vector<T>? _headBias;
    private readonly BOILOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the BOILModel.
    /// </summary>
    /// <param name="baseModel">The base model to use for feature extraction.</param>
    /// <param name="adaptedBodyParams">The adapted body parameters for this task.</param>
    /// <param name="headWeights">The frozen head weight parameters.</param>
    /// <param name="headBias">The frozen head bias parameters (optional).</param>
    /// <param name="options">The BOIL options.</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    public BOILModel(
        IFullModel<T, TInput, TOutput> baseModel,
        Vector<T> adaptedBodyParams,
        Vector<T> headWeights,
        Vector<T>? headBias,
        BOILOptions<T, TInput, TOutput> options)
        : base(baseModel)
    {
        Guard.NotNull(adaptedBodyParams);
        _adaptedBodyParams = adaptedBodyParams;
        Guard.NotNull(headWeights);
        _headWeights = headWeights;
        _headBias = headBias;
        Guard.NotNull(options);
        _options = options;
    }

    /// <summary>
    /// Gets the adapted body parameters.
    /// </summary>
    public Vector<T> AdaptedBodyParams => _adaptedBodyParams;

    /// <summary>
    /// Gets the frozen head weights.
    /// </summary>
    public Vector<T> HeadWeights => _headWeights;

    /// <summary>
    /// Gets the frozen head bias (may be null if not used).
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
        ApplyAdaptedBodyParameters();
        var features = ExtractFeaturesFromBaseModel(input, _options.FeatureDimension);
        var logits = ComputeLogits(features);
        return ConvertVectorToOutput(logits);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        int totalSize = _adaptedBodyParams.Length + _headWeights.Length + (_headBias?.Length ?? 0);
        var combined = new Vector<T>(totalSize);

        int idx = 0;
        for (int i = 0; i < _adaptedBodyParams.Length; i++)
        {
            combined[idx++] = _adaptedBodyParams[i];
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
        int expectedLength = _adaptedBodyParams.Length + _headWeights.Length + (_headBias?.Length ?? 0);
        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException(
                $"Parameter count mismatch: expected {expectedLength}, got {parameters.Length}.",
                nameof(parameters));
        }

        int idx = 0;
        for (int i = 0; i < _adaptedBodyParams.Length; i++)
        {
            _adaptedBodyParams[i] = parameters[idx++];
        }
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
        var clonedBase = BaseModel.DeepCopy();
        return new BOILModel<T, TInput, TOutput>(
            clonedBase, _adaptedBodyParams.Clone(), _headWeights.Clone(),
            _headBias?.Clone(), _options);
    }

    private void ApplyAdaptedBodyParameters()
    {
        var currentParams = BaseModel.GetParameters();
        var updatedParams = new Vector<T>(currentParams.Length);

        int copyLen = Math.Min(_adaptedBodyParams.Length, currentParams.Length);
        for (int i = 0; i < copyLen; i++)
        {
            updatedParams[i] = _adaptedBodyParams[i];
        }

        for (int i = copyLen; i < currentParams.Length; i++)
        {
            updatedParams[i] = currentParams[i];
        }

        BaseModel.SetParameters(updatedParams);
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
