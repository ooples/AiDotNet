using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Modules;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Models;

/// <summary>
/// Relation Network model for few-shot classification.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This model stores the adapted state of a Relation Network after seeing support examples.
/// It can then classify query examples by computing relation scores with the stored support set.
/// </para>
/// <para><b>For Beginners:</b> After the Relation Network sees the support examples (the few
/// labeled examples for each class), this model remembers them and uses them to classify
/// new query examples. It does this by computing how "related" the query is to each
/// support example.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("Learning to Compare: Relation Network for Few-Shot Learning", "https://arxiv.org/abs/1711.06025", Year = 2018, Authors = "Flood Sung, Yongxin Yang, Li Zhang, Tao Xiang, Philip H.S. Torr, Timothy M. Hospedales")]
public class RelationNetworkModel<T, TInput, TOutput> : MetaLearningModelBase<T, TInput, TOutput>
{
    private readonly RelationModule<T> _relationModule;
    private readonly TInput _supportInputs;
    private readonly TOutput _supportOutputs;
    private readonly RelationNetworkOptions<T, TInput, TOutput> _options;
    private readonly List<Vector<T>> _supportFeatures;
    private readonly List<int> _supportLabels;

    /// <summary>
    /// Initializes a new instance of the RelationNetworkModel.
    /// </summary>
    /// <param name="featureEncoder">The feature encoder network.</param>
    /// <param name="relationModule">The relation module for computing similarity.</param>
    /// <param name="supportInputs">The support set inputs.</param>
    /// <param name="supportOutputs">The support set labels.</param>
    /// <param name="options">The Relation Network options.</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    public RelationNetworkModel(
        IFullModel<T, TInput, TOutput> featureEncoder,
        RelationModule<T> relationModule,
        TInput supportInputs,
        TOutput supportOutputs,
        RelationNetworkOptions<T, TInput, TOutput> options)
        : base(featureEncoder)
    {
        Guard.NotNull(relationModule);
        _relationModule = relationModule;
        _supportInputs = supportInputs;
        _supportOutputs = supportOutputs;
        Guard.NotNull(options);
        _options = options;

        _supportFeatures = new List<Vector<T>>();
        _supportLabels = new List<int>();
        PrecomputeSupportFeatures();
    }

    private void PrecomputeSupportFeatures()
    {
        if (_supportInputs is Tensor<T> supportTensor)
        {
            int numSamples = supportTensor.Shape[0];
            int sampleSize = 1;
            for (int i = 1; i < supportTensor.Shape.Length; i++)
            {
                sampleSize *= supportTensor.Shape[i];
            }

            for (int i = 0; i < numSamples; i++)
            {
                var sampleTensor = new Tensor<T>(supportTensor.Shape.Skip(1).ToArray());
                for (int j = 0; j < sampleSize; j++)
                {
                    sampleTensor.SetFlat(j, supportTensor.GetFlat(i * sampleSize + j));
                }

                var encoded = EncodeSample(sampleTensor);
                _supportFeatures.Add(encoded);
            }
        }
        else if (_supportInputs is Matrix<T> supportMatrix)
        {
            for (int i = 0; i < supportMatrix.Rows; i++)
            {
                var row = supportMatrix.GetRow(i);
                var encoded = EncodeVector(row);
                _supportFeatures.Add(encoded);
            }
        }

        if (_supportOutputs is Vector<T> labelVector)
        {
            for (int i = 0; i < labelVector.Length; i++)
            {
                _supportLabels.Add((int)NumOps.ToDouble(labelVector[i]));
            }
        }
        else if (_supportOutputs is Tensor<T> labelTensor)
        {
            int numLabels = labelTensor.Shape[0];
            for (int i = 0; i < numLabels; i++)
            {
                _supportLabels.Add((int)NumOps.ToDouble(labelTensor.GetFlat(i)));
            }
        }
    }

    private Vector<T> EncodeSample(Tensor<T> sample)
    {
        if (sample is TInput input)
        {
            var output = BaseModel.Predict(input);
            return ConversionsHelper.ConvertToVector<T, TOutput>(output);
        }

        var vector = new Vector<T>(sample.Length);
        for (int i = 0; i < sample.Length; i++)
        {
            vector[i] = sample.GetFlat(i);
        }
        return vector;
    }

    private Vector<T> EncodeVector(Vector<T> sample)
    {
        if (sample is TInput input)
        {
            var output = BaseModel.Predict(input);
            return ConversionsHelper.ConvertToVector<T, TOutput>(output);
        }

        return sample;
    }

    /// <inheritdoc/>
    public override TOutput Predict(TInput input)
    {
        var queryOutput = BaseModel.Predict(input);
        var queryFeatures = ConversionsHelper.ConvertToVector<T, TOutput>(queryOutput);

        var relationScores = new List<T>();
        for (int i = 0; i < _supportFeatures.Count; i++)
        {
            var supportFeature = _supportFeatures[i];
            var score = ComputeRelationScore(queryFeatures, supportFeature);
            relationScores.Add(score);
        }

        var classScores = AggregateScoresByClass(relationScores);
        var probabilities = ApplySoftmax(classScores);
        return ConvertVectorToOutput(probabilities);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var encoderParams = BaseModel.GetParameters();
        var relationParams = _relationModule.GetParameters();

        var combined = new Vector<T>(encoderParams.Length + relationParams.Length);
        for (int i = 0; i < encoderParams.Length; i++)
            combined[i] = encoderParams[i];
        for (int i = 0; i < relationParams.Length; i++)
            combined[encoderParams.Length + i] = relationParams[i];

        return combined;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        Guard.NotNull(parameters);
        var encoderParams = BaseModel.GetParameters();
        var relationParams = _relationModule.GetParameters();
        int expectedLength = encoderParams.Length + relationParams.Length;
        if (parameters.Length != expectedLength)
        {
            throw new ArgumentException(
                $"Parameter count mismatch: expected {expectedLength}, got {parameters.Length}.",
                nameof(parameters));
        }

        var newEncoderParams = new Vector<T>(encoderParams.Length);
        for (int i = 0; i < encoderParams.Length; i++)
        {
            newEncoderParams[i] = parameters[i];
        }
        BaseModel.SetParameters(newEncoderParams);

        var newRelationParams = new Vector<T>(relationParams.Length);
        for (int i = 0; i < relationParams.Length; i++)
        {
            newRelationParams[i] = parameters[encoderParams.Length + i];
        }
        _relationModule.SetParameters(newRelationParams);
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
        return new RelationNetworkModel<T, TInput, TOutput>(
            clonedEncoder, _relationModule, _supportInputs, _supportOutputs, _options);
    }

    private T ComputeRelationScore(Vector<T> queryFeatures, Vector<T> supportFeatures)
    {
        var combinedFeatures = ConcatenateFeatures(queryFeatures, supportFeatures);
        var scoreOutput = _relationModule.Forward(combinedFeatures);
        return scoreOutput.GetFlat(0);
    }

    private Tensor<T> ConcatenateFeatures(Vector<T> query, Vector<T> support)
    {
        int totalLength = query.Length + support.Length;
        var combined = new Tensor<T>(new int[] { totalLength });

        for (int i = 0; i < query.Length; i++)
        {
            combined.SetFlat(i, query[i]);
        }
        for (int i = 0; i < support.Length; i++)
        {
            combined.SetFlat(query.Length + i, support[i]);
        }

        return combined;
    }

    private Vector<T> AggregateScoresByClass(List<T> relationScores)
    {
        var classScores = new Vector<T>(_options.NumClasses);
        var classCounts = new int[_options.NumClasses];

        for (int c = 0; c < _options.NumClasses; c++)
        {
            classScores[c] = NumOps.Zero;
        }

        for (int i = 0; i < relationScores.Count && i < _supportLabels.Count; i++)
        {
            int classIdx = _supportLabels[i];
            if (classIdx >= 0 && classIdx < _options.NumClasses)
            {
                classScores[classIdx] = NumOps.Add(classScores[classIdx], relationScores[i]);
                classCounts[classIdx]++;
            }
        }

        for (int c = 0; c < _options.NumClasses; c++)
        {
            if (classCounts[c] > 0)
            {
                classScores[c] = NumOps.Divide(classScores[c], NumOps.FromDouble(classCounts[c]));
            }
        }

        return classScores;
    }

    private Vector<T> ApplySoftmax(Vector<T> scores)
    {
        var probabilities = new Vector<T>(scores.Length);

        T maxScore = scores[0];
        for (int i = 1; i < scores.Length; i++)
        {
            if (NumOps.GreaterThan(scores[i], maxScore))
            {
                maxScore = scores[i];
            }
        }

        T sum = NumOps.Zero;
        for (int i = 0; i < scores.Length; i++)
        {
            double expValue = Math.Exp(NumOps.ToDouble(scores[i]) - NumOps.ToDouble(maxScore));
            probabilities[i] = NumOps.FromDouble(expValue);
            sum = NumOps.Add(sum, probabilities[i]);
        }

        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            for (int i = 0; i < probabilities.Length; i++)
            {
                probabilities[i] = NumOps.Divide(probabilities[i], sum);
            }
        }

        return probabilities;
    }
}
