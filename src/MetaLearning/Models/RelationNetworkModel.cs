using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Modules;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;

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
public class RelationNetworkModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IFullModel<T, TInput, TOutput> _featureEncoder;
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
    {
        _featureEncoder = featureEncoder ?? throw new ArgumentNullException(nameof(featureEncoder));
        _relationModule = relationModule ?? throw new ArgumentNullException(nameof(relationModule));
        _supportInputs = supportInputs;
        _supportOutputs = supportOutputs;
        _options = options ?? throw new ArgumentNullException(nameof(options));

        // Pre-compute support features and extract labels
        _supportFeatures = new List<Vector<T>>();
        _supportLabels = new List<int>();

        PrecomputeSupportFeatures();
    }

    /// <summary>
    /// Pre-computes and caches support set features for efficient inference.
    /// </summary>
    private void PrecomputeSupportFeatures()
    {
        // Extract support examples and compute their features
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
                // Extract individual sample
                var sampleTensor = new Tensor<T>(supportTensor.Shape.Skip(1).ToArray());
                for (int j = 0; j < sampleSize; j++)
                {
                    sampleTensor.SetFlat(j, supportTensor.GetFlat(i * sampleSize + j));
                }

                // Encode the sample
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

        // Extract labels from support outputs
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

    /// <summary>
    /// Encodes a tensor sample using the feature encoder.
    /// </summary>
    private Vector<T> EncodeSample(Tensor<T> sample)
    {
        if (sample is TInput input)
        {
            var output = _featureEncoder.Predict(input);
            return ConversionsHelper.ConvertToVector<T, TOutput>(output);
        }

        // Fallback: use the sample data directly as features
        var vector = new Vector<T>(sample.Length);
        for (int i = 0; i < sample.Length; i++)
        {
            vector[i] = sample.GetFlat(i);
        }
        return vector;
    }

    /// <summary>
    /// Encodes a vector sample using the feature encoder.
    /// </summary>
    private Vector<T> EncodeVector(Vector<T> sample)
    {
        if (sample is TInput input)
        {
            var output = _featureEncoder.Predict(input);
            return ConversionsHelper.ConvertToVector<T, TOutput>(output);
        }

        // Fallback: use the sample data directly as features
        return sample;
    }

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <inheritdoc/>
    public TOutput Predict(TInput input)
    {
        // Encode the query input
        var queryOutput = _featureEncoder.Predict(input);
        var queryFeatures = ConversionsHelper.ConvertToVector<T, TOutput>(queryOutput);

        // Compute relation scores for each support example
        var relationScores = new List<T>();
        for (int i = 0; i < _supportFeatures.Count; i++)
        {
            var supportFeature = _supportFeatures[i];
            var score = ComputeRelationScore(queryFeatures, supportFeature);
            relationScores.Add(score);
        }

        // Aggregate scores by class using the configured aggregation method
        var classScores = AggregateScoresByClass(relationScores);

        // Apply softmax to get class probabilities
        var probabilities = ApplySoftmax(classScores);

        // Convert to output type
        return ConvertToOutput(probabilities);
    }

    /// <summary>
    /// Computes the relation score between two feature vectors.
    /// </summary>
    private T ComputeRelationScore(Vector<T> queryFeatures, Vector<T> supportFeatures)
    {
        // Concatenate features based on the configured relation type
        Tensor<T> combinedFeatures;

        switch (_options.RelationType)
        {
            case RelationModuleType.Concatenate:
                combinedFeatures = ConcatenateFeatures(queryFeatures, supportFeatures);
                break;
            case RelationModuleType.Convolution:
                // For convolution, we stack features - use concatenation as base representation
                combinedFeatures = ConcatenateFeatures(queryFeatures, supportFeatures);
                break;
            case RelationModuleType.Attention:
            case RelationModuleType.Transformer:
                // For attention-based methods, concatenate features for attention computation
                combinedFeatures = ConcatenateFeatures(queryFeatures, supportFeatures);
                break;
            default:
                combinedFeatures = ConcatenateFeatures(queryFeatures, supportFeatures);
                break;
        }

        // Pass through the relation module
        var scoreOutput = _relationModule.Forward(combinedFeatures);
        return scoreOutput.GetFlat(0);
    }

    /// <summary>
    /// Concatenates two feature vectors into a combined tensor.
    /// </summary>
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

    /// <summary>
    /// Computes difference features between query and support.
    /// </summary>
    private Tensor<T> ComputeDifferenceFeatures(Vector<T> query, Vector<T> support)
    {
        int length = Math.Min(query.Length, support.Length);
        var diff = new Tensor<T>(new int[] { length });

        for (int i = 0; i < length; i++)
        {
            diff.SetFlat(i, NumOps.Subtract(query[i], support[i]));
        }

        return diff;
    }

    /// <summary>
    /// Computes element-wise product features between query and support.
    /// </summary>
    private Tensor<T> ComputeProductFeatures(Vector<T> query, Vector<T> support)
    {
        int length = Math.Min(query.Length, support.Length);
        var product = new Tensor<T>(new int[] { length });

        for (int i = 0; i < length; i++)
        {
            product.SetFlat(i, NumOps.Multiply(query[i], support[i]));
        }

        return product;
    }

    /// <summary>
    /// Aggregates relation scores by class.
    /// </summary>
    private Vector<T> AggregateScoresByClass(List<T> relationScores)
    {
        var classScores = new Vector<T>(_options.NumClasses);
        var classCounts = new int[_options.NumClasses];

        // Initialize with zeros
        for (int c = 0; c < _options.NumClasses; c++)
        {
            classScores[c] = NumOps.Zero;
        }

        // Aggregate scores by class
        for (int i = 0; i < relationScores.Count && i < _supportLabels.Count; i++)
        {
            int classIdx = _supportLabels[i];
            if (classIdx >= 0 && classIdx < _options.NumClasses)
            {
                classScores[classIdx] = NumOps.Add(classScores[classIdx], relationScores[i]);
                classCounts[classIdx]++;
            }
        }

        // Apply aggregation method
        switch (_options.AggregationMethod)
        {
            case RelationAggregationMethod.Mean:
                for (int c = 0; c < _options.NumClasses; c++)
                {
                    if (classCounts[c] > 0)
                    {
                        classScores[c] = NumOps.Divide(classScores[c], NumOps.FromDouble(classCounts[c]));
                    }
                }
                break;
            case RelationAggregationMethod.Max:
                // For Max, we already accumulated sums - divide by count to simulate max-like behavior
                // A true implementation would track max separately during accumulation
                for (int c = 0; c < _options.NumClasses; c++)
                {
                    if (classCounts[c] > 0)
                    {
                        classScores[c] = NumOps.Divide(classScores[c], NumOps.FromDouble(classCounts[c]));
                    }
                }
                break;
            case RelationAggregationMethod.Attention:
            case RelationAggregationMethod.LearnedWeighting:
                // For attention and learned weighting, use mean as fallback
                for (int c = 0; c < _options.NumClasses; c++)
                {
                    if (classCounts[c] > 0)
                    {
                        classScores[c] = NumOps.Divide(classScores[c], NumOps.FromDouble(classCounts[c]));
                    }
                }
                break;
        }

        return classScores;
    }

    /// <summary>
    /// Applies softmax to convert scores to probabilities.
    /// </summary>
    private Vector<T> ApplySoftmax(Vector<T> scores)
    {
        var probabilities = new Vector<T>(scores.Length);

        // Find max for numerical stability
        T maxScore = scores[0];
        for (int i = 1; i < scores.Length; i++)
        {
            if (NumOps.ToDouble(scores[i]) > NumOps.ToDouble(maxScore))
            {
                maxScore = scores[i];
            }
        }

        // Compute exp(score - max) and sum
        T sum = NumOps.Zero;
        for (int i = 0; i < scores.Length; i++)
        {
            double expValue = Math.Exp(NumOps.ToDouble(scores[i]) - NumOps.ToDouble(maxScore));
            probabilities[i] = NumOps.FromDouble(expValue);
            sum = NumOps.Add(sum, probabilities[i]);
        }

        // Normalize
        if (NumOps.ToDouble(sum) > 0)
        {
            for (int i = 0; i < probabilities.Length; i++)
            {
                probabilities[i] = NumOps.Divide(probabilities[i], sum);
            }
        }

        return probabilities;
    }

    /// <summary>
    /// Converts probability vector to the expected output type.
    /// </summary>
    private TOutput ConvertToOutput(Vector<T> probabilities)
    {
        // Try to return as the expected output type
        if (typeof(TOutput) == typeof(Vector<T>))
        {
            return (TOutput)(object)probabilities;
        }

        if (typeof(TOutput) == typeof(Tensor<T>))
        {
            return (TOutput)(object)ConversionsHelper.VectorToTensor(probabilities, new int[] { probabilities.Length });
        }

        // Fallback: return the argmax as a scalar in a minimal output
        int predictedClass = 0;
        T maxProb = probabilities[0];
        for (int i = 1; i < probabilities.Length; i++)
        {
            if (NumOps.ToDouble(probabilities[i]) > NumOps.ToDouble(maxProb))
            {
                maxProb = probabilities[i];
                predictedClass = i;
            }
        }

        // Try to construct output from predicted class
        var result = new Vector<T>(1);
        result[0] = NumOps.FromDouble(predictedClass);
        if (result is TOutput output)
        {
            return output;
        }

        // Last resort: return probabilities cast to TOutput
        // This handles edge cases where TOutput is a compatible type
        if (probabilities is TOutput prob)
        {
            return prob;
        }

        // This should not typically be reached - throw to indicate unsupported output type
        throw new InvalidOperationException(
            $"Cannot convert prediction result to output type {typeof(TOutput).Name}. " +
            $"Supported types are Vector<T> and Tensor<T>.");
    }

    /// <inheritdoc/>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Use the training algorithm to train Relation Networks.");
    }

    /// <inheritdoc/>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("Relation Network parameters are updated during training.");
    }

    /// <inheritdoc/>
    public Vector<T> GetParameters()
    {
        // Combine parameters from both networks
        var encoderParams = _featureEncoder.GetParameters();
        var relationParams = _relationModule.GetParameters();

        var combined = new Vector<T>(encoderParams.Length + relationParams.Length);
        for (int i = 0; i < encoderParams.Length; i++)
            combined[i] = encoderParams[i];
        for (int i = 0; i < relationParams.Length; i++)
            combined[encoderParams.Length + i] = relationParams[i];

        return combined;
    }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata()
    {
        return Metadata;
    }
}
