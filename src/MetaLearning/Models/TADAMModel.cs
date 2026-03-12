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
/// TADAM model for few-shot classification with task conditioning and metric scaling.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This model stores the adapted state of TADAM after computing task-conditioned prototypes.
/// It uses learned metric scaling and temperature to classify new query examples.
/// </para>
/// <para><b>For Beginners:</b> After TADAM sees the support examples and computes
/// task-conditioned prototypes, this model stores those prototypes along with the
/// learned metric scaling parameters. It can then classify new examples by measuring
/// scaled distances to these prototypes.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ModelPaper("TADAM: Task dependent adaptive metric for improved few-shot learning", "https://arxiv.org/abs/1805.10123", Year = 2018, Authors = "Boris Oreshkin, Pau Rodriguez Lopez, Alexandre Lacoste")]
public class TADAMModel<T, TInput, TOutput> : MetaLearningModelBase<T, TInput, TOutput>
{
    private readonly Dictionary<int, Tensor<T>> _prototypes;
    private Vector<T> _metricScale;
    private T _temperature;
    private readonly TADAMOptions<T, TInput, TOutput> _options;

    /// <summary>
    /// Initializes a new instance of the TADAMModel.
    /// </summary>
    /// <param name="featureEncoder">The feature encoder network.</param>
    /// <param name="prototypes">The computed class prototypes.</param>
    /// <param name="metricScale">The learned metric scaling parameters.</param>
    /// <param name="temperature">The learned temperature parameter.</param>
    /// <param name="options">The TADAM options.</param>
    /// <exception cref="ArgumentNullException">Thrown when required parameters are null.</exception>
    public TADAMModel(
        IFullModel<T, TInput, TOutput> featureEncoder,
        Dictionary<int, Tensor<T>> prototypes,
        Vector<T> metricScale,
        T temperature,
        TADAMOptions<T, TInput, TOutput> options)
        : base(featureEncoder)
    {
        Guard.NotNull(prototypes);
        _prototypes = prototypes;
        Guard.NotNull(metricScale);
        _metricScale = metricScale;
        _temperature = temperature;
        Guard.NotNull(options);
        _options = options;
    }

    /// <inheritdoc/>
    public override TOutput Predict(TInput input)
    {
        var encoderOutput = BaseModel.Predict(input);
        var queryEmbedding = ConversionsHelper.ConvertToVector<T, TOutput>(encoderOutput);

        if (_options.NormalizeEmbeddings)
        {
            queryEmbedding = VectorHelper.Normalize(queryEmbedding);
        }

        var distances = ComputeScaledDistances(queryEmbedding);
        var logits = ComputeLogits(distances);
        var probabilities = ApplySoftmax(logits);
        return ConvertVectorToOutput(probabilities);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        return BaseModel.GetParameters();
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        Guard.NotNull(parameters);
        BaseModel.SetParameters(parameters);
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
        var clonedPrototypes = new Dictionary<int, Tensor<T>>();
        foreach (var kvp in _prototypes)
        {
            var clonedTensor = new Tensor<T>(kvp.Value.Shape);
            for (int i = 0; i < kvp.Value.Length; i++)
            {
                clonedTensor.SetFlat(i, kvp.Value.GetFlat(i));
            }
            clonedPrototypes[kvp.Key] = clonedTensor;
        }
        return new TADAMModel<T, TInput, TOutput>(
            clonedEncoder, clonedPrototypes, _metricScale.Clone(), _temperature, _options);
    }

    private Vector<T> ComputeScaledDistances(Vector<T> query)
    {
        var distances = new Vector<T>(_options.NumClasses);

        foreach (var kvp in _prototypes)
        {
            int classIdx = kvp.Key;
            var prototype = kvp.Value;

            if (classIdx >= 0 && classIdx < _options.NumClasses)
            {
                T distance = ComputeScaledDistance(query, prototype);
                distances[classIdx] = distance;
            }
        }

        return distances;
    }

    private T ComputeScaledDistance(Vector<T> query, Tensor<T> prototype)
    {
        T distanceSum = NumOps.Zero;
        int dim = Math.Min(query.Length, prototype.Length);

        for (int i = 0; i < dim; i++)
        {
            T diff = NumOps.Subtract(query[i], prototype.GetFlat(i));
            T scaledDiff = diff;

            if (_options.UseMetricScaling && i < _metricScale.Length)
            {
                scaledDiff = NumOps.Multiply(diff, _metricScale[i]);
            }

            distanceSum = NumOps.Add(distanceSum, NumOps.Multiply(scaledDiff, scaledDiff));
        }

        return distanceSum;
    }

    private Vector<T> ComputeLogits(Vector<T> distances)
    {
        var logits = new Vector<T>(distances.Length);
        double temp = NumOps.ToDouble(_temperature);
        if (temp < 1e-10)
        {
            temp = 1.0;
        }

        for (int i = 0; i < distances.Length; i++)
        {
            double distValue = NumOps.ToDouble(distances[i]);
            logits[i] = NumOps.FromDouble(-distValue / temp);
        }

        return logits;
    }

    private Vector<T> ApplySoftmax(Vector<T> logits)
    {
        var probabilities = new Vector<T>(logits.Length);

        T maxLogit = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (NumOps.GreaterThan(logits[i], maxLogit))
            {
                maxLogit = logits[i];
            }
        }

        T sum = NumOps.Zero;
        for (int i = 0; i < logits.Length; i++)
        {
            double expValue = Math.Exp(NumOps.ToDouble(logits[i]) - NumOps.ToDouble(maxLogit));
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
