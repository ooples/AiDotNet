using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Tensors.LinearAlgebra;
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
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("TADAM: Task Dependent Adaptive Metric for Improved Few-Shot Learning",
    "https://arxiv.org/abs/1805.10123",
    Year = 2018,
    Authors = "Oreshkin, B. N., Rodriguez, P., & Lacoste, A.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class TADAMModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    protected static IEngine Engine => AiDotNetEngine.Current;

    private readonly IFullModel<T, TInput, TOutput> _featureEncoder;
    private readonly Dictionary<int, Tensor<T>> _prototypes;
    [AiDotNet.Attributes.TrainableParameter]
    private readonly Vector<T> _metricScale;
    private readonly T _temperature;
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
    {
        Guard.NotNull(featureEncoder);
        _featureEncoder = featureEncoder;
        Guard.NotNull(prototypes);
        _prototypes = prototypes;
        Guard.NotNull(metricScale);
        _metricScale = metricScale;
        _temperature = temperature;
        Guard.NotNull(options);
        _options = options;
    }

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <inheritdoc/>
    /// <remarks>
    /// One score row per example, <c>[rows, NumClasses]</c>: <c>softmax_k(-||s (q - c_k)||^2 / temperature)</c> over the
    /// stored prototypes, with <c>s</c> the task's per-dimension scale when one is supplied for every feature. A class
    /// with no prototype gets probability 0 (it used to be scored at distance 0, the most probable of all).
    /// </remarks>
    public TOutput Predict(TInput input)
    {
        var rows = PrototypeMetric<T>.Normalized(
            ClassifierOutputs<T>.AsRows(_featureEncoder.Predict(input)), _options.NormalizeEmbeddings);
        int count = rows.Shape[0], width = rows.Shape[1];

        var slots = _prototypes.Keys.Where(k => k >= 0 && k < _options.NumClasses).OrderBy(k => k).ToArray();
        if (slots.Length == 0)
            return ClassifierOutputs<T>.ToOutput<TOutput>(new Tensor<T>(new[] { count, _options.NumClasses }));

        var prototypes = Engine.TensorConcatenate(
            slots.Select(k => _prototypes[k].Reshape(new[] { 1, width })).ToArray(), 0);

        // Metric scaling multiplies every query-prototype difference by s, i.e. compares s q with s c_k.
        if (_metricScale.Length == width)
        {
            var scale = Tensor<T>.FromVector(_metricScale, new[] { 1, width });
            rows = Engine.TensorMultiply(rows, scale);
            prototypes = Engine.TensorMultiply(prototypes, scale);
        }

        // Logit = -distance / temperature (negative because closer = higher probability); a non-positive temperature
        // would divide by zero, so it falls back to 1.
        double temperature = NumOps.ToDouble(_temperature);
        var logits = PrototypeMetric<T>.Logits(
            rows, prototypes, ProtoNetsDistanceFunction.Euclidean, 1.0, null, null, slots,
            temperature < 1e-10 ? 1.0 : temperature);
        var probabilities = Engine.TensorSoftmax(logits, 1);

        // Spread the present classes' columns over all NumClasses columns.
        var placement = Engine.TensorOneHot<T>(new Tensor<int>(slots, new[] { slots.Length }), _options.NumClasses);
        return ClassifierOutputs<T>.ToOutput<TOutput>(Engine.TensorMatMul(probabilities, placement));
    }

    /// <inheritdoc/>
    public void Train(TInput inputs, TOutput targets)
    {
        throw new NotSupportedException("Use the training algorithm to train TADAM.");
    }

    /// <inheritdoc/>
    public void UpdateParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("TADAM parameters are updated during training.");
    }

    /// <inheritdoc/>
    public Vector<T> GetParameters()
    {
        return InterfaceGuard.Parameterizable(_featureEncoder).GetParameters();
    }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata()
    {
        return Metadata;
    }
}
