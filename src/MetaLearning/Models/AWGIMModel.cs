using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Models;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;

namespace AiDotNet.MetaLearning.Models;

/// <summary>
/// A task-adapted AWGIM classifier: the support set's embeddings and the meta-learned weight generator,
/// which writes a classifier for each example it is asked to classify.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// AWGIM's classifier depends on the query (the attentive path reads the support set from the query's
/// point of view), so adaptation cannot reduce to one fixed weight matrix. Prediction runs the generator
/// in evaluation mode: no dropout, the mean of each weight distribution.
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Attentive Weights Generation for Few Shot Learning via Information Maximization",
    "https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Attentive_Weights_Generation_for_Few_Shot_Learning_via_Information_Maximization_CVPR_2020_paper.pdf",
    Year = 2020,
    Authors = "Yiluan Guo, Ngai-Man Cheung")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class AWGIMModel<T, TInput, TOutput> : IModel<TInput, TOutput, ModelMetadata<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly IFullModel<T, TInput, TOutput> _featureEncoder;
    private readonly AwgimNetwork<T> _network;
    [TrainableParameter]
    private readonly Vector<T> _weights;
    private readonly Tensor<T> _support;
    private readonly Tensor<T> _membership;
    private readonly Tensor<T> _classAverager;
    private readonly int[] _classSlots;
    private readonly AWGIMOptions<T, TInput, TOutput> _options;

    internal AWGIMModel(
        IFullModel<T, TInput, TOutput> featureEncoder,
        AwgimNetwork<T> network,
        Vector<T> weights,
        Tensor<T> support,
        Tensor<T> membership,
        Tensor<T> classAverager,
        int[] classSlots,
        AWGIMOptions<T, TInput, TOutput> options)
    {
        Guard.NotNull(featureEncoder);
        Guard.NotNull(network);
        Guard.NotNull(weights);
        Guard.NotNull(support);
        Guard.NotNull(membership);
        Guard.NotNull(classAverager);
        Guard.NotNull(classSlots);
        Guard.NotNull(options);
        if (weights.Length != network.ParameterCount)
            throw new ArgumentException($"{weights.Length} weights are not the generator's {network.ParameterCount}.", nameof(weights));

        _featureEncoder = featureEncoder.DeepCopy();
        _network = network;
        _weights = weights;
        _support = support;
        _membership = membership;
        _classAverager = classAverager;
        _classSlots = classSlots;
        _options = options;
    }

    /// <inheritdoc/>
    public ModelMetadata<T> Metadata { get; } = new ModelMetadata<T>();

    /// <summary>The number of classes the model scores.</summary>
    public int NumClasses => _options.NumClasses;

    /// <summary>
    /// Scores each input row. For a <see cref="Vector{T}"/> output, the predicted class index per row;
    /// otherwise class probabilities over <see cref="NumClasses"/> columns.
    /// </summary>
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

        var noise = AwgimNetwork<T>.Noise.None;
        var generated = _network.Generate(_network.CreateLeaves(_weights), _support, rows, _membership, _classAverager, noise);
        var logits = AwgimNetwork<T>.QueryLogits(engine, rows, generated.ClassWeights);
        var probabilities = engine.Softmax(logits, axis: 1);
        int classes = _classSlots.Length;

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
        => throw new NotSupportedException("Use the AWGIM algorithm to train the generator; adaptation is the support set.");

    /// <inheritdoc/>
    public void UpdateParameters(Vector<T> parameters)
        => throw new NotSupportedException("AWGIM model parameters are fixed at adaptation.");

    /// <summary>The feature encoder's parameters followed by the generator weights.</summary>
    public Vector<T> GetParameters()
    {
        var encoder = InterfaceGuard.Parameterizable(_featureEncoder).GetParameters();
        var combined = new Vector<T>(encoder.Length + _weights.Length);
        for (int i = 0; i < encoder.Length; i++) combined[i] = encoder[i];
        for (int i = 0; i < _weights.Length; i++) combined[encoder.Length + i] = _weights[i];
        return combined;
    }

    /// <inheritdoc/>
    public ModelMetadata<T> GetModelMetadata() => Metadata;
}
