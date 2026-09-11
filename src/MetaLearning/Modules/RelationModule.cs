using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

using AiDotNet.Models.Parameters;

namespace AiDotNet.MetaLearning.Modules;

/// <summary>
/// Relation module that computes similarity between feature pairs for Relation Networks.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// The relation module <c>g</c> of Sung et al. 2018 maps a pair of embeddings to a relation score in (0, 1). The
/// paper's module concatenates the pair and ends with a ReLU layer and a sigmoid unit (Figure 2); see
/// <see cref="RelationModuleType"/> for the other architectures. The input is the concatenation of two equally wide
/// embeddings, the sample's first, as one vector or one row per pair.
/// </para>
/// <para>
/// The weights are sized on the first forward pass, when the embedding width is known, and initialised as PyTorch
/// initialises <c>nn.Linear</c>. This used to be a single dot product with at most
/// <see cref="HiddenDimension"/> inputs and a sigmoid - no hidden layer, and the rest of the input ignored.
/// </para>
/// <para><b>For Beginners:</b> Instead of using a fixed formula to measure similarity
/// (like Euclidean distance), the relation module is a small neural network that LEARNS
/// how to compare examples. It takes two feature vectors as input (concatenated together)
/// and outputs a number between 0 and 1 indicating how related they are.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.MetaLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Learning to Compare: Relation Network for Few-Shot Learning",
    "https://arxiv.org/abs/1711.06025",
    Year = 2018,
    Authors = "Sung, F., Yang, Y., Zhang, L., Xiang, T., Torr, P. H. S., & Hospedales, T. M.")]
[ComponentType(ComponentType.MetaLearner)]
[PipelineStage(PipelineStage.Training)]
public partial class RelationModule<T> : ModelBase<T, Tensor<T>, Tensor<T>>
{

    /// <inheritdoc />
    /// <remarks>The relation module's weights; a restore of a different length resizes them to it.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new VectorFieldParameterSource<T>(
            () => _weights,
            value =>
            {
                if (value.Length != _weights.Length) _weights = new Vector<T>(value.Length);
                for (int i = 0; i < _weights.Length; i++) _weights[i] = value[i];
            }));
    }
    // NumOps inherited from ModelBase

    private int _hiddenDimension;
    private RelationModuleType _relationType;
    private int _inputWidth;
    [AiDotNet.Attributes.TrainableParameter]
    private Vector<T> _weights = new Vector<T>(0);
    private bool _isTraining;

    /// <summary>
    /// Initializes a new instance of RelationModule with the paper's concatenation architecture.
    /// </summary>
    /// <param name="hiddenDimension">Width of the pair representation before the sigmoid unit.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> The hidden dimension controls how complex the
    /// relation function can be. Larger values allow more complex comparisons
    /// but require more data to train effectively.
    /// </para>
    /// </remarks>
    public RelationModule(int hiddenDimension)
        : this(hiddenDimension, RelationModuleType.Concatenate)
    {
    }

    /// <summary>
    /// Initializes a new instance of RelationModule with the given architecture.
    /// </summary>
    /// <param name="hiddenDimension">Width of the pair representation before the sigmoid unit.</param>
    /// <param name="relationType">How the module combines the pair.</param>
    /// <exception cref="ArgumentOutOfRangeException">The hidden dimension is not positive.</exception>
    public RelationModule(int hiddenDimension, RelationModuleType relationType)
    {
        if (hiddenDimension <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenDimension), "The hidden dimension must be positive.");
        }

        _hiddenDimension = hiddenDimension;
        _relationType = relationType;
        _isTraining = false;
    }

    /// <summary>Gets the width of the pair representation before the sigmoid unit.</summary>
    public int HiddenDimension => _hiddenDimension;

    /// <summary>Gets how the module combines the pair.</summary>
    public RelationModuleType RelationType => _relationType;

    /// <summary>Gets the width of each embedding in a pair; zero until the first forward pass.</summary>
    public int InputWidth => _inputWidth;

    /// <summary>The live weight vector (no copy).</summary>
    internal Vector<T> Weights => _weights;

    /// <summary>
    /// Sizes and initialises the weights for embeddings of <paramref name="width"/>, once.
    /// </summary>
    /// <exception cref="InvalidOperationException">The module was already sized for another width.</exception>
    internal void EnsureInitialized(int width, Random random)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width), "Embeddings must be at least one wide.");
        if (_weights.Length > 0)
        {
            if (width != _inputWidth)
            {
                throw new InvalidOperationException(
                    $"This relation module compares {_inputWidth}-wide embeddings, not {width}-wide ones.");
            }

            return;
        }

        _inputWidth = width;
        _weights = new Vector<T>(RelationFunction<T>.ParameterCount(_relationType, width, _hiddenDimension));
        RelationFunction<T>.Initialize(_relationType, _weights, 0, width, _hiddenDimension, random);
    }

    /// <summary>
    /// Performs forward pass through the relation module.
    /// </summary>
    /// <param name="combinedFeatures">
    /// The pair's concatenated embeddings, sample first: <c>[2 * width]</c> for one pair or <c>[pairs, 2 * width]</c>.
    /// </param>
    /// <returns>Relation scores in (0, 1): <c>[1]</c> for one pair, <c>[pairs]</c> for rows.</returns>
    /// <exception cref="ArgumentException">The input is not one or more concatenated pairs.</exception>
    public Tensor<T> Forward(Tensor<T> combinedFeatures)
    {
        if (combinedFeatures is null) throw new ArgumentNullException(nameof(combinedFeatures));
        int rank = combinedFeatures.Shape.Length;
        if (rank != 1 && rank != 2)
        {
            throw new ArgumentException(
                $"A relation module reads one concatenated pair or one pair per row, not a rank-{rank} tensor.",
                nameof(combinedFeatures));
        }

        int pairWidth = combinedFeatures.Shape[rank - 1];
        if (pairWidth == 0 || pairWidth % 2 != 0)
        {
            throw new ArgumentException(
                $"A relation module reads the concatenation of two equally wide embeddings, but a pair here is "
                + $"{pairWidth} wide.", nameof(combinedFeatures));
        }

        int width = pairWidth / 2;
        EnsureInitialized(width, RandomHelper.CreateSecureRandom());
        var rows = rank == 1 ? combinedFeatures.Reshape(1, pairWidth) : combinedFeatures;
        int pairs = rows.Shape[0];

        using var noGrad = new NoGradScope<T>();
        var engine = AiDotNetEngine.Current;
        var sample = engine.TensorMatMul(rows, Half(pairWidth, width, 0));
        var query = engine.TensorMatMul(rows, Half(pairWidth, width, width));
        var scores = new RelationFunction<T>(_relationType, _weights, 0, width, _hiddenDimension).Scores(sample, query, null);
        return rank == 1 ? scores.Reshape(1) : scores.Reshape(pairs);
    }

    /// <summary>
    /// Sets the training mode.
    /// </summary>
    /// <param name="isTraining">True for training mode, false for inference mode.</param>
    public void SetTrainingMode(bool isTraining)
    {
        _isTraining = isTraining;
    }

    /// <summary>
    /// Creates a deep copy of the relation module.
    /// </summary>
    /// <returns>A new RelationModule with the same architecture and copied weights.</returns>
    public new RelationModule<T> Clone()
    {
        var cloned = new RelationModule<T>(_hiddenDimension, _relationType)
        {
            _inputWidth = _inputWidth,
            _weights = new Vector<T>(_weights.Length),
            _isTraining = _isTraining,
        };
        for (int i = 0; i < _weights.Length; i++)
        {
            cloned._weights[i] = _weights[i];
        }

        return cloned;
    }

    /// <summary><c>[pairWidth, width]</c>: picks the <paramref name="width"/> columns from <paramref name="start"/>.</summary>
    private Tensor<T> Half(int pairWidth, int width, int start)
    {
        var select = new Tensor<T>(new[] { pairWidth, width });
        for (int i = 0; i < width; i++) select[(start + i) * width + i] = NumOps.One;
        return select;
    }

    #region ModelBase Overrides

    /// <inheritdoc />
    public override Tensor<T> Predict(Tensor<T> input) => Forward(input);

    /// <inheritdoc />
    /// <remarks>
    /// A relation module is trained as part of a Relation Network, whose loss reaches it through the relation scores
    /// of a whole episode. This used to do nothing silently.
    /// </remarks>
    /// <exception cref="NotSupportedException">Always.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
        => throw new NotSupportedException(
            "A relation module is trained by RelationNetworkAlgorithm, through the relation scores of each episode.");

    /// <inheritdoc />
    public override ILossFunction<T> DefaultLossFunction => new MeanSquaredErrorLoss<T>();

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var copy = DeepCopy();
        ((IParameterizable<T, Tensor<T>, Tensor<T>>)copy).SetParameters(parameters);
        return copy;
    }

    #endregion
}
