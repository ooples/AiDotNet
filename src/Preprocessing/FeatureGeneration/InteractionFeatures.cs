using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureGeneration;

/// <summary>
/// Generates pairwise interaction features between input features.
/// </summary>
/// <remarks>
/// <para>
/// InteractionFeatures creates new features by multiplying pairs of existing features.
/// Unlike PolynomialFeatures with degree=2, this only produces interaction terms,
/// not squared terms.
/// </para>
/// <para>
/// For features [a, b, c], this produces: [ab, ac, bc]
/// </para>
/// <para><b>For Beginners:</b> Interaction features capture combined effects:
/// - If both "age" and "income" matter together (not just separately)
/// - Creating age × income might help the model
/// - This is simpler than full polynomial features (no squared terms)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.FeatureGenerator)]
[PipelineStage(PipelineStage.Preprocessing)]
public class InteractionFeatures<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly bool _includeOriginal;
    private readonly InteractionType _interactionType;

    // Fitted parameters
    private int _nInputFeatures;
    private int _nOutputFeatures;
    private List<(int, int)>? _interactionPairs;

    /// <summary>
    /// Gets whether original features are included in output.
    /// </summary>
    public bool IncludeOriginal => _includeOriginal;

    /// <summary>
    /// Gets the type of interactions generated.
    /// </summary>
    public InteractionType InteractionType => _interactionType;

    /// <summary>
    /// Gets the number of output features.
    /// </summary>
    public int NOutputFeatures => _nOutputFeatures;

    /// <summary>
    /// Gets the interaction pairs (feature indices).
    /// </summary>
    public List<(int, int)>? InteractionPairs => _interactionPairs;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="InteractionFeatures{T}"/>.
    /// </summary>
    /// <param name="includeOriginal">Whether to include original features in output. Defaults to true.</param>
    /// <param name="interactionType">Type of interactions to generate. Defaults to Pairwise.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public InteractionFeatures(
        bool includeOriginal = true,
        InteractionType interactionType = InteractionType.Pairwise,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _includeOriginal = includeOriginal;
        _interactionType = interactionType;
    }

    /// <summary>
    /// Fits the transformer by computing interaction pairs.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        // Generate interaction pairs based on type
        _interactionPairs = new List<(int, int)>();

        switch (_interactionType)
        {
            case InteractionType.Pairwise:
                // All pairs without self-interaction
                for (int i = 0; i < columnsToProcess.Length; i++)
                {
                    for (int j = i + 1; j < columnsToProcess.Length; j++)
                    {
                        _interactionPairs.Add((columnsToProcess[i], columnsToProcess[j]));
                    }
                }
                break;

            case InteractionType.WithSelf:
                // All pairs including self-interaction (squared terms)
                for (int i = 0; i < columnsToProcess.Length; i++)
                {
                    for (int j = i; j < columnsToProcess.Length; j++)
                    {
                        _interactionPairs.Add((columnsToProcess[i], columnsToProcess[j]));
                    }
                }
                break;

            case InteractionType.AllPairs:
                // All ordered pairs (a*b and b*a both included)
                for (int i = 0; i < columnsToProcess.Length; i++)
                {
                    for (int j = 0; j < columnsToProcess.Length; j++)
                    {
                        if (i != j)
                        {
                            _interactionPairs.Add((columnsToProcess[i], columnsToProcess[j]));
                        }
                    }
                }
                break;
        }

        _nOutputFeatures = _interactionPairs.Count;
        if (_includeOriginal)
        {
            _nOutputFeatures += _nInputFeatures;
        }

        IsFitted = true;
    }

    /// <summary>
    /// Transforms data by generating interaction features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_interactionPairs is null)
        {
            throw new InvalidOperationException("InteractionFeatures has not been fitted.");
        }

        if (data.Columns != _nInputFeatures)
        {
            throw new ArgumentException(
                $"Input data has {data.Columns} columns, but the transformer was fitted with {_nInputFeatures} columns.",
                nameof(data));
        }

        var engine = Engine;
        int numRows = data.Rows;
        var x = Tensor<T>.FromMatrix(data);
        var blocks = new List<Tensor<T>>(2);

        // Include original features if requested
        if (_includeOriginal)
            blocks.Add(x);

        // Generate interaction features: column idx1 times column idx2 for every pair
        int pairs = _interactionPairs.Count;
        if (pairs > 0 && numRows > 0)
        {
            var first = new int[pairs];
            var second = new int[pairs];
            for (int p = 0; p < pairs; p++)
            {
                first[p] = _interactionPairs[p].Item1;
                second[p] = _interactionPairs[p].Item2;
            }

            blocks.Add(engine.TensorMultiply(
                engine.TensorIndexSelect(x, new Tensor<int>(first, new[] { pairs }), 1),
                engine.TensorIndexSelect(x, new Tensor<int>(second, new[] { pairs }), 1)));
        }

        if (numRows == 0 || blocks.Count == 0)
            return new Matrix<T>(numRows, _nOutputFeatures);
        return (blocks.Count == 1 ? blocks[0] : engine.TensorConcatenate(blocks.ToArray(), 1)).ToMatrix();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("InteractionFeatures does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_interactionPairs is null)
        {
            return Array.Empty<string>();
        }

        var names = new List<string>();

        // Generate default names if not provided
        inputFeatureNames ??= Enumerable.Range(0, _nInputFeatures).Select(i => $"x{i}").ToArray();

        // Original feature names
        if (_includeOriginal)
        {
            for (int j = 0; j < _nInputFeatures; j++)
            {
                names.Add(j < inputFeatureNames.Length ? inputFeatureNames[j] : $"x{j}");
            }
        }

        // Interaction feature names
        foreach (var (idx1, idx2) in _interactionPairs)
        {
            string name1 = idx1 < inputFeatureNames.Length ? inputFeatureNames[idx1] : $"x{idx1}";
            string name2 = idx2 < inputFeatureNames.Length ? inputFeatureNames[idx2] : $"x{idx2}";
            names.Add($"{name1}*{name2}");
        }

        return names.ToArray();
    }
}

/// <summary>
/// Specifies the type of feature interactions to generate.
/// </summary>
public enum InteractionType
{
    /// <summary>
    /// Pairwise interactions only (a×b, a×c, b×c). No self-interaction.
    /// </summary>
    Pairwise,

    /// <summary>
    /// Pairwise plus self-interaction (includes a², b², c²).
    /// </summary>
    WithSelf,

    /// <summary>
    /// All ordered pairs (a×b and b×a both included).
    /// </summary>
    AllPairs
}
