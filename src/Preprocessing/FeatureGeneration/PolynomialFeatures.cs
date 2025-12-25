using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureGeneration;

/// <summary>
/// Generates polynomial and interaction features.
/// </summary>
/// <remarks>
/// <para>
/// PolynomialFeatures generates a new feature matrix consisting of all polynomial combinations
/// of the features with degree less than or equal to the specified degree. For example, with
/// degree=2 and input [a, b], generates [1, a, b, a², ab, b²].
/// </para>
/// <para><b>For Beginners:</b> This transformer creates new features from existing ones:
/// - Polynomial terms: a, a², a³, etc.
/// - Interaction terms: a*b, a*b*c, etc.
/// - Useful for capturing non-linear relationships
///
/// Example with degree=2 and features [x₁, x₂]:
/// Output: [1, x₁, x₂, x₁², x₁x₂, x₂²]
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
public class PolynomialFeatures<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _degree;
    private readonly bool _interactionOnly;
    private readonly bool _includeBias;

    // Fitted parameters
    private int _nInputFeatures;
    private int _nOutputFeatures;
    private List<int[]>? _powers;

    /// <summary>
    /// Gets the degree of polynomial features.
    /// </summary>
    public int Degree => _degree;

    /// <summary>
    /// Gets whether only interaction features are generated.
    /// </summary>
    public bool InteractionOnly => _interactionOnly;

    /// <summary>
    /// Gets whether a bias (constant) column is included.
    /// </summary>
    public bool IncludeBias => _includeBias;

    /// <summary>
    /// Gets the number of input features.
    /// </summary>
    public int NInputFeatures => _nInputFeatures;

    /// <summary>
    /// Gets the number of output features after transformation.
    /// </summary>
    public int NOutputFeatures => _nOutputFeatures;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    /// <remarks>
    /// Inverse transform is not supported because polynomial feature generation is not reversible.
    /// </remarks>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="PolynomialFeatures{T}"/>.
    /// </summary>
    /// <param name="degree">The maximum degree of polynomial features. Defaults to 2.</param>
    /// <param name="interactionOnly">If true, only interaction features are produced (no x², x³, etc.). Defaults to false.</param>
    /// <param name="includeBias">If true, includes a bias (constant 1) column. Defaults to true.</param>
    /// <param name="columnIndices">The column indices to use for polynomial generation. If null, uses all columns.</param>
    public PolynomialFeatures(
        int degree = 2,
        bool interactionOnly = false,
        bool includeBias = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (degree < 1)
        {
            throw new ArgumentException("Degree must be at least 1.", nameof(degree));
        }

        _degree = degree;
        _interactionOnly = interactionOnly;
        _includeBias = includeBias;
    }

    /// <summary>
    /// Learns the polynomial feature combinations from the training data.
    /// </summary>
    /// <param name="data">The training data matrix where each column is a feature.</param>
    protected override void FitCore(Matrix<T> data)
    {
        var columnsToProcess = GetColumnsToProcess(data.Columns);
        _nInputFeatures = columnsToProcess.Length;

        // Generate all power combinations
        _powers = new List<int[]>();

        // Generate powers for each degree from 0 (or 1) to _degree
        int minDegree = _includeBias ? 0 : 1;

        for (int d = minDegree; d <= _degree; d++)
        {
            GenerateCombinations(_nInputFeatures, d, _powers);
        }

        _nOutputFeatures = _powers.Count;
    }

    private void GenerateCombinations(int nFeatures, int degree, List<int[]> result)
    {
        if (degree == 0)
        {
            result.Add(new int[nFeatures]); // All zeros = bias term
            return;
        }

        // Generate all combinations of powers that sum to 'degree'
        var current = new int[nFeatures];
        GenerateCombinationsRecursive(nFeatures, degree, 0, current, result);
    }

    private void GenerateCombinationsRecursive(int nFeatures, int remaining, int startIdx, int[] current, List<int[]> result)
    {
        if (startIdx == nFeatures - 1)
        {
            current[startIdx] = remaining;
            if (IsValidCombination(current))
            {
                result.Add((int[])current.Clone());
            }
            current[startIdx] = 0;
            return;
        }

        int maxPower = _interactionOnly ? Math.Min(1, remaining) : remaining;
        for (int power = 0; power <= maxPower; power++)
        {
            current[startIdx] = power;
            GenerateCombinationsRecursive(nFeatures, remaining - power, startIdx + 1, current, result);
        }
        current[startIdx] = 0;
    }

    private bool IsValidCombination(int[] powers)
    {
        if (_interactionOnly)
        {
            // For interaction only, each power must be 0 or 1
            foreach (int p in powers)
            {
                if (p > 1) return false;
            }
        }
        return true;
    }

    /// <summary>
    /// Transforms the data by generating polynomial features.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The data with polynomial features.</returns>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_powers is null)
        {
            throw new InvalidOperationException("Transformer has not been fitted.");
        }

        int numRows = data.Rows;
        var columnsToProcess = GetColumnsToProcess(data.Columns);
        var result = new T[numRows, _nOutputFeatures];

        for (int i = 0; i < numRows; i++)
        {
            // Extract relevant features
            var features = new T[_nInputFeatures];
            for (int j = 0; j < _nInputFeatures; j++)
            {
                features[j] = data[i, columnsToProcess[j]];
            }

            // Compute each polynomial term
            for (int p = 0; p < _powers.Count; p++)
            {
                T term = NumOps.One;
                for (int j = 0; j < _nInputFeatures; j++)
                {
                    for (int power = 0; power < _powers[p][j]; power++)
                    {
                        term = NumOps.Multiply(term, features[j]);
                    }
                }
                result[i, p] = term;
            }
        }

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Inverse transformation is not supported for polynomial features.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>Never returns - always throws.</returns>
    /// <exception cref="NotSupportedException">Always thrown because polynomial expansion is not reversible.</exception>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException(
            "PolynomialFeatures does not support inverse transformation. " +
            "Polynomial feature generation is not reversible.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names.</param>
    /// <returns>The output feature names with polynomial notation.</returns>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_powers is null)
        {
            return Array.Empty<string>();
        }

        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);
        string[] names = inputFeatureNames ??
            Enumerable.Range(0, _nInputFeatures).Select(i => $"x{i}").ToArray();

        // Map to actual column names
        if (inputFeatureNames is not null && columnsToProcess.Length < inputFeatureNames.Length)
        {
            names = columnsToProcess.Select(c => inputFeatureNames[c]).ToArray();
        }

        var outputNames = new string[_powers.Count];
        for (int p = 0; p < _powers.Count; p++)
        {
            var parts = new List<string>();
            for (int j = 0; j < _nInputFeatures; j++)
            {
                if (_powers[p][j] == 1)
                {
                    parts.Add(names[j]);
                }
                else if (_powers[p][j] > 1)
                {
                    parts.Add($"{names[j]}^{_powers[p][j]}");
                }
            }

            if (parts.Count == 0)
            {
                outputNames[p] = "1"; // Bias term
            }
            else
            {
                outputNames[p] = string.Join("*", parts);
            }
        }

        return outputNames;
    }
}
