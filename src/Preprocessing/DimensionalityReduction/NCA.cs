using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Neighborhood Components Analysis (NCA) for supervised dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// NCA is a supervised dimensionality reduction algorithm that learns a linear transformation
/// to maximize the expected leave-one-out classification accuracy in the transformed space.
/// It uses stochastic neighbor assignment for soft nearest-neighbor classification.
/// </para>
/// <para>
/// The algorithm:
/// 1. Define soft neighbor probabilities using softmax over distances
/// 2. Compute expected leave-one-out classification accuracy
/// 3. Optimize transformation matrix using gradient descent
/// 4. Project data using learned transformation
/// </para>
/// <para><b>For Beginners:</b> NCA learns a space where k-NN works well:
/// - Points with same label are pulled closer together
/// - Points with different labels are pushed apart
/// - The learned transformation is linear (a matrix)
/// - Can be used for feature extraction before classification
///
/// Use cases:
/// - Preprocessing for k-NN classifiers
/// - Metric learning for distance-based methods
/// - Visualization with class structure preserved
/// - Feature extraction for classification tasks
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class NCA<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _maxIter;
    private readonly double _learningRate;
    private readonly double _tol;
    private readonly int? _randomState;

    // Fitted parameters
    private Matrix<T>? _transformationMatrix; // A: d x p (projects from p to d dimensions)
    private int _nFeatures;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the learned transformation matrix.
    /// </summary>
    public Matrix<T>? TransformationMatrix => _transformationMatrix;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="NCA{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="maxIter">Maximum number of optimization iterations. Defaults to 100.</param>
    /// <param name="learningRate">Learning rate for gradient descent. Defaults to 0.01.</param>
    /// <param name="tol">Tolerance for convergence. Defaults to 1e-5.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public NCA(
        int nComponents = 2,
        int maxIter = 100,
        double learningRate = 0.01,
        double tol = 1e-5,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (maxIter < 1)
        {
            throw new ArgumentException("Maximum iterations must be at least 1.", nameof(maxIter));
        }

        if (learningRate <= 0)
        {
            throw new ArgumentException("Learning rate must be positive.", nameof(learningRate));
        }

        if (tol <= 0)
        {
            throw new ArgumentException("Tolerance must be positive.", nameof(tol));
        }

        _nComponents = nComponents;
        _maxIter = maxIter;
        _learningRate = learningRate;
        _tol = tol;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits NCA using the provided data and labels.
    /// </summary>
    /// <remarks>
    /// This method is internal to maintain the transformer pattern.
    /// Use <see cref="FitTransformSupervised"/> for supervised fitting with labels.
    /// </remarks>
    /// <param name="data">The input data matrix.</param>
    /// <param name="labels">The class labels for each sample.</param>
    internal void Fit(Matrix<T> data, int[] labels)
    {
        var engine = Engine;
        int n = data.Rows;

        if (labels.Length != n)
        {
            throw new ArgumentException("Labels length must match number of samples.", nameof(labels));
        }

        // Apply column selection if specified
        var columnsToProcess = GetColumnsToProcess(data.Columns);
        int p = columnsToProcess.Length;
        _nFeatures = p;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSeededRandom(42);

        var x = DimensionalityReductionOps.SelectColumns(
            Tensor<T>.FromMatrix(data), DimensionalityReductionOps.IndexTensor(columnsToProcess));

        // Initialize transformation matrix A (d x p) with small random values in [-0.005, 0.005)
        var a = engine.TensorMultiplyScalar(
            engine.TensorSubtractScalar(DimensionalityReductionOps.Uniform<T>(random, _nComponents, p), NumOps.FromDouble(0.5)),
            NumOps.FromDouble(0.01));

        // same[i, j] = 1 when labels[i] == labels[j]
        var labelValues = new Tensor<int>(labels, new[] { n }).Cast<T>();
        var same = engine.TensorEquals(
            engine.TensorSubtract(DimensionalityReductionOps.AsColumn(labelValues), DimensionalityReductionOps.AsRow(labelValues)),
            NumOps.Zero);

        T tol = NumOps.FromDouble(_tol);
        T sumFloor = NumOps.FromDouble(1e-10);
        T infinity = NumOps.FromDouble(double.PositiveInfinity);
        T two = NumOps.FromDouble(2.0);
        T prevObjective = NumOps.Zero;
        bool hasPrevious = false;
        double lr = _learningRate;

        for (int iter = 0; iter < _maxIter; iter++)
        {
            // Transformed data Y = X A^T and squared distances with the diagonal excluded (+inf)
            var y = DimensionalityReductionOps.MatMulTranspose(x, a);
            var distances = DimensionalityReductionOps.SquaredDistances(y, y);
            engine.TensorFillDiagonalInPlace(distances, infinity);

            // Softmax p_ij = exp(-(d_ij - min_i)) / sum, rows with sum <= 1e-10 left unnormalized
            var minDist = engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(distances), new[] { 1 }, true));
            var exponentials = engine.TensorExp(engine.TensorNegate(engine.TensorSubtract(distances, minDist)));
            var sumExp = engine.ReduceSum(exponentials, new[] { 1 }, true);
            var divisor = engine.TensorWhere(
                engine.TensorGreaterThan(sumExp, sumFloor), sumExp, engine.TensorOnesLike(sumExp));
            var probabilities = engine.TensorDivide(exponentials, divisor);

            // p_i = sum over same-class j of p_ij; objective = sum_i p_i
            var pi = engine.ReduceSum(engine.TensorMultiply(probabilities, same), new[] { 1 }, true);   // [n, 1]
            T objective = engine.TensorSum(pi);

            // Check convergence
            if (hasPrevious && DimensionalityReductionOps.ChangeBelow(objective, prevObjective, tol))
            {
                break;
            }

            prevObjective = objective;
            hasPrevious = true;

            // Gradient of the objective (Goldberger et al. 2005):
            // df/dA = 2 A sum_ik c_ik x_ik x_ik^T with c_ik = p_ik (p_i - [y_i == y_k]),
            // and sum_ik c_ik x_ik x_ik^T = X^T (diag(rowsum C + colsum C) - C - C^T) X.
            var c = engine.TensorMultiply(probabilities, engine.TensorSubtract(pi, same));
            var degree = engine.TensorAdd(DimensionalityReductionOps.RowSums(c), DimensionalityReductionOps.ColumnSums(c));
            var laplacian = engine.TensorSubtract(
                engine.TensorDiag(degree), engine.TensorAdd(c, engine.TensorTranspose(c)));
            var gradient = engine.TensorMultiplyScalar(
                DimensionalityReductionOps.TransposeMatMul(y, engine.TensorMatMul(laplacian, x)), two);

            // Gradient ascent (maximizing the objective)
            a = engine.TensorAdd(a, engine.TensorMultiplyScalar(gradient, DimensionalityReductionOps.Constant<T>(lr)));

            // Decay learning rate
            if (iter > 0 && iter % 20 == 0)
            {
                lr *= 0.9;
            }
        }

        _transformationMatrix = a.ToMatrix();
    }

    /// <summary>
    /// Fits NCA (without labels - uses unsupervised initialization).
    /// </summary>
    /// <remarks>
    /// This method provides a fallback when no labels are available.
    /// For best results, use the Fit(data, labels) overload with class labels.
    /// </remarks>
    protected override void FitCore(Matrix<T> data)
    {
        // Without labels, initialize with PCA-like transformation
        var engine = Engine;
        _nFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Center data and compute the covariance matrix (divided by n)
        var x = Tensor<T>.FromMatrix(data);
        var centered = DimensionalityReductionOps.SubtractRow(x, DimensionalityReductionOps.ColumnMeans(x));
        var covariance = engine.TensorDivideScalar(
            DimensionalityReductionOps.TransposeMatMul(centered, centered), NumOps.FromDouble(n));

        // Top eigenvectors as the rows of A (zero rows past the feature count)
        var (_, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(covariance, descending: true);
        int available = Math.Min(_nComponents, p);
        var rows = engine.TensorTranspose(DimensionalityReductionOps.FirstColumns(eigenvectors, available));
        var a = available == _nComponents
            ? rows
            : engine.TensorConcatenate(new[] { rows, new Tensor<T>(new[] { _nComponents - available, p }) }, 0);

        _transformationMatrix = a.ToMatrix();
    }

    /// <summary>
    /// Fits NCA using the provided data and labels, then transforms the data.
    /// </summary>
    /// <param name="data">The input data matrix.</param>
    /// <param name="labels">The class labels for each sample.</param>
    /// <returns>The transformed data projected onto the learned space.</returns>
    public Matrix<T> FitTransformSupervised(Matrix<T> data, int[] labels)
    {
        Fit(data, labels);
        return Transform(data);
    }

    /// <summary>
    /// Transforms data using the learned transformation matrix.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_transformationMatrix is null)
        {
            throw new InvalidOperationException("NCA has not been fitted.");
        }

        int p = data.Columns;

        if (p != _nFeatures)
        {
            throw new ArgumentException(
                $"Input has {p} features but NCA was fitted with {_nFeatures} features.",
                nameof(data));
        }

        return DimensionalityReductionOps.MatMulTranspose(
            Tensor<T>.FromMatrix(data), Tensor<T>.FromMatrix(_transformationMatrix)).ToMatrix();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("NCA does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"NCA{i + 1}";
        }
        return names;
    }
}
