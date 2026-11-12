using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.DecompositionMethods.TimeSeriesDecomposition;

/// <summary>
/// Implements Singular Spectrum Analysis (SSA) for time series decomposition.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> SSA is a technique that helps break down a time series (sequence of data points) into 
/// meaningful components like trends, seasonal patterns, and noise. Think of it like separating the 
/// ingredients of a mixed smoothie - you can identify the fruits, yogurt, and other components that were 
/// blended together.
/// </para>
/// <para>
/// SSA works by transforming your time series into a matrix, analyzing patterns using mathematical 
/// techniques, and then reconstructing the important components.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelDomain(ModelDomain.TimeSeries)]
[ModelCategory(ModelCategory.TimeSeriesModel)]
[ModelCategory(ModelCategory.Statistical)]
[ModelTask(ModelTask.DimensionalityReduction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Vector<>), typeof(Vector<>))]
    [ResearchPaper("Analysis of Time Series Structure: SSA and Related Techniques", "https://doi.org/10.1201/9781420035841")]
public class SSADecomposition<T> : TimeSeriesDecompositionBase<T>
{
    private readonly int _windowSize;
    private readonly int _numberOfComponents;
    private readonly SSAAlgorithmType _algorithmType;

    /// <summary>
    /// Initializes a new instance of the SSA decomposition algorithm.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The window size determines how much historical data is considered when looking for patterns.
    /// A larger window can capture longer-term patterns but requires more data. The number of components 
    /// controls how many different patterns you want to extract from your data.
    /// </para>
    /// </remarks>
    /// <param name="timeSeries">The time series data to decompose.</param>
    /// <param name="windowSize">The window size for the trajectory matrix. Must be positive and not greater than half the time series length.</param>
    /// <param name="numberOfComponents">The number of components to extract. Must be positive and not greater than the window size.</param>
    /// <param name="algorithmType">The SSA algorithm variant to use. Defaults to Basic.</param>
    /// <exception cref="ArgumentException">Thrown when window size or number of components are invalid.</exception>
    public SSADecomposition(Vector<T> timeSeries, int windowSize, int numberOfComponents, SSAAlgorithmType algorithmType = SSAAlgorithmType.Basic)
        : base(timeSeries)
    {
        if (windowSize <= 0 || windowSize > timeSeries.Length / 2)
        {
            throw new ArgumentException("Window size must be positive and not greater than half the time series length.", nameof(windowSize));
        }

        if (numberOfComponents <= 0 || numberOfComponents > windowSize)
        {
            throw new ArgumentException("Number of components must be positive and not greater than the window size.", nameof(numberOfComponents));
        }

        _windowSize = windowSize;
        _numberOfComponents = numberOfComponents;
        _algorithmType = algorithmType;
        Decompose();
    }

    /// <summary>
    /// Performs the time series decomposition using the selected SSA algorithm.
    /// </summary>
    /// <remarks>
    /// This method orchestrates the entire decomposition process by selecting the appropriate
    /// algorithm, grouping components, and reconstructing the time series.
    /// </remarks>
    protected override void Decompose()
    {
        Matrix<T> trajectoryMatrix;
        Matrix<T> U;
        Vector<T> S;
        Matrix<T> V;

        switch (_algorithmType)
        {
            case SSAAlgorithmType.Basic:
                trajectoryMatrix = CreateTrajectoryMatrix();
                (U, S, V) = PerformSVD(trajectoryMatrix);
                break;
            case SSAAlgorithmType.Sequential:
                (U, S, V) = PerformSequentialSSA();
                break;
            case SSAAlgorithmType.Toeplitz:
                (U, S, V) = PerformToeplitzSSA();
                break;
            default:
                throw new ArgumentException("Invalid SSA algorithm type.");
        }

        var groupedComponents = GroupComponents(U, S, V);
        var reconstructedComponents = ReconstructComponents(groupedComponents);

        AssignComponents(reconstructedComponents);
    }

    /// <summary>
    /// Performs Singular Value Decomposition on the trajectory matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SVD is a mathematical technique that breaks down a complex matrix into simpler parts.
    /// It's like factoring a number (e.g., 12 = 3 × 4), but for matrices. This helps identify the most 
    /// important patterns in your data.
    /// </para>
    /// </remarks>
    /// <param name="trajectoryMatrix">The trajectory matrix to decompose.</param>
    /// <returns>The U, S, and V components of the SVD.</returns>
    private (Matrix<T> U, Vector<T> S, Matrix<T> V) PerformSVD(Matrix<T> trajectoryMatrix)
    {
        var svdDecomposition = new SvdDecomposition<T>(trajectoryMatrix);
        return (svdDecomposition.U, svdDecomposition.S, svdDecomposition.Vt);
    }

    /// <summary>
    /// Performs Sequential SSA using an iterative approach.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Sequential SSA extracts patterns one by one, starting with the strongest pattern.
    /// After finding each pattern, it removes that pattern from the data before looking for the next one.
    /// This is like identifying the loudest instrument in an orchestra, removing its sound, then identifying
    /// the next loudest, and so on.
    /// </para>
    /// </remarks>
    /// <returns>The U, S, and V components of the decomposition.</returns>
    private (Matrix<T> U, Vector<T> S, Matrix<T> V) PerformSequentialSSA()
    {
        int N = TimeSeries.Length;
        int K = N - _windowSize + 1;

        Matrix<T> X = CreateTrajectoryMatrix();
        Matrix<T> U = new Matrix<T>(_windowSize, _numberOfComponents);
        Matrix<T> V = new Matrix<T>(K, _numberOfComponents);
        var singularValues = new List<Tensor<T>>();

        for (int i = 0; i < _numberOfComponents; i++)
        {
            Vector<T> u = Vector<T>.CreateRandom(_windowSize);
            u = u.Normalize();

            for (int iter = 0; iter < 10; iter++) // Number of iterations for power method
            {
                Vector<T> v1 = X.Transpose().Multiply(u);
                T s1 = v1.Norm();
                v1 = v1.Divide(s1);

                u = X.Multiply(v1);
                u = u.Normalize();
            }

            Vector<T> v = X.Transpose().Multiply(u);
            T s = v.Norm();
            v = v.Divide(s);

            U.SetColumn(i, u);
            singularValues.Add(Tensor<T>.FromScalar(s).Reshape(new[] { 1 }));
            V.SetColumn(i, v);

            // Deflate X
            X = X.Subtract(u.OuterProduct(v).Multiply(s));
        }

        Vector<T> S = singularValues.Count > 0
            ? Engine.TensorConcatenate(singularValues.ToArray(), 0).ToVector()
            : new Vector<T>(0);
        return (U, S, V);
    }

    /// <summary>
    /// Performs Toeplitz SSA using the autocovariance matrix.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Toeplitz SSA is a variation that uses a special matrix structure to find patterns.
    /// It focuses on how data points relate to each other at different time lags. This approach can be 
    /// more efficient for certain types of time series data, especially when the patterns are consistent
    /// over time.
    /// </para>
    /// </remarks>
    /// <returns>The U, S, and V components of the decomposition.</returns>
    private (Matrix<T> U, Vector<T> S, Matrix<T> V) PerformToeplitzSSA()
    {
        int N = TimeSeries.Length;
        int K = N - _windowSize + 1;

        // Create the Toeplitz matrix C[i, j] = c(|i - j|), where the lag-l autocovariance
        // c(l) = sum_{k < N - l} x[k] x[k + l] / (N - l) comes from the zero-padded self-correlation
        // of the series (lag l at output position N - 1 - l).
        var signal = Tensor<T>.FromVector(TimeSeries, new[] { 1, 1, N });
        var correlation = Engine.Conv1D(signal, signal, 1, N - 1).Reshape(new[] { 2 * N - 1 });
        var lagSums = Engine.TensorFlip(
            Engine.TensorSlice(correlation, new[] { N - _windowSize }, new[] { _windowSize }), new[] { 0 });
        var pairCounts = Engine.ScalarMinusTensor(NumOps.FromDouble(N), Engine.TensorArange<T>(0, _windowSize, 1));
        var autocovariance = DivideElementwise(lagSums, pairCounts);

        var index = Engine.TensorArange<int>(0, _windowSize, 1);
        var lags = Engine.TensorAbs(Engine.TensorSubtract(
            Engine.TensorBroadcastTo(index.Reshape(new[] { _windowSize, 1 }), new[] { _windowSize, _windowSize }),
            Engine.TensorBroadcastTo(index.Reshape(new[] { 1, _windowSize }), new[] { _windowSize, _windowSize })));
        Matrix<T> C = Engine.TensorIndexSelect(autocovariance, lags.Reshape(new[] { _windowSize * _windowSize }), 0)
            .Reshape(new[] { _windowSize, _windowSize }).ToMatrix();

        // Perform eigendecomposition
        var eigenDecomposition = new EigenDecomposition<T>(C);
        Matrix<T> U = eigenDecomposition.EigenVectors;
        Vector<T> S = Engine.TensorSqrt(Tensor<T>.FromVector(eigenDecomposition.EigenValues)).ToVector();

        // Compute V: column i is X^T u_i / s_i
        Matrix<T> X = CreateTrajectoryMatrix();
        var leading = Engine.TensorSlice(Tensor<T>.FromMatrix(U), new[] { 0, 0 }, new[] { U.Rows, _numberOfComponents });
        var projected = Engine.TensorMatMul(Engine.TensorTranspose(Tensor<T>.FromMatrix(X)), leading);
        var leadingS = Engine.TensorSlice(Tensor<T>.FromVector(S), new[] { 0 }, new[] { _numberOfComponents });
        Matrix<T> V = DivideElementwise(
            projected,
            Engine.TensorBroadcastTo(leadingS.Reshape(new[] { 1, _numberOfComponents }), new[] { K, _numberOfComponents })).ToMatrix();

        // Truncate to the specified number of components
        U = U.Submatrix(0, _windowSize - 1, 0, _numberOfComponents - 1);
        S = S.Subvector(0, _numberOfComponents - 1);
        V = V.Submatrix(0, K - 1, 0, _numberOfComponents - 1);

        return (U, S, V);
    }

    /// <summary>
    /// Assigns the reconstructed components to their respective types (trend, seasonal, residual).
    /// </summary>
    /// <remarks>
    /// By convention, the first component is typically assigned as the trend, the second as seasonal (if available),
    /// and the remaining components are combined into the residual.
    /// </remarks>
    /// <param name="reconstructedComponents">The array of reconstructed components.</param>
    private void AssignComponents(Vector<T>[] reconstructedComponents)
    {
        AddComponent(DecompositionComponentType.Trend, reconstructedComponents[0]);

        if (reconstructedComponents.Length > 2)
        {
            AddComponent(DecompositionComponentType.Seasonal, reconstructedComponents[1]);

            var residual = new Vector<T>(TimeSeries.Length);
            for (int i = 2; i < reconstructedComponents.Length; i++)
            {
                residual = residual.Add(reconstructedComponents[i]);
            }
            AddComponent(DecompositionComponentType.Residual, residual);
        }
        else if (reconstructedComponents.Length == 2)
        {
            AddComponent(DecompositionComponentType.Residual, reconstructedComponents[1]);
        }
    }

    /// <summary>
    /// Creates a trajectory matrix from the time series data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A trajectory matrix is a way to reorganize your time series data into a matrix format.
    /// Imagine taking a sliding window of size _windowSize and moving it through your data one step at a time.
    /// Each position of this window becomes a column in the matrix. This transformation helps us identify
    /// patterns that might not be obvious in the original data.
    /// </para>
    /// </remarks>
    /// <returns>A matrix where each column represents a segment of the original time series.</returns>
    private Matrix<T> CreateTrajectoryMatrix()
    {
        int K = TimeSeries.Length - _windowSize + 1;

        // Hankel matrix: trajectoryMatrix[j, i] = x[i + j]
        var rowIndex = Engine.TensorArange<int>(0, _windowSize, 1).Reshape(new[] { _windowSize, 1 });
        var columnIndex = Engine.TensorArange<int>(0, K, 1).Reshape(new[] { 1, K });
        var positions = Engine.TensorAdd(
            Engine.TensorBroadcastTo(rowIndex, new[] { _windowSize, K }),
            Engine.TensorBroadcastTo(columnIndex, new[] { _windowSize, K }));
        return Engine.TensorIndexSelect(Tensor<T>.FromVector(TimeSeries), positions.Reshape(new[] { _windowSize * K }), 0)
            .Reshape(new[] { _windowSize, K }).ToMatrix();
    }

    /// <summary>
    /// Groups the decomposed components based on the SVD results.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> After breaking down our data using SVD (Singular Value Decomposition), 
    /// this method reconstructs each individual component. Think of it like reassembling specific 
    /// instruments from an orchestra recording - we're recreating each instrument's contribution 
    /// to the overall sound.
    /// </para>
    /// <para>
    /// The mathematical operation used here (outer product) combines the patterns found in the U and V 
    /// matrices, weighted by their importance (S values).
    /// </para>
    /// </remarks>
    /// <param name="U">The left singular vectors from SVD.</param>
    /// <param name="S">The singular values from SVD.</param>
    /// <param name="V">The right singular vectors from SVD.</param>
    /// <returns>An array of matrices, each representing a component in the trajectory matrix space.</returns>
    private Matrix<T>[] GroupComponents(Matrix<T> U, Vector<T> S, Matrix<T> V)
    {
        var groupedComponents = new Matrix<T>[_numberOfComponents];
        var singularValues = Tensor<T>.FromVector(S);

        for (int i = 0; i < _numberOfComponents; i++)
        {
            var Ui = U.GetColumn(i);
            var Vi = V.GetColumn(i);
            T si = Engine.TensorSum(Engine.TensorSlice(singularValues, new[] { i }, new[] { 1 }));
            groupedComponents[i] = Ui.OuterProduct(Vi).Multiply(si);
        }

        return groupedComponents;
    }

    /// <summary>
    /// Reconstructs the time series components from the grouped components.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts the matrix components back into time series format.
    /// Since we transformed our original time series into a matrix earlier, we now need to reverse
    /// that process to get meaningful time series components.
    /// </para>
    /// <para>
    /// The process involves a technique called "diagonal averaging" where we average values along 
    /// diagonals of the matrix. Imagine the matrix as a grid - we take all values that are the same 
    /// "distance" from the top-left corner and average them together.
    /// </para>
    /// </remarks>
    /// <param name="groupedComponents">The grouped components in matrix form.</param>
    /// <returns>An array of vectors, each representing a reconstructed component of the original time series.</returns>
    private Vector<T>[] ReconstructComponents(Matrix<T>[] groupedComponents)
    {
        var reconstructedComponents = new Vector<T>[_numberOfComponents];

        // Every grouped component has the same [W, K] shape, so the anti-diagonal counts are shared.
        Tensor<T>? counts = null;
        for (int k = 0; k < _numberOfComponents; k++)
        {
            var X = Tensor<T>.FromMatrix(groupedComponents[k]);
            var sums = AntiDiagonalSums(X);
            counts ??= AntiDiagonalSums(Engine.TensorOnesLike(X));
            reconstructedComponents[k] = DivideElementwise(sums, counts).ToVector();
        }

        return reconstructedComponents;
    }

    /// <summary>
    /// Sums of a [W, K] matrix along its W + K - 1 anti-diagonals (i = row + column), by skewing:
    /// [X | 0_W] flattened and re-read with row length W + K - 1 shifts row j right by j, so the
    /// column sums of the skewed matrix are the anti-diagonal sums.
    /// </summary>
    private Tensor<T> AntiDiagonalSums(Tensor<T> matrix)
    {
        int rows = matrix.Shape[0];
        int columns = matrix.Shape[1];
        int length = rows + columns - 1;
        var widened = Engine.TensorConcatenate(new[] { matrix, new Tensor<T>(new[] { rows, rows }) }, 1)
            .Reshape(new[] { rows * (columns + rows) });
        var skewed = Engine.TensorSlice(widened, new[] { 0 }, new[] { rows * length }).Reshape(new[] { rows, length });
        return Engine.ReduceSum(skewed, new[] { 0 }, false);
    }
}
