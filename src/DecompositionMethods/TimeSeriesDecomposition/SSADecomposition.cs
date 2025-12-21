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
        Vector<T> S = new Vector<T>(_numberOfComponents);
        Matrix<T> V = new Matrix<T>(K, _numberOfComponents);

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
            S[i] = s;
            V.SetColumn(i, v);

            // Deflate X
            X = X.Subtract(u.OuterProduct(v).Multiply(s));
        }

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

        // Create the Toeplitz matrix
        Matrix<T> C = new Matrix<T>(_windowSize, _windowSize);
        for (int i = 0; i < _windowSize; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                T sum = NumOps.Zero;
                for (int k = 0; k < N - i + j; k++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(TimeSeries[k], TimeSeries[k + i - j]));
                }
                C[i, j] = C[j, i] = NumOps.Divide(sum, NumOps.FromDouble(N - i + j));
            }
        }

        // Perform eigendecomposition
        var eigenDecomposition = new EigenDecomposition<T>(C);
        Matrix<T> U = eigenDecomposition.EigenVectors;
        Vector<T> S = eigenDecomposition.EigenValues.Transform(x => NumOps.Sqrt(x));

        // Compute V
        Matrix<T> V = new Matrix<T>(K, _numberOfComponents);
        Matrix<T> X = CreateTrajectoryMatrix();
        for (int i = 0; i < _numberOfComponents; i++)
        {
            Vector<T> v = X.Transpose().Multiply(U.GetColumn(i));
            v = v.Divide(S[i]);
            V.SetColumn(i, v);
        }

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
        var trajectoryMatrix = new Matrix<T>(_windowSize, K);

        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < _windowSize; j++)
            {
                trajectoryMatrix[j, i] = TimeSeries[i + j];
            }
        }

        return trajectoryMatrix;
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

        for (int i = 0; i < _numberOfComponents; i++)
        {
            var Ui = U.GetColumn(i);
            var Vi = V.GetColumn(i);
            groupedComponents[i] = Ui.OuterProduct(Vi).Multiply(S[i]);
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

        for (int k = 0; k < _numberOfComponents; k++)
        {
            var component = new Vector<T>(TimeSeries.Length);
            var X = groupedComponents[k];

            for (int i = 0; i < TimeSeries.Length; i++)
            {
                T sum = NumOps.Zero;
                int count = 0;

                for (int j = Math.Max(0, i - X.Columns + 1); j <= Math.Min(i, X.Rows - 1); j++)
                {
                    sum = NumOps.Add(sum, X[j, i - j]);
                    count++;
                }

                component[i] = NumOps.Divide(sum, NumOps.FromDouble(count));
            }

            reconstructedComponents[k] = component;
        }

        return reconstructedComponents;
    }
}
