global using AiDotNet.NumericOperations;

namespace AiDotNet.Helpers;

public static class MatrixHelper<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    public static T CalculateDeterminantRecursive(Matrix<T> matrix)
    {
        var rows = matrix.Rows;

        if (rows != matrix.Columns)
        {
            throw new ArgumentException("Matrix must be square.");
        }

        if (rows == 1)
        {
            return matrix[0, 0];
        }

        T determinant = NumOps.Zero;

        for (var i = 0; i < rows; i++)
        {
            var subMatrix = CreateSubMatrix(matrix, 0, i);
            T subDeterminant = CalculateDeterminantRecursive(subMatrix);
            T sign = NumOps.FromDouble(i % 2 == 0 ? 1 : -1);
            T product = NumOps.Multiply(NumOps.Multiply(sign, matrix[0, i]), subDeterminant);
            determinant = NumOps.Add(determinant, product);
        }

        return determinant;
    }

    private static Matrix<T> CreateSubMatrix(Matrix<T> matrix, int excludeRowIndex, int excludeColumnIndex)
    {
        var rows = matrix.Rows;
        var subMatrix = new Matrix<T>(rows - 1, rows - 1);

        var r = 0;
        for (var i = 0; i < rows; i++)
        {
            if (i == excludeRowIndex)
            {
                continue;
            }

            var c = 0;
            for (var j = 0; j < rows; j++)
            {
                if (j == excludeColumnIndex)
                {
                    continue;
                }

                subMatrix[r, c] = matrix[i, j];
                c++;
            }

            r++;
        }

        return subMatrix;
    }

    public static Matrix<T> ReduceToHessenbergFormat(Matrix<T> matrix)
    {
        var rows = matrix.Rows;
        var result = Matrix<T>.CreateMatrix<T>(rows, rows);

        for (int k = 0; k < rows - 2; k++)
        {
            var xVector = new Vector<T>(rows - k - 1);
            for (int i = 0; i < rows - k - 1; i++)
            {
                xVector[i] = matrix[k + 1 + i, k];
            }

            var hVector = CreateHouseholderVector(xVector);
            matrix = ApplyHouseholderTransformation(matrix, hVector, k);
        }

        return matrix;
    }

    public static Vector<T> ExtractDiagonal(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        Vector<T> diagonal = new(n);
        for (int i = 0; i < n; i++)
        {
            diagonal[i] = matrix[i, i];
        }

        return diagonal;
    }

    public static Matrix<T> OuterProduct(Vector<T> v1, Vector<T> v2)
    {
        int n = v1.Length;
        Matrix<T> result = new(n, n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                result[i, j] = NumOps.Multiply(v1[i], v2[j]);
            }
        }

        return result;
    }

    
    public static T Hypotenuse(T x, T y)
    {
        T xabs = NumOps.Abs(x), yabs = NumOps.Abs(y), min, max;

        if (NumOps.LessThan(xabs, yabs))
        {
            min = xabs; max = yabs;
        }
        else
        {
            min = yabs; max = xabs;
        }

        if (NumOps.Equals(min, NumOps.Zero))
        {
            return max;
        }

        T u = NumOps.Divide(min, max);

        return NumOps.Multiply(max, NumOps.Sqrt(NumOps.Add(NumOps.One, NumOps.Multiply(u, u))));
    }

    public static T Hypotenuse(params T[] values)
    {
        T sum = NumOps.Zero;
        foreach (var value in values)
        {
            sum = NumOps.Add(sum, NumOps.Multiply(value, value));
        }

        return NumOps.Sqrt(sum);
    }

    public static bool IsUpperHessenberg(Matrix<T> matrix, T tolerance)
    {
        for (int i = 2; i < matrix.Rows; i++)
        {
            for (int j = 0; j < i - 1; j++)
            {
                if (NumOps.GreaterThan(NumOps.Abs(matrix[i, j]), tolerance))
                {
                    return false;
                }
            }
        }

        return true;
    }

    public static Matrix<T> OrthogonalizeColumns(Matrix<T> matrix)
    {
        int m = matrix.Rows;
        int n = matrix.Columns;
        Matrix<T> Q = new(m, n);

        for (int j = 0; j < n; j++)
        {
            Vector<T> v = matrix.GetColumn(j);
            for (int i = 0; i < j; i++)
            {
                Vector<T> qi = Q.GetColumn(i);
                T r = NumOps.Divide(v.DotProduct(qi), qi.DotProduct(qi));
                v = v.Subtract(qi.Multiply(r));
            }
            T norm = v.Norm();
            if (!NumOps.Equals(norm, NumOps.Zero))
            {
                v = v.Divide(norm);
            }
            Q.SetColumn(j, v);
        }

        return Q;
    }

    public static (T c, T s) ComputeGivensRotation(T a, T b)
    {
        if (NumOps.Equals(b, NumOps.Zero))
        {
            return (NumOps.One, NumOps.Zero);
        }

        var t = NumOps.Divide(a, b);
        var u = NumOps.Sqrt(NumOps.Add(NumOps.One, NumOps.Multiply(t, t)));
        var c = NumOps.Divide(NumOps.One, u);
        var s = NumOps.Multiply(c, t);

        return (c, s);
    }

    public static void ApplyGivensRotation(Matrix<T> H, T c, T s, int i, int j, int kStart, int kEnd)
    {
        for (int k = kStart; k < kEnd; k++)
        {
            var temp1 = H[i, k];
            var temp2 = H[j, k];
            H[i, k] = NumOps.Add(NumOps.Multiply(c, temp1), NumOps.Multiply(s, temp2));
            H[j, k] = NumOps.Subtract(NumOps.Multiply(NumOps.Negate(s), temp1), NumOps.Multiply(c, temp2));
        }
    }

    
    public static Matrix<T> ApplyHouseholderTransformation(Matrix<T> matrix, Vector<T> vector, int k)
    {
        var rows = matrix.Rows;

        for (int i = k + 1; i < rows; i++)
        {
            T sum = NumOps.Zero;
            for (int j = k + 1; j < rows; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(vector[j - k - 1], matrix[j, i]));
            }
            for (int j = k + 1; j < rows; j++)
            {
                matrix[j, i] = NumOps.Subtract(matrix[j, i], NumOps.Multiply(NumOps.Multiply(NumOps.FromDouble(2), vector[j - k - 1]), sum));
            }
        }

        for (int i = 0; i < rows; i++)
        {
            T sum = NumOps.Zero;
            for (int j = k + 1; j < rows; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(vector[j - k - 1], matrix[i, j]));
            }
            for (int j = k + 1; j < rows; j++)
            {
                matrix[i, j] = NumOps.Subtract(matrix[i, j], NumOps.Multiply(NumOps.Multiply(NumOps.FromDouble(2), vector[j - k - 1]), sum));
            }
        }

        return matrix;
    }


    public static Vector<T> CreateHouseholderVector(Vector<T> xVector)
    {
        var result = new Vector<T>(xVector.Length);
        T norm = NumOps.Zero;

        for (int i = 0; i < xVector.Length; i++)
        {
            norm = NumOps.Add(norm, NumOps.Multiply(xVector[i], xVector[i]));
        }
        norm = NumOps.Sqrt(norm);

        result[0] = NumOps.Add(xVector[0], NumOps.Multiply(NumOps.SignOrZero(xVector[0]), norm));
        for (int i = 1; i < xVector.Length; i++)
        {
            result[i] = xVector[i];
        }

        T vNorm = NumOps.Zero;
        for (int i = 0; i < result.Length; i++)
        {
            vNorm = NumOps.Add(vNorm, NumOps.Multiply(result[i], result[i]));
        }
        vNorm = NumOps.Sqrt(vNorm);

        for (int i = 0; i < result.Length; i++)
        {
            result[i] = NumOps.Divide(result[i], vNorm);
        }

        return result;
    }

    public static (T, Vector<T>) PowerIteration(Matrix<T> aMatrix, int maxIterations, T tolerance)
    {
        var rows = aMatrix.Rows;
        var bVector = new Vector<T>(rows);
        var b2Vector = new Vector<T>(rows);
        T eigenvalue = NumOps.Zero;

        // Initial guess for the eigenvector
        for (int i = 0; i < rows; i++)
        {
            bVector[i] = NumOps.One;
        }

        for (int iter = 0; iter < maxIterations; iter++)
        {
            // Multiply A by the vector b
            for (int i = 0; i < rows; i++)
            {
                b2Vector[i] = NumOps.Zero;
                for (int j = 0; j < rows; j++)
                {
                    b2Vector[i] = NumOps.Add(b2Vector[i], NumOps.Multiply(aMatrix[i, j], bVector[j]));
                }
            }

            // Normalize the vector
            T norm = NumOps.Zero;
            for (int i = 0; i < rows; i++)
            {
                norm = NumOps.Add(norm, NumOps.Multiply(b2Vector[i], b2Vector[i]));
            }
            norm = NumOps.Sqrt(norm);
            for (int i = 0; i < rows; i++)
            {
                b2Vector[i] = NumOps.Divide(b2Vector[i], norm);
            }

            // Estimate the eigenvalue
            T newEigenvalue = NumOps.Zero;
            for (int i = 0; i < rows; i++)
            {
                newEigenvalue = NumOps.Add(newEigenvalue, NumOps.Multiply(b2Vector[i], b2Vector[i]));
            }

            // Check for convergence
            if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(newEigenvalue, eigenvalue)), tolerance))
            {
                break;
            }
            eigenvalue = newEigenvalue;
            for (int i = 0; i < rows; i++)
            {
                bVector[i] = b2Vector[i];
            }
        }

        return (eigenvalue, b2Vector);
    }

    public static T SpectralNorm(Matrix<T> matrix)
    {
        // Use the power iteration method to estimate the spectral norm
        int maxIterations = 100;
        T tolerance = NumOps.FromDouble(1e-10);
        
        Vector<T> v = Vector<T>.CreateRandom(matrix.Columns);
        v = v.Divide(v.Norm());

        for (int i = 0; i < maxIterations; i++)
        {
            Vector<T> w = matrix.Transpose().Multiply(matrix.Multiply(v));
            T lambda = v.DotProduct(w);
            Vector<T> vNew = w.Divide(w.Norm());

            if (NumOps.LessThan(vNew.Subtract(v).Norm(), tolerance))
            {
                break;
            }

            v = vNew;
        }

        return NumOps.Sqrt(v.DotProduct(matrix.Transpose().Multiply(matrix.Multiply(v))));
    }

    public static bool IsInvertible(Matrix<T> matrix)
    {
        // Check if the matrix is square
        if (matrix.Rows != matrix.Columns)
        {
            return false;
        }

        // Check if the determinant is zero (or very close to zero)
        T determinant = matrix.Determinant();
        T tolerance = NumOps.FromDouble(1e-10);

        return NumOps.GreaterThan(NumOps.Abs(determinant), tolerance);
    }

    public static Matrix<T> InvertUsingDecomposition(IMatrixDecomposition<T> decomposition)
    {
        int n = decomposition.A.Rows;
        var inverse = new Matrix<T>(n, n);

        for (int j = 0; j < n; j++)
        {
            var ej = Vector<T>.CreateStandardBasis(n, j);
            var xj = decomposition.Solve(ej);

            for (int i = 0; i < n; i++)
            {
                inverse[i, j] = xj[i];
            }
        }

        return inverse;
    }

    public static void TridiagonalSolve(Vector<T> vector1, Vector<T> vector2, Vector<T> vector3,
    Vector<T> solutionVector, Vector<T> actualVector)
    {
        var size = vector1.Length;
        T bet;
        var gamVector = new Vector<T>(size);

        if (NumOps.Equals(vector2[0], NumOps.Zero))
        {
            throw new InvalidOperationException("Not a tridiagonal matrix!");
        }

        bet = vector2[0];
        solutionVector[0] = NumOps.Divide(actualVector[0], bet);
        for (int i = 1; i < size; i++)
        {
            gamVector[i] = NumOps.Divide(vector3[i - 1], bet);
            bet = NumOps.Subtract(vector2[i], NumOps.Multiply(vector1[i], gamVector[i]));

            if (NumOps.Equals(bet, NumOps.Zero))
            {
                throw new InvalidOperationException("Not a tridiagonal matrix!");
            }

            solutionVector[i] = NumOps.Divide(
                NumOps.Subtract(actualVector[i], NumOps.Multiply(vector1[i], solutionVector[i - 1])),
                bet
            );
        }

        for (int i = size - 2; i >= 0; i--)
        {
            solutionVector[i] = NumOps.Subtract(
                solutionVector[i],
                NumOps.Multiply(gamVector[i + 1], solutionVector[i + 1])
            );
        }
    }

    public static void BandDiagonalMultiply(int leftSide, int rightSide, Matrix<T> matrix, Vector<T> solutionVector, Vector<T> actualVector)
    {
        var size = matrix.Rows;

        for (int i = 0; i < size; i++)
        {
            var k = i - leftSide;
            var temp = Math.Min(leftSide + rightSide + 1, size - k);
            solutionVector[i] = NumOps.Zero;
            for (int j = Math.Max(0, -k); j < temp; j++)
            {
                solutionVector[i] = NumOps.Add(solutionVector[i], NumOps.Multiply(matrix[i, j], actualVector[j + k]));
            }
        }
    }

    public static Matrix<T> CalculateHatMatrix(Matrix<T> features)
    {
        var transposeFeatures = features.Transpose();
        var inverseMatrix = transposeFeatures.Multiply(features).Inverse();

        return features.Multiply(inverseMatrix.Multiply(transposeFeatures));
    }
}