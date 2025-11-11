using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Modified Gradient Descent optimizer for Hope architecture.
/// Based on Equations 27-29 from "Nested Learning" paper.
///
/// Traditional GD: W(t+1) = W(t) - eta * gradient(L(W(t); x(t))) outer-product x(t)
/// Modified GD:    W(t+1) = W(t) * (I - x(t)*x(t)^T) - eta * gradient(L(W(t); x(t))) outer-product x(t)
///
/// This formulation uses L2 regression objective instead of dot-product similarity,
/// resulting in better handling of data dependencies in token space.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
public class ModifiedGradientDescentOptimizer<T>
{
    private readonly T _learningRate;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Creates a modified gradient descent optimizer.
    /// </summary>
    /// <param name="learningRate">Learning rate (eta)</param>
    public ModifiedGradientDescentOptimizer(T learningRate)
    {
        _learningRate = learningRate;
    }

    /// <summary>
    /// Updates parameters using modified gradient descent (Equations 27-29).
    ///
    /// min_W ||W*x(t) - gradient_y(L(W(t); x(t)))||^2
    ///
    /// Results in:
    /// W(t+1) = W(t) * (I - x(t)*x(t)^T) - eta * gradient_y(L(W(t); x(t))) outer-product x(t)
    /// </summary>
    /// <param name="currentParameters">Current parameter matrix W(t)</param>
    /// <param name="input">Input vector x(t)</param>
    /// <param name="outputGradient">Gradient gradient_y(L(W(t); x(t)))</param>
    /// <returns>Updated parameters W(t+1)</returns>
    public Matrix<T> UpdateMatrix(Matrix<T> currentParameters, Vector<T> input, Vector<T> outputGradient)
    {
        int rows = currentParameters.Rows;
        int cols = currentParameters.Columns;

        // Compute (I - x(t)*x(t)^T)
        var identityMinusOuterProduct = ComputeIdentityMinusOuterProduct(input);

        // Compute W(t) * (I - x(t)*x(t)^T)
        var firstTerm = currentParameters.Multiply(identityMinusOuterProduct);

        // Compute gradient_y(L(W(t); x(t))) outer-product x(t)
        var gradientUpdate = ComputeOuterProduct(outputGradient, input);

        // Scale by learning rate: eta * (gradient_y(L) outer-product x(t))
        var scaledGradient = gradientUpdate.Multiply(_learningRate);

        // Final update: W(t+1) = W(t) * (I - x(t)*x(t)^T) - eta * (gradient_y(L) outer-product x(t))
        var updated = firstTerm.Subtract(scaledGradient);

        return updated;
    }

    /// <summary>
    /// Updates a parameter vector using modified gradient descent.
    ///
    /// NOTE: This is a simplified scalar approximation of the matrix operation.
    /// The matrix form W_t * (I - x_t x_t^T) is always stable, but this scalar
    /// version using (1 - ||x_t||²) requires clipping to prevent instability
    /// when input norm exceeds 1.
    /// </summary>
    /// <param name="currentParameters">Current parameters</param>
    /// <param name="input">Input vector</param>
    /// <param name="outputGradient">Output gradient</param>
    /// <returns>Updated parameters</returns>
    public Vector<T> UpdateVector(Vector<T> currentParameters, Vector<T> input, Vector<T> outputGradient)
    {
        var updated = new Vector<T>(currentParameters.Length);

        // For vector form: apply element-wise operations
        // This is a simplified version that preserves the spirit of the modification
        T inputNormSquared = _numOps.Zero;
        for (int i = 0; i < input.Length; i++)
        {
            inputNormSquared = _numOps.Add(inputNormSquared, _numOps.Square(input[i]));
        }

        // Apply modified update rule
        for (int i = 0; i < currentParameters.Length; i++)
        {
            // Standard GD component: -eta * gradient
            T gradComponent = _numOps.Multiply(outputGradient[i], _learningRate);

            // Modification: scale by (1 - ||x(t)||^2) factor for regularization
            // CRITICAL: Clip to prevent negative scaling when ||x(t)||^2 > 1
            // Without clipping, parameters would explode when input norm exceeds 1
            T modFactor = _numOps.Subtract(_numOps.One, inputNormSquared);
            if (_numOps.LessThan(modFactor, _numOps.Zero))
            {
                modFactor = _numOps.Zero;
            }

            T paramComponent = _numOps.Multiply(currentParameters[i], modFactor);

            updated[i] = _numOps.Subtract(paramComponent, gradComponent);
        }

        return updated;
    }

    /// <summary>
    /// Computes (I - x(t)*x(t)^T) where x(t) is the input vector.
    /// This is the modification term that accounts for data dependencies.
    /// </summary>
    private Matrix<T> ComputeIdentityMinusOuterProduct(Vector<T> input)
    {
        int dim = input.Length;
        var result = new Matrix<T>(dim, dim);

        // Start with identity matrix
        for (int i = 0; i < dim; i++)
        {
            result[i, i] = _numOps.One;
        }

        // Subtract outer product: x(t)*x(t)^T
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                T outerProduct = _numOps.Multiply(input[i], input[j]);
                result[i, j] = _numOps.Subtract(result[i, j], outerProduct);
            }
        }

        return result;
    }

    /// <summary>
    /// Computes outer product of two vectors: a outer-product b = a*b^T
    /// </summary>
    private Matrix<T> ComputeOuterProduct(Vector<T> a, Vector<T> b)
    {
        var result = new Matrix<T>(a.Length, b.Length);

        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < b.Length; j++)
            {
                result[i, j] = _numOps.Multiply(a[i], b[j]);
            }
        }

        return result;
    }

    /// <summary>
    /// Gets the learning rate.
    /// </summary>
    public T LearningRate => _learningRate;
}
