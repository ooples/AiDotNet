using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Optimizers;

/// <summary>
/// Modified Gradient Descent optimizer for Hope architecture.
/// Based on Equations 27-29 from "Nested Learning" paper.
///
/// Traditional GD: W_{t+1} = W_t - η * ∇L(W_t; x_t) ⊗ x_t
/// Modified GD:    W_{t+1} = W_t * (I - x_t*x_t^T) - η * ∇L(W_t; x_t) ⊗ x_t
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
    /// <param name="learningRate">Learning rate η</param>
    public ModifiedGradientDescentOptimizer(T learningRate)
    {
        _learningRate = learningRate;
    }

    /// <summary>
    /// Updates parameters using modified gradient descent (Equations 27-29).
    ///
    /// min_W ||W*x_t - ∇_y L(W_t; x_t)||²
    ///
    /// Results in:
    /// W_{t+1} = W_t * (I - x_t*x_t^T) - η * ∇_y L(W_t; x_t) ⊗ x_t
    /// </summary>
    /// <param name="currentParameters">Current parameter matrix W_t</param>
    /// <param name="input">Input vector x_t</param>
    /// <param name="outputGradient">Gradient ∇_y L(W_t; x_t)</param>
    /// <returns>Updated parameters W_{t+1}</returns>
    public Matrix<T> UpdateMatrix(Matrix<T> currentParameters, Vector<T> input, Vector<T> outputGradient)
    {
        int rows = currentParameters.Rows;
        int cols = currentParameters.Columns;

        // Compute (I - x_t*x_t^T)
        var identityMinusOuterProduct = ComputeIdentityMinusOuterProduct(input);

        // Compute W_t * (I - x_t*x_t^T)
        var firstTerm = currentParameters.Multiply(identityMinusOuterProduct);

        // Compute ∇_y L(W_t; x_t) ⊗ x_t (outer product)
        var gradientUpdate = ComputeOuterProduct(outputGradient, input);

        // Scale by learning rate: η * (∇_y L ⊗ x_t)
        var scaledGradient = gradientUpdate.Multiply(_learningRate);

        // Final update: W_{t+1} = W_t * (I - x_t*x_t^T) - η * (∇_y L ⊗ x_t)
        var updated = firstTerm.Subtract(scaledGradient);

        return updated;
    }

    /// <summary>
    /// Updates a parameter vector using modified gradient descent.
    ///
    /// For a vector parameter w, the matrix operation W * (I - x x^T) becomes:
    /// w_new = w * (I - x x^T) = w - x*(x^T*w) = w - x*dot(w,x)
    ///
    /// Full update: w_{t+1} = w_t - x_t*dot(w_t,x_t) - η * gradient
    /// </summary>
    /// <param name="currentParameters">Current parameter vector w_t</param>
    /// <param name="input">Input vector x_t</param>
    /// <param name="outputGradient">Output gradient ∇_y L(w_t; x_t)</param>
    /// <returns>Updated parameters w_{t+1}</returns>
    public Vector<T> UpdateVector(Vector<T> currentParameters, Vector<T> input, Vector<T> outputGradient)
    {
        if (currentParameters.Length != input.Length)
            throw new ArgumentException($"Parameter length ({currentParameters.Length}) must match input length ({input.Length})");

        if (currentParameters.Length != outputGradient.Length)
            throw new ArgumentException($"Parameter length ({currentParameters.Length}) must match gradient length ({outputGradient.Length})");

        var updated = new Vector<T>(currentParameters.Length);

        // Compute dot(w_t, x_t) = x_t^T * w_t
        T dotProduct = _numOps.Zero;
        for (int i = 0; i < currentParameters.Length; i++)
        {
            dotProduct = _numOps.Add(dotProduct, _numOps.Multiply(currentParameters[i], input[i]));
        }

        // Apply modified update rule: w_{t+1} = w_t - x_t*dot(w_t,x_t) - η*gradient
        for (int i = 0; i < currentParameters.Length; i++)
        {
            // Projection term: w_t - x_t*dot(w_t,x_t)
            // This is the vector equivalent of W_t * (I - x_t*x_t^T)
            T projectionComponent = _numOps.Multiply(input[i], dotProduct);
            T projectedParam = _numOps.Subtract(currentParameters[i], projectionComponent);

            // Gradient term: -η * gradient
            T gradComponent = _numOps.Multiply(outputGradient[i], _learningRate);

            // Final update: w_{t+1} = (w_t - x_t*dot(w_t,x_t)) - η*gradient
            updated[i] = _numOps.Subtract(projectedParam, gradComponent);
        }

        return updated;
    }

    /// <summary>
    /// Computes (I - x_t*x_t^T) where x_t is the input vector.
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

        // Subtract outer product: x_t*x_t^T
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
    /// Computes outer product of two vectors: a ⊗ b = a*b^T
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
