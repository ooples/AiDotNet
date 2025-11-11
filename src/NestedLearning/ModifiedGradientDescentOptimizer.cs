using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NestedLearning;

/// <summary>
/// Modified Gradient Descent optimizer for Hope architecture.
/// Based on Equations 27-29 from "Nested Learning" paper.
///
/// Traditional GD: Wt+1 = Wt - η * ∇L(Wt; xt) ⊗ xt
/// Modified GD:    Wt+1 = Wt * (I - xt*xt^T) - η * ∇L(Wt; xt) ⊗ xt
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
    /// min_W ||W*xt - ∇ytL(Wt; xt)||²_2
    ///
    /// Results in:
    /// Wt+1 = Wt * (I - xt*xt^T) - η * ∇ytL(Wt; xt) ⊗ xt
    /// </summary>
    /// <param name="currentParameters">Current parameter matrix Wt</param>
    /// <param name="input">Input vector xt</param>
    /// <param name="outputGradient">Gradient ∇ytL(Wt; xt)</param>
    /// <returns>Updated parameters Wt+1</returns>
    public Matrix<T> UpdateMatrix(Matrix<T> currentParameters, Vector<T> input, Vector<T> outputGradient)
    {
        int rows = currentParameters.Rows;
        int cols = currentParameters.Columns;

        // Compute (I - xt*xt^T)
        var identityMinusOuterProduct = ComputeIdentityMinusOuterProduct(input);

        // Compute Wt * (I - xt*xt^T)
        var firstTerm = currentParameters.Multiply(identityMinusOuterProduct);

        // Compute ∇ytL(Wt; xt) ⊗ xt (outer product)
        var gradientUpdate = ComputeOuterProduct(outputGradient, input);

        // Scale by learning rate: η * (∇ytL ⊗ xt)
        var scaledGradient = gradientUpdate.Multiply(_learningRate);

        // Final update: Wt+1 = Wt * (I - xt*xt^T) - η * (∇ytL ⊗ xt)
        var updated = firstTerm.Subtract(scaledGradient);

        return updated;
    }

    /// <summary>
    /// Updates a parameter vector using modified gradient descent.
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
            // Standard GD component: -η * gradient
            T gradComponent = _numOps.Multiply(outputGradient[i], _learningRate);

            // Modification: scale by (1 - ||xt||²) factor for regularization
            T modFactor = _numOps.Subtract(_numOps.One, inputNormSquared);
            T paramComponent = _numOps.Multiply(currentParameters[i], modFactor);

            updated[i] = _numOps.Subtract(paramComponent, gradComponent);
        }

        return updated;
    }

    /// <summary>
    /// Computes (I - xt*xt^T) where xt is the input vector.
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

        // Subtract outer product: xt*xt^T
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
