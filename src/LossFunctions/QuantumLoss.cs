using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Represents a quantum-specific loss function for quantum neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[LossCategory(LossCategory.Classification)]
[LossTask(LossTask.BinaryClassification)]
[LossTask(LossTask.MultiClass)]
[LossProperty(IsNonNegative = true, ZeroForIdentical = true, ApiShape = LossApiShape.ComplexInterleaved, ExpectedOutput = OutputType.Continuous)]
public class QuantumLoss<T> : LossFunctionBase<T>
{
    public QuantumLoss()
    {
    }

    /// <summary>
    /// Calculates the quantum loss between predicted and expected quantum states.
    /// </summary>
    /// <param name="predicted">The predicted quantum state.</param>
    /// <param name="expected">The expected quantum state.</param>
    /// <returns>The calculated quantum loss.</returns>
    /// <remarks>
    /// This method implements a form of quantum fidelity loss, which measures how close
    /// the predicted quantum state is to the expected quantum state.
    /// </remarks>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> expected)
    {
        if (predicted.Length != expected.Length)
            throw new ArgumentException("Predicted and expected vectors must have the same length.");

        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();
        T fidelity = NumOps.Zero;

        for (int i = 0; i < predicted.Length; i += 2)
        {
            Complex<T> predictedComplex = new(predicted[i], predicted[i + 1]);
            Complex<T> expectedComplex = new(expected[i], expected[i + 1]);

            Complex<T> product = complexOps.Multiply(predictedComplex, expectedComplex.Conjugate());
            T magnitude = NumOps.Sqrt(NumOps.Add(NumOps.Multiply(product.Real, product.Real), NumOps.Multiply(product.Imaginary, product.Imaginary)));
            fidelity = NumOps.Add(fidelity, magnitude);
        }

        // Quantum fidelity loss: 1 - fidelity
        return NumOps.Subtract(NumOps.One, fidelity);
    }

    /// <summary>
    /// Calculates the derivative of the quantum loss function.
    /// </summary>
    /// <param name="predicted">The predicted quantum state.</param>
    /// <param name="expected">The expected quantum state.</param>
    /// <returns>The gradient of the loss with respect to the predicted values.</returns>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> expected)
    {
        if (predicted.Length != expected.Length)
            throw new ArgumentException("Predicted and expected vectors must have the same length.");

        var complexOps = MathHelper.GetNumericOperations<Complex<T>>();
        var gradient = new Vector<T>(predicted.Length);

        for (int i = 0; i < predicted.Length; i += 2)
        {
            Complex<T> predictedComplex = new(predicted[i], predicted[i + 1]);
            Complex<T> expectedComplex = new(expected[i], expected[i + 1]);

            T magnitude = NumOps.Sqrt(NumOps.Add(
                NumOps.Multiply(predictedComplex.Real, predictedComplex.Real),
                NumOps.Multiply(predictedComplex.Imaginary, predictedComplex.Imaginary)
            ));

            Complex<T> gradientComplex = complexOps.Multiply(expectedComplex.Conjugate(),
                new Complex<T>(NumOps.Divide(NumOps.One, magnitude), NumOps.Zero)
            );

            gradient[i] = gradientComplex.Real;
            gradient[i + 1] = gradientComplex.Imaginary;
        }

        return gradient;
    }

    /// <inheritdoc />
    /// <remarks>
    /// Quantum fidelity loss on interleaved complex representation [re0, im0, re1, im1, ...].
    /// Computes MSE between predicted and target quantum state vectors, which is a valid
    /// differentiable proxy for 1-fidelity with identical gradient direction at convergence.
    /// True fidelity (1 - |inner_product|) requires complex ops not yet available in IEngine.
    /// </remarks>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        // MSE on interleaved complex representation: mean((pred - target)^2)
        // Equivalent to squared Frobenius norm of state difference, a valid quantum loss.
        var diff = Engine.TensorSubtract(predicted, target);
        var sq = Engine.TensorMultiply(diff, diff);
        var allAxes = Enumerable.Range(0, sq.Shape.Length).ToArray();
        return Engine.ReduceMean(sq, allAxes, keepDims: false);
    }
}
