namespace AiDotNet.LossFunctions;

/// <summary>
/// Represents a quantum-specific loss function for quantum neural networks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
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
}
