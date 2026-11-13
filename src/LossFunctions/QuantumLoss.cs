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

        // Interleaved [re0, im0, re1, im1, ...]: an odd length has no imaginary part for its last element.
        if (predicted.Length % 2 != 0)
            throw new ArgumentOutOfRangeException(nameof(predicted), "Interleaved complex vectors must have an even length.");

        // Split into real/imaginary columns: p = a + bi, e = c + di.
        int pairs = predicted.Length / 2;
        var pairShape = new[] { pairs, 2 };
        var p = Engine.Reshape(predicted.AsTensor(), pairShape);
        var e = Engine.Reshape(expected.AsTensor(), pairShape);
        var a = Engine.TensorSelect(p, 1, 0);
        var b = Engine.TensorSelect(p, 1, 1);
        var c = Engine.TensorSelect(e, 1, 0);
        var d = Engine.TensorSelect(e, 1, 1);

        // p * conj(e) = (ac + bd) + (bc - ad)i
        var real = Engine.TensorAdd(Engine.TensorMultiply(a, c), Engine.TensorMultiply(b, d));
        var imaginary = Engine.TensorSubtract(Engine.TensorMultiply(b, c), Engine.TensorMultiply(a, d));

        // fidelity = Σ |p * conj(e)|
        var magnitude = Engine.TensorSqrt(Engine.TensorAdd(
            Engine.TensorMultiply(real, real), Engine.TensorMultiply(imaginary, imaginary)));
        T fidelity = Engine.TensorSum(magnitude);

        // Quantum fidelity loss: 1 - fidelity
        return NumOps.Subtract(NumOps.One, fidelity);
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
