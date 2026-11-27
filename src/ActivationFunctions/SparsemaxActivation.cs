using AiDotNet.Attributes;
using AiDotNet.Enums;

namespace AiDotNet.ActivationFunctions;

/// <summary>
/// Implements the Sparsemax activation function, which is an alternative to Softmax that can produce sparse probability distributions.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Sparsemax maps input vectors to probability distributions, similar to Softmax, but with the key difference
/// that Sparsemax can assign exactly zero probability to low-scoring classes, creating sparse outputs.
/// </para>
/// <para>
/// <b>For Beginners:</b> Sparsemax is an advanced activation function used primarily in the output layer of 
/// neural networks for classification tasks. While Softmax always gives some probability to every possible 
/// class (even if very small), Sparsemax can assign exactly zero probability to unlikely classes.
/// 
/// Think of it like this:
/// - Softmax is like saying "I'm 80% sure it's a dog, 15% sure it's a cat, and 5% sure it's something else"
/// - Sparsemax might say "I'm 90% sure it's a dog, 10% sure it's a cat, and 0% sure it's anything else"
/// 
/// This "sparsity" (having many zeros) can be useful when you have many possible classes but only a few
/// are likely to be correct. It makes the model's predictions more focused and interpretable.
/// </para>
/// </remarks>
[ActivationCategory(ActivationCategory.Normalization)]
[ActivationTask(ActivationTask.AttentionGating)]
[ActivationTask(ActivationTask.NormalizationOutput)]
[ActivationProperty(IsMonotonic = false, ZeroPreserving = false, IsBounded = true, IsVectorActivation = true, Cost = ComputeCost.High)]
public class SparsemaxActivation<T> : ActivationFunctionBase<T>
{
    /// <summary>
    /// Indicates whether this activation function supports scalar operations.
    /// </summary>
    /// <returns>Always returns false as Sparsemax requires a vector of values.</returns>
    /// <remarks>
    /// <para>
    /// Unlike functions like ReLU that can operate on individual values, Sparsemax needs to consider
    /// all elements in a vector together to compute the probability distribution.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method returning false means that Sparsemax cannot work on just one number at a time.
    /// It needs to see all the values together (like all scores for different classes) to determine
    /// which ones should get non-zero probabilities.
    /// </para>
    /// </remarks>
    protected override bool SupportsScalarOperations() => false;

    /// <summary>
    /// Applies the Sparsemax activation function to a vector of inputs.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A vector where some elements may be exactly zero, and all elements sum to 1.</returns>
    /// <remarks>
    /// <para>
    /// Sparsemax projects the input vector onto the probability simplex, potentially setting some values to exactly zero.
    /// The algorithm finds a threshold value and sets all elements below this threshold to zero,
    /// while shifting the remaining elements to ensure they sum to 1.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method transforms a vector of numbers (like scores for different classes) into 
    /// probabilities that add up to 1. Here's how it works:
    /// 
    /// 1. Sort all the input values from highest to lowest
    /// 2. Find a "threshold" value by checking when the running average becomes greater than the next value
    /// 3. Subtract this threshold from all input values
    /// 4. Set any negative results to zero
    /// 
    /// The result is a probability distribution where:
    /// - Some values may be exactly zero (unlike Softmax where all values are positive)
    /// - All values add up to 1
    /// - Higher input values get higher probabilities
    /// 
    /// This creates "sparse" outputs where only the most important classes get non-zero probabilities,
    /// making the model's predictions easier to interpret.
    /// </para>
    /// </remarks>
    public override Vector<T> Activate(Vector<T> input)
    {
        if (input.Length == 0)
        {
            return new Vector<T>(0);
        }

        // Sort descending, find k(z) = max{k in [d] : 1 + k * z_(k) > sum_{j<=k} z_(j)}, take the
        // threshold tau = (sum_{j=1}^{k} z_(j) - 1) / k and return max(0, z - tau): the engine's
        // sparsemax, the same operation the tensor overload uses
        // sparsemax(z) = sparsemax(z - c): taking the maximum off first keeps the support test
        // 1 + k z_(k) > sum_{j<=k} z_(j) away from magnitudes where the 1 is absorbed (2e7 in float), where the
        // unshifted projection returned the raw inputs. A NaN input stays NaN instead of becoming 0.
        var engine = Engine;
        var z = Tensor<T>.FromVector(input);
        var shifted = engine.TensorSubtractScalar(z, engine.TensorMaxValue(z));
        return engine.TensorAdd(engine.Sparsemax(shifted, axis: -1), NaNOrZero(z)).ToVector();
    }

    /// <summary>
    /// Applies Sparsemax to a tensor via <c>Engine.Sparsemax</c> so the gradient tape
    /// records the op. The vector-level implementation is correct for inference but
    /// bypasses the tape entirely.
    /// </summary>
    /// <remarks>
    /// Each slice is shifted by its own maximum first, as the vector overload does: sparsemax(z) = sparsemax(z - c),
    /// and without the shift the support test 1 + k z_(k) &gt; sum z_(j) loses its 1 at large magnitudes (2e7 in
    /// float), so no element entered the support and the raw logits came back. The maximum is taken off the tape:
    /// every row of the sparsemax Jacobian sums to zero, so a shift common to a slice has no gradient, and the
    /// engine's own Sparsemax backward stays the recorded operation.
    /// </remarks>
    public override Tensor<T> Activate(Tensor<T> input)
    {
        var engine = Engine;
        if (input.Rank == 0 || input.Length == 0)
        {
            return engine.Sparsemax(input, axis: -1);
        }

        Tensor<T> maximum;
        using (new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>())
        {
            maximum = engine.TensorBroadcastTo(engine.ReduceMax(input, new[] { -1 }, true), input.Shape.ToArray());
        }

        return engine.Sparsemax(engine.TensorSubtract(input, maximum), axis: -1);
    }

    /// <summary>
    /// Calculates the Jacobian matrix of the Sparsemax function for a given input vector.
    /// </summary>
    /// <param name="input">The input vector.</param>
    /// <returns>A matrix representing the partial derivatives of each output with respect to each input.</returns>
    /// <remarks>
    /// <para>
    /// The Jacobian matrix contains the partial derivatives of each output element with respect to each input element.
    /// For Sparsemax, the Jacobian has a specific structure based on which outputs are non-zero.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The derivative of Sparsemax is more complex than simpler activation functions because
    /// changing one input can affect multiple outputs. This method returns a "Jacobian matrix" which shows
    /// how each output value changes when each input value changes.
    /// 
    /// Some key points about the Sparsemax derivative:
    /// - For outputs that are zero, the derivatives are also zero (these outputs don't respond to small input changes)
    /// - The non-zero outputs (the "support", k of them) all equal their input minus one shared threshold, and that
    ///   threshold is the average of the support's inputs minus 1/k. So:
    ///   - When i=j (diagonal elements): the derivative is 1 - 1/k
    ///   - When i and j are different members of the support: the derivative is -1/k
    ///   - When j is outside the support: the derivative is 0
    ///
    /// In matrix form this is diag(s) - s s^T / k, where s is 1 on the support and 0 elsewhere (Martins and
    /// Astudillo, 2016, equation 11).
    /// 
    /// During neural network training, this matrix helps determine how to adjust the weights based on errors.
    /// The sparsity of Sparsemax can make training more efficient because many derivatives will be zero.
    /// </para>
    /// </remarks>
    public override Matrix<T> Derivative(Vector<T> input)
    {
        int d = input.Length;
        if (d == 0)
        {
            return new Matrix<T>(0, 0);
        }

        var engine = Engine;
        if (UseScalarJacobian(d))
        {
            return ScalarJacobian(Activate(input));
        }

        var output = Tensor<T>.FromVector(Activate(input));

        // s[i] = 1 where the output is positive (the support), 0 elsewhere
        var support = engine.TensorGreaterThan(output, NumOps.Zero);
        T supportSize = engine.TensorSum(support);
        if (!NumOps.GreaterThan(supportSize, NumOps.Zero))
        {
            // Only a NaN input leaves the support empty; there is nothing to differentiate
            return new Matrix<T>(d, d);
        }

        // On the support the output is input - tau with tau = (sum of the support's inputs - 1) / |S|,
        // so J = diag(s) - s s^T / |S|
        var shared = engine.TensorDivideScalar(engine.TensorOuter(support, support), supportSize);
        return Matrix<T>.FromMemory(engine.TensorSubtract(engine.TensorDiag(support), shared).ToArray(), d, d);
    }

    [AiDotNet.Attributes.ScalarKernel("host vector in, host matrix out: below the measured break-even length, and on a device engine at any "
        + "length, the fixed cost of the engine operations exceeds the scalar arithmetic")]
    private Matrix<T> ScalarJacobian(Vector<T> output)
    {
        int d = output.Length;
        var o = output.AsSpan();
        int supportSize = 0;
        for (int i = 0; i < d; i++)
        {
            if (NumOps.GreaterThan(o[i], NumOps.Zero))
            {
                supportSize++;
            }
        }

        var jacobian = new Matrix<T>(d, d);
        if (supportSize == 0)
        {
            return jacobian;
        }

        // diag(s) - s s^T / |S| with s the support indicator
        T shared = NumOps.Divide(NumOps.One, NumOps.FromDouble(supportSize));
        T offDiagonal = NumOps.Negate(shared);
        T diagonal = NumOps.Subtract(NumOps.One, shared);
        var cells = jacobian.AsWritableSpan();
        for (int i = 0; i < d; i++)
        {
            if (!NumOps.GreaterThan(o[i], NumOps.Zero))
            {
                continue;
            }

            int row = i * d;
            for (int j = 0; j < d; j++)
            {
                if (NumOps.GreaterThan(o[j], NumOps.Zero))
                {
                    cells[row + j] = offDiagonal;
                }
            }

            cells[row + i] = diagonal;
        }

        return jacobian;
    }

}
