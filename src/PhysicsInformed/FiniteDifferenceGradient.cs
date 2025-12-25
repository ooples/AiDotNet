using System;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.PhysicsInformed
{
    /// <summary>
    /// Computes gradients using finite difference approximation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>IMPORTANT:</b> This class should NOT be used for neural network training gradients.
    /// Neural network training should use standard backpropagation via the Backward() method.
    /// </para>
    /// <para>
    /// <b>Valid use cases:</b>
    /// - Computing physics/simulation gradients (e.g., spatial derivatives for PDEs)
    /// - Debugging/verification of analytic gradients
    /// - Computing gradients where backpropagation is not available
    /// </para>
    /// <para>
    /// <b>Why not for training?</b>
    /// - Finite difference is slow (requires 2N forward passes for N parameters)
    /// - Less accurate than analytic gradients from backpropagation
    /// - Standard neural network layers all support backpropagation
    /// </para>
    /// <para>
    /// For neural network training, use the standard pattern:
    /// <code>
    /// var prediction = Forward(input);
    /// Backward(outputGradient);
    /// _optimizer.UpdateParameters(Layers);
    /// </code>
    /// </para>
    /// </remarks>
    internal static class FiniteDifferenceGradient
    {
        /// <summary>
        /// Computes gradients using central finite difference approximation.
        /// </summary>
        /// <typeparam name="T">The numeric type.</typeparam>
        /// <param name="lossFunction">Function that computes the loss given current parameters.</param>
        /// <param name="parameters">Current parameter values.</param>
        /// <param name="applyParameters">Action to apply updated parameters.</param>
        /// <param name="numOps">Numeric operations for type T.</param>
        /// <param name="epsilon">Perturbation size for finite difference (default: 1e-5).</param>
        /// <returns>Gradient vector with same length as parameters.</returns>
        /// <remarks>
        /// Uses central difference formula: df/dx ≈ (f(x+h) - f(x-h)) / (2h)
        /// This is O(h^2) accurate, meaning error decreases with the square of epsilon.
        /// </remarks>
        public static Vector<T> Compute<T>(
            Func<T> lossFunction,
            Vector<T> parameters,
            Action<Vector<T>> applyParameters,
            INumericOperations<T> numOps,
            double epsilon = 1e-5)
        {
            if (lossFunction == null)
            {
                throw new ArgumentNullException(nameof(lossFunction));
            }

            if (parameters == null)
            {
                throw new ArgumentNullException(nameof(parameters));
            }

            if (applyParameters == null)
            {
                throw new ArgumentNullException(nameof(applyParameters));
            }

            if (numOps == null)
            {
                throw new ArgumentNullException(nameof(numOps));
            }

            if (epsilon <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(epsilon), epsilon,
                    "Epsilon must be positive to avoid division by zero and ensure accurate gradients.");
            }

            if (double.IsNaN(epsilon) || double.IsInfinity(epsilon))
            {
                throw new ArgumentOutOfRangeException(nameof(epsilon), epsilon,
                    "Epsilon must be a finite positive number.");
            }

            var gradient = new Vector<T>(parameters.Length);
            var epsilonT = numOps.FromDouble(epsilon);
            var twoEpsilon = numOps.FromDouble(2.0 * epsilon);

            for (int i = 0; i < parameters.Length; i++)
            {
                var original = parameters[i];

                parameters[i] = numOps.Add(original, epsilonT);
                applyParameters(parameters);
                var lossPlus = lossFunction();

                parameters[i] = numOps.Subtract(original, epsilonT);
                applyParameters(parameters);
                var lossMinus = lossFunction();

                gradient[i] = numOps.Divide(numOps.Subtract(lossPlus, lossMinus), twoEpsilon);
                parameters[i] = original;
            }

            applyParameters(parameters);

            return gradient;
        }
    }
}
