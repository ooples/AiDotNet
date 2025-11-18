using System;
using System.Numerics;
using AiDotNet.PhysicsInformed.Interfaces;

namespace AiDotNet.PhysicsInformed
{
    /// <summary>
    /// Provides automatic differentiation capabilities for computing derivatives needed in PINNs.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Automatic Differentiation (AD) is a technique for computing derivatives exactly and efficiently.
    /// Unlike numerical differentiation (finite differences) which is approximate and slow,
    /// AD uses the chain rule of calculus to compute exact derivatives.
    ///
    /// How It Works:
    /// 1. Forward Mode AD: Computes derivatives along with the function value in one pass
    /// 2. Reverse Mode AD (backpropagation): Used in neural networks for efficiency
    ///
    /// For PINNs, we need:
    /// - First derivatives (∂u/∂x, ∂u/∂t, etc.)
    /// - Second derivatives (∂²u/∂x², ∂²u/∂x∂y, etc.)
    /// - Mixed derivatives
    ///
    /// Why This Matters:
    /// PDEs involve derivatives of the solution. To train a PINN, we need to:
    /// 1. Compute u(x,t) using the neural network (forward pass)
    /// 2. Compute ∂u/∂x, ∂u/∂t automatically
    /// 3. Use these in the PDE residual
    /// 4. Backpropagate through everything to train the network
    ///
    /// This is "differentiating the differentiator" - we differentiate the network's output
    /// with respect to its inputs, then use those derivatives in the loss, then differentiate
    /// the loss with respect to network parameters. Mind-bending but powerful!
    /// </remarks>
    public static class AutomaticDifferentiation<T> where T : struct, INumber<T>
    {
        /// <summary>
        /// Computes first and second derivatives using finite differences (numerical approximation).
        /// </summary>
        /// <param name="networkFunction">The neural network function to differentiate.</param>
        /// <param name="inputs">The point at which to compute derivatives.</param>
        /// <param name="outputDim">The dimension of the network output.</param>
        /// <param name="epsilon">The step size for finite differences (smaller = more accurate but less stable).</param>
        /// <returns>A PDEDerivatives object containing first and second derivatives.</returns>
        /// <remarks>
        /// For Beginners:
        /// This uses finite differences: f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
        /// We compute the function at nearby points and estimate the slope.
        ///
        /// Limitations:
        /// - Approximation (not exact)
        /// - Sensitive to epsilon choice (too small → numerical errors, too large → approximation errors)
        /// - Slow for high dimensions
        ///
        /// For production PINNs, you'd want to use true automatic differentiation from a framework
        /// like TensorFlow or PyTorch. This implementation is educational and works for small problems.
        /// </remarks>
        public static PDEDerivatives<T> ComputeDerivatives(
            Func<T[], T[]> networkFunction,
            T[] inputs,
            int outputDim,
            T? epsilon = null)
        {
            T h = epsilon ?? T.CreateChecked(1e-5); // Default step size
            int inputDim = inputs.Length;

            // Initialize derivative structures
            var derivatives = new PDEDerivatives<T>
            {
                FirstDerivatives = new T[outputDim, inputDim],
                SecondDerivatives = new T[outputDim, inputDim, inputDim]
            };

            // Compute base output
            T[] baseOutput = networkFunction(inputs);

            // Compute first derivatives using central differences
            for (int i = 0; i < inputDim; i++)
            {
                T[] inputsPlus = (T[])inputs.Clone();
                T[] inputsMinus = (T[])inputs.Clone();

                inputsPlus[i] += h;
                inputsMinus[i] -= h;

                T[] outputPlus = networkFunction(inputsPlus);
                T[] outputMinus = networkFunction(inputsMinus);

                for (int j = 0; j < outputDim; j++)
                {
                    // Central difference: (f(x+h) - f(x-h)) / (2h)
                    derivatives.FirstDerivatives[j, i] = (outputPlus[j] - outputMinus[j]) / (T.CreateChecked(2) * h);
                }
            }

            // Compute second derivatives
            for (int i = 0; i < inputDim; i++)
            {
                for (int k = 0; k < inputDim; k++)
                {
                    T[] inputsPP = (T[])inputs.Clone();
                    T[] inputsPM = (T[])inputs.Clone();
                    T[] inputsMP = (T[])inputs.Clone();
                    T[] inputsMM = (T[])inputs.Clone();

                    inputsPP[i] += h;
                    inputsPP[k] += h;

                    inputsPM[i] += h;
                    inputsPM[k] -= h;

                    inputsMP[i] -= h;
                    inputsMP[k] += h;

                    inputsMM[i] -= h;
                    inputsMM[k] -= h;

                    T[] outputPP = networkFunction(inputsPP);
                    T[] outputPM = networkFunction(inputsPM);
                    T[] outputMP = networkFunction(inputsMP);
                    T[] outputMM = networkFunction(inputsMM);

                    for (int j = 0; j < outputDim; j++)
                    {
                        // Mixed partial derivative: (f(x+h,y+h) - f(x+h,y-h) - f(x-h,y+h) + f(x-h,y-h)) / (4h²)
                        if (i == k)
                        {
                            // Second derivative with respect to same variable: (f(x+h) - 2f(x) + f(x-h)) / h²
                            T[] inputsPlus = (T[])inputs.Clone();
                            T[] inputsMinus = (T[])inputs.Clone();
                            inputsPlus[i] += h;
                            inputsMinus[i] -= h;

                            T[] outputPlus = networkFunction(inputsPlus);
                            T[] outputMinus = networkFunction(inputsMinus);

                            derivatives.SecondDerivatives[j, i, k] =
                                (outputPlus[j] - T.CreateChecked(2) * baseOutput[j] + outputMinus[j]) / (h * h);
                        }
                        else
                        {
                            // Mixed partial derivative
                            derivatives.SecondDerivatives[j, i, k] =
                                (outputPP[j] - outputPM[j] - outputMP[j] + outputMM[j]) /
                                (T.CreateChecked(4) * h * h);
                        }
                    }
                }
            }

            return derivatives;
        }

        /// <summary>
        /// Computes only first derivatives (faster when second derivatives aren't needed).
        /// </summary>
        public static T[,] ComputeGradient(
            Func<T[], T[]> networkFunction,
            T[] inputs,
            int outputDim,
            T? epsilon = null)
        {
            T h = epsilon ?? T.CreateChecked(1e-5);
            int inputDim = inputs.Length;
            T[,] gradient = new T[outputDim, inputDim];

            for (int i = 0; i < inputDim; i++)
            {
                T[] inputsPlus = (T[])inputs.Clone();
                T[] inputsMinus = (T[])inputs.Clone();

                inputsPlus[i] += h;
                inputsMinus[i] -= h;

                T[] outputPlus = networkFunction(inputsPlus);
                T[] outputMinus = networkFunction(inputsMinus);

                for (int j = 0; j < outputDim; j++)
                {
                    gradient[j, i] = (outputPlus[j] - outputMinus[j]) / (T.CreateChecked(2) * h);
                }
            }

            return gradient;
        }

        /// <summary>
        /// Computes the Jacobian matrix of a vector-valued function.
        /// </summary>
        /// <param name="networkFunction">The function f: R^n → R^m to differentiate.</param>
        /// <param name="inputs">The point at which to compute the Jacobian.</param>
        /// <param name="outputDim">The dimension m of the output.</param>
        /// <param name="epsilon">The step size for finite differences.</param>
        /// <returns>The Jacobian matrix J[i,j] = ∂f_i/∂x_j.</returns>
        /// <remarks>
        /// For Beginners:
        /// The Jacobian is a matrix of all first-order partial derivatives.
        /// For a function f: R^n → R^m, the Jacobian is an m×n matrix where:
        /// - Row i contains all partial derivatives of output i
        /// - Column j contains derivatives with respect to input j
        ///
        /// Example:
        /// If f(x,y) = [x², xy, y²], then the Jacobian is:
        /// J = [2x   0  ]
        ///     [y    x  ]
        ///     [0    2y ]
        /// </remarks>
        public static T[,] ComputeJacobian(
            Func<T[], T[]> networkFunction,
            T[] inputs,
            int outputDim,
            T? epsilon = null)
        {
            return ComputeGradient(networkFunction, inputs, outputDim, epsilon);
        }

        /// <summary>
        /// Computes the Hessian matrix of a scalar function.
        /// </summary>
        /// <param name="scalarFunction">The function f: R^n → R to differentiate.</param>
        /// <param name="inputs">The point at which to compute the Hessian.</param>
        /// <param name="epsilon">The step size for finite differences.</param>
        /// <returns>The Hessian matrix H[i,j] = ∂²f/∂x_i∂x_j.</returns>
        /// <remarks>
        /// For Beginners:
        /// The Hessian is a square matrix of all second-order partial derivatives.
        /// It describes the curvature of a function.
        ///
        /// Properties:
        /// - Symmetric: H[i,j] = H[j,i] (by Schwarz's theorem)
        /// - Positive definite → local minimum
        /// - Negative definite → local maximum
        /// - Indefinite → saddle point
        ///
        /// Example:
        /// For f(x,y) = x² + xy + y²:
        /// H = [2  1]
        ///     [1  2]
        /// </remarks>
        public static T[,] ComputeHessian(
            Func<T[], T> scalarFunction,
            T[] inputs,
            T? epsilon = null)
        {
            T h = epsilon ?? T.CreateChecked(1e-5);
            int n = inputs.Length;
            T[,] hessian = new T[n, n];

            T baseValue = scalarFunction(inputs);

            for (int i = 0; i < n; i++)
            {
                for (int j = i; j < n; j++) // Only compute upper triangle (symmetric)
                {
                    if (i == j)
                    {
                        // Diagonal: ∂²f/∂x_i²
                        T[] inputsPlus = (T[])inputs.Clone();
                        T[] inputsMinus = (T[])inputs.Clone();
                        inputsPlus[i] += h;
                        inputsMinus[i] -= h;

                        T valuePlus = scalarFunction(inputsPlus);
                        T valueMinus = scalarFunction(inputsMinus);

                        hessian[i, j] = (valuePlus - T.CreateChecked(2) * baseValue + valueMinus) / (h * h);
                    }
                    else
                    {
                        // Off-diagonal: ∂²f/∂x_i∂x_j
                        T[] inputsPP = (T[])inputs.Clone();
                        T[] inputsPM = (T[])inputs.Clone();
                        T[] inputsMP = (T[])inputs.Clone();
                        T[] inputsMM = (T[])inputs.Clone();

                        inputsPP[i] += h;
                        inputsPP[j] += h;
                        inputsPM[i] += h;
                        inputsPM[j] -= h;
                        inputsMP[i] -= h;
                        inputsMP[j] += h;
                        inputsMM[i] -= h;
                        inputsMM[j] -= h;

                        T valuePP = scalarFunction(inputsPP);
                        T valuePM = scalarFunction(inputsPM);
                        T valueMP = scalarFunction(inputsMP);
                        T valueMM = scalarFunction(inputsMM);

                        hessian[i, j] = (valuePP - valuePM - valueMP + valueMM) / (T.CreateChecked(4) * h * h);
                        hessian[j, i] = hessian[i, j]; // Symmetry
                    }
                }
            }

            return hessian;
        }
    }
}
