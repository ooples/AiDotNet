using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.PhysicsInformed.PDEs
{
    /// <summary>
    /// Represents the Poisson Equation: ∇²u = f(x,y)
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// The Poisson Equation is one of the most important equations in physics and engineering:
    /// - ∇²u (Laplacian of u) = ∂²u/∂x² + ∂²u/∂y² (+ ∂²u/∂z² in 3D)
    /// - f(x,y) is a source term (known function)
    ///
    /// Physical Interpretation:
    /// - Models steady-state (time-independent) phenomena
    /// - u could represent: temperature, electric potential, pressure, concentration, etc.
    /// - f represents sources (+) and sinks (-) in the domain
    ///
    /// Special Case:
    /// When f = 0, it becomes Laplace's Equation (∇²u = 0), which models equilibrium states.
    ///
    /// Applications:
    /// - Electrostatics: Electric potential from charge distribution
    /// - Steady heat conduction: Temperature distribution with heat sources
    /// - Fluid dynamics: Pressure field in incompressible flow
    /// - Gravitational potential: From mass distribution
    /// - Image processing: Image reconstruction and smoothing
    ///
    /// Example:
    /// Temperature in a metal plate with heat sources/sinks reaches a steady state
    /// described by the Poisson equation.
    /// </remarks>
    public class PoissonEquation<T> : PDESpecificationBase<T>, IPDEResidualGradient<T>
    {
        private readonly Func<Tensor<T>, Tensor<T>>? _sourceFunction;
        private readonly int _spatialDimension;
        private readonly bool _isLaplace;

        /// <summary>
        /// Initializes a new instance of the Poisson Equation.
        /// </summary>
        /// <param name="sourceFunction">
        /// The source term f, evaluated for a batch of points: it maps coordinates [N, spatialDimension] to source
        /// values [N] with engine operations. Null gives Laplace's equation (f = 0).
        /// </param>
        /// <param name="spatialDimension">The number of spatial dimensions (default 2).</param>
        public PoissonEquation(Func<Tensor<T>, Tensor<T>>? sourceFunction = null, int spatialDimension = 2)
        {
            if (spatialDimension < 1 || spatialDimension > 3)
            {
                throw new ArgumentException("Spatial dimension must be 1, 2, or 3.", nameof(spatialDimension));
            }

            _isLaplace = sourceFunction == null;
            _sourceFunction = sourceFunction;
            _spatialDimension = spatialDimension;
        }

        /// <inheritdoc/>
        public override T ComputeResidual(Vector<T> inputs, Vector<T> outputs, PDEDerivatives<T> derivatives)
        {
            if (derivatives.SecondDerivatives is null)
            {
                throw new ArgumentException("Poisson equation requires second derivatives.");
            }

            return EvaluatePointResidual(inputs, outputs, derivatives);
        }

        /// <inheritdoc/>
        /// <remarks>The residual is laplacian(u) - f.</remarks>
        public override Tensor<T> ComputeTapeResidual(Tensor<T> inputs, Tensor<T> outputs, PDEDerivatives<T> derivatives)
        {
            if (derivatives.SecondDerivatives is null)
            {
                throw new ArgumentException("Poisson equation requires second derivatives.");
            }

            var engine = TapeEngine;
            var laplacian = D2(derivatives, 0, 0, 0);
            for (int i = 1; i < _spatialDimension; i++)
            {
                laplacian = engine.TensorAdd(laplacian, D2(derivatives, 0, i, i));
            }

            if (_sourceFunction is null)
            {
                return laplacian;
            }

            var source = _sourceFunction(inputs);
            if (source.Length != laplacian.Length)
            {
                throw new InvalidOperationException(
                    $"The Poisson source must return one value per point ({laplacian.Length}); got {source.Length}.");
            }

            return engine.TensorSubtract(laplacian, engine.Reshape(source, laplacian.Shape.ToArray()));
        }

        /// <inheritdoc/>
        public PDEResidualGradient<T> ComputeResidualGradient(Vector<T> inputs, Vector<T> outputs, PDEDerivatives<T> derivatives)
        {
            if (derivatives.SecondDerivatives is null)
            {
                throw new ArgumentException("Poisson equation requires second derivatives.");
            }

            // d(laplacian u)/d(u_ii) = 1 on the diagonal of the Hessian.
            var gradient = CreateGradient();
            var identity = TapeEngine.TensorEye<T>(_spatialDimension);
            identity.AsSpan().CopyTo(gradient.SecondDerivatives.AsWritableSpan());
            return gradient;
        }

        /// <inheritdoc/>
        public override int InputDimension => _spatialDimension;

        /// <inheritdoc/>
        public override int OutputDimension => 1; // [u]

        /// <inheritdoc/>
        public override string Name => _isLaplace
            ? $"Laplace Equation ({_spatialDimension}D)"
            : $"Poisson Equation ({_spatialDimension}D)";
    }
}

