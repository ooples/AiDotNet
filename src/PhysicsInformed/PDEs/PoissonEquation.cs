using System;
using System.Numerics;
using AiDotNet.PhysicsInformed.Interfaces;

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
    public class PoissonEquation<T> : IPDESpecification<T> where T : struct, INumber<T>
    {
        private readonly Func<T[], T> _sourceFunction;
        private readonly int _spatialDimension;

        /// <summary>
        /// Initializes a new instance of the Poisson Equation.
        /// </summary>
        /// <param name="sourceFunction">The source term f(x,y,...). Set to null or return zero for Laplace's equation.</param>
        /// <param name="spatialDimension">The number of spatial dimensions (1, 2, or 3).</param>
        public PoissonEquation(Func<T[], T>? sourceFunction = null, int spatialDimension = 2)
        {
            if (spatialDimension < 1 || spatialDimension > 3)
            {
                throw new ArgumentException("Spatial dimension must be 1, 2, or 3.", nameof(spatialDimension));
            }

            _sourceFunction = sourceFunction ?? (x => T.Zero);
            _spatialDimension = spatialDimension;
        }

        /// <inheritdoc/>
        public T ComputeResidual(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            if (derivatives.SecondDerivatives == null)
            {
                throw new ArgumentException("Poisson equation requires second derivatives.");
            }

            // Compute Laplacian: ∇²u = ∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²
            T laplacian = T.Zero;

            for (int i = 0; i < _spatialDimension; i++)
            {
                laplacian += derivatives.SecondDerivatives[0, i, i]; // ∂²u/∂xi²
            }

            T sourceValue = _sourceFunction(inputs);

            // PDE: ∇²u - f = 0
            T residual = laplacian - sourceValue;
            return residual;
        }

        /// <inheritdoc/>
        public int InputDimension => _spatialDimension;

        /// <inheritdoc/>
        public int OutputDimension => 1; // [u]

        /// <inheritdoc/>
        public string Name => _sourceFunction == null
            ? $"Laplace Equation ({_spatialDimension}D)"
            : $"Poisson Equation ({_spatialDimension}D)";
    }
}
