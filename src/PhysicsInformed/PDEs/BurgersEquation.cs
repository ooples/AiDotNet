using System;
using System.Numerics;
using AiDotNet.PhysicsInformed.Interfaces;

namespace AiDotNet.PhysicsInformed.PDEs
{
    /// <summary>
    /// Represents the Burgers' Equation: ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Burgers' Equation is a fundamental PDE that combines:
    /// 1. Nonlinear convection (u * ∂u/∂x): The solution advects (moves) at its own speed
    /// 2. Diffusion (ν * ∂²u/∂x²): The solution spreads out over time
    ///
    /// Physical Interpretation:
    /// - Models simplified fluid dynamics (1D version of Navier-Stokes)
    /// - u(x,t) can represent fluid velocity at position x and time t
    /// - ν (nu) is the viscosity - controls how much the solution smooths out
    /// - The nonlinear term creates shock waves and turbulence-like behavior
    ///
    /// Key Feature:
    /// The nonlinearity makes this equation challenging - it can develop discontinuities (shocks)
    /// even from smooth initial conditions. This makes it a perfect benchmark for PINNs.
    ///
    /// Applications:
    /// - Gas dynamics
    /// - Traffic flow modeling
    /// - Shock wave formation
    /// - Turbulence studies
    /// </remarks>
    public class BurgersEquation<T> : IPDESpecification<T> where T : struct, INumber<T>
    {
        private readonly T _viscosity;

        /// <summary>
        /// Initializes a new instance of Burgers' Equation.
        /// </summary>
        /// <param name="viscosity">The viscosity coefficient ν (must be non-negative). Set to 0 for inviscid Burgers.</param>
        public BurgersEquation(T viscosity)
        {
            if (viscosity < T.Zero)
            {
                throw new ArgumentException("Viscosity must be non-negative.", nameof(viscosity));
            }
            _viscosity = viscosity;
        }

        /// <inheritdoc/>
        public T ComputeResidual(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            if (derivatives.FirstDerivatives == null)
            {
                throw new ArgumentException("Burgers' equation requires first derivatives.");
            }

            // For 1D Burgers equation: inputs = [x, t], outputs = [u]
            // PDE: ∂u/∂t + u * ∂u/∂x - ν * ∂²u/∂x² = 0

            T u = outputs[0];
            T dudt = derivatives.FirstDerivatives[0, 1]; // ∂u/∂t
            T dudx = derivatives.FirstDerivatives[0, 0]; // ∂u/∂x

            T convectionTerm = u * dudx; // Nonlinear convection
            T diffusionTerm = T.Zero;

            if (_viscosity > T.Zero && derivatives.SecondDerivatives != null)
            {
                T d2udx2 = derivatives.SecondDerivatives[0, 0, 0]; // ∂²u/∂x²
                diffusionTerm = _viscosity * d2udx2;
            }

            T residual = dudt + convectionTerm - diffusionTerm;
            return residual;
        }

        /// <inheritdoc/>
        public int InputDimension => 2; // [x, t]

        /// <inheritdoc/>
        public int OutputDimension => 1; // [u]

        /// <inheritdoc/>
        public string Name => $"Burgers' Equation (ν={_viscosity})";
    }
}
