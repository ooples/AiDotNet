using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
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
    public class BurgersEquation<T> : PDESpecificationBase<T>, IPDEResidualGradient<T>
    {
        private readonly T _viscosity;

        /// <summary>
        /// Initializes a new instance of Burgers' Equation.
        /// </summary>
        /// <param name="viscosity">The viscosity coefficient ν (must be non-negative). Set to 0 for inviscid Burgers.</param>
        public BurgersEquation(T viscosity)
        {
            ValidateNonNegative(viscosity, nameof(viscosity));
            _viscosity = viscosity;
        }

        /// <summary>
        /// Initializes a new instance of Burgers' Equation with double parameter.
        /// </summary>
        /// <param name="viscosity">The viscosity coefficient ν (default 0.01, must be non-negative).</param>
        public BurgersEquation(double viscosity = 0.01)
            : this(MathHelper.GetNumericOperations<T>().FromDouble(viscosity))
        {
        }

        /// <inheritdoc/>
        public override T ComputeResidual(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateFirstDerivatives(derivatives);

            var firstDerivs = derivatives.FirstDerivatives;
            if (firstDerivs is null)
            {
                throw new InvalidOperationException("First derivatives were null after validation.");
            }

            // For 1D Burgers equation: inputs = [x, t], outputs = [u]
            // PDE: ∂u/∂t + u * ∂u/∂x - ν * ∂²u/∂x² = 0

            T u = outputs[0];
            T dudt = firstDerivs[0, 1]; // ∂u/∂t
            T dudx = firstDerivs[0, 0]; // ∂u/∂x

            T convectionTerm = NumOps.Multiply(u, dudx); // Nonlinear convection
            T diffusionTerm = NumOps.Zero;

            if (NumOps.GreaterThan(_viscosity, NumOps.Zero))
            {
                var secondDerivs = derivatives.SecondDerivatives;
                if (secondDerivs is null)
                {
                    throw new ArgumentException("Burgers' equation with viscosity requires second derivatives.");
                }

                T d2udx2 = secondDerivs[0, 0, 0]; // ∂²u/∂x²
                diffusionTerm = NumOps.Multiply(_viscosity, d2udx2);
            }

            T residual = NumOps.Subtract(NumOps.Add(dudt, convectionTerm), diffusionTerm);
            return residual;
        }

        /// <inheritdoc/>
        public PDEResidualGradient<T> ComputeResidualGradient(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            var firstDerivs = derivatives.FirstDerivatives;
            if (firstDerivs is null)
            {
                throw new ArgumentException("Burgers' equation requires first derivatives.");
            }

            if (NumOps.GreaterThan(_viscosity, NumOps.Zero) && derivatives.SecondDerivatives is null)
            {
                throw new ArgumentException("Burgers' equation with viscosity requires second derivatives.");
            }

            var gradient = CreateGradient();
            T dudx = firstDerivs[0, 0];
            gradient.OutputGradients[0] = dudx;
            gradient.FirstDerivatives[0, 0] = outputs[0];
            gradient.FirstDerivatives[0, 1] = NumOps.One;

            if (NumOps.GreaterThan(_viscosity, NumOps.Zero))
            {
                gradient.SecondDerivatives[0, 0, 0] = NumOps.Negate(_viscosity);
            }

            return gradient;
        }

        /// <inheritdoc/>
        public override int InputDimension => 2; // [x, t]

        /// <inheritdoc/>
        public override int OutputDimension => 1; // [u]

        /// <inheritdoc/>
        public override string Name => $"Burgers' Equation (ν={_viscosity})";
    }
}
