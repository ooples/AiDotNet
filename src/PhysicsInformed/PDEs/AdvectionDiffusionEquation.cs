using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;

namespace AiDotNet.PhysicsInformed.PDEs
{
    /// <summary>
    /// Represents the Advection-Diffusion Equation:
    /// ∂c/∂t + v·∇c = D∇²c + S
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// The Advection-Diffusion equation models the transport of a substance (like heat,
    /// pollutants, or chemicals) by both fluid flow (advection) and random molecular
    /// motion (diffusion).
    ///
    /// Variables:
    /// - c(x,y,t) or c(x,t) = Concentration of the transported quantity
    /// - v = (vₓ, vᵧ) = Velocity field (how fast the fluid flows)
    /// - D = Diffusion coefficient (how fast the substance spreads randomly)
    /// - S = Source/sink term (where substance is added or removed)
    ///
    /// Physical Interpretation:
    /// - Advection: The substance is carried along by the flowing fluid (like leaves in a river)
    /// - Diffusion: The substance spreads from high to low concentration (like a drop of ink in water)
    /// - The equation combines both effects
    ///
    /// Dimensionless Numbers:
    /// - Péclet number (Pe = vL/D) measures advection vs diffusion strength
    ///   * Pe >> 1: Advection dominates (sharp fronts)
    ///   * Pe << 1: Diffusion dominates (smooth profiles)
    ///
    /// Applications:
    /// - Environmental engineering (pollution transport in air/water)
    /// - Chemical engineering (reactor design)
    /// - Hydrology (groundwater contamination)
    /// - Heat transfer in moving fluids
    /// - Drug delivery in blood flow
    ///
    /// Example: Smoke dispersing from a chimney is advected by wind while
    /// simultaneously diffusing into the surrounding air.
    /// </remarks>
    public class AdvectionDiffusionEquation<T> : PDESpecificationBase<T>, IPDEResidualGradient<T>
    {
        private readonly T _diffusionCoeff;
        private readonly T _velocityX;
        private readonly T _velocityY;
        private readonly T _sourceterm;
        private readonly bool _is2D;

        /// <summary>
        /// Initializes a new instance of the 1D Advection-Diffusion Equation.
        /// </summary>
        /// <param name="diffusionCoeff">Diffusion coefficient D (must be non-negative).</param>
        /// <param name="velocityX">Advection velocity in x-direction.</param>
        /// <param name="sourceTerm">Source/sink term S (default 0).</param>
        public AdvectionDiffusionEquation(T diffusionCoeff, T velocityX, T? sourceTerm = default)
        {
            ValidateNonNegative(diffusionCoeff, nameof(diffusionCoeff));

            _diffusionCoeff = diffusionCoeff;
            _velocityX = velocityX;
            _velocityY = NumOps.Zero;
            _sourceterm = sourceTerm ?? NumOps.Zero;
            _is2D = false;
        }

        /// <summary>
        /// Initializes a new instance of the 2D Advection-Diffusion Equation.
        /// </summary>
        /// <param name="diffusionCoeff">Diffusion coefficient D (must be non-negative).</param>
        /// <param name="velocityX">Advection velocity in x-direction.</param>
        /// <param name="velocityY">Advection velocity in y-direction.</param>
        /// <param name="sourceTerm">Source/sink term S (default 0).</param>
        public AdvectionDiffusionEquation(T diffusionCoeff, T velocityX, T velocityY, T? sourceTerm = default)
        {
            ValidateNonNegative(diffusionCoeff, nameof(diffusionCoeff));

            _diffusionCoeff = diffusionCoeff;
            _velocityX = velocityX;
            _velocityY = velocityY;
            _sourceterm = sourceTerm ?? NumOps.Zero;
            _is2D = true;
        }

        /// <summary>
        /// Initializes a new instance of the 1D Advection-Diffusion Equation with double parameters.
        /// </summary>
        /// <param name="diffusionCoeff">Diffusion coefficient D (default 0.1, must be non-negative).</param>
        /// <param name="velocityX">Advection velocity in x-direction (default 1.0).</param>
        /// <param name="sourceTerm">Source/sink term S (default 0).</param>
        public AdvectionDiffusionEquation(double diffusionCoeff = 0.1, double velocityX = 1.0, double sourceTerm = 0)
            : this(
                MathHelper.GetNumericOperations<T>().FromDouble(diffusionCoeff),
                MathHelper.GetNumericOperations<T>().FromDouble(velocityX),
                MathHelper.GetNumericOperations<T>().FromDouble(sourceTerm))
        {
        }

        /// <summary>
        /// Initializes a new instance of the 2D Advection-Diffusion Equation with double parameters.
        /// Note: The velocityY parameter must be specified to use 2D mode.
        /// </summary>
        /// <param name="diffusionCoeff">Diffusion coefficient D (must be non-negative).</param>
        /// <param name="velocityX">Advection velocity in x-direction.</param>
        /// <param name="velocityY">Advection velocity in y-direction.</param>
        /// <param name="sourceTerm">Source/sink term S (default 0).</param>
        public AdvectionDiffusionEquation(double diffusionCoeff, double velocityX, double velocityY, double sourceTerm = 0)
            : this(
                MathHelper.GetNumericOperations<T>().FromDouble(diffusionCoeff),
                MathHelper.GetNumericOperations<T>().FromDouble(velocityX),
                MathHelper.GetNumericOperations<T>().FromDouble(velocityY),
                MathHelper.GetNumericOperations<T>().FromDouble(sourceTerm))
        {
        }

        /// <inheritdoc/>
        public override T ComputeResidual(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateSecondDerivatives(derivatives);

            var firstDerivs = derivatives.FirstDerivatives;
            var secondDerivs = derivatives.SecondDerivatives;

            if (firstDerivs is null || secondDerivs is null)
            {
                throw new InvalidOperationException("Derivatives were null after validation.");
            }

            if (_is2D)
            {
                // 2D case: inputs = [x, y, t], outputs = [c]
                // PDE: ∂c/∂t + vₓ∂c/∂x + vᵧ∂c/∂y - D(∂²c/∂x² + ∂²c/∂y²) - S = 0

                T dcdt = firstDerivs[0, 2]; // ∂c/∂t
                T dcdx = firstDerivs[0, 0]; // ∂c/∂x
                T dcdy = firstDerivs[0, 1]; // ∂c/∂y
                T d2cdx2 = secondDerivs[0, 0, 0]; // ∂²c/∂x²
                T d2cdy2 = secondDerivs[0, 1, 1]; // ∂²c/∂y²

                // Advection terms
                T advectionX = NumOps.Multiply(_velocityX, dcdx);
                T advectionY = NumOps.Multiply(_velocityY, dcdy);
                T advection = NumOps.Add(advectionX, advectionY);

                // Diffusion term
                T laplacian = NumOps.Add(d2cdx2, d2cdy2);
                T diffusion = NumOps.Multiply(_diffusionCoeff, laplacian);

                // Residual: ∂c/∂t + v·∇c - D∇²c - S
                T residual = NumOps.Subtract(NumOps.Add(dcdt, advection), diffusion);
                residual = NumOps.Subtract(residual, _sourceterm);

                return residual;
            }
            else
            {
                // 1D case: inputs = [x, t], outputs = [c]
                // PDE: ∂c/∂t + vₓ∂c/∂x - D∂²c/∂x² - S = 0

                T dcdt = firstDerivs[0, 1]; // ∂c/∂t
                T dcdx = firstDerivs[0, 0]; // ∂c/∂x
                T d2cdx2 = secondDerivs[0, 0, 0]; // ∂²c/∂x²

                // Advection term
                T advection = NumOps.Multiply(_velocityX, dcdx);

                // Diffusion term
                T diffusion = NumOps.Multiply(_diffusionCoeff, d2cdx2);

                // Residual: ∂c/∂t + v∂c/∂x - D∂²c/∂x² - S
                T residual = NumOps.Subtract(NumOps.Add(dcdt, advection), diffusion);
                residual = NumOps.Subtract(residual, _sourceterm);

                return residual;
            }
        }

        /// <inheritdoc/>
        public PDEResidualGradient<T> ComputeResidualGradient(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateSecondDerivatives(derivatives);

            var gradient = CreateGradient();

            if (_is2D)
            {
                // 2D gradients
                gradient.FirstDerivatives[0, 2] = NumOps.One; // ∂R/∂(∂c/∂t)
                gradient.FirstDerivatives[0, 0] = _velocityX; // ∂R/∂(∂c/∂x)
                gradient.FirstDerivatives[0, 1] = _velocityY; // ∂R/∂(∂c/∂y)
                gradient.SecondDerivatives[0, 0, 0] = NumOps.Negate(_diffusionCoeff); // ∂R/∂(∂²c/∂x²)
                gradient.SecondDerivatives[0, 1, 1] = NumOps.Negate(_diffusionCoeff); // ∂R/∂(∂²c/∂y²)
            }
            else
            {
                // 1D gradients
                gradient.FirstDerivatives[0, 1] = NumOps.One; // ∂R/∂(∂c/∂t)
                gradient.FirstDerivatives[0, 0] = _velocityX; // ∂R/∂(∂c/∂x)
                gradient.SecondDerivatives[0, 0, 0] = NumOps.Negate(_diffusionCoeff); // ∂R/∂(∂²c/∂x²)
            }

            return gradient;
        }

        /// <inheritdoc/>
        public override int InputDimension => _is2D ? 3 : 2; // [x, y, t] or [x, t]

        /// <inheritdoc/>
        public override int OutputDimension => 1; // [c] - concentration

        /// <inheritdoc/>
        public override string Name => _is2D
            ? $"2D Advection-Diffusion (D={_diffusionCoeff}, v=({_velocityX},{_velocityY}))"
            : $"1D Advection-Diffusion (D={_diffusionCoeff}, v={_velocityX})";
    }
}
