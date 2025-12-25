using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;

namespace AiDotNet.PhysicsInformed.PDEs
{
    /// <summary>
    /// Represents the incompressible Navier-Stokes equations for fluid dynamics.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// The Navier-Stokes equations describe the motion of viscous fluids like water, air, and oil.
    /// They are fundamental to understanding weather patterns, ocean currents, blood flow, and aerodynamics.
    ///
    /// The equations consist of:
    /// 1. **Continuity Equation** (mass conservation): The fluid cannot be created or destroyed
    ///    ∂u/∂x + ∂v/∂y = 0
    ///
    /// 2. **Momentum Equations**: Newton's second law for fluid motion
    ///    X-momentum: ∂u/∂t + u(∂u/∂x) + v(∂u/∂y) = -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
    ///    Y-momentum: ∂v/∂t + u(∂v/∂x) + v(∂v/∂y) = -∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)
    ///
    /// Variables:
    /// - u(x,y,t): Velocity in x-direction
    /// - v(x,y,t): Velocity in y-direction
    /// - p(x,y,t): Pressure
    /// - ν (nu): Kinematic viscosity (how "thick" the fluid is)
    ///
    /// Physical Interpretation:
    /// - Left side (∂u/∂t + u∂u/∂x + ...): How velocity changes following a fluid particle
    /// - Pressure terms (-∂p/∂x, -∂p/∂y): Forces from pressure differences
    /// - Viscous terms (ν∂²u/∂x²): Friction/drag within the fluid
    ///
    /// Applications:
    /// - Aircraft design (aerodynamics)
    /// - Weather prediction
    /// - Blood flow in arteries
    /// - Ocean currents
    /// - Pipe flow engineering
    /// </remarks>
    public class NavierStokesEquation<T> : PDESpecificationBase<T>, IPDEResidualGradient<T>
    {
        private readonly T _viscosity;
        private readonly T _density;

        /// <summary>
        /// Initializes a new instance of the Navier-Stokes equations.
        /// </summary>
        /// <param name="viscosity">The kinematic viscosity ν (must be positive). Water ≈ 1e-6 m²/s, Air ≈ 1.5e-5 m²/s</param>
        /// <param name="density">The fluid density ρ (must be positive). Water ≈ 1000 kg/m³, Air ≈ 1.2 kg/m³</param>
        /// <remarks>
        /// For Beginners:
        /// - Higher viscosity means thicker fluid (honey vs water)
        /// - The Reynolds number Re = UL/ν determines flow behavior:
        ///   Low Re (less than ~2000): Smooth laminar flow
        ///   High Re (greater than ~4000): Turbulent flow
        /// </remarks>
        public NavierStokesEquation(T viscosity, T? density = default)
        {
            ValidatePositive(viscosity, nameof(viscosity));

            _viscosity = viscosity;

            // Use provided density or default to 1
            if (density is null)
            {
                _density = NumOps.One;
            }
            else
            {
                ValidatePositive(density, nameof(density));
                _density = density;
            }
        }

        /// <summary>
        /// Initializes a new instance of the Navier-Stokes equations with double parameters.
        /// </summary>
        /// <param name="viscosity">Kinematic viscosity ν (default 0.01, similar to water)</param>
        /// <param name="density">Fluid density ρ (default 1.0, normalized)</param>
        public NavierStokesEquation(double viscosity = 0.01, double density = 1.0)
            : this(
                MathHelper.GetNumericOperations<T>().FromDouble(viscosity),
                MathHelper.GetNumericOperations<T>().FromDouble(density))
        {
        }

        /// <inheritdoc/>
        /// <remarks>
        /// Computes three residuals (one for each equation):
        /// - R1 (Continuity): ∂u/∂x + ∂v/∂y = 0
        /// - R2 (X-momentum): ∂u/∂t + u(∂u/∂x) + v(∂u/∂y) + ∂p/∂x - ν(∂²u/∂x² + ∂²u/∂y²) = 0
        /// - R3 (Y-momentum): ∂v/∂t + u(∂v/∂x) + v(∂v/∂y) + ∂p/∂y - ν(∂²v/∂x² + ∂²v/∂y²) = 0
        ///
        /// Returns the sum of squared residuals: R1² + R2² + R3²
        /// </remarks>
        public override T ComputeResidual(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateFirstDerivatives(derivatives);
            ValidateSecondDerivatives(derivatives);

            var firstDerivs = derivatives.FirstDerivatives;
            var secondDerivs = derivatives.SecondDerivatives;

            if (firstDerivs is null || secondDerivs is null)
            {
                throw new InvalidOperationException("Derivatives were null after validation.");
            }

            if (outputs.Length != 3)
            {
                throw new ArgumentException("Navier-Stokes expects 3 outputs: [u, v, p].");
            }

            // outputs = [u, v, p] (velocity x, velocity y, pressure)
            // inputs = [x, y, t]
            T u = outputs[0];
            T v = outputs[1];

            // First derivatives: [output_idx, input_idx]
            // Input indices: 0=x, 1=y, 2=t
            T dudx = firstDerivs[0, 0];  // ∂u/∂x
            T dudy = firstDerivs[0, 1];  // ∂u/∂y
            T dudt = firstDerivs[0, 2];  // ∂u/∂t

            T dvdx = firstDerivs[1, 0];  // ∂v/∂x
            T dvdy = firstDerivs[1, 1];  // ∂v/∂y
            T dvdt = firstDerivs[1, 2];  // ∂v/∂t

            T dpdx = firstDerivs[2, 0];  // ∂p/∂x
            T dpdy = firstDerivs[2, 1];  // ∂p/∂y

            // Second derivatives: [output_idx, input_idx1, input_idx2]
            T d2udx2 = secondDerivs[0, 0, 0];  // ∂²u/∂x²
            T d2udy2 = secondDerivs[0, 1, 1];  // ∂²u/∂y²

            T d2vdx2 = secondDerivs[1, 0, 0];  // ∂²v/∂x²
            T d2vdy2 = secondDerivs[1, 1, 1];  // ∂²v/∂y²

            // Continuity equation: ∂u/∂x + ∂v/∂y = 0
            T continuity = NumOps.Add(dudx, dvdy);

            // X-momentum: ∂u/∂t + u(∂u/∂x) + v(∂u/∂y) = -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
            T convectionX = NumOps.Add(
                NumOps.Multiply(u, dudx),
                NumOps.Multiply(v, dudy));
            T diffusionX = NumOps.Multiply(_viscosity, NumOps.Add(d2udx2, d2udy2));
            T pressureX = NumOps.Divide(dpdx, _density);
            T momentumX = NumOps.Subtract(
                NumOps.Add(dudt, NumOps.Add(convectionX, pressureX)),
                diffusionX);

            // Y-momentum: ∂v/∂t + u(∂v/∂x) + v(∂v/∂y) = -∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)
            T convectionY = NumOps.Add(
                NumOps.Multiply(u, dvdx),
                NumOps.Multiply(v, dvdy));
            T diffusionY = NumOps.Multiply(_viscosity, NumOps.Add(d2vdx2, d2vdy2));
            T pressureY = NumOps.Divide(dpdy, _density);
            T momentumY = NumOps.Subtract(
                NumOps.Add(dvdt, NumOps.Add(convectionY, pressureY)),
                diffusionY);

            // Total residual: sum of squared residuals
            T residual = NumOps.Add(
                NumOps.Multiply(continuity, continuity),
                NumOps.Add(
                    NumOps.Multiply(momentumX, momentumX),
                    NumOps.Multiply(momentumY, momentumY)));

            return residual;
        }

        /// <inheritdoc/>
        public PDEResidualGradient<T> ComputeResidualGradient(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateFirstDerivatives(derivatives);
            ValidateSecondDerivatives(derivatives);

            var firstDerivs = derivatives.FirstDerivatives;
            var secondDerivs = derivatives.SecondDerivatives;

            if (firstDerivs is null || secondDerivs is null)
            {
                throw new InvalidOperationException("Derivatives were null after validation.");
            }

            if (outputs.Length != 3)
            {
                throw new ArgumentException("Navier-Stokes expects 3 outputs: [u, v, p].");
            }

            var gradient = CreateGradient();
            T two = NumOps.FromDouble(2.0);

            T u = outputs[0];
            T v = outputs[1];

            T dudx = firstDerivs[0, 0];
            T dudy = firstDerivs[0, 1];
            T dudt = firstDerivs[0, 2];
            T dvdx = firstDerivs[1, 0];
            T dvdy = firstDerivs[1, 1];
            T dvdt = firstDerivs[1, 2];
            T dpdx = firstDerivs[2, 0];
            T dpdy = firstDerivs[2, 1];

            T d2udx2 = secondDerivs[0, 0, 0];
            T d2udy2 = secondDerivs[0, 1, 1];
            T d2vdx2 = secondDerivs[1, 0, 0];
            T d2vdy2 = secondDerivs[1, 1, 1];

            // Compute residuals for gradient scaling
            T continuity = NumOps.Add(dudx, dvdy);
            T convectionX = NumOps.Add(NumOps.Multiply(u, dudx), NumOps.Multiply(v, dudy));
            T diffusionX = NumOps.Multiply(_viscosity, NumOps.Add(d2udx2, d2udy2));
            T pressureX = NumOps.Divide(dpdx, _density);
            T momentumX = NumOps.Subtract(NumOps.Add(dudt, NumOps.Add(convectionX, pressureX)), diffusionX);

            T convectionY = NumOps.Add(NumOps.Multiply(u, dvdx), NumOps.Multiply(v, dvdy));
            T diffusionY = NumOps.Multiply(_viscosity, NumOps.Add(d2vdx2, d2vdy2));
            T pressureY = NumOps.Divide(dpdy, _density);
            T momentumY = NumOps.Subtract(NumOps.Add(dvdt, NumOps.Add(convectionY, pressureY)), diffusionY);

            // Gradient w.r.t. u (output 0)
            // From momentum-x: u appears in convection term u*∂u/∂x → ∂R/∂u += 2*momentumX*∂u/∂x
            // From momentum-y: u appears in convection term u*∂v/∂x → ∂R/∂u += 2*momentumY*∂v/∂x
            gradient.OutputGradients[0] = NumOps.Add(
                NumOps.Multiply(two, NumOps.Multiply(momentumX, dudx)),
                NumOps.Multiply(two, NumOps.Multiply(momentumY, dvdx)));

            // Gradient w.r.t. v (output 1)
            // From momentum-x: v appears in convection term v*∂u/∂y → ∂R/∂v += 2*momentumX*∂u/∂y
            // From momentum-y: v appears in convection term v*∂v/∂y → ∂R/∂v += 2*momentumY*∂v/∂y
            gradient.OutputGradients[1] = NumOps.Add(
                NumOps.Multiply(two, NumOps.Multiply(momentumX, dudy)),
                NumOps.Multiply(two, NumOps.Multiply(momentumY, dvdy)));

            // Gradient w.r.t. first derivatives
            // Continuity: ∂R/∂(∂u/∂x) = 2*continuity, ∂R/∂(∂v/∂y) = 2*continuity
            gradient.FirstDerivatives[0, 0] = NumOps.Add(
                NumOps.Multiply(two, continuity),
                NumOps.Multiply(two, NumOps.Multiply(momentumX, u)));
            gradient.FirstDerivatives[0, 1] = NumOps.Multiply(two, NumOps.Multiply(momentumX, v));
            gradient.FirstDerivatives[0, 2] = NumOps.Multiply(two, momentumX);

            gradient.FirstDerivatives[1, 0] = NumOps.Multiply(two, NumOps.Multiply(momentumY, u));
            gradient.FirstDerivatives[1, 1] = NumOps.Add(
                NumOps.Multiply(two, continuity),
                NumOps.Multiply(two, NumOps.Multiply(momentumY, v)));
            gradient.FirstDerivatives[1, 2] = NumOps.Multiply(two, momentumY);

            T invDensity = NumOps.Divide(NumOps.One, _density);
            gradient.FirstDerivatives[2, 0] = NumOps.Multiply(two, NumOps.Multiply(momentumX, invDensity));
            gradient.FirstDerivatives[2, 1] = NumOps.Multiply(two, NumOps.Multiply(momentumY, invDensity));

            // Gradient w.r.t. second derivatives
            T negTwoViscosity = NumOps.Negate(NumOps.Multiply(two, _viscosity));
            gradient.SecondDerivatives[0, 0, 0] = NumOps.Multiply(negTwoViscosity, momentumX);
            gradient.SecondDerivatives[0, 1, 1] = NumOps.Multiply(negTwoViscosity, momentumX);
            gradient.SecondDerivatives[1, 0, 0] = NumOps.Multiply(negTwoViscosity, momentumY);
            gradient.SecondDerivatives[1, 1, 1] = NumOps.Multiply(negTwoViscosity, momentumY);

            return gradient;
        }

        /// <inheritdoc/>
        public override int InputDimension => 3; // [x, y, t]

        /// <inheritdoc/>
        public override int OutputDimension => 3; // [u, v, p] (velocity x, velocity y, pressure)

        /// <inheritdoc/>
        public override string Name => $"Navier-Stokes (ν={_viscosity}, ρ={_density})";
    }
}
