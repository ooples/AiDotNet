using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;

namespace AiDotNet.PhysicsInformed.PDEs
{
    /// <summary>
    /// Represents Maxwell's equations for electromagnetic wave propagation (2D TE mode).
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// Maxwell's equations are the foundation of all electromagnetic phenomena including:
    /// - Light, radio waves, microwaves, X-rays (all forms of electromagnetic radiation)
    /// - Electric motors and generators
    /// - Wireless communication
    /// - Optical fibers
    ///
    /// The equations describe how electric and magnetic fields interact and propagate.
    /// This implementation uses the 2D Transverse Electric (TE) mode where:
    /// - Electric field lies in the x-y plane: (Ex, Ey)
    /// - Magnetic field points in z-direction: Bz
    ///
    /// The equations are:
    /// **Faraday's Law** (changing magnetic field creates electric field):
    /// ∂Bz/∂t = -(∂Ey/∂x - ∂Ex/∂y)
    ///
    /// **Ampere's Law** (changing electric field creates magnetic field):
    /// ∂Ex/∂t = (1/ε) ∂Bz/∂y
    /// ∂Ey/∂t = -(1/ε) ∂Bz/∂x
    ///
    /// Key Parameters:
    /// - ε (epsilon): Electric permittivity - how easily a material polarizes
    /// - μ (mu): Magnetic permeability - how easily a material magnetizes
    /// - c = 1/sqrt(εμ): Speed of light in the medium
    ///
    /// Physical Interpretation:
    /// - Electromagnetic waves are self-sustaining oscillations of E and B fields
    /// - The wave equation shows they propagate at the speed of light
    /// - Energy flows perpendicular to both E and B (Poynting vector)
    ///
    /// Applications:
    /// - Antenna design
    /// - Optical waveguides
    /// - Photonic crystals
    /// - Metamaterials
    /// - Radar and communication systems
    /// </remarks>
    public class MaxwellEquations<T> : PDESpecificationBase<T>, IPDEResidualGradient<T>
    {
        private readonly T _permittivity;   // ε
        private readonly T _permeability;   // μ

        /// <summary>
        /// Initializes Maxwell's equations with specified electromagnetic properties.
        /// </summary>
        /// <param name="permittivity">Electric permittivity ε (default: 1 for vacuum, normalized)</param>
        /// <param name="permeability">Magnetic permeability μ (default: 1 for vacuum, normalized)</param>
        /// <remarks>
        /// For Beginners:
        /// - In vacuum: ε₀ ≈ 8.85×10⁻¹² F/m, μ₀ ≈ 4π×10⁻⁷ H/m
        /// - Relative permittivity εᵣ = ε/ε₀ (water ≈ 80, glass ≈ 4-10)
        /// - For normalized units (common in simulations): ε = μ = 1
        /// </remarks>
        public MaxwellEquations(T? permittivity = default, T? permeability = default)
        {
            // Use provided values or default to 1 (vacuum/normalized units)
            if (permittivity is null)
            {
                _permittivity = NumOps.One;
            }
            else
            {
                ValidatePositive(permittivity, nameof(permittivity));
                _permittivity = permittivity;
            }

            if (permeability is null)
            {
                _permeability = NumOps.One;
            }
            else
            {
                ValidatePositive(permeability, nameof(permeability));
                _permeability = permeability;
            }
        }

        /// <summary>
        /// Initializes Maxwell's equations with double parameters.
        /// </summary>
        /// <param name="permittivity">Electric permittivity ε (default 1.0 for vacuum/normalized)</param>
        /// <param name="permeability">Magnetic permeability μ (default 1.0 for vacuum/normalized)</param>
        public MaxwellEquations(double permittivity = 1.0, double permeability = 1.0)
            : this(
                MathHelper.GetNumericOperations<T>().FromDouble(permittivity),
                MathHelper.GetNumericOperations<T>().FromDouble(permeability))
        {
        }

        /// <inheritdoc/>
        /// <remarks>
        /// Computes three residuals:
        /// - R1 (Faraday): ∂Bz/∂t + (∂Ey/∂x - ∂Ex/∂y) = 0
        /// - R2 (Ampere-x): ε ∂Ex/∂t - ∂Bz/∂y = 0
        /// - R3 (Ampere-y): ε ∂Ey/∂t + ∂Bz/∂x = 0
        ///
        /// Returns the sum of squared residuals.
        /// </remarks>
        public override T ComputeResidual(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateFirstDerivatives(derivatives);

            var firstDerivs = derivatives.FirstDerivatives;
            if (firstDerivs is null)
            {
                throw new InvalidOperationException("First derivatives were null after validation.");
            }

            if (outputs.Length != 3)
            {
                throw new ArgumentException("Maxwell's equations expect 3 outputs: [Ex, Ey, Bz].");
            }

            // outputs = [Ex, Ey, Bz]
            // inputs = [x, y, t]

            // First derivatives: [output_idx, input_idx]
            // Output indices: 0=Ex, 1=Ey, 2=Bz
            // Input indices: 0=x, 1=y, 2=t
            T dExdx = firstDerivs[0, 0];
            T dExdy = firstDerivs[0, 1];
            T dExdt = firstDerivs[0, 2];

            T dEydx = firstDerivs[1, 0];
            T dEydy = firstDerivs[1, 1];
            T dEydt = firstDerivs[1, 2];

            T dBzdx = firstDerivs[2, 0];
            T dBzdy = firstDerivs[2, 1];
            T dBzdt = firstDerivs[2, 2];

            // Faraday's Law: ∂Bz/∂t = -(∂Ey/∂x - ∂Ex/∂y)
            // Residual: ∂Bz/∂t + ∂Ey/∂x - ∂Ex/∂y = 0
            T faraday = NumOps.Add(dBzdt, NumOps.Subtract(dEydx, dExdy));

            // Ampere's Law (x-component): ε ∂Ex/∂t = ∂Bz/∂y
            // Residual: ε ∂Ex/∂t - ∂Bz/∂y = 0
            T ampereX = NumOps.Subtract(NumOps.Multiply(_permittivity, dExdt), dBzdy);

            // Ampere's Law (y-component): ε ∂Ey/∂t = -∂Bz/∂x
            // Residual: ε ∂Ey/∂t + ∂Bz/∂x = 0
            T ampereY = NumOps.Add(NumOps.Multiply(_permittivity, dEydt), dBzdx);

            // Total residual: sum of squared residuals
            T residual = NumOps.Add(
                NumOps.Multiply(faraday, faraday),
                NumOps.Add(
                    NumOps.Multiply(ampereX, ampereX),
                    NumOps.Multiply(ampereY, ampereY)));

            return residual;
        }

        /// <inheritdoc/>
        public PDEResidualGradient<T> ComputeResidualGradient(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateFirstDerivatives(derivatives);

            var firstDerivs = derivatives.FirstDerivatives;
            if (firstDerivs is null)
            {
                throw new InvalidOperationException("First derivatives were null after validation.");
            }

            var gradient = CreateGradient();
            T two = NumOps.FromDouble(2.0);

            T dExdy = firstDerivs[0, 1];
            T dExdt = firstDerivs[0, 2];
            T dEydx = firstDerivs[1, 0];
            T dEydt = firstDerivs[1, 2];
            T dBzdx = firstDerivs[2, 0];
            T dBzdy = firstDerivs[2, 1];
            T dBzdt = firstDerivs[2, 2];

            // Compute residuals for gradient scaling
            T faraday = NumOps.Add(dBzdt, NumOps.Subtract(dEydx, dExdy));
            T ampereX = NumOps.Subtract(NumOps.Multiply(_permittivity, dExdt), dBzdy);
            T ampereY = NumOps.Add(NumOps.Multiply(_permittivity, dEydt), dBzdx);

            // Gradients w.r.t. first derivatives
            // Faraday: R = dBzdt + dEydx - dExdy
            // ∂R²/∂(dExdy) = 2*faraday*(-1) = -2*faraday
            gradient.FirstDerivatives[0, 1] = NumOps.Multiply(two, NumOps.Negate(faraday));
            // ∂R²/∂(dExdt) = 2*ampereX*ε
            gradient.FirstDerivatives[0, 2] = NumOps.Multiply(two, NumOps.Multiply(ampereX, _permittivity));

            // ∂R²/∂(dEydx) = 2*faraday
            gradient.FirstDerivatives[1, 0] = NumOps.Multiply(two, faraday);
            // ∂R²/∂(dEydt) = 2*ampereY*ε
            gradient.FirstDerivatives[1, 2] = NumOps.Multiply(two, NumOps.Multiply(ampereY, _permittivity));

            // ∂R²/∂(dBzdx) = 2*ampereY
            gradient.FirstDerivatives[2, 0] = NumOps.Multiply(two, ampereY);
            // ∂R²/∂(dBzdy) = -2*ampereX
            gradient.FirstDerivatives[2, 1] = NumOps.Multiply(two, NumOps.Negate(ampereX));
            // ∂R²/∂(dBzdt) = 2*faraday
            gradient.FirstDerivatives[2, 2] = NumOps.Multiply(two, faraday);

            return gradient;
        }

        /// <inheritdoc/>
        public override int InputDimension => 3; // [x, y, t]

        /// <inheritdoc/>
        public override int OutputDimension => 3; // [Ex, Ey, Bz]

        /// <inheritdoc/>
        public override string Name => $"Maxwell Equations (ε={_permittivity}, μ={_permeability})";
    }
}
