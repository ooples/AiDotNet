using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;

namespace AiDotNet.PhysicsInformed.PDEs
{
    /// <summary>
    /// Represents the time-dependent Schrodinger equation for quantum mechanics.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// The Schrodinger equation is the fundamental equation of quantum mechanics.
    /// It describes how the quantum state (wavefunction) of a physical system evolves over time.
    ///
    /// The equation in 1D (with normalized units ℏ = m = 1):
    /// i ∂ψ/∂t = -½ ∂²ψ/∂x² + V(x)ψ
    ///
    /// Since ψ is complex (ψ = ψ_r + i*ψ_i), we split into real/imaginary parts:
    /// ∂ψ_r/∂t = -½ ∂²ψ_i/∂x² + V*ψ_i
    /// ∂ψ_i/∂t =  ½ ∂²ψ_r/∂x² - V*ψ_r
    ///
    /// Key Concepts:
    /// - ψ(x,t): Wavefunction - its square magnitude |ψ|² gives probability density
    /// - V(x): Potential energy function (e.g., harmonic oscillator, particle in box)
    /// - ℏ: Reduced Planck's constant (set to 1 in normalized units)
    /// - m: Particle mass (set to 1 in normalized units)
    ///
    /// Physical Interpretation:
    /// - The wavefunction encodes all quantum information about the system
    /// - |ψ(x,t)|² = ψ_r² + ψ_i² is the probability density at position x, time t
    /// - Total probability is conserved: ∫|ψ|²dx = 1
    ///
    /// Applications:
    /// - Atomic and molecular physics
    /// - Quantum chemistry
    /// - Semiconductor physics
    /// - Quantum computing simulations
    /// - Tunneling phenomena
    ///
    /// This implementation supports a user-defined potential function V(x).
    /// </remarks>
    public class SchrodingerEquation<T> : PDESpecificationBase<T>, IPDEResidualGradient<T>
    {
        private readonly Func<T, T> _potentialFunction;
        private readonly T _halfCoeff;

        /// <summary>
        /// Initializes the Schrodinger equation with a specified potential.
        /// </summary>
        /// <param name="potentialFunction">
        /// The potential energy function V(x). Examples:
        /// - Free particle: V(x) = 0
        /// - Harmonic oscillator: V(x) = ½ω²x²
        /// - Particle in box: V(x) = 0 for |x| less than L, ∞ otherwise
        /// - Gaussian barrier: V(x) = V₀ exp(-x²/2σ²)
        /// </param>
        /// <remarks>
        /// For Beginners:
        /// The potential function V(x) determines the forces acting on the quantum particle.
        /// Different potentials lead to different quantum behaviors:
        /// - Bound states: Particle confined by potential wells (like electrons in atoms)
        /// - Scattering: Particle interacts with potential barriers (like quantum tunneling)
        /// - Free motion: No potential means simple wave propagation
        /// </remarks>
        public SchrodingerEquation(Func<T, T> potentialFunction)
        {
            if (potentialFunction is null)
            {
                throw new ArgumentNullException(nameof(potentialFunction), "Potential function cannot be null.");
            }

            _potentialFunction = potentialFunction;
            _halfCoeff = NumOps.FromDouble(0.5);
        }

        /// <summary>
        /// Creates a Schrodinger equation with zero potential (free particle).
        /// </summary>
        /// <remarks>
        /// For Beginners:
        /// A free particle has no forces acting on it, so it moves freely.
        /// The wavefunction spreads out over time (wave packet dispersion).
        /// </remarks>
        public SchrodingerEquation()
            : this(CreateZeroPotential())
        {
        }

        /// <summary>
        /// Creates a zero potential function (free particle case).
        /// Caches the zero value to avoid repeated lookups.
        /// </summary>
        private static Func<T, T> CreateZeroPotential()
        {
            var zero = MathHelper.GetNumericOperations<T>().Zero;
            return _ => zero;
        }

        /// <inheritdoc/>
        /// <remarks>
        /// The Schrodinger equation i∂ψ/∂t = -½∂²ψ/∂x² + Vψ separates into:
        /// - Real: ∂ψ_r/∂t = -½ ∂²ψ_i/∂x² + V*ψ_i
        /// - Imag: ∂ψ_i/∂t = ½ ∂²ψ_r/∂x² - V*ψ_r
        ///
        /// Residuals (set to zero):
        /// - R1: ∂ψ_r/∂t + ½ ∂²ψ_i/∂x² - V*ψ_i = 0
        /// - R2: ∂ψ_i/∂t - ½ ∂²ψ_r/∂x² + V*ψ_r = 0
        ///
        /// Returns the sum of squared residuals: R1² + R2²
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

            if (outputs.Length != 2)
            {
                throw new ArgumentException("Schrodinger equation expects 2 outputs: [psi_r, psi_i].");
            }

            // inputs = [x, t], outputs = [psi_r, psi_i]
            T x = inputs[0];
            T psiR = outputs[0];  // Real part of wavefunction
            T psiI = outputs[1];  // Imaginary part of wavefunction

            // Evaluate potential at current position
            T V = _potentialFunction(x);

            // First derivatives: [output_idx, input_idx]
            // Input indices: 0=x, 1=t
            T dPsiRdt = firstDerivs[0, 1];  // ∂ψ_r/∂t
            T dPsiIdt = firstDerivs[1, 1];  // ∂ψ_i/∂t

            // Second derivatives: [output_idx, input_idx1, input_idx2]
            T d2PsiRdx2 = secondDerivs[0, 0, 0];  // ∂²ψ_r/∂x²
            T d2PsiIdx2 = secondDerivs[1, 0, 0];  // ∂²ψ_i/∂x²

            // Real part equation: ∂ψ_r/∂t = -½ ∂²ψ_i/∂x² + V*ψ_i
            // Residual: ∂ψ_r/∂t + ½ ∂²ψ_i/∂x² - V*ψ_i = 0
            T residualReal = NumOps.Subtract(
                NumOps.Add(dPsiRdt, NumOps.Multiply(_halfCoeff, d2PsiIdx2)),
                NumOps.Multiply(V, psiI));

            // Imaginary part equation: ∂ψ_i/∂t = ½ ∂²ψ_r/∂x² - V*ψ_r
            // Residual: ∂ψ_i/∂t - ½ ∂²ψ_r/∂x² + V*ψ_r = 0
            T residualImag = NumOps.Add(
                NumOps.Subtract(dPsiIdt, NumOps.Multiply(_halfCoeff, d2PsiRdx2)),
                NumOps.Multiply(V, psiR));

            // Total residual: sum of squared residuals
            T residual = NumOps.Add(
                NumOps.Multiply(residualReal, residualReal),
                NumOps.Multiply(residualImag, residualImag));

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

            var gradient = CreateGradient();
            T two = NumOps.FromDouble(2.0);

            T x = inputs[0];
            T psiR = outputs[0];
            T psiI = outputs[1];
            T V = _potentialFunction(x);

            T dPsiRdt = firstDerivs[0, 1];
            T dPsiIdt = firstDerivs[1, 1];
            T d2PsiRdx2 = secondDerivs[0, 0, 0];
            T d2PsiIdx2 = secondDerivs[1, 0, 0];

            // Compute residuals for gradient scaling
            // R_real = ∂ψ_r/∂t + ½ ∂²ψ_i/∂x² - V*ψ_i
            T residualReal = NumOps.Subtract(
                NumOps.Add(dPsiRdt, NumOps.Multiply(_halfCoeff, d2PsiIdx2)),
                NumOps.Multiply(V, psiI));

            // R_imag = ∂ψ_i/∂t - ½ ∂²ψ_r/∂x² + V*ψ_r
            T residualImag = NumOps.Add(
                NumOps.Subtract(dPsiIdt, NumOps.Multiply(_halfCoeff, d2PsiRdx2)),
                NumOps.Multiply(V, psiR));

            // Gradient w.r.t. outputs
            // ∂R²/∂ψ_r = 2*residualImag*V (from R_imag term +V*ψ_r)
            gradient.OutputGradients[0] = NumOps.Multiply(two, NumOps.Multiply(V, residualImag));
            // ∂R²/∂ψ_i = 2*residualReal*(-V) = -2*V*residualReal (from R_real term -V*ψ_i)
            gradient.OutputGradients[1] = NumOps.Multiply(two, NumOps.Negate(NumOps.Multiply(V, residualReal)));

            // Gradient w.r.t. first derivatives
            // ∂R²/∂(∂ψ_r/∂t) = 2*residualReal
            gradient.FirstDerivatives[0, 1] = NumOps.Multiply(two, residualReal);
            // ∂R²/∂(∂ψ_i/∂t) = 2*residualImag
            gradient.FirstDerivatives[1, 1] = NumOps.Multiply(two, residualImag);

            // Gradient w.r.t. second derivatives
            // ∂R²/∂(∂²ψ_r/∂x²) = 2*residualImag*(-½) = -residualImag (from R_imag term -½∂²ψ_r/∂x²)
            gradient.SecondDerivatives[0, 0, 0] = NumOps.Negate(NumOps.Multiply(_halfCoeff, NumOps.Multiply(two, residualImag)));
            // ∂R²/∂(∂²ψ_i/∂x²) = 2*residualReal*(½) = residualReal (from R_real term +½∂²ψ_i/∂x²)
            gradient.SecondDerivatives[1, 0, 0] = NumOps.Multiply(_halfCoeff, NumOps.Multiply(two, residualReal));

            return gradient;
        }

        /// <inheritdoc/>
        public override int InputDimension => 2; // [x, t]

        /// <inheritdoc/>
        public override int OutputDimension => 2; // [psi_r, psi_i] (real and imaginary parts)

        /// <inheritdoc/>
        public override string Name => "Schrodinger Equation";
    }
}
