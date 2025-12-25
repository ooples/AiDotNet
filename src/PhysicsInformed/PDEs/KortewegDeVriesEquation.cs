using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;

namespace AiDotNet.PhysicsInformed.PDEs
{
    /// <summary>
    /// Represents the Korteweg-de Vries (KdV) Equation:
    /// ∂u/∂t + αu∂u/∂x + β∂³u/∂x³ = 0
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// The Korteweg-de Vries equation is one of the most famous nonlinear PDEs in physics.
    /// It describes waves in shallow water and is remarkable for having "soliton" solutions.
    ///
    /// Variables:
    /// - u(x,t) = Wave amplitude or displacement
    /// - x = Spatial coordinate
    /// - t = Time
    /// - α = Nonlinear coefficient (strength of steepening)
    /// - β = Dispersion coefficient (wave spreading)
    ///
    /// Physical Interpretation:
    /// - The u∂u/∂x term causes wave steepening (like shock waves)
    /// - The ∂³u/∂x³ term causes dispersion (different frequencies travel at different speeds)
    /// - When these effects balance, you get solitons - stable traveling wave packets
    ///
    /// Solitons:
    /// - Solitons maintain their shape while traveling at constant speed
    /// - Two solitons can pass through each other without changing shape
    /// - First observed by John Scott Russell in 1834 watching a wave in a canal
    ///
    /// Standard Forms:
    /// - Canonical form: ∂u/∂t + 6u∂u/∂x + ∂³u/∂x³ = 0 (α=6, β=1)
    /// - Physical form: ∂u/∂t + u∂u/∂x + ∂³u/∂x³ = 0 (α=1, β=1)
    ///
    /// Applications:
    /// - Water waves in shallow channels
    /// - Internal waves in oceans
    /// - Plasma physics (ion-acoustic waves)
    /// - Optical fiber communications
    /// - Tsunami modeling (in simplified cases)
    ///
    /// Example: A solitary wave traveling along a canal maintains its bell-shaped
    /// profile indefinitely, unlike ordinary waves that disperse.
    /// </remarks>
    public class KortewegDeVriesEquation<T> : PDESpecificationBase<T>, IPDEResidualGradient<T>
    {
        private readonly T _alpha; // Nonlinear coefficient
        private readonly T _beta; // Dispersion coefficient

        /// <summary>
        /// Initializes a new instance of the Korteweg-de Vries Equation.
        /// </summary>
        /// <param name="alpha">Nonlinear coefficient α (default 6 for canonical form).</param>
        /// <param name="beta">Dispersion coefficient β (default 1, must be non-zero).</param>
        public KortewegDeVriesEquation(T alpha, T beta)
        {
            if (NumOps.Equals(beta, NumOps.Zero))
            {
                throw new ArgumentException("Dispersion coefficient β must be non-zero.", nameof(beta));
            }

            _alpha = alpha;
            _beta = beta;
        }

        /// <summary>
        /// Initializes a new instance of the Korteweg-de Vries Equation with double parameters.
        /// </summary>
        /// <param name="alpha">Nonlinear coefficient α (default 6 for canonical form).</param>
        /// <param name="beta">Dispersion coefficient β (default 1, must be non-zero).</param>
        public KortewegDeVriesEquation(double alpha = 6.0, double beta = 1.0)
            : this(
                MathHelper.GetNumericOperations<T>().FromDouble(alpha),
                MathHelper.GetNumericOperations<T>().FromDouble(beta))
        {
        }

        /// <summary>
        /// Creates a Korteweg-de Vries Equation in canonical form (α=6, β=1).
        /// </summary>
        /// <returns>A KdV equation with standard coefficients.</returns>
        public static KortewegDeVriesEquation<T> Canonical()
        {
            return new KortewegDeVriesEquation<T>(6.0, 1.0);
        }

        /// <summary>
        /// Creates a Korteweg-de Vries Equation in physical form (α=1, β=1).
        /// </summary>
        /// <returns>A KdV equation with unit coefficients.</returns>
        public static KortewegDeVriesEquation<T> Physical()
        {
            return new KortewegDeVriesEquation<T>(1.0, 1.0);
        }

        /// <inheritdoc/>
        public override T ComputeResidual(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateFirstDerivatives(derivatives);
            ValidateThirdDerivatives(derivatives);

            var firstDerivs = derivatives.FirstDerivatives;
            var thirdDerivs = derivatives.ThirdDerivatives;

            if (firstDerivs is null || thirdDerivs is null)
            {
                throw new InvalidOperationException("Derivatives were null after validation.");
            }

            // inputs = [x, t], outputs = [u]
            // PDE: ∂u/∂t + αu∂u/∂x + β∂³u/∂x³ = 0

            T u = outputs[0]; // Wave amplitude

            T dudt = firstDerivs[0, 1]; // ∂u/∂t
            T dudx = firstDerivs[0, 0]; // ∂u/∂x
            T d3udx3 = thirdDerivs[0, 0, 0, 0]; // ∂³u/∂x³

            // Nonlinear term: αu∂u/∂x
            T nonlinearTerm = NumOps.Multiply(_alpha, NumOps.Multiply(u, dudx));

            // Dispersion term: β∂³u/∂x³
            T dispersionTerm = NumOps.Multiply(_beta, d3udx3);

            // Residual: ∂u/∂t + αu∂u/∂x + β∂³u/∂x³
            T residual = NumOps.Add(dudt, nonlinearTerm);
            residual = NumOps.Add(residual, dispersionTerm);

            return residual;
        }

        /// <inheritdoc/>
        public PDEResidualGradient<T> ComputeResidualGradient(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateFirstDerivatives(derivatives);
            ValidateThirdDerivatives(derivatives);

            var firstDerivs = derivatives.FirstDerivatives;
            if (firstDerivs is null)
            {
                throw new InvalidOperationException("First derivatives were null after validation.");
            }

            T u = outputs[0];
            T dudx = firstDerivs[0, 0];

            var gradient = CreateGradient();

            // ∂R/∂(∂u/∂t) = 1
            gradient.FirstDerivatives[0, 1] = NumOps.One;

            // ∂R/∂(∂u/∂x) = αu
            gradient.FirstDerivatives[0, 0] = NumOps.Multiply(_alpha, u);

            // ∂R/∂(∂³u/∂x³) = β
            gradient.ThirdDerivatives[0, 0, 0, 0] = _beta;

            // ∂R/∂u = α∂u/∂x (from the nonlinear term)
            gradient.OutputGradients[0] = NumOps.Multiply(_alpha, dudx);

            return gradient;
        }

        /// <inheritdoc/>
        public override int InputDimension => 2; // [x, t]

        /// <inheritdoc/>
        public override int OutputDimension => 1; // [u] - wave amplitude

        /// <inheritdoc/>
        public override string Name => $"Korteweg-de Vries Equation (α={_alpha}, β={_beta})";
    }
}
