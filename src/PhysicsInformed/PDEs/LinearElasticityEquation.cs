using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;

namespace AiDotNet.PhysicsInformed.PDEs
{
    /// <summary>
    /// Represents the 2D Linear Elasticity Equations (Navier-Cauchy equations):
    /// (λ + μ)∂(∂u/∂x + ∂v/∂y)/∂x + μ∇²u + fₓ = 0
    /// (λ + μ)∂(∂u/∂x + ∂v/∂y)/∂y + μ∇²v + fᵧ = 0
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// The Linear Elasticity equations describe how solid materials deform under stress.
    ///
    /// Variables:
    /// - u(x,y) = Displacement in x-direction
    /// - v(x,y) = Displacement in y-direction
    /// - λ (lambda) = First Lamé parameter (related to bulk modulus)
    /// - μ (mu) = Second Lamé parameter (shear modulus, measures resistance to shearing)
    /// - fₓ, fᵧ = Body forces (like gravity)
    ///
    /// Physical Interpretation:
    /// - When you push or pull on a solid object, it deforms
    /// - The equations balance internal stresses with external forces
    /// - The Lamé parameters describe how stiff the material is
    ///
    /// Material Properties:
    /// - λ and μ can be computed from Young's modulus E and Poisson's ratio ν:
    ///   * λ = Eν / ((1+ν)(1-2ν))
    ///   * μ = E / (2(1+ν))
    ///
    /// Applications:
    /// - Structural engineering (buildings, bridges)
    /// - Mechanical design (stress analysis)
    /// - Geology (tectonic plate deformation)
    /// - Biomechanics (bone and tissue mechanics)
    ///
    /// Example: A beam bending under load, a pressure vessel expanding,
    /// or a rubber band stretching.
    /// </remarks>
    public class LinearElasticityEquation<T> : PDESpecificationBase<T>, IPDEResidualGradient<T>
    {
        private readonly T _lambda; // First Lamé parameter
        private readonly T _mu; // Second Lamé parameter (shear modulus)
        private readonly T _lambdaPlusMu;
        private readonly T _bodyForceX;
        private readonly T _bodyForceY;

        /// <summary>
        /// Initializes a new instance of the Linear Elasticity Equation.
        /// </summary>
        /// <param name="lambda">First Lamé parameter λ.</param>
        /// <param name="mu">Second Lamé parameter μ (shear modulus, must be positive).</param>
        /// <param name="bodyForceX">Body force in x-direction (default 0).</param>
        /// <param name="bodyForceY">Body force in y-direction (default 0, use -ρg for gravity).</param>
        public LinearElasticityEquation(T lambda, T mu, T? bodyForceX = default, T? bodyForceY = default)
        {
            ValidatePositive(mu, nameof(mu));

            _lambda = lambda;
            _mu = mu;
            _lambdaPlusMu = NumOps.Add(lambda, mu);
            _bodyForceX = bodyForceX ?? NumOps.Zero;
            _bodyForceY = bodyForceY ?? NumOps.Zero;
        }

        /// <summary>
        /// Initializes a new instance of the Linear Elasticity Equation with double parameters.
        /// </summary>
        /// <param name="lambda">First Lamé parameter λ (default 1.0 for normalized problems).</param>
        /// <param name="mu">Second Lamé parameter μ (shear modulus, default 1.0, must be positive).</param>
        /// <param name="bodyForceX">Body force in x-direction (default 0).</param>
        /// <param name="bodyForceY">Body force in y-direction (default 0).</param>
        public LinearElasticityEquation(double lambda = 1.0, double mu = 1.0, double bodyForceX = 0, double bodyForceY = 0)
            : this(
                MathHelper.GetNumericOperations<T>().FromDouble(lambda),
                MathHelper.GetNumericOperations<T>().FromDouble(mu),
                MathHelper.GetNumericOperations<T>().FromDouble(bodyForceX),
                MathHelper.GetNumericOperations<T>().FromDouble(bodyForceY))
        {
        }

        /// <summary>
        /// Creates a Linear Elasticity Equation from Young's modulus and Poisson's ratio.
        /// </summary>
        /// <param name="youngsModulus">Young's modulus E (must be positive).</param>
        /// <param name="poissonsRatio">Poisson's ratio ν (must be between -1 and 0.5).</param>
        /// <param name="bodyForceX">Body force in x-direction (default 0).</param>
        /// <param name="bodyForceY">Body force in y-direction (default 0).</param>
        /// <returns>A new LinearElasticityEquation instance.</returns>
        public static LinearElasticityEquation<T> FromEngineeringConstants(
            double youngsModulus, double poissonsRatio,
            double bodyForceX = 0, double bodyForceY = 0)
        {
            if (youngsModulus <= 0)
            {
                throw new ArgumentException("Young's modulus must be positive.", nameof(youngsModulus));
            }

            if (poissonsRatio <= -1 || poissonsRatio >= 0.5)
            {
                throw new ArgumentException("Poisson's ratio must be between -1 and 0.5.", nameof(poissonsRatio));
            }

            // Convert E and ν to Lamé parameters
            // λ = Eν / ((1+ν)(1-2ν))
            // μ = E / (2(1+ν))
            double lambda = youngsModulus * poissonsRatio / ((1 + poissonsRatio) * (1 - 2 * poissonsRatio));
            double mu = youngsModulus / (2 * (1 + poissonsRatio));

            return new LinearElasticityEquation<T>(lambda, mu, bodyForceX, bodyForceY);
        }

        /// <inheritdoc/>
        public override T ComputeResidual(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateSecondDerivatives(derivatives);

            var secondDerivs = derivatives.SecondDerivatives;
            if (secondDerivs is null)
            {
                throw new InvalidOperationException("Second derivatives were null after validation.");
            }

            // inputs = [x, y], outputs = [u, v]
            // Equation 1: (λ+μ)∂²u/∂x² + (λ+μ)∂²v/∂x∂y + μ(∂²u/∂x² + ∂²u/∂y²) + fₓ = 0
            // Equation 2: (λ+μ)∂²u/∂x∂y + (λ+μ)∂²v/∂y² + μ(∂²v/∂x² + ∂²v/∂y²) + fᵧ = 0

            // Second derivatives for u (output 0)
            T d2udx2 = secondDerivs[0, 0, 0]; // ∂²u/∂x²
            T d2udy2 = secondDerivs[0, 1, 1]; // ∂²u/∂y²
            T d2udxdy = secondDerivs[0, 0, 1]; // ∂²u/∂x∂y

            // Second derivatives for v (output 1)
            T d2vdx2 = secondDerivs[1, 0, 0]; // ∂²v/∂x²
            T d2vdy2 = secondDerivs[1, 1, 1]; // ∂²v/∂y²
            T d2vdxdy = secondDerivs[1, 0, 1]; // ∂²v/∂x∂y

            // Equation 1 residual: (λ+2μ)∂²u/∂x² + μ∂²u/∂y² + (λ+μ)∂²v/∂x∂y + fₓ
            T lambdaPlus2Mu = NumOps.Add(_lambda, NumOps.Multiply(NumOps.FromDouble(2), _mu));
            T res1_term1 = NumOps.Multiply(lambdaPlus2Mu, d2udx2);
            T res1_term2 = NumOps.Multiply(_mu, d2udy2);
            T res1_term3 = NumOps.Multiply(_lambdaPlusMu, d2vdxdy);
            T residual1 = NumOps.Add(NumOps.Add(NumOps.Add(res1_term1, res1_term2), res1_term3), _bodyForceX);

            // Equation 2 residual: (λ+μ)∂²u/∂x∂y + μ∂²v/∂x² + (λ+2μ)∂²v/∂y² + fᵧ
            T res2_term1 = NumOps.Multiply(_lambdaPlusMu, d2udxdy);
            T res2_term2 = NumOps.Multiply(_mu, d2vdx2);
            T res2_term3 = NumOps.Multiply(lambdaPlus2Mu, d2vdy2);
            T residual2 = NumOps.Add(NumOps.Add(NumOps.Add(res2_term1, res2_term2), res2_term3), _bodyForceY);

            // Return sum of squared residuals
            T res1Sq = NumOps.Multiply(residual1, residual1);
            T res2Sq = NumOps.Multiply(residual2, residual2);
            return NumOps.Add(res1Sq, res2Sq);
        }

        /// <inheritdoc/>
        /// <remarks>
        /// The gradient is computed for R = R₁² + R₂² where:
        /// - R₁ = (λ+2μ)∂²u/∂x² + μ∂²u/∂y² + (λ+μ)∂²v/∂x∂y + fₓ
        /// - R₂ = (λ+μ)∂²u/∂x∂y + μ∂²v/∂x² + (λ+2μ)∂²v/∂y² + fᵧ
        ///
        /// The gradient includes both same-equation and cross-equation coupling terms.
        /// For example, ∂²u/∂x∂y appears in R₂, so its gradient is 2*R₂*(λ+μ),
        /// while ∂²v/∂x∂y appears in R₁, so its gradient is 2*R₁*(λ+μ).
        /// </remarks>
        public PDEResidualGradient<T> ComputeResidualGradient(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateSecondDerivatives(derivatives);

            var secondDerivs = derivatives.SecondDerivatives;
            if (secondDerivs is null)
            {
                throw new InvalidOperationException("Second derivatives were null after validation.");
            }

            var gradient = CreateGradient();
            T two = NumOps.FromDouble(2.0);
            T lambdaPlus2Mu = NumOps.Add(_lambda, NumOps.Multiply(two, _mu));

            // First, compute the residuals (needed for chain rule: d/dx(R²) = 2*R*dR/dx)
            T d2udx2 = secondDerivs[0, 0, 0];
            T d2udy2 = secondDerivs[0, 1, 1];
            T d2udxdy = secondDerivs[0, 0, 1];
            T d2vdx2 = secondDerivs[1, 0, 0];
            T d2vdy2 = secondDerivs[1, 1, 1];
            T d2vdxdy = secondDerivs[1, 0, 1];

            // Residual 1: (λ+2μ)∂²u/∂x² + μ∂²u/∂y² + (λ+μ)∂²v/∂x∂y + fₓ
            T res1_term1 = NumOps.Multiply(lambdaPlus2Mu, d2udx2);
            T res1_term2 = NumOps.Multiply(_mu, d2udy2);
            T res1_term3 = NumOps.Multiply(_lambdaPlusMu, d2vdxdy);
            T residual1 = NumOps.Add(NumOps.Add(NumOps.Add(res1_term1, res1_term2), res1_term3), _bodyForceX);

            // Residual 2: (λ+μ)∂²u/∂x∂y + μ∂²v/∂x² + (λ+2μ)∂²v/∂y² + fᵧ
            T res2_term1 = NumOps.Multiply(_lambdaPlusMu, d2udxdy);
            T res2_term2 = NumOps.Multiply(_mu, d2vdx2);
            T res2_term3 = NumOps.Multiply(lambdaPlus2Mu, d2vdy2);
            T residual2 = NumOps.Add(NumOps.Add(NumOps.Add(res2_term1, res2_term2), res2_term3), _bodyForceY);

            // Compute 2*R₁ and 2*R₂ for chain rule
            T twoR1 = NumOps.Multiply(two, residual1);
            T twoR2 = NumOps.Multiply(two, residual2);

            // Gradients for R = R₁² + R₂²
            // ∂R/∂(∂²u/∂x²) = 2*R₁*(λ+2μ) - appears only in R₁
            gradient.SecondDerivatives[0, 0, 0] = NumOps.Multiply(twoR1, lambdaPlus2Mu);

            // ∂R/∂(∂²u/∂y²) = 2*R₁*μ - appears only in R₁
            gradient.SecondDerivatives[0, 1, 1] = NumOps.Multiply(twoR1, _mu);

            // ∂R/∂(∂²u/∂x∂y) = 2*R₂*(λ+μ) - CROSS-COUPLING: appears in R₂, not R₁
            gradient.SecondDerivatives[0, 0, 1] = NumOps.Multiply(twoR2, _lambdaPlusMu);

            // ∂R/∂(∂²v/∂x²) = 2*R₂*μ - appears only in R₂
            gradient.SecondDerivatives[1, 0, 0] = NumOps.Multiply(twoR2, _mu);

            // ∂R/∂(∂²v/∂y²) = 2*R₂*(λ+2μ) - appears only in R₂
            gradient.SecondDerivatives[1, 1, 1] = NumOps.Multiply(twoR2, lambdaPlus2Mu);

            // ∂R/∂(∂²v/∂x∂y) = 2*R₁*(λ+μ) - CROSS-COUPLING: appears in R₁, not R₂
            gradient.SecondDerivatives[1, 0, 1] = NumOps.Multiply(twoR1, _lambdaPlusMu);

            return gradient;
        }

        /// <inheritdoc/>
        public override int InputDimension => 2; // [x, y]

        /// <inheritdoc/>
        public override int OutputDimension => 2; // [u, v] - displacements

        /// <inheritdoc/>
        public override string Name => $"Linear Elasticity (λ={_lambda}, μ={_mu})";
    }
}
