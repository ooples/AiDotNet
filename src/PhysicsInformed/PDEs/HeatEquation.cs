using System;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.PhysicsInformed.Interfaces;

namespace AiDotNet.PhysicsInformed.PDEs
{
    /// <summary>
    /// Represents the Heat Equation (or Diffusion Equation): ∂u/∂t = α ∂²u/∂x²
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// The Heat Equation models how heat diffuses through a material over time.
    /// - u(x,t) is the temperature at position x and time t
    /// - α (alpha) is the thermal diffusivity, which controls how fast heat spreads
    /// - The equation says: The rate of temperature change equals how curved the temperature profile is
    ///
    /// Physical Interpretation:
    /// - If the temperature profile is concave (curves down), heat flows in → temperature increases
    /// - If the temperature profile is convex (curves up), heat flows out → temperature decreases
    /// - At inflection points (no curvature), temperature stays constant
    ///
    /// Example: A metal rod with one end heated - the heat gradually spreads along the rod.
    /// </remarks>
    public class HeatEquation<T> : PDESpecificationBase<T>, IPDEResidualGradient<T>
    {
        private readonly T _thermalDiffusivity;

        /// <summary>
        /// Initializes a new instance of the Heat Equation.
        /// </summary>
        /// <param name="thermalDiffusivity">The thermal diffusivity coefficient α (must be positive).</param>
        public HeatEquation(T thermalDiffusivity)
        {
            ValidatePositive(thermalDiffusivity, nameof(thermalDiffusivity));
            _thermalDiffusivity = thermalDiffusivity;
        }

        /// <summary>
        /// Initializes a new instance of the Heat Equation with double parameter.
        /// </summary>
        /// <param name="thermalDiffusivity">The thermal diffusivity coefficient α (default 1.0, must be positive).</param>
        public HeatEquation(double thermalDiffusivity = 1.0)
            : this(MathHelper.GetNumericOperations<T>().FromDouble(thermalDiffusivity))
        {
        }

        /// <inheritdoc/>
        public override T ComputeResidual(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateSecondDerivatives(derivatives);

            // For 1D heat equation: inputs = [x, t], outputs = [u]
            // PDE: ∂u/∂t - α * ∂²u/∂x² = 0

            var firstDerivs = derivatives.FirstDerivatives;
            var secondDerivs = derivatives.SecondDerivatives;

            if (firstDerivs is null || secondDerivs is null)
            {
                throw new InvalidOperationException("Derivatives were null after validation.");
            }

            T dudt = firstDerivs[0, 1]; // ∂u/∂t (output 0, input 1 which is t)
            T d2udx2 = secondDerivs[0, 0, 0]; // ∂²u/∂x² (output 0, both inputs are x)

            T residual = NumOps.Subtract(dudt, NumOps.Multiply(_thermalDiffusivity, d2udx2));
            return residual;
        }

        /// <inheritdoc/>
        public PDEResidualGradient<T> ComputeResidualGradient(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            ValidateSecondDerivatives(derivatives);

            var gradient = CreateGradient();
            gradient.FirstDerivatives[0, 1] = NumOps.One;
            gradient.SecondDerivatives[0, 0, 0] = NumOps.Negate(_thermalDiffusivity);
            return gradient;
        }

        /// <inheritdoc/>
        public override int InputDimension => 2; // [x, t]

        /// <inheritdoc/>
        public override int OutputDimension => 1; // [u]

        /// <inheritdoc/>
        public override string Name => $"Heat Equation (α={_thermalDiffusivity})";
    }
}
