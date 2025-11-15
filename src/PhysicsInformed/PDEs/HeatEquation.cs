using System;
using System.Numerics;
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
    public class HeatEquation<T> : IPDESpecification<T> where T : struct, INumber<T>
    {
        private readonly T _thermalDiffusivity;

        /// <summary>
        /// Initializes a new instance of the Heat Equation.
        /// </summary>
        /// <param name="thermalDiffusivity">The thermal diffusivity coefficient α (must be positive).</param>
        public HeatEquation(T thermalDiffusivity)
        {
            if (thermalDiffusivity <= T.Zero)
            {
                throw new ArgumentException("Thermal diffusivity must be positive.", nameof(thermalDiffusivity));
            }
            _thermalDiffusivity = thermalDiffusivity;
        }

        /// <inheritdoc/>
        public T ComputeResidual(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            if (derivatives.FirstDerivatives == null || derivatives.SecondDerivatives == null)
            {
                throw new ArgumentException("Heat equation requires first and second derivatives.");
            }

            // For 1D heat equation: inputs = [x, t], outputs = [u]
            // PDE: ∂u/∂t - α * ∂²u/∂x² = 0

            T dudt = derivatives.FirstDerivatives[0, 1]; // ∂u/∂t (output 0, input 1 which is t)
            T d2udx2 = derivatives.SecondDerivatives[0, 0, 0]; // ∂²u/∂x² (output 0, both inputs are x)

            T residual = dudt - _thermalDiffusivity * d2udx2;
            return residual;
        }

        /// <inheritdoc/>
        public int InputDimension => 2; // [x, t]

        /// <inheritdoc/>
        public int OutputDimension => 1; // [u]

        /// <inheritdoc/>
        public string Name => $"Heat Equation (α={_thermalDiffusivity})";
    }
}
