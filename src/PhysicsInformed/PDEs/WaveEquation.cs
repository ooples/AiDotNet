using System;
using System.Numerics;
using AiDotNet.PhysicsInformed.Interfaces;

namespace AiDotNet.PhysicsInformed.PDEs
{
    /// <summary>
    /// Represents the Wave Equation: ∂²u/∂t² = c² * ∇²u
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <remarks>
    /// For Beginners:
    /// The Wave Equation describes how waves propagate through a medium:
    /// - u(x,t) is the wave amplitude (displacement) at position x and time t
    /// - c is the wave speed (how fast disturbances propagate)
    /// - ∇²u is the Laplacian (spatial curvature) of u
    ///
    /// Physical Interpretation:
    /// - The acceleration (∂²u/∂t²) is proportional to the spatial curvature
    /// - If the wave is curved upward, it accelerates upward (and vice versa)
    /// - This creates oscillations that propagate at speed c
    ///
    /// Key Properties:
    /// - Solutions are superpositions of traveling waves: u(x±ct)
    /// - Energy is conserved
    /// - Waves can reflect, refract, diffract, and interfere
    ///
    /// Applications:
    /// - Sound waves in air/water
    /// - Vibrating strings (guitar, piano)
    /// - Electromagnetic waves (light, radio)
    /// - Seismic waves (earthquakes)
    /// - Water waves (ocean, lake)
    ///
    /// Example:
    /// A guitar string vibrates according to the wave equation when plucked.
    /// The wave speed depends on the string's tension and mass.
    /// </remarks>
    public class WaveEquation<T> : IPDESpecification<T> where T : struct, INumber<T>
    {
        private readonly T _waveSpeed;
        private readonly int _spatialDimension;

        /// <summary>
        /// Initializes a new instance of the Wave Equation.
        /// </summary>
        /// <param name="waveSpeed">The wave propagation speed c (must be positive).</param>
        /// <param name="spatialDimension">The number of spatial dimensions (1, 2, or 3).</param>
        public WaveEquation(T waveSpeed, int spatialDimension = 1)
        {
            if (waveSpeed <= T.Zero)
            {
                throw new ArgumentException("Wave speed must be positive.", nameof(waveSpeed));
            }

            if (spatialDimension < 1 || spatialDimension > 3)
            {
                throw new ArgumentException("Spatial dimension must be 1, 2, or 3.", nameof(spatialDimension));
            }

            _waveSpeed = waveSpeed;
            _spatialDimension = spatialDimension;
        }

        /// <inheritdoc/>
        public T ComputeResidual(T[] inputs, T[] outputs, PDEDerivatives<T> derivatives)
        {
            if (derivatives.SecondDerivatives == null)
            {
                throw new ArgumentException("Wave equation requires second derivatives.");
            }

            // For 1D wave equation: inputs = [x, t], outputs = [u]
            // PDE: ∂²u/∂t² - c² * ∂²u/∂x² = 0
            // For 2D: ∂²u/∂t² - c² * (∂²u/∂x² + ∂²u/∂y²) = 0

            int timeIndex = _spatialDimension; // Time is always the last input dimension
            T d2udt2 = derivatives.SecondDerivatives[0, timeIndex, timeIndex]; // ∂²u/∂t²

            // Compute spatial Laplacian: ∇²u
            T laplacian = T.Zero;
            for (int i = 0; i < _spatialDimension; i++)
            {
                laplacian += derivatives.SecondDerivatives[0, i, i]; // ∂²u/∂xi²
            }

            T c2 = _waveSpeed * _waveSpeed;
            T residual = d2udt2 - c2 * laplacian;
            return residual;
        }

        /// <inheritdoc/>
        public int InputDimension => _spatialDimension + 1; // Space + time

        /// <inheritdoc/>
        public int OutputDimension => 1; // [u]

        /// <inheritdoc/>
        public string Name => $"Wave Equation (c={_waveSpeed}, {_spatialDimension}D)";
    }
}
