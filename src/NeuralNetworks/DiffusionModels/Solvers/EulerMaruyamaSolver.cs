using System;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.DiffusionModels.Solvers
{
    /// <summary>
    /// Implements the Euler-Maruyama method for numerical solution of Stochastic Differential Equations (SDEs).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Euler-Maruyama method is the stochastic equivalent of the Euler method for ordinary differential equations.
    /// It provides a simple first-order numerical scheme for solving SDEs of the form:
    /// dX(t) = a(X(t), t)dt + b(t)dW(t)
    /// where a is the drift coefficient, b is the diffusion coefficient, and W(t) is a Wiener process (Brownian motion).
    /// </para>
    /// <para><b>For Beginners:</b> This is the simplest method for solving equations with randomness.
    /// 
    /// Think of it like predicting a drunk person's walk:
    /// - They have a general direction they're trying to go (drift)
    /// - But each step has some randomness (diffusion)
    /// - This solver calculates each step by combining both effects
    /// 
    /// The method works by:
    /// 1. Taking a small time step
    /// 2. Moving in the drift direction
    /// 3. Adding random noise scaled by the diffusion
    /// 
    /// It's simple and fast but may need very small time steps for accuracy.
    /// Good for:
    /// - Quick approximations
    /// - When speed is more important than precision
    /// - Testing and prototyping
    /// </para>
    /// </remarks>
    public class EulerMaruyamaSolver : ISolver
    {
        private readonly double _minTimeStep;
        private readonly double _maxTimeStep;
        
        /// <summary>
        /// Initializes a new instance of the EulerMaruyamaSolver class.
        /// </summary>
        /// <param name="minTimeStep">Minimum allowed time step (default: 1e-6)</param>
        /// <param name="maxTimeStep">Maximum allowed time step (default: 0.1)</param>
        public EulerMaruyamaSolver(double minTimeStep = 1e-6, double maxTimeStep = 0.1)
        {
            if (minTimeStep <= 0)
                throw new ArgumentException("Minimum time step must be positive", nameof(minTimeStep));
            if (maxTimeStep <= minTimeStep)
                throw new ArgumentException("Maximum time step must be greater than minimum time step", nameof(maxTimeStep));
            
            _minTimeStep = minTimeStep;
            _maxTimeStep = maxTimeStep;
        }
        
        /// <summary>
        /// Performs one step of the Euler-Maruyama integration scheme.
        /// </summary>
        /// <param name="x">Current state vector</param>
        /// <param name="t">Current time</param>
        /// <param name="dt">Time step size</param>
        /// <param name="drift">Drift function a(x,t)</param>
        /// <param name="diffusion">Diffusion coefficient function b(t)</param>
        /// <param name="random">Random number generator</param>
        /// <returns>Updated state vector</returns>
        /// <exception cref="ArgumentNullException">Thrown when required parameters are null</exception>
        /// <exception cref="ArgumentException">Thrown when time step is invalid</exception>
        public Tensor<double> Step(
            Tensor<double> x, 
            double t, 
            double dt,
            Func<Tensor<double>, double, Tensor<double>> drift,
            Func<double, double> diffusion,
            Random random)
        {
            // Validate inputs
            if (x == null)
                throw new ArgumentNullException(nameof(x));
            if (drift == null)
                throw new ArgumentNullException(nameof(drift));
            if (diffusion == null)
                throw new ArgumentNullException(nameof(diffusion));
            if (random == null)
                throw new ArgumentNullException(nameof(random));
            
            if (dt < _minTimeStep || dt > _maxTimeStep)
                throw new ArgumentException($"Time step must be between {_minTimeStep} and {_maxTimeStep}", nameof(dt));
            
            if (double.IsNaN(t) || double.IsInfinity(t))
                throw new ArgumentException("Time must be a valid finite number", nameof(t));
            
            try
            {
                // Compute drift term: a(x,t) * dt
                var driftTerm = drift(x, t).Multiply(dt);
                
                // Compute diffusion coefficient
                var diffusionCoeff = diffusion(t);
                
                // Generate Brownian motion increment: dW ~ N(0, dt)
                var noise = GenerateNoise(x.Shape, random);
                var diffusionTerm = noise.Multiply(diffusionCoeff * Math.Sqrt(dt));
                
                // Euler-Maruyama update: x_{n+1} = x_n + a(x_n, t_n)*dt + b(t_n)*sqrt(dt)*Z
                var result = x.Add(driftTerm).Add(diffusionTerm);
                
                // Check for numerical stability
                ValidateResult(result);
                
                return result;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to perform Euler-Maruyama step: {ex.Message}", ex);
            }
        }
        
        /// <summary>
        /// Generates Gaussian noise with the same shape as the input.
        /// </summary>
        private Tensor<double> GenerateNoise(int[] shape, Random random)
        {
            var totalElements = 1;
            foreach (var dim in shape)
                totalElements *= dim;
            
            var noiseData = new double[totalElements];
            
            // Generate samples from standard normal distribution using Box-Muller transform
            for (int i = 0; i < totalElements; i += 2)
            {
                var u1 = random.NextDouble();
                var u2 = random.NextDouble();
                
                // Box-Muller transform
                var radius = Math.Sqrt(-2.0 * Math.Log(u1));
                var theta = 2.0 * Math.PI * u2;
                
                noiseData[i] = radius * Math.Cos(theta);
                if (i + 1 < totalElements)
                {
                    noiseData[i + 1] = radius * Math.Sin(theta);
                }
            }
            
            return new Tensor<double>(shape, new Vector<double>(noiseData));
        }
        
        /// <summary>
        /// Validates the result for numerical stability.
        /// </summary>
        private void ValidateResult(Tensor<double> result)
        {
            var data = result.ToVector();
            for (int i = 0; i < data.Length; i++)
            {
                if (double.IsNaN(data[i]) || double.IsInfinity(data[i]))
                {
                    throw new InvalidOperationException("Numerical instability detected: result contains NaN or infinity values");
                }
            }
        }
    }
}