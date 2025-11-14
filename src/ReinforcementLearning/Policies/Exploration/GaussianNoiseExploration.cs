using AiDotNet.LinearAlgebra;
using AiDotNet.Helpers;
using System;

namespace AiDotNet.ReinforcementLearning.Policies.Exploration
{
    /// <summary>
    /// Gaussian noise exploration for continuous action spaces.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class GaussianNoiseExploration<T> : IExplorationStrategy<T>
    {
        private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

        private double _noiseStdDev;
        private readonly double _noiseDecay;
        private readonly double _minNoise;

        public GaussianNoiseExploration(double initialStdDev = 0.1, double noiseDecay = 0.995, double minNoise = 0.01)
        {
            _noiseStdDev = initialStdDev;
            _noiseDecay = noiseDecay;
            _minNoise = minNoise;
        }

        public Vector<T> GetExplorationAction(Vector<T> state, Vector<T> policyAction, int actionSpaceSize, Random random)
        {
            var noisyAction = new Vector<T>(actionSpaceSize);

            for (int i = 0; i < actionSpaceSize; i++)
            {
                // Box-Muller transform for Gaussian noise
                double u1 = random.NextDouble();
                double u2 = random.NextDouble();
                double noise = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2) * _noiseStdDev;

                double actionValue = NumOps.ToDouble(policyAction[i]) + noise;
                // Clamp to [-1, 1] (Math.Clamp not available in net462)
                double clampedValue = Math.Max(-1.0, Math.Min(1.0, actionValue));
                noisyAction[i] = NumOps.FromDouble(clampedValue);
            }

            return noisyAction;
        }

        public void Update()
        {
            _noiseStdDev = Math.Max(_minNoise, _noiseStdDev * _noiseDecay);
        }

        public void Reset()
        {
            // Noise doesn't typically reset between episodes
        }

        public double CurrentNoiseStdDev => _noiseStdDev;
    }
}
