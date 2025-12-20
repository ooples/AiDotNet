using System;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.ReinforcementLearning.Policies.Exploration
{
    /// <summary>
    /// Gaussian noise exploration for continuous action spaces.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class GaussianNoiseExploration<T> : ExplorationStrategyBase<T>
    {
        private double _noiseStdDev;
        private readonly double _noiseDecay;
        private readonly double _minNoise;

        public GaussianNoiseExploration(double initialStdDev = 0.1, double noiseDecay = 0.995, double minNoise = 0.01)
        {
            _noiseStdDev = initialStdDev;
            _noiseDecay = noiseDecay;
            _minNoise = minNoise;
        }

        public override Vector<T> GetExplorationAction(Vector<T> state, Vector<T> policyAction, int actionSpaceSize, Random random)
        {
            var noisyAction = new Vector<T>(actionSpaceSize);

            for (int i = 0; i < actionSpaceSize; i++)
            {
                // Use BoxMullerSample from base class
                double noise = NumOps.ToDouble(BoxMullerSample(random)) * _noiseStdDev;

                double actionValue = NumOps.ToDouble(policyAction[i]) + noise;
                noisyAction[i] = NumOps.FromDouble(actionValue);
            }

            // Use ClampAction from base class (net462-compatible)
            return ClampAction(noisyAction);
        }

        public override void Update()
        {
            _noiseStdDev = Math.Max(_minNoise, _noiseStdDev * _noiseDecay);
        }

        public override void Reset()
        {
            // Noise doesn't typically reset between episodes
        }

        public double CurrentNoiseStdDev => _noiseStdDev;
    }
}
