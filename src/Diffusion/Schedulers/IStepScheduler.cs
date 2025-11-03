using System;

namespace AiDotNet.Diffusion.Schedulers
{
    public interface IStepScheduler
    {
        int[] Timesteps { get; }

        void SetTimesteps(int inferenceSteps);

        // Deterministic step when eta == 0; noise is ignored in that case.
        double[] Step(double[] modelOutput, int timestep, double[] sample, double eta = 0.0, double[]? noise = null);
    }
}
