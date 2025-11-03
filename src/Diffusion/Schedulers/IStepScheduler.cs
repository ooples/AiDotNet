using System;

namespace AiDotNet.Diffusion.Schedulers
{
    public interface IStepScheduler<T>
    {
        int[] Timesteps { get; }

        void SetTimesteps(int inferenceSteps);

        // Deterministic step when eta == 0; noise is ignored in that case.
        AiDotNet.LinearAlgebra.Vector<T> Step(AiDotNet.LinearAlgebra.Vector<T> modelOutput, int timestep, AiDotNet.LinearAlgebra.Vector<T> sample, T eta, AiDotNet.LinearAlgebra.Vector<T>? noise = null);
    }
}
