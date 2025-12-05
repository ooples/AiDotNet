using Xunit;
using AiDotNet.Models.Generative.Diffusion;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.LinearAlgebra;

namespace UnitTests.Models.Generative.Diffusion
{
    public class DDPMModelTests
    {
        private sealed class NoopScheduler : IStepScheduler<double>
        {
            public int[] Timesteps { get; private set; } = System.Array.Empty<int>();
            public void SetTimesteps(int inferenceSteps) { Timesteps = new[] { 0 }; }
            public Vector<double> Step(Vector<double> modelOutput, int timestep, Vector<double> sample, double eta, Vector<double>? noise = null)
                => sample; // identity
        }

        [Fact]
        public void Predict_Uses_Scheduler()
        {
            var scheduler = new NoopScheduler();
            var model = new DDPMModel<double>(scheduler);
            var input = new Tensor<double>(new[] { 3 }, new Vector<double>(new double[] { 1, 2, 3 }));
            var outTensor = model.Predict(input);
            Assert.Equal(input.Length, outTensor.Length);
        }
    }
}

