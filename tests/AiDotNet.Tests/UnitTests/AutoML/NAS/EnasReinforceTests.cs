using System;
using System.Linq;
using AiDotNet.AutoML.NAS;
using AiDotNet.AutoML.SearchSpace;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.NAS
{
    /// <summary>
    /// ENAS controller training (Pham et al. 2018, Sec. 2.2): REINFORCE with the gradient
    /// -(R - b) * d/dtheta sum_t log pi(a_t) over the decisions of the rewarded sample, an EMA baseline b, and Adam.
    /// </summary>
    public class EnasReinforceTests
    {
        [Fact]
        public void UpdateController_ReinforcesTheSampleWhoseLogProbIsGiven_NotTheLastOne()
        {
            const double decay = 0.9, lambda = 0.1;
            var enas = new ENAS<double>(new SearchSpaceBase<double>(), numNodes: 3, controllerHiddenSize: 4,
                baselineDecay: decay, entropyWeight: lambda);

            var (_, logProbA, entropyA) = enas.SampleArchitecture();
            var decisionsA = enas.LastDecisions.ToList();

            // Sample a second, different architecture so that A is no longer the most recent sample.
            var decisionsB = decisionsA;
            for (int attempt = 0; attempt < 50 && decisionsB.SequenceEqual(decisionsA); attempt++)
            {
                enas.SampleArchitecture();
                decisionsB = enas.LastDecisions.ToList();
            }

            Assert.False(decisionsB.SequenceEqual(decisionsA));

            var scoreA = enas.LogProbabilityGradient(decisionsA);
            enas.UpdateController(1.5, logProbA, entropyA);

            double shaped = 1.5 + lambda * entropyA;
            double baseline = (1 - decay) * shaped;
            var gradients = enas.GetControllerGradients();
            for (int i = 0; i < gradients.Count; i++)
            {
                for (int k = 0; k < gradients[i].Length; k++)
                {
                    Assert.Equal(-(shaped - baseline) * scoreA[i][k], gradients[i][k], 12);
                }
            }
        }

        [Fact]
        public void UpdateController_UnknownOrAlreadyRewardedLogProb_Throws()
        {
            var enas = new ENAS<double>(new SearchSpaceBase<double>(), numNodes: 3, controllerHiddenSize: 4);
            var (_, logProb, entropy) = enas.SampleArchitecture();

            Assert.Throws<InvalidOperationException>(() => enas.UpdateController(1.0, logProb + 1.0, entropy));

            enas.UpdateController(1.0, logProb, entropy);
            Assert.Throws<InvalidOperationException>(() => enas.UpdateController(1.0, logProb, entropy));
        }

        [Fact]
        public void Options_DefaultsAreThePapers()
        {
            var options = new ENASOptions();

            Assert.Equal(3.5e-4, options.ControllerLearningRate);
            Assert.Equal(0.05, options.SharedMaxLearningRate);
            Assert.Equal(0.001, options.SharedMinLearningRate);
            Assert.Equal(0.9, options.SharedMomentum);
            Assert.Equal(1e-4, options.SharedWeightDecay);
            Assert.Equal(5.0, options.SharedGradientClipNorm);
            Assert.Equal(2.5, options.ControllerTanhConstant);
            Assert.Equal(5.0, options.ControllerTemperature);
            Assert.Equal(0.1, options.ControllerInitRange);
        }

        [Fact]
        public void Options_ControllerLearningRate_SetsTheAdamStepSize()
        {
            // The first bias-corrected Adam step moves each parameter by -lr * g / (|g| + eps).
            const double lr = 0.01, eps = 1e-8;
            var enas = new ENAS<double>(new SearchSpaceBase<double>(),
                new ENASOptions { NumNodes = 3, ControllerHiddenSize = 4, ControllerLearningRate = lr });
            Assert.Equal(lr, enas.Options.ControllerLearningRate);

            var (_, logProb, entropy) = enas.SampleArchitecture();
            var before = enas.GetControllerParameters().Select(v => v.ToArray()).ToList();
            enas.UpdateController(2.0, logProb, entropy);

            var gradients = enas.GetControllerGradients();
            var after = enas.GetControllerParameters();
            for (int i = 0; i < gradients.Count; i++)
            {
                for (int k = 0; k < gradients[i].Length; k++)
                {
                    double g = gradients[i][k];
                    Assert.Equal(before[i][k] - lr * g / (Math.Abs(g) + eps), after[i][k], 12);
                }
            }
        }

        [Fact]
        public void Options_InvalidValues_Throw()
        {
            var space = new SearchSpaceBase<double>();
            Assert.Throws<ArgumentOutOfRangeException>(() => new ENAS<double>(space, new ENASOptions { DeriveSamples = 0 }));
            Assert.Throws<ArgumentOutOfRangeException>(() => new ENAS<double>(space, new ENASOptions { BaselineDecay = 1.5 }));
        }
    }
}
