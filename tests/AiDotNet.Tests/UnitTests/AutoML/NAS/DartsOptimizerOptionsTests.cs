using System;
using AiDotNet.AutoML;
using AiDotNet.Enums;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AutoML.NAS
{
    /// <summary>
    /// The DARTS optimizer settings default to Liu et al. 2019, Sec. 5.1, and can be overridden.
    /// </summary>
    public class DartsOptimizerOptionsTests
    {
        [Fact]
        public void Defaults_AreThePapers()
        {
            var options = new DartsOptimizerOptions();

            Assert.Equal(3e-4, options.ArchitectureLearningRate);
            Assert.Equal(0.5, options.ArchitectureBeta1);
            Assert.Equal(0.999, options.ArchitectureBeta2);
            Assert.Equal(1e-3, options.ArchitectureWeightDecay);
            Assert.Equal(0.025, options.WeightMaxLearningRate);
            Assert.Equal(0.001, options.WeightMinLearningRate);
            Assert.Equal(0.9, options.WeightMomentum);
            Assert.Equal(3e-4, options.WeightDecay);
            Assert.Equal(5.0, options.GradientClipNorm);
        }

        [Fact]
        public void Constructor_KeepsCustomOptions_AndTheDefaultConstructorUsesThePapers()
        {
            var custom = new DartsOptimizerOptions { WeightMaxLearningRate = 0.1, WeightMinLearningRate = 0.01 };
            var nas = new NeuralArchitectureSearch<double>(NeuralArchitectureSearchStrategy.GradientBased, 3, custom);
            Assert.Same(custom, nas.OptimizerOptions);

            var defaults = new NeuralArchitectureSearch<double>();
            Assert.Equal(0.025, defaults.OptimizerOptions.WeightMaxLearningRate);
        }

        [Fact]
        public void Constructor_RejectsInvalidLearningRates()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new NeuralArchitectureSearch<double>(
                NeuralArchitectureSearchStrategy.GradientBased, 3,
                new DartsOptimizerOptions { WeightMaxLearningRate = 0.001, WeightMinLearningRate = 0.01 }));
        }
    }
}
