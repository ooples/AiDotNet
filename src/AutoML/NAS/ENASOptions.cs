namespace AiDotNet.AutoML.NAS
{
    /// <summary>
    /// Hyperparameters of <see cref="ENAS{T}"/>. The defaults are those of Pham et al. 2018, "Efficient Neural
    /// Architecture Search via Parameter Sharing" (Sec. 3) and its reference implementation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> ENAS trains two things in turn: a controller that proposes architectures, and
    /// one set of shared weights that every proposed architecture reuses. These settings control how fast each is
    /// trained and how long the search runs. The defaults reproduce the paper; change them when a search space
    /// or dataset needs a longer search or a different learning rate.</para>
    /// </remarks>
    public class ENASOptions
    {
        /// <summary>Number of nodes in the searched cell (default: 4).</summary>
        public int NumNodes { get; set; } = 4;

        /// <summary>Hidden size of the controller LSTM (default: 100, as in the paper).</summary>
        public int ControllerHiddenSize { get; set; } = 100;

        /// <summary>Decay of the exponential-moving-average REINFORCE baseline (default: 0.95).</summary>
        public double BaselineDecay { get; set; } = 0.95;

        /// <summary>Weight of the policy entropy added to the reward (default: 0.01).</summary>
        public double EntropyWeight { get; set; } = 0.01;

        /// <summary>Controller parameters start uniform in [-range, range] (default: 0.1).</summary>
        public double ControllerInitRange { get; set; } = 0.1;

        /// <summary>Logits are squashed as c * tanh(logits / temperature); this is c (default: 2.5).</summary>
        public double ControllerTanhConstant { get; set; } = 2.5;

        /// <summary>Logits are squashed as c * tanh(logits / temperature); this is the temperature (default: 5).</summary>
        public double ControllerTemperature { get; set; } = 5.0;

        /// <summary>Adam learning rate of the controller (default: 0.00035).</summary>
        public double ControllerLearningRate { get; set; } = 3.5e-4;

        /// <summary>Adam beta1 of the controller (default: 0.9).</summary>
        public double ControllerBeta1 { get; set; } = 0.9;

        /// <summary>Adam beta2 of the controller (default: 0.999).</summary>
        public double ControllerBeta2 { get; set; } = 0.999;

        /// <summary>Adam epsilon of the controller (default: 1e-8).</summary>
        public double ControllerEpsilon { get; set; } = 1e-8;

        /// <summary>Initial (maximum) learning rate of the cosine schedule for the shared weights (default: 0.05).</summary>
        public double SharedMaxLearningRate { get; set; } = 0.05;

        /// <summary>Final (minimum) learning rate of the cosine schedule for the shared weights (default: 0.001).</summary>
        public double SharedMinLearningRate { get; set; } = 0.001;

        /// <summary>SGD momentum of the shared weights (default: 0.9).</summary>
        public double SharedMomentum { get; set; } = 0.9;

        /// <summary>L2 weight decay of the shared weights (default: 1e-4).</summary>
        public double SharedWeightDecay { get; set; } = 1e-4;

        /// <summary>Global gradient-norm clip of the shared weights (default: 5).</summary>
        public double SharedGradientClipNorm { get; set; } = 5.0;

        /// <summary>Number of alternating shared-weight and controller steps in a search (default: 50).</summary>
        public int SearchSteps { get; set; } = 50;

        /// <summary>Architectures sampled from the trained policy to pick the final one from (default: 10).</summary>
        public int DeriveSamples { get; set; } = 10;

        /// <summary>Seed of the sampling random generator (default: 42).</summary>
        public int Seed { get; set; } = 42;
    }
}
