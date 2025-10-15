namespace AiDotNet.FederatedLearning.MetaLearning.Parameters
{
    /// <summary>
    /// Model-Agnostic Meta-Learning (MAML) parameters configuration
    /// </summary>
    public class MAMLParameters
    {
        /// <summary>
        /// Inner loop learning rate for task-specific adaptation
        /// </summary>
        public double InnerLearningRate { get; set; } = 0.01;

        /// <summary>
        /// Outer loop learning rate for meta-parameter updates
        /// </summary>
        public double OuterLearningRate { get; set; } = 0.001;

        /// <summary>
        /// Number of gradient descent steps in the inner loop
        /// </summary>
        public int InnerSteps { get; set; } = 5;

        /// <summary>
        /// Convergence threshold for early stopping in inner loop
        /// </summary>
        public double ConvergenceThreshold { get; set; } = 1e-6;

        /// <summary>
        /// Use first-order approximation (FOMAML) to reduce computational cost
        /// </summary>
        public bool UseFirstOrder { get; set; } = false;

        /// <summary>
        /// Number of support examples per task for adaptation
        /// </summary>
        public int SupportSize { get; set; } = 5;

        /// <summary>
        /// Number of query examples per task for evaluation
        /// </summary>
        public int QuerySize { get; set; } = 15;

        /// <summary>
        /// Number of tasks to process in parallel for meta-updates
        /// </summary>
        public int TaskBatchSize { get; set; } = 4;

        /// <summary>
        /// Maximum number of meta-learning rounds
        /// </summary>
        public int MaxMetaRounds { get; set; } = 100;

        /// <summary>
        /// Whether to use adaptive learning rates
        /// </summary>
        public bool UseAdaptiveLearningRate { get; set; } = true;

        /// <summary>
        /// Gradient clipping threshold to prevent exploding gradients
        /// </summary>
        public double GradientClipThreshold { get; set; } = 1.0;

        /// <summary>
        /// Whether to normalize gradients before aggregation
        /// </summary>
        public bool NormalizeGradients { get; set; } = true;

        /// <summary>
        /// Momentum coefficient for outer loop optimizer
        /// </summary>
        public double OuterMomentum { get; set; } = 0.9;

        /// <summary>
        /// Validate parameters
        /// </summary>
        public void Validate()
        {
            if (InnerLearningRate <= 0 || InnerLearningRate > 1)
                throw new ArgumentException("Inner learning rate must be in (0, 1]");
            
            if (OuterLearningRate <= 0 || OuterLearningRate > 1)
                throw new ArgumentException("Outer learning rate must be in (0, 1]");
            
            if (InnerSteps <= 0)
                throw new ArgumentException("Inner steps must be positive");
            
            if (ConvergenceThreshold <= 0)
                throw new ArgumentException("Convergence threshold must be positive");
            
            if (SupportSize <= 0)
                throw new ArgumentException("Support size must be positive");
            
            if (QuerySize <= 0)
                throw new ArgumentException("Query size must be positive");
            
            if (TaskBatchSize <= 0)
                throw new ArgumentException("Task batch size must be positive");
            
            if (GradientClipThreshold <= 0)
                throw new ArgumentException("Gradient clip threshold must be positive");
        }
    }
}