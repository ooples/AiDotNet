namespace AiDotNet.MetaLearning;

/// <summary>
/// Specifies the type of meta-learning algorithm used for few-shot learning and quick adaptation.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Meta-learning algorithms are designed to "learn how to learn."
/// Instead of learning a single task, they learn to quickly adapt to new tasks with minimal data.
/// This enum lists all supported meta-learning algorithms in the framework.
/// </para>
/// <para>
/// <b>Algorithm Categories:</b>
/// <list type="bullet">
/// <item><b>Optimization-based:</b> MAML, Reptile, Meta-SGD, iMAML, ANIL, BOIL, LEO</item>
/// <item><b>Metric-based:</b> ProtoNets, MatchingNetworks, RelationNetwork, TADAM</item>
/// <item><b>Memory-based:</b> MANN, NTM</item>
/// <item><b>Hybrid/Advanced:</b> CNAP, SEAL, GNNMeta, MetaOptNet</item>
/// </list>
/// </para>
/// </remarks>
public enum MetaLearningAlgorithmType
{
    /// <summary>
    /// Model-Agnostic Meta-Learning (Finn et al., 2017).
    /// The foundational gradient-based meta-learning algorithm that learns an initialization
    /// that can be quickly fine-tuned to new tasks with a few gradient steps.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Find initial parameters that are sensitive to task-specific changes,
    /// so that small gradient updates produce large improvements in task performance.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You need a general-purpose meta-learning approach that works across
    /// different domains (classification, regression, reinforcement learning).
    /// </para>
    /// </remarks>
    MAML,

    /// <summary>
    /// Reptile meta-learning algorithm (Nichol et al., 2018).
    /// A simpler alternative to MAML that avoids computing second-order derivatives.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Repeatedly sample a task, train on it, and move the initialization
    /// towards the trained weights. Simpler gradient computation than MAML.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want MAML-like performance with lower computational cost and
    /// simpler implementation.
    /// </para>
    /// </remarks>
    Reptile,

    /// <summary>
    /// Meta-SGD with per-parameter learning rates (Li et al., 2017).
    /// Extends MAML by learning not just the initialization but also per-parameter learning rates.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Different parameters may need different learning rates for optimal
    /// adaptation. Meta-SGD learns these rates as part of the meta-learning process.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You suspect that uniform learning rates are suboptimal for your
    /// model architecture.
    /// </para>
    /// </remarks>
    MetaSGD,

    /// <summary>
    /// Implicit MAML with implicit gradients (Rajeswaran et al., 2019).
    /// Uses implicit differentiation to compute meta-gradients more efficiently.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Instead of differentiating through the optimization path, use
    /// the implicit function theorem to compute gradients. Enables more inner-loop steps.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You need many inner-loop adaptation steps and MAML's memory
    /// requirements become prohibitive.
    /// </para>
    /// </remarks>
    iMAML,

    /// <summary>
    /// Conditional Neural Adaptive Processes (Requeima et al., 2019).
    /// Combines neural processes with task-specific adaptation using FiLM layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Generate task-specific parameters by conditioning on the support set,
    /// enabling fast adaptation without gradient-based fine-tuning at test time.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You need fast inference-time adaptation without gradient computation.
    /// </para>
    /// </remarks>
    CNAP,

    /// <summary>
    /// Self-Explanatory Attention Learning.
    /// Combines attention mechanisms with meta-learning for interpretable few-shot learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Use attention to focus on relevant features and provide
    /// explanations for predictions in few-shot scenarios.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You need interpretable meta-learning with attention-based explanations.
    /// </para>
    /// </remarks>
    SEAL,

    /// <summary>
    /// Task-Dependent Adaptive Metric (Oreshkin et al., 2018).
    /// Combines metric-based learning with task-dependent feature scaling.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Learn to adapt the metric space based on the task at hand,
    /// combining prototypical networks with task-conditional scaling.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want metric-based learning with task-specific adaptation.
    /// </para>
    /// </remarks>
    TADAM,

    /// <summary>
    /// Graph Neural Network for Meta-Learning.
    /// Uses graph neural networks to model relationships between examples in few-shot learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Treat the support and query examples as nodes in a graph and use
    /// message passing to propagate information for classification.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want to explicitly model relationships between all examples
    /// in a task.
    /// </para>
    /// </remarks>
    GNNMeta,

    /// <summary>
    /// Neural Turing Machine for meta-learning.
    /// Uses external memory with read/write heads for meta-learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Use a differentiable external memory to store and retrieve
    /// task-relevant information across examples.
    /// </para>
    /// <para>
    /// <b>Use When:</b> Tasks require storing and retrieving specific examples or patterns.
    /// </para>
    /// </remarks>
    NTM,

    /// <summary>
    /// Memory-Augmented Neural Network (Santoro et al., 2016).
    /// Uses external memory for one-shot learning without explicit training phases.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Store examples in external memory and learn to retrieve similar
    /// examples for classification. No explicit support/query split at inference.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You need online learning capabilities where examples arrive sequentially.
    /// </para>
    /// </remarks>
    MANN,

    /// <summary>
    /// Matching Networks for One Shot Learning (Vinyals et al., 2016).
    /// Uses attention over support examples for one-shot classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Embed examples in a shared space and classify by computing
    /// attention-weighted similarity to support examples.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You need simple, non-parametric few-shot classification with
    /// attention mechanisms.
    /// </para>
    /// </remarks>
    MatchingNetworks,

    /// <summary>
    /// Prototypical Networks (Snell et al., 2017).
    /// Learns a metric space where classification is performed by computing distances to class prototypes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Represent each class by the mean (prototype) of its support examples
    /// in embedding space. Classify by nearest prototype.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want simple, effective metric-based few-shot learning with
    /// strong baselines.
    /// </para>
    /// </remarks>
    ProtoNets,

    /// <summary>
    /// Relation Network for few-shot learning (Sung et al., 2018).
    /// Learns to compare query and support examples through a learned relation module.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Instead of using a fixed distance metric, learn a neural network
    /// that computes relation scores between example pairs.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want to learn complex, non-linear similarity functions.
    /// </para>
    /// </remarks>
    RelationNetwork,

    /// <summary>
    /// Almost No Inner Loop (Raghu et al., 2020).
    /// A simplified version of MAML that only adapts the final classification layer.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> The feature extractor is frozen during inner-loop adaptation;
    /// only the classifier head is updated. Much faster than full MAML.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want faster adaptation with comparable performance to MAML.
    /// </para>
    /// </remarks>
    ANIL,

    /// <summary>
    /// Latent Embedding Optimization (Rusu et al., 2019).
    /// Performs optimization in a low-dimensional latent space for faster adaptation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Learn a low-dimensional latent space for model parameters.
    /// Adaptation happens in this latent space, then maps back to full parameters.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You need to adapt very large models quickly with limited data.
    /// </para>
    /// </remarks>
    LEO,

    /// <summary>
    /// Meta-learning with differentiable convex optimization (Lee et al., 2019).
    /// Uses a differentiable SVM or ridge regression for the final classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Replace the inner-loop gradient descent with a closed-form
    /// convex optimization (like ridge regression or SVM) that is differentiable.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want theoretically grounded, stable optimization in the inner loop.
    /// </para>
    /// </remarks>
    MetaOptNet,

    /// <summary>
    /// Body Only Inner Loop (Oh et al., 2021).
    /// Opposite of ANIL - only adapts the feature extractor, keeping the head frozen.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> The classifier head is frozen; only the feature extractor (body)
    /// is adapted during the inner loop. Provides different inductive biases than ANIL.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You believe task-specific features are more important than
    /// task-specific classifiers.
    /// </para>
    /// </remarks>
    BOIL,

    /// <summary>
    /// Fast Context Adaptation via Meta-Learning (Zintgraf et al., ICML 2019).
    /// Separates model parameters into shared body and task-specific context, adapting only context in the inner loop.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Divide parameters into shared body parameters (updated in outer loop)
    /// and a small context vector (adapted per task in inner loop). Much faster than MAML
    /// because only the context vector is differentiated through.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want MAML-like adaptation speed with reduced meta-overfitting
    /// and lower computational cost. Especially effective when tasks share common structure
    /// but differ in specific aspects.
    /// </para>
    /// </remarks>
    CAVIA,

    /// <summary>
    /// Warped Gradient Descent meta-learning (Flennerhag et al., ICLR 2020).
    /// Learns preconditioning warp-layers that transform gradients for more efficient inner-loop adaptation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Instead of just learning a good initialization (like MAML), learn a
    /// gradient preconditioning transformation that makes gradient descent more effective.
    /// Warp-layers reshape the optimization landscape without requiring second-order gradients.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want efficient adaptation without second-order gradient cost,
    /// or when tasks benefit from learning both initialization and optimization geometry.
    /// </para>
    /// </remarks>
    WarpGrad,

    /// <summary>
    /// MAML++ - How to Train Your MAML (Antoniou et al., ICLR 2019).
    /// Production-hardened MAML with multi-step loss, per-step learning rates,
    /// derivative-order annealing, and batch normalization fixes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Addresses training instabilities in MAML through engineering improvements:
    /// multi-step loss optimization (MSL), learned step-size learning rates (LSLR),
    /// derivative-order annealing, and per-step batch normalization.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want MAML's approach but with production-grade stability and performance.
    /// </para>
    /// </remarks>
    MAMLPlusPlus,

    /// <summary>
    /// R2-D2 - Meta-learning with Differentiable Closed-form Solvers (Bertinetto et al., ICLR 2019).
    /// Uses differentiable ridge regression as a closed-form inner-loop solver.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Replace iterative inner-loop gradient descent with a closed-form
    /// ridge regression solver. The exact solution w = (X^T X + lambda I)^-1 X^T y is
    /// differentiable, enabling efficient meta-gradient computation.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want extremely fast inner-loop adaptation with a mathematically
    /// optimal classifier, especially when tasks have linearly separable features.
    /// </para>
    /// </remarks>
    R2D2
}
