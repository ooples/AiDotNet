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
    R2D2,

    /// <summary>
    /// VERSA - Versatile and Efficient Few-shot Learning (Gordon et al., ICLR 2019).
    /// Uses an amortization network to produce task-specific classifier parameters
    /// in a single forward pass, eliminating the need for inner-loop optimization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Train a separate "amortization network" that takes aggregated
    /// support set features and directly outputs classifier weights. This replaces
    /// iterative inner-loop optimization with a single forward pass.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You need the fastest possible adaptation (no inner-loop optimization)
    /// and want to learn a general mapping from support sets to classifiers.
    /// </para>
    /// </remarks>
    VERSA,

    /// <summary>
    /// SNAIL - Simple Neural Attentive Meta-Learner (Mishra et al., ICLR 2018).
    /// Combines temporal convolutions with causal attention to perform
    /// sequence-to-sequence meta-learning on few-shot tasks.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Treat few-shot learning as a sequence modeling problem.
    /// Feed support examples (with labels) as a sequence, then feed query examples.
    /// Temporal convolutions capture local patterns, causal attention captures global patterns.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want to leverage powerful sequence modeling architectures
    /// for few-shot learning, especially when the order of examples matters or when
    /// you need both local and global pattern recognition.
    /// </para>
    /// </remarks>
    SNAIL,

    /// <summary>
    /// SimpleShot - Nearest-centroid classification with feature normalization (Wang et al., 2019).
    /// Shows that simple methods with proper normalization match complex meta-learning algorithms.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> A well-trained feature extractor + L2 or centered L2 normalization +
    /// nearest-centroid classification is a surprisingly strong few-shot baseline.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You need a strong baseline, want fast inference, or want to
    /// evaluate whether complex meta-learning methods are truly adding value.
    /// </para>
    /// </remarks>
    SimpleShot,

    /// <summary>
    /// DeepEMD - Earth Mover's Distance for few-shot learning (Zhang et al., CVPR 2020).
    /// Uses optimal transport to compare local feature sets between examples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Compare examples by finding the optimal matching between their local
    /// features using the Earth Mover's Distance. This captures fine-grained structural
    /// similarity that global feature comparison misses.
    /// </para>
    /// <para>
    /// <b>Use When:</b> Tasks involve structured data where part-to-part correspondence
    /// matters (e.g., fine-grained image classification, structural comparison).
    /// </para>
    /// </remarks>
    DeepEMD,

    /// <summary>
    /// FEAT - Few-shot Embedding Adaptation with Transformer (Ye et al., CVPR 2020).
    /// Uses a set-to-set transformer to adapt class prototypes based on inter-class relationships.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Prototypes should be task-aware. A transformer lets prototypes
    /// "see" each other and adjust their positions in feature space for better discrimination
    /// within each specific task.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want task-adaptive prototypes that capture inter-class
    /// relationships, improving over standard ProtoNets.
    /// </para>
    /// </remarks>
    FEAT,

    /// <summary>
    /// TIM - Transductive Information Maximization (Boudiaf et al., NeurIPS 2020).
    /// Refines query predictions by maximizing mutual information across the query set.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Use ALL query examples jointly by maximizing mutual information:
    /// each prediction should be confident (low conditional entropy) and class assignments
    /// should be balanced (high marginal entropy).
    /// </para>
    /// <para>
    /// <b>Use When:</b> You have access to all query examples at once (transductive setting)
    /// and want to exploit query set structure for better predictions.
    /// </para>
    /// </remarks>
    TIM,

    /// <summary>
    /// LaplacianShot - Laplacian Regularized Few-Shot Learning (Ziko et al., ICML 2020).
    /// Adds graph-based label propagation to nearest-centroid classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Build a kNN graph over query features and smooth predictions using
    /// the graph Laplacian. Similar queries get similar predictions, propagating confident
    /// labels to uncertain ones.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want a simple yet effective transductive method that improves
    /// upon SimpleShot with graph-based refinement.
    /// </para>
    /// </remarks>
    LaplacianShot,

    /// <summary>
    /// SIB - Sequential Information Bottleneck (Hu et al., 2020).
    /// Uses the information bottleneck principle for transductive few-shot clustering.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Iteratively refine cluster assignments by balancing information
    /// retention (useful for classification) with compression (removing noise).
    /// Multiple random restarts avoid local optima.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want a principled transductive method based on information
    /// theory with theoretical guarantees.
    /// </para>
    /// </remarks>
    SIB,

    /// <summary>
    /// PMF - Pre-train, Meta-train, Fine-tune (Hu et al., ICLR 2022).
    /// Three-stage pipeline combining standard pretraining, episodic meta-training,
    /// and optional task-specific fine-tuning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> The best of all worlds: use pretraining for good features,
    /// episodic training for few-shot adaptation, and optional fine-tuning for final refinement.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want a strong, well-studied pipeline that systematically
    /// combines the best practices from both transfer learning and meta-learning.
    /// </para>
    /// </remarks>
    PMF,

    /// <summary>
    /// Meta-Baseline - Simple pre-train then meta-train with cosine classifier (Chen et al., ICLR 2021).
    /// Shows that simple methods with cosine classification are surprisingly strong baselines.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Pre-train a feature extractor with standard classification, then
    /// fine-tune with episodic training using cosine-similarity nearest-centroid. Simplicity
    /// is competitive with complex meta-learning.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want a strong, simple baseline or when complex methods
    /// aren't clearly justified for your task.
    /// </para>
    /// </remarks>
    MetaBaseline,

    /// <summary>
    /// CAML - Context-Aware Meta-Learning (Fifty et al., NeurIPS 2023).
    /// Uses frozen pretrained backbones with lightweight context-aware adaptation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Modern pretrained models produce excellent features. Instead of
    /// fine-tuning the backbone, learn a small context module that adapts classification
    /// based on the support set structure.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You have access to a strong pretrained model and want efficient
    /// adaptation without backbone fine-tuning.
    /// </para>
    /// </remarks>
    CAML,

    /// <summary>
    /// Open-MAML - MAML extended for open-set recognition.
    /// Handles scenarios where query examples may belong to unseen classes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Extend MAML with a confidence-based rejection mechanism.
    /// The model learns to produce low confidence for out-of-distribution examples,
    /// enabling it to say "I don't know" instead of forcing a wrong classification.
    /// </para>
    /// <para>
    /// <b>Use When:</b> Your application may encounter classes not seen during support
    /// set construction, requiring robust unknown detection.
    /// </para>
    /// </remarks>
    OpenMAML,

    /// <summary>
    /// HyperShot - Kernel hypernetwork for few-shot learning.
    /// Generates task-specific kernel parameters from support set statistics.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Different tasks need different similarity functions. A hypernetwork
    /// generates custom kernel parameters for each task based on support set statistics,
    /// enabling adaptive distance computation.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You believe a fixed distance metric is suboptimal and want
    /// task-adaptive similarity computation.
    /// </para>
    /// </remarks>
    HyperShot,

    /// <summary>
    /// HyperMAML - Hypernetwork-based MAML initialization.
    /// Generates task-specific initial parameters rather than using a shared initialization.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Instead of one initialization for all tasks, use a hypernetwork
    /// that looks at the support set and generates a task-specific starting point.
    /// The custom initialization is already close to the task optimum.
    /// </para>
    /// <para>
    /// <b>Use When:</b> Tasks are highly diverse and a single MAML initialization
    /// can't serve all tasks well, or you want faster adaptation with fewer inner steps.
    /// </para>
    /// </remarks>
    HyperMAML,

    /// <summary>
    /// SetFeat - Matching Feature Sets for Few-Shot Classification (Afrasiyabi et al., CVPR 2022).
    /// Learns set-level features that capture intra-class variation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Represent each class as a SET of features rather than a single
    /// prototype. A set encoder captures how the class varies, and optional cross-attention
    /// lets classes inform each other.
    /// </para>
    /// <para>
    /// <b>Use When:</b> Intra-class variation matters for your task and simple mean
    /// prototypes lose important distributional information.
    /// </para>
    /// </remarks>
    SetFeat,

    /// <summary>
    /// FewTURE - Few-shot Transformer with Uncertainty and Reliable Estimation (Hiller et al., ECCV 2022).
    /// Token-level matching with uncertainty estimation for reliable prediction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Compare images at the patch/token level instead of globally.
    /// Estimate uncertainty for each token comparison and weight reliable matches more.
    /// This focuses on informative image regions automatically.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You need fine-grained comparison with uncertainty quantification,
    /// especially for visual tasks where discriminative features are localized.
    /// </para>
    /// </remarks>
    FewTURE,

    /// <summary>
    /// NPBML - Neural Process-Based Meta-Learning.
    /// Probabilistic meta-learner that captures task-level uncertainty.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Encode support sets into a latent DISTRIBUTION (not just a point).
    /// Multiple samples from this distribution give different predictions, and their
    /// disagreement quantifies uncertainty about the task.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You need uncertainty estimates for your few-shot predictions,
    /// such as safety-critical applications or active learning.
    /// </para>
    /// </remarks>
    NPBML,

    /// <summary>
    /// MCL - Meta-learning with Contrastive Learning.
    /// Combines episodic meta-learning with supervised contrastive learning.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Train features with two objectives: (1) meta-learning loss for
    /// few-shot task performance, and (2) contrastive loss for well-clustered embeddings.
    /// Features that are both task-adapted and well-organized transfer better.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want features that are simultaneously good for few-shot
    /// classification AND produce well-structured embedding spaces.
    /// </para>
    /// </remarks>
    MCL
}
