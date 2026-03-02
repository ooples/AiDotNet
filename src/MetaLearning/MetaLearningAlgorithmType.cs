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
/// <item><b>Neural Processes:</b> CNP, NP, ANP, ConvCNP, ConvNP, TNP, SwinTNP, TETNP, EquivCNP, SteerCNP, RCNP, LBANP</item>
/// <item><b>Foundation Model Era:</b> MetaLoRA, LoRARecycle, ICMFusion, MetaLoRABank, AutoLoRA, MetaDiff, MetaDM, MetaDDPM</item>
/// <item><b>Bayesian Extensions:</b> PACOH, MetaPACOH, BMAML, BayProNet, FlexPACBayes</item>
/// <item><b>Cross-Domain:</b> MetaFDMixup, FreqPrior, MetaCollaborative, SDCL, FreqPrompt, OpenMAMLPlus</item>
/// <item><b>Meta-RL:</b> PEARL, DREAM, DiscoRL, InContextRL, HyperNetMetaRL, ContextMetaRL</item>
/// <item><b>Continual/Online:</b> ACL, iTAML, MetaContinualAL, MePo, OML, MOCA</item>
/// <item><b>Task Augmentation:</b> MetaTask, ATAML, MPTS, DynamicTaskSampling, UnsupervisedMetaLearn</item>
/// <item><b>Transductive:</b> GCDPLNet, BayTransProto, JMP, ETPN, ActiveTransFSL</item>
/// <item><b>Hypernetwork:</b> TaskCondHyperNet, HyperCLIP, RecurrentHyperNet, HyperNeRFMeta</item>
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
    MCL,

    /// <summary>
    /// DKT - Deep Kernel Transfer (Patacchiola et al., ICLR 2020).
    /// Combines deep feature extractors with Gaussian processes for Bayesian few-shot classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Use a neural network to learn a feature space where a GP classifier
    /// provides principled Bayesian predictions with uncertainty estimates. The deep kernel
    /// is trained end-to-end.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You need uncertainty estimates for predictions, or when
    /// principled Bayesian reasoning is important for your application.
    /// </para>
    /// </remarks>
    DKT,

    /// <summary>
    /// DPGN - Distribution Propagation Graph Network (Yang et al., CVPR 2020).
    /// Dual graph structure propagating both point estimates and distribution information.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Use two complementary graphs - a point graph for feature propagation
    /// and a distribution graph for uncertainty propagation. Both refine each other
    /// through multi-layer message passing.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want to model both feature similarity and confidence
    /// explicitly in a graph-based framework.
    /// </para>
    /// </remarks>
    DPGN,

    /// <summary>
    /// EPNet - Embedding Propagation Network (Rodriguez et al., CVPR 2020).
    /// Refines embeddings through label propagation on a nearest-neighbor graph.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Build a kNN graph over all examples and propagate features
    /// through diffusion. This smooths the feature manifold, making features more
    /// discriminative and robust to noise.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want a simple transductive method that improves features
    /// through manifold smoothing.
    /// </para>
    /// </remarks>
    EPNet,

    /// <summary>
    /// PT+MAP - Power Transform + Maximum A Posteriori (Hu et al., ICLR 2021).
    /// Simple power transform normalization with Bayesian MAP classification.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Apply a power transform to make features Gaussian, then use
    /// MAP estimation for optimal Bayesian classification. Surprisingly effective
    /// despite its simplicity.
    /// </para>
    /// <para>
    /// <b>Use When:</b> You want a strong transductive baseline with minimal complexity,
    /// or when feature distributions are highly non-Gaussian.
    /// </para>
    /// </remarks>
    PTMAP,

    /// <summary>
    /// FRN - Few-shot Classification via Feature Map Reconstruction (Wertheimer et al., CVPR 2021).
    /// Classifies by reconstruction error from class-specific support features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Instead of comparing features by distance, try to reconstruct
    /// the query features using each class's support features via ridge regression.
    /// The class with lowest reconstruction error is chosen.
    /// </para>
    /// <para>
    /// <b>Use When:</b> Simple distance metrics are insufficient and you want
    /// reconstruction-based classification that can combine multiple support examples.
    /// </para>
    /// </remarks>
    FRN,

    /// <summary>
    /// ConstellationNet - Structured part-based few-shot learning (Xu et al., ICLR 2021).
    /// Detects discriminative parts and models their spatial relationships.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Key Idea:</b> Detect K discriminative parts per example and model their
    /// spatial arrangement as a "constellation." Classification matches both
    /// individual parts and their structural arrangement.
    /// </para>
    /// <para>
    /// <b>Use When:</b> Discriminative information lies in the arrangement of parts,
    /// such as fine-grained recognition where spatial structure matters.
    /// </para>
    /// </remarks>
    ConstellationNet,

    // ===== Neural Process Family =====

    /// <summary>
    /// Conditional Neural Process (Garnelo et al., ICML 2018).
    /// Encodes context (support) points independently and aggregates them to predict target (query) points.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Encode each context pair (x,y) independently, aggregate via mean pooling,
    /// and decode to predict targets. Fast but underfits due to lack of latent variable.
    /// <para><b>Use When:</b> You need fast amortized inference without gradient-based adaptation.</para>
    /// </remarks>
    CNP,

    /// <summary>
    /// Neural Process (Garnelo et al., 2018).
    /// Extends CNP with a latent variable for modeling uncertainty and correlations.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Adds a global latent z ~ N(mu, sigma) from context encoding.
    /// Samples from z allow capturing coherent function samples and uncertainty.
    /// <para><b>Use When:</b> You need uncertainty estimates and coherent predictions across targets.</para>
    /// </remarks>
    NP,

    /// <summary>
    /// Attentive Neural Process (Kim et al., ICLR 2019).
    /// Adds cross-attention from targets to context for better predictions.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Instead of mean-pooling context, use attention so each target attends to relevant
    /// context points. Combines deterministic attention path with latent variable path.
    /// <para><b>Use When:</b> You want NP-quality uncertainty with CNP-quality point predictions.</para>
    /// </remarks>
    ANP,

    /// <summary>
    /// Convolutional Conditional Neural Process (Gordon et al., ICLR 2020).
    /// Places context on a grid and uses convolutions for translation equivariance.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Map context to a discrete grid, apply CNN layers, then interpolate for targets.
    /// Translation equivariance is a powerful inductive bias for spatial/temporal data.
    /// <para><b>Use When:</b> Data has spatial or temporal structure benefiting from translation equivariance.</para>
    /// </remarks>
    ConvCNP,

    /// <summary>
    /// Convolutional Neural Process (Foong et al., 2020).
    /// Extends ConvCNP with a latent variable for uncertainty modeling.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Combines ConvCNP's grid-based convolutions with NP's latent variable.
    /// Provides both translation equivariance and coherent uncertainty estimates.
    /// <para><b>Use When:</b> You need ConvCNP's inductive bias plus NP-style uncertainty.</para>
    /// </remarks>
    ConvNP,

    /// <summary>
    /// Transformer Neural Process (Nguyen &amp; Grover, ICML 2023).
    /// Uses transformer architecture for neural process encoding and decoding.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Replace set encoders with transformer blocks. Self-attention over context
    /// and cross-attention to targets provides flexible, powerful function approximation.
    /// <para><b>Use When:</b> You want state-of-the-art NP performance with transformer scalability.</para>
    /// </remarks>
    TNP,

    /// <summary>
    /// Swin Transformer Neural Process (2024).
    /// Combines Swin Transformer's hierarchical attention with neural processes.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Use shifted window attention for efficient multi-scale processing of context sets.
    /// Hierarchical features capture both local and global context patterns.
    /// <para><b>Use When:</b> Large context sets where full attention is too expensive.</para>
    /// </remarks>
    SwinTNP,

    /// <summary>
    /// Translation-Equivariant Transformer Neural Process (2024).
    /// Combines TNP's transformer attention with translation equivariance via relative positional encoding.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Standard TNP uses absolute positions in attention. TE-TNP replaces this
    /// with relative positional encodings so that the prediction function is equivariant to input translations.
    /// This gives a strong inductive bias for spatial and temporal data.
    /// <para><b>Use When:</b> Your regression/prediction task has translational symmetry (e.g., spatial processes,
    /// time series where absolute position is irrelevant).</para>
    /// </remarks>
    TETNP,

    /// <summary>
    /// Equivariant Conditional Neural Process (Kawano et al., 2021).
    /// Incorporates symmetry equivariance into the CNP architecture.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Encoder and decoder respect group symmetries (rotation, reflection).
    /// Equivariance provides strong inductive bias for data with known symmetries.
    /// <para><b>Use When:</b> Your data has known symmetries (e.g., physical systems, molecular data).</para>
    /// </remarks>
    EquivCNP,

    /// <summary>
    /// Steerable Conditional Neural Process (Holderrieth et al., 2021).
    /// Uses steerable kernels for continuous equivariance in neural processes.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Steerable convolutions provide continuous rotation equivariance rather than
    /// discrete group equivariance, giving smoother function predictions.
    /// <para><b>Use When:</b> You need continuous rotation equivariance (e.g., orientation-sensitive tasks).</para>
    /// </remarks>
    SteerCNP,

    /// <summary>
    /// Recurrent Conditional Neural Process (2024).
    /// Processes context points sequentially with a recurrent encoder for streaming data.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Replace set aggregation with recurrent processing. Enables online updating
    /// as new context points arrive without reprocessing the entire set.
    /// <para><b>Use When:</b> Context points arrive in a stream and you need online updates.</para>
    /// </remarks>
    RCNP,

    /// <summary>
    /// Latent Bottleneck Attentive Neural Process (Feng et al., ICML 2023).
    /// Introduces a small set of latent bottleneck tokens for efficient attention.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Instead of O(n*m) cross-attention, use K latent tokens as a bottleneck.
    /// Context compresses into latent tokens, targets attend to them. O(n*K + m*K) complexity.
    /// <para><b>Use When:</b> Large context or target sets where full attention is too expensive.</para>
    /// </remarks>
    LBANP,

    // ===== Foundation Model Era Methods =====

    /// <summary>
    /// Meta-LoRA - Meta-learning with Low-Rank Adaptation (2024).
    /// Learns a meta-initialization for LoRA parameters that adapts quickly to new tasks.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Apply MAML-style meta-learning to LoRA adapter parameters instead of full model.
    /// Much more parameter-efficient than full-model meta-learning.
    /// <para><b>Use When:</b> You want to meta-learn adaptation of large pretrained models efficiently.</para>
    /// </remarks>
    MetaLoRA,

    /// <summary>
    /// LoRA-Recycle - Recycling LoRA adapters across tasks (2024).
    /// Reuses and recombines LoRA adapters from previously seen tasks for new tasks.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Maintain a library of task-specific LoRA adapters. For new tasks, select and
    /// combine relevant adapters via learned routing, avoiding training from scratch.
    /// <para><b>Use When:</b> You have a growing library of task adapters and want to transfer between them.</para>
    /// </remarks>
    LoRARecycle,

    /// <summary>
    /// ICM Fusion - In-Context Model Fusion (2024).
    /// Merges multiple adapted models in-context without additional training.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Fuse predictions from multiple task-specific models by treating their outputs
    /// as context for a meta-fusion module. No gradient updates needed at fusion time.
    /// <para><b>Use When:</b> You have multiple expert models and want to combine them for new tasks.</para>
    /// </remarks>
    ICMFusion,

    /// <summary>
    /// Meta-LoRA Bank - Library of meta-learned LoRA modules (2024).
    /// Maintains a bank of reusable LoRA modules that can be composed for new tasks.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Decompose adaptation into modular LoRA components. A routing network selects
    /// and weights relevant modules from the bank for each new task.
    /// <para><b>Use When:</b> Tasks share sub-components and modular composition is beneficial.</para>
    /// </remarks>
    MetaLoRABank,

    /// <summary>
    /// AutoLoRA - Automatic LoRA rank and configuration selection (2024).
    /// Uses meta-learning to automatically determine optimal LoRA hyperparameters per layer.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Different layers benefit from different LoRA ranks. AutoLoRA meta-learns
    /// per-layer rank allocation and initialization for optimal adaptation efficiency.
    /// <para><b>Use When:</b> You want automated LoRA configuration without manual hyperparameter tuning.</para>
    /// </remarks>
    AutoLoRA,

    /// <summary>
    /// MetaDiff - Meta-learning with Diffusion Models (2024).
    /// Uses diffusion process for generating task-specific model parameters.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Frame parameter generation as a denoising diffusion process. The diffusion model
    /// learns to generate task-adapted parameters from noise conditioned on the support set.
    /// <para><b>Use When:</b> You want to model the full distribution of task-optimal parameters.</para>
    /// </remarks>
    MetaDiff,

    /// <summary>
    /// MetaDM - Meta Diffusion Model for few-shot generation (2024).
    /// Adapts diffusion models to new domains with few examples using meta-learning.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Meta-train a diffusion model across diverse domains so it can quickly adapt
    /// its generation process to new domains with just a few examples.
    /// <para><b>Use When:</b> You need few-shot generative modeling across diverse domains.</para>
    /// </remarks>
    MetaDM,

    /// <summary>
    /// MetaDDPM - Meta Denoising Diffusion Probabilistic Model (2024).
    /// Combines DDPM with meta-learning for task-conditional generation.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Meta-learn the denoising schedule and noise prediction network so that
    /// the DDPM can be quickly adapted to generate samples for new task distributions.
    /// <para><b>Use When:</b> You need task-adaptive generation with principled probabilistic modeling.</para>
    /// </remarks>
    MetaDDPM,

    // ===== Bayesian Extensions =====

    /// <summary>
    /// PACOH - PAC-Bayesian Meta-Learning with Optimal Hyperparameters (Rothfuss et al., ICLR 2021).
    /// Provides PAC-Bayesian generalization bounds for meta-learning.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Use PAC-Bayesian theory to derive principled meta-learning objectives with
    /// provable generalization guarantees. The meta-learned prior provides tight bounds.
    /// <para><b>Use When:</b> You need theoretical guarantees on meta-learning generalization.</para>
    /// </remarks>
    PACOH,

    /// <summary>
    /// Meta-PACOH - Extended PACOH with hierarchical Bayesian meta-learning (2023).
    /// Adds hierarchical structure to PACOH for multi-level meta-learning.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Extend PACOH with hierarchical Bayesian structure for learning at multiple
    /// levels of abstraction, with PAC-Bayesian bounds at each level.
    /// <para><b>Use When:</b> Tasks have hierarchical structure (e.g., domain -> subdomain -> task).</para>
    /// </remarks>
    MetaPACOH,

    /// <summary>
    /// BMAML - Bayesian MAML (Yoon et al., NeurIPS 2018).
    /// Combines MAML with Stein Variational Gradient Descent for Bayesian inference.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Maintain a particle ensemble of model initializations and use SVGD to
    /// approximate the posterior over task-adapted parameters, providing uncertainty estimates.
    /// <para><b>Use When:</b> You need uncertainty-aware MAML with posterior approximation.</para>
    /// </remarks>
    BMAML,

    /// <summary>
    /// BayProNet - Bayesian Prototypical Networks (2024).
    /// Extends ProtoNets with Bayesian inference over prototypes.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Instead of point-estimate prototypes, maintain distributions over prototype
    /// locations. Classification integrates over prototype uncertainty for robust predictions.
    /// <para><b>Use When:</b> You want ProtoNets with principled uncertainty over class representations.</para>
    /// </remarks>
    BayProNet,

    /// <summary>
    /// Flex-PAC-Bayes - Flexible PAC-Bayes bounds for meta-learning (2024).
    /// Provides tighter, more flexible PAC-Bayesian bounds for meta-learners.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Use data-dependent priors and flexible divergence measures to derive
    /// tighter PAC-Bayesian bounds that better reflect actual meta-learning performance.
    /// <para><b>Use When:</b> You need the tightest available generalization guarantees for meta-learning.</para>
    /// </remarks>
    FlexPACBayes,

    // ===== Cross-Domain Few-Shot =====

    /// <summary>
    /// Meta-FDMixup - Feature Distribution Mixup for cross-domain few-shot learning (2021).
    /// Mixes feature distributions across domains for better cross-domain transfer.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Mix feature statistics (mean, variance) across domains during training.
    /// This creates interpolated domains that bridge the gap between source and target distributions.
    /// <para><b>Use When:</b> Source and target domains differ significantly in feature distributions.</para>
    /// </remarks>
    MetaFDMixup,

    /// <summary>
    /// FreqPrior - Frequency-based Prior for cross-domain few-shot learning (2024).
    /// Decomposes features into frequency components and applies domain-invariant priors.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Low-frequency components tend to be domain-invariant while high-frequency
    /// components are domain-specific. Learn to weight frequency bands appropriately.
    /// <para><b>Use When:</b> Cross-domain tasks where frequency decomposition reveals transferable patterns.</para>
    /// </remarks>
    FreqPrior,

    /// <summary>
    /// MetaCollaborative - Collaborative meta-learning across multiple source domains (2024).
    /// Multiple domain-specific meta-learners collaborate for target domain adaptation.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Train separate meta-learners on each source domain, then learn to
    /// collaborate by weighting their contributions based on relevance to the target task.
    /// <para><b>Use When:</b> Multiple diverse source domains are available for meta-training.</para>
    /// </remarks>
    MetaCollaborative,

    /// <summary>
    /// SDCL - Self-Distillation Contrastive Learning for cross-domain FSL (2024).
    /// Combines self-distillation with contrastive learning for domain-robust features.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Use self-distillation to transfer knowledge between augmented views and
    /// contrastive learning to ensure features are discriminative across domains.
    /// <para><b>Use When:</b> You need domain-robust representations for diverse target domains.</para>
    /// </remarks>
    SDCL,

    /// <summary>
    /// FreqPrompt - Frequency-aware Prompt tuning for cross-domain FSL (2024).
    /// Uses frequency-domain prompts to bridge domain gaps efficiently.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Instead of adapting the full model, learn small frequency-domain prompts
    /// that adjust the model's behavior for different target domains.
    /// <para><b>Use When:</b> You want parameter-efficient cross-domain adaptation.</para>
    /// </remarks>
    FreqPrompt,

    /// <summary>
    /// Open-MAML++ - Enhanced MAML for open-set cross-domain recognition (2024).
    /// Extends OpenMAML with cross-domain robustness and improved open-set detection.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Combine domain-adaptive features with calibrated confidence scores
    /// for robust open-set recognition across different domains.
    /// <para><b>Use When:</b> You face both domain shift and open-set challenges simultaneously.</para>
    /// </remarks>
    OpenMAMLPlus,

    // ===== Meta-Reinforcement Learning =====

    /// <summary>
    /// PEARL - Probabilistic Embeddings for Actor-critic RL (Rakelly et al., ICML 2019).
    /// Off-policy meta-RL with probabilistic context encoding.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Encode task information into a probabilistic latent context from experience.
    /// The policy conditions on this context for task-specific behavior without gradient updates.
    /// <para><b>Use When:</b> You need sample-efficient meta-RL with probabilistic task inference.</para>
    /// </remarks>
    PEARL,

    /// <summary>
    /// DREAM - Decoupled Reward-Environment Adaptation Meta-learning (2022).
    /// Separately adapts to reward and transition dynamics changes.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Decouple task variation into reward function changes and dynamics changes.
    /// Separate encoders for each provide more structured task representations.
    /// <para><b>Use When:</b> Meta-RL tasks vary in both reward structure and environment dynamics.</para>
    /// </remarks>
    DREAM,

    /// <summary>
    /// DiscoRL - Discovering Meta-RL Objectives (2024).
    /// Automatically discovers effective meta-RL training objectives.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Instead of hand-designing the meta-RL objective, use a learned objective
    /// that is itself meta-optimized to produce good task adaptation.
    /// <para><b>Use When:</b> Standard meta-RL objectives underperform and you want automated objective design.</para>
    /// </remarks>
    DiscoRL,

    /// <summary>
    /// In-Context RL - Reinforcement learning through in-context learning (2023).
    /// Uses transformer-based in-context learning for RL adaptation.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Frame RL adaptation as in-context learning: the agent receives prior
    /// trajectories as context and learns to adapt its policy purely through attention.
    /// <para><b>Use When:</b> You want gradient-free RL adaptation using powerful sequence models.</para>
    /// </remarks>
    InContextRL,

    /// <summary>
    /// HyperNet Meta-RL - Hypernetwork-based Meta-Reinforcement Learning (2024).
    /// Uses hypernetworks to generate task-specific RL policies.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> A hypernetwork generates policy parameters conditioned on the task context,
    /// enabling instant task-specific policy generation without gradient-based adaptation.
    /// <para><b>Use When:</b> You need zero-shot policy generation for new RL tasks.</para>
    /// </remarks>
    HyperNetMetaRL,

    /// <summary>
    /// Context Meta-RL - Context-based Meta-Reinforcement Learning (2024).
    /// Uses explicit context vectors for meta-RL task representation.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Learn a structured context space where similar tasks have similar contexts.
    /// The context vector conditions the policy for task-specific behavior.
    /// <para><b>Use When:</b> Tasks have interpretable structure that can be captured in a context vector.</para>
    /// </remarks>
    ContextMetaRL,

    // ===== Continual / Online Meta-Learning =====

    /// <summary>
    /// ACL - Adaptive Continual Learning with meta-learning (2024).
    /// Combines continual learning with meta-learning for non-stationary task distributions.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Meta-learn to adapt continually without forgetting. The meta-learner
    /// discovers update rules that balance plasticity (learning new) and stability (remembering old).
    /// <para><b>Use When:</b> Task distribution changes over time and you need to avoid catastrophic forgetting.</para>
    /// </remarks>
    ACL,

    /// <summary>
    /// iTAML - Incremental Task-Agnostic Meta-Learning (Rajasegaran et al., 2020).
    /// Handles incremental class learning through meta-learning without task boundaries.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Meta-learn to incorporate new classes without knowing task boundaries.
    /// A task-agnostic approach that uses meta-optimization for balanced old/new performance.
    /// <para><b>Use When:</b> New classes arrive incrementally without clear task boundaries.</para>
    /// </remarks>
    iTAML,

    /// <summary>
    /// MetaContinualAL - Meta-learning for Continual Active Learning (2024).
    /// Combines active learning with continual meta-learning for efficient data selection.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Meta-learn an acquisition function that selects the most informative
    /// examples for continual learning, maximizing learning efficiency over time.
    /// <para><b>Use When:</b> You have a labeling budget and need to select examples wisely over time.</para>
    /// </remarks>
    MetaContinualAL,

    /// <summary>
    /// MePo - Meta-learning for Policy optimization in continual RL (2024).
    /// Meta-learns policy optimization strategies for continual reinforcement learning.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Learn update rules that enable a policy to adapt to new tasks while
    /// maintaining performance on previously learned tasks, in an RL setting.
    /// <para><b>Use When:</b> You need continual RL where new tasks arrive sequentially.</para>
    /// </remarks>
    MePo,

    /// <summary>
    /// OML - Online Meta-Learning (Javed &amp; White, 2019).
    /// Performs meta-learning in an online streaming setting without episodic boundaries.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Learn a representation online that enables fast learning on future data.
    /// No episodic training needed — continuously meta-learns from a data stream.
    /// <para><b>Use When:</b> Data arrives in a stream without clear task/episode boundaries.</para>
    /// </remarks>
    OML,

    /// <summary>
    /// MOCA - Meta-learning Online Continual Adaptation (2024).
    /// Combines online learning with meta-learned adaptation for streaming scenarios.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Meta-learn fast online adaptation rules that work for streaming data.
    /// The meta-learner discovers update strategies that generalize across changing distributions.
    /// <para><b>Use When:</b> You need robust online adaptation to non-stationary streaming data.</para>
    /// </remarks>
    MOCA,

    // ===== Task Augmentation / Sampling =====

    /// <summary>
    /// MetaTask - Meta-learning task generation for improved generalization (2024).
    /// Learns to generate synthetic training tasks that improve meta-learner performance.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Instead of relying on manually defined task distributions, learn a task
    /// generator that creates maximally informative tasks for training the meta-learner.
    /// <para><b>Use When:</b> Your task distribution is limited and you want to augment it effectively.</para>
    /// </remarks>
    MetaTask,

    /// <summary>
    /// ATAML - Adaptive Task Augmentation for Meta-Learning (2024).
    /// Adaptively augments tasks during meta-training for improved robustness.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Apply task-level augmentation (not just data augmentation) such as
    /// varying N-way, K-shot, adding noise classes, or mixing task domains.
    /// <para><b>Use When:</b> Meta-training tasks are too easy or lack diversity.</para>
    /// </remarks>
    ATAML,

    /// <summary>
    /// MPTS - Meta-learning with Prioritized Task Sampling (2024).
    /// Prioritizes informative tasks during meta-training for faster convergence.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Not all tasks are equally informative. Prioritize tasks where the
    /// meta-learner has high loss or high uncertainty for more efficient training.
    /// <para><b>Use When:</b> You have a large task pool and want to train more efficiently.</para>
    /// </remarks>
    MPTS,

    /// <summary>
    /// Dynamic Task Sampling - Adaptive task distribution during meta-training (2024).
    /// Dynamically adjusts the task sampling distribution based on meta-learner performance.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Continuously adjust which tasks are sampled based on current meta-learner
    /// capabilities, focusing on tasks at the frontier of the learner's ability.
    /// <para><b>Use When:</b> You want curriculum-like training without manually defining stages.</para>
    /// </remarks>
    DynamicTaskSampling,

    /// <summary>
    /// Unsupervised Meta-Learning - Meta-learning without task labels (Hsu et al., 2019).
    /// Generates pseudo-tasks from unlabeled data for meta-training.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Use clustering or generative models to create pseudo-tasks from unlabeled
    /// data, enabling meta-learning without curated task distributions.
    /// <para><b>Use When:</b> Labeled tasks are scarce but unlabeled data is abundant.</para>
    /// </remarks>
    UnsupervisedMetaLearn,

    // ===== Transductive Few-Shot =====

    /// <summary>
    /// GCDPLNet - Graph-based Class Distribution Propagation and Label Network (2024).
    /// Uses graph neural networks with class distribution propagation for transductive FSL.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Build a graph over all examples and propagate both features and class
    /// distribution information through message passing for joint classification.
    /// <para><b>Use When:</b> You want graph-based transductive inference with distribution modeling.</para>
    /// </remarks>
    GCDPLNet,

    /// <summary>
    /// BayTransProto - Bayesian Transductive Prototypical Networks (2024).
    /// Combines Bayesian prototype estimation with transductive refinement.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Start with Bayesian prototypes from the support set, then refine them
    /// transductively using query examples via iterative Bayesian updates.
    /// <para><b>Use When:</b> You want principled uncertainty with transductive improvements.</para>
    /// </remarks>
    BayTransProto,

    /// <summary>
    /// JMP - Joint Meta-learning and Propagation for transductive FSL (2024).
    /// Jointly optimizes meta-learning and label propagation objectives.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Instead of treating meta-learning and label propagation as separate stages,
    /// jointly optimize both objectives end-to-end for better transductive performance.
    /// <para><b>Use When:</b> You want an end-to-end transductive meta-learning approach.</para>
    /// </remarks>
    JMP,

    /// <summary>
    /// ETPN - Enhanced Transductive Prototypical Networks (2024).
    /// Iteratively refines prototypes using query set information.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Start with standard prototypes, then iteratively update them using
    /// soft assignments from query examples. Prototypes converge to better estimates.
    /// <para><b>Use When:</b> You want improved prototypes through transductive refinement.</para>
    /// </remarks>
    ETPN,

    /// <summary>
    /// ActiveTransFSL - Active Transductive Few-Shot Learning (2024).
    /// Combines active learning with transductive few-shot classification.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> In the transductive setting, actively select which query examples to
    /// label first based on expected information gain, then propagate to remaining queries.
    /// <para><b>Use When:</b> You can label a few query examples and want optimal selection.</para>
    /// </remarks>
    ActiveTransFSL,

    // ===== Hypernetwork Methods =====

    /// <summary>
    /// Task-Conditioned HyperNet - Hypernetwork conditioned on task representations (2024).
    /// Generates full model parameters from compact task descriptions.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Encode the task into a compact representation, then use a hypernetwork to
    /// generate all model parameters in a single forward pass. Zero inner-loop gradient steps.
    /// <para><b>Use When:</b> You need instant task-specific model generation without adaptation.</para>
    /// </remarks>
    TaskCondHyperNet,

    /// <summary>
    /// HyperCLIP - Hypernetwork with CLIP-based task encoding (2024).
    /// Uses CLIP-style encodings to condition hypernetworks on multimodal task descriptions.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Encode task descriptions using CLIP embeddings and generate task-specific
    /// model parameters. Enables natural language task specification.
    /// <para><b>Use When:</b> Tasks can be described in natural language or with example images.</para>
    /// </remarks>
    HyperCLIP,

    /// <summary>
    /// Recurrent HyperNet - Hypernetwork with recurrent task encoding (2024).
    /// Uses recurrent processing of support examples to generate model parameters.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Process support examples sequentially through a recurrent network, then
    /// generate model parameters from the final hidden state. Supports variable support set sizes.
    /// <para><b>Use When:</b> Support set size varies and you need flexible hypernetwork conditioning.</para>
    /// </remarks>
    RecurrentHyperNet,

    /// <summary>
    /// HyperNeRF Meta - Hypernetwork for Meta-learning Neural Radiance Fields (2024).
    /// Uses hypernetworks to generate NeRF parameters for few-shot 3D reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>Key Idea:</b> Generate NeRF parameters from few input views using a hypernetwork.
    /// Meta-learning enables rapid 3D scene reconstruction from minimal observations.
    /// <para><b>Use When:</b> You need few-shot 3D reconstruction with neural radiance fields.</para>
    /// </remarks>
    HyperNeRFMeta
}
