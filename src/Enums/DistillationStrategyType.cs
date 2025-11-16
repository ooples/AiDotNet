namespace AiDotNet.Enums;

/// <summary>
/// Specifies the type of knowledge distillation strategy to use for transferring knowledge
/// from teacher to student models.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Different distillation strategies focus on different aspects of
/// the teacher's knowledge. Some match final outputs, others match intermediate features or
/// relationships between samples.</para>
///
/// <para><b>Choosing a Strategy:</b>
/// - Use **ResponseBased** for most cases (standard Hinton distillation)
/// - Use **FeatureBased** when student architecture differs significantly from teacher
/// - Use **AttentionBased** for transformer models (BERT, GPT)
/// - Use **RelationBased** to preserve relationships between samples
/// - Use **Contrastive** for self-supervised learning scenarios</para>
/// </remarks>
public enum DistillationStrategyType
{
    /// <summary>
    /// Response-based distillation (Hinton et al., 2015).
    /// Matches the teacher's final output predictions using temperature-scaled softmax.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Standard classification tasks, general-purpose distillation.</para>
    /// <para><b>Key Parameters:</b> Temperature (2-10), Alpha (0.3-0.5).</para>
    /// <para><b>Pros:</b> Simple, effective, widely used.</para>
    /// <para><b>Cons:</b> Doesn't capture intermediate representations.</para>
    /// </remarks>
    ResponseBased = 0,

    /// <summary>
    /// Feature-based distillation / FitNets (Romero et al., 2014).
    /// Matches intermediate layer representations between teacher and student.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Different architectures (e.g., CNN → MobileNet), transfer across domains.</para>
    /// <para><b>Key Parameters:</b> Layer pairs to match, feature weight.</para>
    /// <para><b>Pros:</b> Transfers deeper knowledge, works across architectures.</para>
    /// <para><b>Cons:</b> Requires layer mapping, may need projection layers.</para>
    /// </remarks>
    FeatureBased = 1,

    /// <summary>
    /// Attention-based distillation (Zagoruyko & Komodakis, 2017).
    /// Transfers attention maps from teacher to student, showing what the model focuses on.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Transformer models (BERT, GPT, Vision Transformers).</para>
    /// <para><b>Key Parameters:</b> Attention layers to match, attention weight.</para>
    /// <para><b>Pros:</b> Captures where model attends, great for transformers.</para>
    /// <para><b>Cons:</b> Only applicable to attention-based models.</para>
    /// </remarks>
    AttentionBased = 2,

    /// <summary>
    /// Relational Knowledge Distillation / RKD (Park et al., 2019).
    /// Preserves relationships (distances and angles) between sample representations.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Metric learning, few-shot learning, embedding models.</para>
    /// <para><b>Key Parameters:</b> Distance weight, angle weight.</para>
    /// <para><b>Pros:</b> Preserves structural relationships, robust to architecture changes.</para>
    /// <para><b>Cons:</b> More computationally expensive (pairwise comparisons).</para>
    /// </remarks>
    RelationBased = 3,

    /// <summary>
    /// Contrastive Representation Distillation / CRD (Tian et al., 2020).
    /// Uses contrastive learning to match teacher and student representations.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Self-supervised learning, representation learning.</para>
    /// <para><b>Key Parameters:</b> Temperature, negative samples, contrast weight.</para>
    /// <para><b>Pros:</b> Strong theoretical foundation, works without labels.</para>
    /// <para><b>Cons:</b> Requires careful tuning, needs negative sampling.</para>
    /// </remarks>
    ContrastiveBased = 4,

    /// <summary>
    /// Similarity-Preserving Distillation / SP (Tung & Mori, 2019).
    /// Preserves pairwise similarity structure in the feature space.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Image retrieval, similarity-based tasks.</para>
    /// <para><b>Key Parameters:</b> Similarity metric, layer selection.</para>
    /// <para><b>Pros:</b> Preserves similarity structure, good for retrieval.</para>
    /// <para><b>Cons:</b> Computationally expensive for large batches.</para>
    /// </remarks>
    SimilarityPreserving = 5,

    /// <summary>
    /// Flow of Solution Procedure / FSP (Yim et al., 2017).
    /// Transfers the flow of information between layers.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Deep networks, capturing layer-to-layer flow.</para>
    /// <para><b>Key Parameters:</b> Layer pairs for flow matrices.</para>
    /// <para><b>Pros:</b> Captures information flow, good for deep networks.</para>
    /// <para><b>Cons:</b> Requires multiple layer pairs, complex to configure.</para>
    /// </remarks>
    FlowBased = 6,

    /// <summary>
    /// Probabilistic Knowledge Transfer / PKT (Passalis & Tefas, 2018).
    /// Transfers probability distributions of activations.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Capturing activation statistics, distribution matching.</para>
    /// <para><b>Key Parameters:</b> Distribution type, activation layers.</para>
    /// <para><b>Pros:</b> Statistically principled, captures distributions.</para>
    /// <para><b>Cons:</b> Computationally intensive, requires sufficient samples.</para>
    /// </remarks>
    ProbabilisticTransfer = 7,

    /// <summary>
    /// Variational Information Distillation / VID (Ahn et al., 2019).
    /// Uses variational bounds to maximize mutual information between teacher and student.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Information-theoretic distillation, maximizing information transfer.</para>
    /// <para><b>Key Parameters:</b> Variational parameters, MI estimator.</para>
    /// <para><b>Pros:</b> Theoretical guarantees, maximizes information.</para>
    /// <para><b>Cons:</b> Complex implementation, harder to tune.</para>
    /// </remarks>
    VariationalInformation = 8,

    /// <summary>
    /// Factor Transfer (Kim et al., 2018).
    /// Transfers factors (paraphrased representations) from teacher to student.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Cross-architecture transfer, efficient distillation.</para>
    /// <para><b>Key Parameters:</b> Paraphraser network, factor layers.</para>
    /// <para><b>Pros:</b> Flexible, works across different architectures.</para>
    /// <para><b>Cons:</b> Requires additional paraphraser network.</para>
    /// </remarks>
    FactorTransfer = 9,

    /// <summary>
    /// Neuron Selectivity Transfer / NST (Huang & Wang, 2017).
    /// Matches the selectivity patterns of neurons between teacher and student.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Preserving neuron semantics, interpretable distillation.</para>
    /// <para><b>Key Parameters:</b> Selectivity layers, matching criterion.</para>
    /// <para><b>Pros:</b> Preserves neuron-level semantics.</para>
    /// <para><b>Cons:</b> Requires careful neuron alignment.</para>
    /// </remarks>
    NeuronSelectivity = 10,

    /// <summary>
    /// Self-distillation (Zhang et al., 2019; Furlanello et al., 2018).
    /// Model learns from its own predictions to improve calibration and generalization.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Improving calibration, no separate teacher needed.</para>
    /// <para><b>Key Parameters:</b> Generations, temperature, EMA decay.</para>
    /// <para><b>Pros:</b> No separate teacher, improves calibration.</para>
    /// <para><b>Cons:</b> Requires multiple training runs.</para>
    /// </remarks>
    SelfDistillation = 11,

    /// <summary>
    /// Combined/Hybrid distillation.
    /// Combines multiple strategies (e.g., Response + Feature + Attention).
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Maximizing knowledge transfer, complex models.</para>
    /// <para><b>Key Parameters:</b> Weights for each strategy.</para>
    /// <para><b>Pros:</b> Transfers knowledge at multiple levels.</para>
    /// <para><b>Cons:</b> More hyperparameters to tune, computationally expensive.</para>
    /// </remarks>
    Hybrid = 12
}
