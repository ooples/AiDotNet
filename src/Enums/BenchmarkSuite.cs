namespace AiDotNet.Enums;

/// <summary>
/// Defines the supported benchmark suites available through the AiDotNet facade.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A benchmark suite is like a standardized test you can run to measure how well your
/// model performs on a specific family of problems. You select a suite using this enum, and AiDotNet runs
/// the benchmark and returns a structured report.
/// </para>
/// </remarks>
public enum BenchmarkSuite
{
    /// <summary>
    /// LEAF - federated benchmark suite (JSON-based train/test splits).
    /// </summary>
    LEAF,

    /// <summary>
    /// FEMNIST - LEAF federated handwritten character classification (per-writer partitioning).
    /// </summary>
    FEMNIST,

    /// <summary>
    /// Sent140 - LEAF federated sentiment classification benchmark based on tweets.
    /// </summary>
    Sent140,

    /// <summary>
    /// Shakespeare - LEAF federated next-character prediction benchmark.
    /// </summary>
    Shakespeare,

    /// <summary>
    /// Reddit - federated next-token prediction benchmark (LEAF Reddit dataset).
    /// </summary>
    Reddit,

    /// <summary>
    /// StackOverflow - federated next-token prediction benchmark (StackOverflow corpus).
    /// </summary>
    StackOverflow,

    /// <summary>
    /// CIFAR-10 - federated image classification (synthetic partitioning of standard CIFAR-10).
    /// </summary>
    CIFAR10,

    /// <summary>
    /// CIFAR-100 - federated image classification (synthetic partitioning of standard CIFAR-100).
    /// </summary>
    CIFAR100,

    /// <summary>
    /// Generic tabular suite with synthetic non-IID client partitions.
    /// </summary>
    TabularNonIID,

    /// <summary>
    /// Grade School Math 8K (GSM8K) - multi-step math word problems.
    /// </summary>
    GSM8K,

    /// <summary>
    /// MATH - competition-style mathematics problems.
    /// </summary>
    MATH,

    /// <summary>
    /// MMLU - broad multi-subject multiple-choice benchmark.
    /// </summary>
    MMLU,

    /// <summary>
    /// TruthfulQA - evaluates truthfulness and resistance to hallucination.
    /// </summary>
    TruthfulQA,

    /// <summary>
    /// ARC-AGI - abstract reasoning puzzles.
    /// </summary>
    ARCAGI,

    /// <summary>
    /// DROP - reading comprehension with discrete reasoning over paragraphs.
    /// </summary>
    DROP,

    /// <summary>
    /// BoolQ - yes/no question answering.
    /// </summary>
    BoolQ,

    /// <summary>
    /// PIQA - physical commonsense reasoning.
    /// </summary>
    PIQA,

    /// <summary>
    /// CommonsenseQA - commonsense multiple-choice QA.
    /// </summary>
    CommonsenseQA,

    /// <summary>
    /// WinoGrande - pronoun resolution / commonsense reasoning.
    /// </summary>
    WinoGrande,

    /// <summary>
    /// HellaSwag - commonsense inference in narrative completion.
    /// </summary>
    HellaSwag,

    /// <summary>
    /// HumanEval - code generation / program synthesis evaluation.
    /// </summary>
    HumanEval,

    /// <summary>
    /// MBPP - mostly basic programming problems.
    /// </summary>
    MBPP,

    /// <summary>
    /// LogiQA - logical reasoning benchmark.
    /// </summary>
    LogiQA
}
