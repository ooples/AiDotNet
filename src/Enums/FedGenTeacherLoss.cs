namespace AiDotNet.Enums;

/// <summary>
/// Selects the teacher term of the FedGen server generator objective (Zhu et al. 2021).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The server generator is trained so that the clients' classifiers recognise the
/// features it produces. These options choose how the clients' opinions are combined into that training
/// signal.</para>
/// </remarks>
public enum FedGenTeacherLoss
{
    /// <summary>
    /// Cross-entropy of the label-count-weighted ensemble of client predictor logits,
    /// CE(Σ_k α_k^y h_k(z), y) (the paper's generator objective).
    /// </summary>
    EnsembleCrossEntropy,

    /// <summary>
    /// Label-count-weighted sum of the clients' own cross-entropies, Σ_k α_k^y CE(h_k(z), y) (the teacher loss of
    /// the authors' released code, serverpFedGen.train_generator).
    /// </summary>
    WeightedClientCrossEntropy,
}
