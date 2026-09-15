namespace AiDotNet.Evolution.Programs;

/// <summary>A program proposal backend that receipts all of its work and exposes cumulative model usage.</summary>
/// <remarks>Include mutation/refinement, model requests, parsing, compilation, repairs and audit writes performed
/// inside the backend in its returned resources. Do not charge those same operations separately. Evaluator and
/// one-time setup costs remain separate ledger operations. Failed or unknown work must not become a zero charge.</remarks>
public interface ICostedProgramProposalSource : ICostedEvolutionProposalSource<ProgramGenome>
{
    /// <summary>Returns cumulative provider-reported usage, including failed requests and repairs.</summary>
    ProgramEvolutionLlmUsage GetUsage();
}
