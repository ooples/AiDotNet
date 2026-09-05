namespace AiDotNet.Enums;

/// <summary>Says why an example program was chosen for a prompt, which decides how it is labelled.</summary>
/// <remarks>
/// <para>
/// A prompt shows other programs for two different reasons: to demonstrate what scores well, and to demonstrate
/// that other shapes of solution exist. Telling the model which is which changes what it does with them, so the
/// kind travels with each example rather than being inferred from its score.
/// </para>
/// <para>
/// The reference OpenEvolve implementation infers the label from score bands whenever no metadata flag is
/// present, so a genuinely diverse low-scoring example is presented to the model as "Exploratory" — a judgement
/// about its quality rather than a statement about why it is in the prompt.
/// </para>
/// <para><b>For Beginners:</b> When the library shows the AI other programs alongside yours, it says what each
/// one is there for: a top scorer to imitate, a deliberately different solution to draw ideas from, or one that
/// travelled in from another part of the search. That label is a hint about how to use the example, not a
/// verdict on how good it is.</para>
/// </remarks>
public enum ProgramPromptExampleKind
{
    /// <summary>A high-scoring program shown as a target to match or beat.</summary>
    TopProgram = 0,

    /// <summary>A program chosen for occupying a different region of the feature space.</summary>
    Diverse = 1,

    /// <summary>A program supplied by the engine's selection policy as an inspiration.</summary>
    Inspiration = 2,

    /// <summary>A program that migrated in from another island.</summary>
    Migrant = 3,

    /// <summary>A program drawn at random from the archive.</summary>
    Random = 4
}
