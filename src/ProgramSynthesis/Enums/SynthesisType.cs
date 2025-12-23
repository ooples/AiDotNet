namespace AiDotNet.ProgramSynthesis.Enums;

/// <summary>
/// Defines the different types of program synthesis approaches available.
/// </summary>
/// <remarks>
/// <para>
/// This enumeration categorizes the various methodologies used for automated program synthesis.
/// Each approach has different strengths and is suited for different types of programming tasks.
/// </para>
/// <para><b>For Beginners:</b> Think of these as different strategies for automatically creating programs.
///
/// Just like there are different approaches to solving a puzzle (looking at the picture, starting
/// from corners, sorting by color), there are different ways to automatically generate code:
/// - Neural: Uses neural networks that learn from examples
/// - Symbolic: Uses logical rules and grammar
/// - Hybrid: Combines neural and symbolic approaches
/// - GeneticProgramming: Evolves programs through selection and mutation
/// </para>
/// </remarks>
public enum SynthesisType
{
    /// <summary>
    /// Neural network-based program synthesis using deep learning models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Neural synthesis uses trained neural networks to generate programs by learning patterns
    /// from a large corpus of existing code. This approach is data-driven and can produce
    /// creative solutions but may lack guarantees of correctness.
    /// </para>
    /// <para><b>For Beginners:</b> Neural synthesis is like learning to code by studying lots of examples.
    ///
    /// The AI looks at thousands of code examples and learns patterns, then generates new code
    /// based on what it has learned. Similar to how you might learn to write by reading many books.
    /// </para>
    /// </remarks>
    Neural,

    /// <summary>
    /// Symbolic program synthesis using formal logic, grammars, and search algorithms.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Symbolic synthesis uses formal methods, programming language grammars, and logical
    /// constraints to systematically explore the space of possible programs. This approach
    /// provides stronger correctness guarantees but may be limited in creativity.
    /// </para>
    /// <para><b>For Beginners:</b> Symbolic synthesis is like following a recipe or instruction manual.
    ///
    /// It uses strict rules about what code should look like and systematically tries different
    /// combinations until it finds one that works. Like solving a math problem step by step.
    /// </para>
    /// </remarks>
    Symbolic,

    /// <summary>
    /// Hybrid approach combining both neural and symbolic techniques.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Hybrid synthesis combines the strengths of both neural and symbolic approaches,
    /// using neural networks for creative exploration and symbolic methods for verification
    /// and constraint satisfaction.
    /// </para>
    /// <para><b>For Beginners:</b> Hybrid synthesis combines the best of both worlds.
    ///
    /// It uses neural networks to come up with creative ideas quickly, then uses symbolic
    /// methods to check and refine them. Like brainstorming ideas (neural) then fact-checking them (symbolic).
    /// </para>
    /// </remarks>
    Hybrid,

    /// <summary>
    /// Genetic programming approach using evolutionary algorithms.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Genetic programming evolves programs through processes inspired by biological evolution,
    /// including selection, crossover (combining parts of programs), and mutation (random changes).
    /// Programs that perform better are more likely to survive and reproduce.
    /// </para>
    /// <para><b>For Beginners:</b> Genetic programming is like evolution in nature.
    ///
    /// It creates a population of random programs, tests them, keeps the best ones, and
    /// creates new programs by mixing and mutating the good ones. Over many generations,
    /// the programs get better and better, like species evolving over time.
    /// </para>
    /// </remarks>
    GeneticProgramming,

    /// <summary>
    /// Inductive program synthesis that learns from input-output examples.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Inductive synthesis generates programs by generalizing from a set of input-output
    /// examples. This is particularly useful when users can provide examples of desired
    /// behavior but may not know how to express the logic formally.
    /// </para>
    /// <para><b>For Beginners:</b> Inductive synthesis learns from examples of what you want.
    ///
    /// Instead of telling the computer exactly what to do, you show it examples:
    /// "When input is [1,2,3], output should be 6"
    /// "When input is [4,5], output should be 9"
    /// The system figures out you want it to sum the numbers.
    /// </para>
    /// </remarks>
    Inductive,

    /// <summary>
    /// Deductive program synthesis from formal specifications.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Deductive synthesis constructs programs from formal specifications that precisely
    /// describe the desired behavior. This approach provides strong correctness guarantees
    /// but requires users to provide detailed formal specifications.
    /// </para>
    /// <para><b>For Beginners:</b> Deductive synthesis works from precise descriptions.
    ///
    /// You provide a detailed specification of exactly what the program should do using
    /// mathematical logic or formal notation, and the system constructs a program that
    /// provably meets that specification. Like building from detailed blueprints.
    /// </para>
    /// </remarks>
    Deductive
}
