using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.ProgramSynthesis.Enums;
using AiDotNet.ProgramSynthesis.Models;
using AiDotNet.ProgramSynthesis.Requests;
using AiDotNet.ProgramSynthesis.Results;

namespace AiDotNet.ProgramSynthesis.Interfaces;

/// <summary>
/// Represents a code understanding model capable of processing and analyzing source code.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para>
/// ICodeModel defines the interface for models that can understand, encode, and analyze
/// source code. These models are typically pre-trained on large corpora of code and can
/// perform tasks like code completion, bug detection, and code summarization.
/// </para>
/// <para><b>For Beginners:</b> A code model is like an AI that understands programming.
///
/// Just as language models understand human languages, code models understand programming
/// languages. They can:
/// - Read and comprehend code
/// - Suggest completions while you're writing
/// - Find bugs and issues
/// - Explain what code does
/// - Translate between programming languages
///
/// This interface defines what capabilities a code model should have.
/// </para>
/// </remarks>
public interface ICodeModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    /// <summary>
    /// Gets the target programming language for this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Specifies which programming language this model is designed to work with.
    /// Some models are language-specific, while others can work with multiple languages.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you which programming language the model knows.
    ///
    /// Like a translator who specializes in French or Spanish, code models often specialize
    /// in specific programming languages like Python or Java.
    /// </para>
    /// </remarks>
    ProgramLanguage TargetLanguage { get; }

    /// <summary>
    /// Gets the maximum sequence length (in tokens) that the model can process.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Code models process code as sequences of tokens. This property specifies the
    /// maximum number of tokens the model can handle at once.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the maximum length of code the model can read at once.
    ///
    /// Code is broken into pieces called "tokens" (like words in a sentence). This number
    /// tells you the maximum number of tokens the model can process, which roughly
    /// corresponds to how long a code file can be.
    /// </para>
    /// </remarks>
    int MaxSequenceLength { get; }

    /// <summary>
    /// Gets the vocabulary size of the model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The vocabulary consists of all the tokens (keywords, operators, identifiers, etc.)
    /// that the model knows and can work with.
    /// </para>
    /// <para><b>For Beginners:</b> This is like the model's dictionary size.
    ///
    /// It tells you how many different code tokens the model knows. A larger vocabulary
    /// means the model can handle more diverse code patterns and identifiers.
    /// </para>
    /// </remarks>
    int VocabularySize { get; }

    /// <summary>
    /// Encodes source code into a vector representation.
    /// </summary>
    /// <param name="code">The source code to encode.</param>
    /// <returns>A tensor representing the encoded code.</returns>
    /// <remarks>
    /// <para>
    /// Encoding transforms source code (text) into a numerical representation that
    /// the model can process. This representation captures semantic information about the code.
    /// </para>
    /// <para><b>For Beginners:</b> Encoding converts code text into numbers the AI can understand.
    ///
    /// Computers can't directly work with text, so we convert code into numerical form.
    /// This encoding captures the meaning of the code, not just the characters.
    /// Like translating emotions into emoji - different form, same meaning.
    /// </para>
    /// </remarks>
    Tensor<T> EncodeCode(string code);

    /// <summary>
    /// Decodes a vector representation back into source code.
    /// </summary>
    /// <param name="encoding">The encoded representation to decode.</param>
    /// <returns>The decoded source code as a string.</returns>
    /// <remarks>
    /// <para>
    /// Decoding transforms the model's internal numerical representation back into
    /// human-readable source code.
    /// </para>
    /// <para><b>For Beginners:</b> Decoding converts the AI's numerical format back to readable code.
    ///
    /// After the AI processes code in numerical form, we need to convert it back to
    /// text that humans can read and computers can execute. This is the reverse of encoding.
    /// </para>
    /// </remarks>
    string DecodeCode(Tensor<T> encoding);

    /// <summary>
    /// Performs a code-related task on the input code.
    /// </summary>
    /// <param name="code">The source code to process.</param>
    /// <param name="task">The type of task to perform.</param>
    /// <returns>The result of the task as a string.</returns>
    /// <remarks>
    /// <para>
    /// This method allows the model to perform various code-related tasks such as
    /// completion, summarization, bug detection, etc. based on the specified task type.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you tell the model what to do with the code.
    ///
    /// You provide code and specify what you want done with it:
    /// - Complete it
    /// - Summarize it
    /// - Find bugs
    /// - Generate documentation
    ///
    /// The model then performs that specific task and returns the result.
    /// </para>
    /// </remarks>
    [Obsolete("Use PerformTask(CodeTaskRequestBase) for structured outputs.")]
    string PerformTask(string code, CodeTask task);

    /// <summary>
    /// Performs a code-related task and returns a structured result type.
    /// </summary>
    /// <param name="request">The task request.</param>
    /// <returns>A structured task result.</returns>
    CodeTaskResultBase PerformTask(CodeTaskRequestBase request);

    /// <summary>
    /// Gets embeddings for code tokens.
    /// </summary>
    /// <param name="code">The source code to get embeddings for.</param>
    /// <returns>A tensor containing token embeddings.</returns>
    /// <remarks>
    /// <para>
    /// Embeddings are dense vector representations of code tokens that capture semantic
    /// similarities. Similar code constructs have similar embeddings.
    /// </para>
    /// <para><b>For Beginners:</b> Embeddings represent each piece of code as a point in space.
    ///
    /// Code with similar meaning is placed close together in this space. For example,
    /// "for loop" and "while loop" would be near each other because they're both loops,
    /// but far from "function definition" because that's a different concept.
    /// </para>
    /// </remarks>
    Tensor<T> GetEmbeddings(string code);
}
