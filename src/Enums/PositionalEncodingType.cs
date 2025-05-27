namespace AiDotNet.Enums;

/// <summary>
/// Defines different types of positional encoding methods for transformer models.
/// </summary>
/// <remarks>
/// <para>
/// Positional encoding is crucial for transformer models because these models don't have an inherent
/// understanding of sequence order. Unlike RNNs or LSTMs which process tokens sequentially,
/// transformers process all tokens in parallel, so positional information must be explicitly added.
/// </para>
/// <para><b>For Beginners:</b> Positional encoding is like adding page numbers to a book.
/// 
/// Without positional encoding, a transformer model would see all words in a sentence at once,
/// but wouldn't know which word comes first, second, and so on. This would be like reading a book
/// where all the words are jumbled - you can see all the words, but their order matters for understanding
/// the story.
/// 
/// Different types of positional encoding are like different ways of numbering the pages - 
/// some methods work better for certain types of books or reading scenarios.
/// </para>
/// </remarks>
public enum PositionalEncodingType
{
    /// <summary>
    /// No positional encoding is applied.
    /// </summary>
    /// <remarks>
    /// This is generally not recommended for sequence tasks, as the model will have
    /// no way to understand token order.
    /// </remarks>
    None = 0,
    
    /// <summary>
    /// The original sinusoidal position encoding from the "Attention Is All You Need" paper.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Uses sine and cosine functions of different frequencies to encode positions.
    /// This method has the advantage of allowing the model to extrapolate to sequence
    /// lengths longer than those seen during training.
    /// </para>
    /// <para><b>For Beginners:</b> This is like using a special mathematical pattern to mark positions.
    /// The pattern repeats in a way that each position gets a unique "signature" based on sine and cosine
    /// waves. It's the classic method used in the original transformer paper.
    /// </para>
    /// </remarks>
    Sinusoidal = 1,
    
    /// <summary>
    /// Learnable absolute position embeddings, as used in BERT and many other models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each position gets its own learned embedding vector. This method is more flexible
    /// but cannot generalize to positions beyond those seen during training.
    /// </para>
    /// <para><b>For Beginners:</b> This is like giving each position a name tag, and the model
    /// learns what each position means during training. This works well but only for positions
    /// the model has seen before - it can't handle sentences longer than what it was trained on.
    /// </para>
    /// </remarks>
    Learned = 2,
    
    /// <summary>
    /// Relative positional encoding, as used in models like Transformer-XL.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Instead of absolute positions, this encodes the relative distance between tokens.
    /// This allows better handling of longer sequences and can improve performance on
    /// tasks requiring understanding of relationships between distant tokens.
    /// </para>
    /// <para><b>For Beginners:</b> Rather than saying "this is position 5," this method
    /// says "this token is 3 positions away from that token." This helps the model focus
    /// on relationships between words rather than their absolute positions.
    /// </para>
    /// </remarks>
    Relative = 3,
    
    /// <summary>
    /// Rotary position embedding (RoPE) as used in models like GPT-Neo-X and PaLM.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Encodes position information by rotating word vectors in a way that preserves
    /// their relative distances. This method has strong theoretical properties and works
    /// well in practice for large language models.
    /// </para>
    /// <para><b>For Beginners:</b> This method "rotates" word representations based on their
    /// position, creating a unique pattern for each position while preserving the relationships
    /// between words. It's a newer method that combines advantages of both absolute and relative
    /// position encoding.
    /// </para>
    /// </remarks>
    Rotary = 4,
    
    /// <summary>
    /// ALiBi (Attention with Linear Biases) as used in models like Bloom.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Instead of adding position encodings to token embeddings, ALiBi directly adds
    /// a bias to attention scores based on the distance between tokens. It has shown
    /// excellent ability to extrapolate to longer sequences.
    /// </para>
    /// <para><b>For Beginners:</b> This method modifies how words "pay attention" to each other
    /// based on how far apart they are. It adds a predictable penalty to attention scores based
    /// on distance, which helps the model handle longer texts than it was trained on.
    /// </para>
    /// </remarks>
    ALiBi = 5,
    
    /// <summary>
    /// T5-style relative positional bias, used in text-to-text transfer transformer models.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Uses learned relative position representations that are added as biases to the
    /// attention scores rather than being incorporated into the token representations.
    /// </para>
    /// <para><b>For Beginners:</b> This approach, used in Google's T5 model, learns special
    /// values that represent the relationship between positions and adds these values
    /// to the attention scores. It's a hybrid approach that works well for many tasks.
    /// </para>
    /// </remarks>
    T5RelativeBias = 6
}