namespace AiDotNet.Enums;

/// <summary>
/// Defines different methods for pooling (downsampling) data in neural networks, particularly in convolutional neural networks.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Pooling is like summarizing information to make it more manageable. Imagine you have a large, 
/// detailed photograph and you want to create a smaller version that still captures the important features. 
/// Pooling does this for AI models by taking groups of numbers (like pixels) and combining them into single values.
/// 
/// This serves two important purposes:
/// 1. It reduces the amount of data the model needs to process, making it faster and more efficient
/// 2. It helps the model focus on important features regardless of their exact position (called "positional invariance")
/// 
/// For example, if a model is trying to recognize a cat in a photo, pooling helps it identify cat features 
/// whether the cat is in the center, corner, or any other position in the image.
/// </para>
/// </remarks>
public enum PoolingType
{
    /// <summary>
    /// Takes the maximum value from each group of values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Max Pooling works by looking at small groups of numbers and keeping only the largest 
    /// value from each group. 
    /// 
    /// For example, if we have this 4×4 grid of numbers:
    /// 
    /// 3  7  5  2
    /// 1  4  6  9
    /// 2  8  3  5
    /// 6  1  4  7
    /// 
    /// And we use Max Pooling with 2×2 groups, we'd get:
    /// 
    /// 7  9
    /// 8  7
    /// 
    /// Max Pooling is especially good at preserving important features like edges and textures. It's like 
    /// looking at a landscape and remembering only the tallest mountains - you might miss some details, 
    /// but you'll definitely remember the most prominent features.
    /// </para>
    /// </remarks>
    Max,

    /// <summary>
    /// Takes the average value from each group of values.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Average Pooling works by looking at small groups of numbers and taking the average 
    /// (mean) of all values in each group.
    /// 
    /// Using the same 4×4 grid example:
    /// 
    /// 3  7  5  2
    /// 1  4  6  9
    /// 2  8  3  5
    /// 6  1  4  7
    /// 
    /// With Average Pooling using 2×2 groups, we'd get:
    /// 
    /// (3+7+1+4)/4 = 3.75    (5+2+6+9)/4 = 5.5
    /// (2+8+6+1)/4 = 4.25    (3+5+4+7)/4 = 4.75
    /// 
    /// Which rounds to:
    /// 
    /// 4  6
    /// 4  5
    /// 
    /// Average Pooling is good at preserving background information and overall context. It's like describing 
    /// a neighborhood by its average house size - you get a general sense of the area rather than just focusing 
    /// on the biggest houses.
    /// </para>
    /// </remarks>
    Average
}
