namespace AiDotNet.Enums;

/// <summary>
/// Defines the different types of biorthogonal wavelets that can be used for signal processing and analysis.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Wavelets are mathematical functions that cut up data into different frequency components.
/// 
/// Think of wavelets like special lenses that let you look at your data in different ways:
/// - They can zoom in to see fine details (high frequencies)
/// - They can zoom out to see the big picture (low frequencies)
/// 
/// Biorthogonal wavelets (Bior) are a special family of wavelets that have useful properties for 
/// signal processing. The "Reverse" prefix indicates these are the reconstruction filters.
/// 
/// The numbers in each wavelet name (like "11" in ReverseBior11) represent:
/// - First number: Decomposition filter length
/// - Second number: Reconstruction filter length
/// 
/// Different wavelets are better suited for different types of data and applications.
/// </remarks>
public enum WaveletType
{
    /// <summary>
    /// Reverse Biorthogonal 1.1 wavelet - the simplest biorthogonal wavelet with one vanishing moment in both decomposition and reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the simplest biorthogonal wavelet.
    /// 
    /// The "1.1" means it has one vanishing moment in both decomposition and reconstruction.
    /// A vanishing moment determines how well the wavelet can represent polynomial behavior.
    /// 
    /// This wavelet is good for:
    /// - Simple signals with minimal complexity
    /// - Cases where computational efficiency is important
    /// - Situations where you need a symmetric wavelet with minimal filter length
    /// </remarks>
    ReverseBior11,

    /// <summary>
    /// Reverse Biorthogonal 1.3 wavelet - has one vanishing moment for decomposition and three for reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This wavelet has one vanishing moment for decomposition and three for reconstruction.
    /// 
    /// The increased reconstruction moments (3) means it can better represent complex patterns
    /// during the reconstruction phase while keeping the decomposition simple.
    /// 
    /// This wavelet provides a good balance between:
    /// - Computational efficiency (from the simple decomposition)
    /// - Reconstruction quality (from the more complex reconstruction)
    /// </remarks>
    ReverseBior13,

    /// <summary>
    /// Reverse Biorthogonal 2.2 wavelet - has two vanishing moments in both decomposition and reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This wavelet has two vanishing moments in both decomposition and reconstruction.
    /// 
    /// With two vanishing moments, this wavelet can accurately represent linear trends in your data.
    /// 
    /// This wavelet is useful for:
    /// - Signals with linear components
    /// - Applications where symmetry is important
    /// - Cases where you need a balance between simplicity and representation power
    /// </remarks>
    ReverseBior22,

    /// <summary>
    /// Reverse Biorthogonal 2.4 wavelet - has two vanishing moments for decomposition and four for reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This wavelet has two vanishing moments for decomposition and four for reconstruction.
    /// 
    /// The higher number of reconstruction moments (4) allows it to better capture complex patterns
    /// during reconstruction while maintaining a relatively simple decomposition.
    /// 
    /// This wavelet is good for:
    /// - Applications where reconstruction quality is more important than decomposition
    /// - Signals with both linear trends and more complex components
    /// </remarks>
    ReverseBior24,

    /// <summary>
    /// Reverse Biorthogonal 2.6 wavelet - has two vanishing moments for decomposition and six for reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This wavelet has two vanishing moments for decomposition and six for reconstruction.
    /// 
    /// With six vanishing moments in reconstruction, this wavelet can represent more complex patterns
    /// during the reconstruction phase while keeping decomposition relatively simple.
    /// 
    /// This wavelet is suitable for:
    /// - Applications requiring high-quality reconstruction
    /// - Signals with complex patterns that need to be preserved during processing
    /// </remarks>
    ReverseBior26,

    /// <summary>
    /// Reverse Biorthogonal 2.8 wavelet - has two vanishing moments for decomposition and eight for reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This wavelet has two vanishing moments for decomposition and eight for reconstruction.
    /// 
    /// The high number of reconstruction moments (8) makes this wavelet excellent at preserving
    /// complex details during reconstruction while maintaining a simpler decomposition.
    /// 
    /// This wavelet is particularly useful for:
    /// - Applications where detail preservation is critical
    /// - Complex signals with many frequency components
    /// - Cases where reconstruction quality is significantly more important than decomposition
    /// </remarks>
    ReverseBior28,

    /// <summary>
    /// Reverse Biorthogonal 3.1 wavelet - has three vanishing moments for decomposition and one for reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This wavelet has three vanishing moments for decomposition and one for reconstruction.
    /// 
    /// Unlike previous wavelets, this one has more complexity in decomposition than reconstruction.
    /// This makes it good at analyzing complex patterns but with simpler reconstruction.
    /// 
    /// This wavelet is useful for:
    /// - Applications where analysis (decomposition) is more important than synthesis (reconstruction)
    /// - Detecting quadratic trends in data
    /// - Feature extraction where simple reconstruction is sufficient
    /// </remarks>
    ReverseBior31,

    /// <summary>
    /// Reverse Biorthogonal 3.3 wavelet - has three vanishing moments in both decomposition and reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This wavelet has three vanishing moments in both decomposition and reconstruction.
    /// 
    /// With three vanishing moments on both sides, this wavelet can accurately represent quadratic
    /// trends in your data during both analysis and synthesis.
    /// 
    /// This wavelet provides:
    /// - Balanced performance between decomposition and reconstruction
    /// - Good representation of quadratic patterns
    /// - Symmetry properties that are useful in image processing
    /// </remarks>
    ReverseBior33,

    /// <summary>
    /// Reverse Biorthogonal 3.5 wavelet - has three vanishing moments for decomposition and five for reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This wavelet has three vanishing moments for decomposition and five for reconstruction.
    /// 
    /// This combination provides good analysis of complex patterns with even better reconstruction quality.
    /// 
    /// This wavelet is suitable for:
    /// - Applications requiring both good analysis and high-quality reconstruction
    /// - Signals with quadratic trends and additional complexity
    /// - Image processing tasks where detail preservation is important
    /// </remarks>
    ReverseBior35,

    /// <summary>
    /// Reverse Biorthogonal 3.7 wavelet - has three vanishing moments for decomposition and seven for reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This wavelet has three vanishing moments for decomposition and seven for reconstruction.
    /// 
    /// The high number of reconstruction moments (7) combined with good decomposition properties
    /// makes this wavelet excellent for detailed analysis with high-quality reconstruction.
    /// 
    /// This wavelet is good for:
    /// - Applications requiring detailed analysis and high-fidelity reconstruction
    /// - Complex signals with multiple frequency components
    /// - Image processing where preserving fine details is critical
    /// </remarks>
    ReverseBior37,

    /// <summary>
    /// Reverse Biorthogonal 3.9 wavelet - has three vanishing moments for decomposition and nine for reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This wavelet has three vanishing moments for decomposition and nine for reconstruction.
    /// 
    /// With nine vanishing moments in reconstruction, this wavelet excels at preserving very complex
    /// patterns during reconstruction while maintaining good analysis capabilities.
    /// 
    /// This wavelet is particularly useful for:
    /// - Applications where extremely high-quality reconstruction is needed
    /// - Signals with very fine details that must be preserved
    /// - Advanced image processing tasks requiring maximum detail preservation
    /// </remarks>
    ReverseBior39,

    /// <summary>
    /// Reverse Biorthogonal 4.4 wavelet - has four vanishing moments in both decomposition and reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This wavelet has four vanishing moments in both decomposition and reconstruction.
    /// 
    /// With four vanishing moments on both sides, this wavelet can accurately represent cubic
    /// trends in your data during both analysis and synthesis.
    /// 
    /// This wavelet provides:
    /// - Balanced performance for complex signals
    /// - Good representation of cubic patterns
    /// - Symmetry properties beneficial for image processing
    /// - Higher computational complexity but better accuracy for complex signals
    /// </remarks>
    ReverseBior44,

    /// <summary>
    /// Reverse Biorthogonal 4.6 wavelet - has four vanishing moments for decomposition and six for reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This wavelet has four vanishing moments for decomposition and six for reconstruction.
    /// 
    /// This combination provides excellent analysis of complex patterns with even better reconstruction quality.
    /// 
    /// This wavelet is suitable for:
    /// - Applications requiring both detailed analysis and high-quality reconstruction
    /// - Signals with cubic trends and additional complexity
    /// - Advanced signal processing where both decomposition and reconstruction quality matter
    /// </remarks>
    ReverseBior46,

    /// <summary>
    /// Reverse Biorthogonal 4.8 wavelet - has four vanishing moments for decomposition and eight for reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This wavelet has four vanishing moments for decomposition and eight for reconstruction.
    /// 
    /// The high number of moments on both sides makes this a very powerful wavelet for complex signals.
    /// 
    /// This wavelet is excellent for:
    /// - Applications requiring detailed analysis and very high-quality reconstruction
    /// - Complex signals with multiple frequency components and cubic trends
    /// - Advanced image processing where preserving fine details is critical
    /// </remarks>
    ReverseBior48,

    /// <summary>
    /// Reverse Biorthogonal 5.5 wavelet - has five vanishing moments in both decomposition and reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This wavelet has five vanishing moments in both decomposition and reconstruction.
    /// 
    /// With five vanishing moments on both sides, this wavelet can accurately represent quartic
    /// (4th degree polynomial) trends in your data during both analysis and synthesis.
    /// 
    /// This wavelet provides:
    /// - High-performance balanced analysis and synthesis
    /// - Excellent representation of complex polynomial patterns
    /// - Symmetry properties beneficial for advanced signal processing
    /// - Higher computational complexity but superior accuracy for complex signals
    /// </remarks>
    ReverseBior55,

    /// <summary>
    /// Reverse Biorthogonal 6.8 wavelet - has six vanishing moments for decomposition and eight for reconstruction.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This wavelet has six vanishing moments for decomposition and eight for reconstruction.
    /// 
    /// This is one of the most complex biorthogonal wavelets, capable of representing very sophisticated
    /// patterns in both decomposition and reconstruction.
    /// 
    /// This wavelet is ideal for:
    /// - The most demanding signal processing applications
    /// - Signals with very complex polynomial trends (up to 5th degree)
    /// - Applications where maximum accuracy in both analysis and synthesis is required
    /// - Advanced scientific and engineering applications requiring highest precision
    /// </remarks>
    ReverseBior68
}
