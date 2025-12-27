namespace AiDotNet.WaveletFunctions
{

    /// <summary>
    /// Represents a Reverse Biorthogonal wavelet function implementation for signal processing and analysis.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The Reverse Biorthogonal wavelet is a family of symmetric wavelets that provide exact reconstruction
    /// while having symmetric decomposition and reconstruction filters. These wavelets are particularly
    /// useful in image processing and applications where phase information is important. This implementation
    /// supports various orders of the wavelet and different boundary handling methods.
    /// </para>
    /// <para><b>For Beginners:</b> Reverse Biorthogonal wavelets are specialized mathematical tools for analyzing data.
    /// 
    /// Think of Reverse Biorthogonal wavelets like precise measuring instruments that:
    /// - Can analyze your data while preserving its exact shape and features
    /// - Work especially well with images and signals where shape matters
    /// - Come in different "sizes" (orders) for different levels of detail
    /// 
    /// These wavelets are particularly good at preserving the symmetry and shape of features in your data.
    /// This makes them excellent for applications like image compression, where you want to reduce file
    /// size while maintaining visual quality, or in medical imaging where preserving exact shapes is crucial.
    /// </para>
    /// </remarks>
    /// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
    public class ReverseBiorthogonalWavelet<T> : WaveletFunctionBase<T>
    {

        /// <summary>
        /// The low-pass filter coefficients used during signal decomposition.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This field holds the coefficients for the low-pass filter used during the decomposition phase
        /// of the wavelet transform. These coefficients determine how the wavelet separates and analyzes
        /// the low-frequency components of the input signal.
        /// </para>
        /// <para><b>For Beginners:</b> These are the values used to extract smooth, gradually changing features.
        /// 
        /// The decomposition low-pass coefficients:
        /// - Act like a filter that keeps the slow changes in your data
        /// - Help identify the overall shape and trends
        /// - Have specific values depending on which wavelet type you selected
        /// 
        /// This is similar to looking at a landscape through a foggy window - you see the
        /// major shapes and contours but not the fine details.
        /// </para>
        /// </remarks>
        private readonly Vector<T> _decompositionLowPass;

        /// <summary>
        /// The high-pass filter coefficients used during signal decomposition.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This field holds the coefficients for the high-pass filter used during the decomposition phase
        /// of the wavelet transform. These coefficients determine how the wavelet separates and analyzes
        /// the high-frequency components of the input signal.
        /// </para>
        /// <para><b>For Beginners:</b> These are the values used to extract detailed, rapidly changing features.
        /// 
        /// The decomposition high-pass coefficients:
        /// - Act like a filter that keeps the quick changes in your data
        /// - Help identify edges, boundaries, and fine details
        /// - Have specific values depending on which wavelet type you selected
        /// 
        /// This is similar to a tool that highlights only the edges and textures in a landscape
        /// while ignoring the broader shapes.
        /// </para>
        /// </remarks>
        private readonly Vector<T> _decompositionHighPass;

        /// <summary>
        /// The low-pass filter coefficients used during signal reconstruction.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This field holds the coefficients for the low-pass filter used during the reconstruction phase
        /// of the wavelet transform. These coefficients determine how the wavelet recombines the
        /// low-frequency components to rebuild the original signal.
        /// </para>
        /// <para><b>For Beginners:</b> These are the values used to rebuild smooth features during reconstruction.
        /// 
        /// The reconstruction low-pass coefficients:
        /// - Help convert the approximation coefficients back to the original form
        /// - Work together with the high-pass coefficients to ensure perfect reconstruction
        /// - Are designed to complement the decomposition coefficients
        /// 
        /// In the Reverse Biorthogonal wavelet family, these coefficients are specifically designed
        /// to ensure that after decomposition and reconstruction, you get back exactly the original data.
        /// </para>
        /// </remarks>
        private readonly Vector<T> _reconstructionLowPass;

        /// <summary>
        /// The high-pass filter coefficients used during signal reconstruction.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This field holds the coefficients for the high-pass filter used during the reconstruction phase
        /// of the wavelet transform. These coefficients determine how the wavelet recombines the
        /// high-frequency components to rebuild the original signal.
        /// </para>
        /// <para><b>For Beginners:</b> These are the values used to rebuild detailed features during reconstruction.
        /// 
        /// The reconstruction high-pass coefficients:
        /// - Help convert the detail coefficients back to the original form
        /// - Work together with the low-pass coefficients to ensure perfect reconstruction
        /// - Are designed to complement the decomposition coefficients
        /// 
        /// When combined with the reconstruction low-pass coefficients, these ensure that
        /// all the fine details of your original data are preserved during the transform process.
        /// </para>
        /// </remarks>
        private readonly Vector<T> _reconstructionHighPass;

        /// <summary>
        /// The method used to handle boundary conditions when processing signals of finite length.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This field specifies how the wavelet transform handles the edges of the input signal
        /// when applying the filters. Different boundary handling methods can affect the results
        /// of the transform, especially near the edges of the signal.
        /// </para>
        /// <para><b>For Beginners:</b> This determines how the edges of your data are handled.
        /// 
        /// The boundary handling method:
        /// - Solves the problem of what to do at the edges of your data
        /// - Can be periodic (wrap around), symmetric (mirror), or zero-padded (add zeros)
        /// - Affects how accurate the transform is near the boundaries
        /// 
        /// This is similar to deciding what happens when you reach the edge of a map:
        /// do you wrap around to the other side, mirror the existing terrain, or just
        /// assume there's nothing beyond the edge?
        /// </para>
        /// </remarks>
        private readonly BoundaryHandlingMethod _boundaryMethod;

        /// <summary>
        /// The size of data chunks used when processing large signals.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This field specifies the size of chunks used when processing large signals to
        /// manage memory usage and improve performance. Instead of processing the entire signal
        /// at once, it can be broken into smaller chunks and processed sequentially.
        /// </para>
        /// <para><b>For Beginners:</b> This controls how large pieces of data are broken down for processing.
        /// 
        /// The chunk size:
        /// - Helps manage memory usage for large datasets
        /// - Is like cutting a large task into smaller, more manageable pieces
        /// - Can affect performance but not the mathematical results
        /// 
        /// This is similar to washing a large pile of dishes - you might break it into
        /// smaller batches rather than trying to wash everything at once.
        /// </para>
        /// </remarks>
        private readonly int _chunkSize;

        /// <summary>
        /// The specific type of Reverse Biorthogonal wavelet being used.
        /// </summary>
        /// <remarks>
        /// <para>
        /// This field specifies the exact type of Reverse Biorthogonal wavelet being used,
        /// which determines the filter coefficients and characteristics of the wavelet transform.
        /// Different types have different vanishing moments and support sizes.
        /// </para>
        /// <para><b>For Beginners:</b> This specifies which specific "flavor" of wavelet is being used.
        /// 
        /// The wavelet type:
        /// - Defines the precise mathematical properties of this wavelet
        /// - Is typically named with two numbers (like "ReverseBior22") that indicate properties on each side
        /// - Affects how well the wavelet can represent different kinds of patterns
        /// 
        /// Different wavelet types are like different lenses - each one is better at
        /// capturing certain kinds of details or features in your data.
        /// </para>
        /// </remarks>
        private readonly WaveletType _waveletType;

        /// <summary>
        /// Initializes a new instance of the <see cref="ReverseBiorthogonalWavelet{T}"/> class with the specified parameters.
        /// </summary>
        /// <param name="waveletType">The type of Reverse Biorthogonal wavelet to use. Defaults to ReverseBior22.</param>
        /// <param name="boundaryMethod">The boundary handling method to use. Defaults to Periodic.</param>
        /// <param name="chunkSize">The size of data chunks for processing large signals. Defaults to 1024.</param>
        /// <remarks>
        /// <para>
        /// This constructor initializes the Reverse Biorthogonal wavelet with the specified parameters.
        /// It sets up the filter coefficients based on the wavelet type and prepares the wavelet for
        /// signal processing operations with the specified boundary handling method and chunk size.
        /// </para>
        /// <para><b>For Beginners:</b> This sets up the Reverse Biorthogonal wavelet with your chosen settings.
        ///
        /// When creating a Reverse Biorthogonal wavelet:
        /// - You can select the specific wavelet type that best matches your needs
        /// - You can choose how to handle the edges of your data
        /// - You can set how large pieces of data are broken down for processing
        ///
        /// The default settings (ReverseBior22, Periodic, 1024) work well for many
        /// common applications, but you can customize them for your specific needs.
        /// </para>
        /// </remarks>
        public ReverseBiorthogonalWavelet(
            WaveletType waveletType = WaveletType.ReverseBior22,
            BoundaryHandlingMethod boundaryMethod = BoundaryHandlingMethod.Periodic,
            int chunkSize = 1024)
        {
            _boundaryMethod = boundaryMethod;
            _chunkSize = chunkSize;
            _waveletType = waveletType;
            (_decompositionLowPass, _decompositionHighPass, _reconstructionLowPass, _reconstructionHighPass) =
                GetReverseBiorthogonalCoefficients(_waveletType);
        }

        /// <summary>
        /// Calculates the wavelet function value at the specified point.
        /// </summary>
        /// <param name="x">The input point at which to calculate the wavelet value.</param>
        /// <returns>The calculated wavelet function value at the specified point.</returns>
        /// <remarks>
        /// <para>
        /// This method computes the value of the Reverse Biorthogonal wavelet function at the given input point.
        /// It uses the discrete cascade algorithm to approximate the continuous wavelet function based on
        /// the reconstruction low-pass filter coefficients.
        /// </para>
        /// <para><b>For Beginners:</b> This method calculates the height of the wavelet at a specific point.
        /// 
        /// When you use this method:
        /// - You provide a point (x) on the horizontal axis
        /// - The method returns the height of the wavelet at that point
        /// - The calculation uses the cascade algorithm to approximate the continuous wavelet
        /// 
        /// While most wavelet transforms work with discrete data, this method lets you
        /// evaluate the continuous wavelet function, which is useful for visualization
        /// and theoretical understanding of the wavelet shape.
        /// </para>
        /// </remarks>
        public override T Calculate(T x)
        {
            T result = NumOps.Zero;
            int centerIndex = _reconstructionLowPass.Length / 2;

            for (int k = 0; k < _reconstructionLowPass.Length; k++)
            {
                T shiftedX = NumOps.Subtract(x, NumOps.FromDouble(k - centerIndex));
                T phiValue = DiscreteCascadeAlgorithm(shiftedX);
                result = NumOps.Add(result, NumOps.Multiply(_reconstructionLowPass[k], phiValue));
            }

            return result;
        }

        /// <summary>
        /// Implements the discrete cascade algorithm to approximate the continuous scaling function.
        /// </summary>
        /// <param name="x">The input point at which to evaluate the scaling function.</param>
        /// <returns>The approximated value of the scaling function at the specified point.</returns>
        /// <remarks>
        /// <para>
        /// This method implements the discrete cascade algorithm, which iteratively approximates
        /// the continuous scaling function based on the reconstruction filter coefficients.
        /// Starting with a simple initial approximation, it applies the filters repeatedly to
        /// refine the approximation of the scaling function.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method calculates the scaling function value at a specific point.
        /// 
        /// The discrete cascade algorithm:
        /// - Starts with a simple approximation of the scaling function
        /// - Repeatedly applies the wavelet filters to make the approximation better
        /// - Gets closer and closer to the true continuous function with each iteration
        /// 
        /// This is like progressively refining a rough sketch into a detailed drawing,
        /// getting closer to the "true" shape with each pass.
        /// </para>
        /// </remarks>
        private T DiscreteCascadeAlgorithm(T x)
        {
            const int resolution = 1024;
            const int iterations = 7;
            var values = new T[resolution];

            // Initialize with the scaling function
            for (int i = 0; i < resolution; i++)
            {
                T xValue = NumOps.Divide(NumOps.FromDouble(i), NumOps.FromDouble(resolution - 1));
                values[i] = ScalingFunction(xValue);
            }

            // Perform iterations
            for (int iter = 0; iter < iterations; iter++)
            {
                var newValues = new T[resolution];
                for (int i = 0; i < resolution; i++)
                {
                    T sum = NumOps.Zero;
                    for (int k = 0; k < _reconstructionLowPass.Length; k++)
                    {
                        sum = NumOps.Add(sum, NumOps.Multiply(_reconstructionLowPass[k], values[(2 * i - k + resolution) % resolution]));
                    }

                    newValues[i] = sum;
                }

                values = newValues;
            }

            // Interpolate to find the value at x
            int index = (int)(Convert.ToDouble(x) * (resolution - 1));
            index = Math.Max(0, Math.Min(resolution - 2, index));
            T t = NumOps.Subtract(x, NumOps.Divide(NumOps.FromDouble(index), NumOps.FromDouble(resolution - 1)));
            return NumOps.Add(
                NumOps.Multiply(NumOps.Subtract(NumOps.One, t), values[index]),
                NumOps.Multiply(t, values[index + 1])
            );
        }

        /// <summary>
        /// Provides a simple initial approximation of the scaling function.
        /// </summary>
        /// <param name="x">The input point at which to evaluate the initial scaling function.</param>
        /// <returns>The initial approximation of the scaling function value.</returns>
        /// <remarks>
        /// <para>
        /// This method provides a simple initial approximation of the scaling function, which is
        /// used as the starting point for the discrete cascade algorithm. It defines a basic
        /// function that approximates the general shape of the scaling function.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides a starting approximation for the scaling function.
        /// 
        /// The initial scaling function:
        /// - Is a simple function that roughly approximates the true scaling function
        /// - Forms the starting point for the cascade algorithm's refinement process
        /// - Is like a rough sketch before adding the details
        /// 
        /// This is defined as a simple piecewise function that decreases linearly
        /// from 1 to 0 in the range [0,1], and has a quadratic tail in the range [1,2].
        /// </para>
        /// </remarks>
        private T ScalingFunction(T x)
        {
            T absX = NumOps.Abs(x);
            T result = NumOps.Zero;

            if (NumOps.LessThan(absX, NumOps.One))
            {
                result = NumOps.Subtract(NumOps.One, absX);
            }
            else if (NumOps.LessThan(absX, NumOps.FromDouble(2)))
            {
                T temp = NumOps.Subtract(NumOps.FromDouble(2), absX);
                result = NumOps.Multiply(NumOps.FromDouble(0.5), NumOps.Multiply(temp, temp));
            }

            return result;
        }

        /// <summary>
        /// Decomposes an input signal into approximation and detail coefficients using the wavelet transform.
        /// </summary>
        /// <param name="input">The input signal to decompose.</param>
        /// <returns>A tuple containing the approximation and detail coefficients of the decomposed signal.</returns>
        /// <remarks>
        /// <para>
        /// This method implements the Reverse Biorthogonal wavelet transform, which decomposes the input signal into
        /// approximation coefficients (low-frequency components) and detail coefficients (high-frequency components).
        /// It applies the decomposition filters to the input signal and processes the data in chunks if necessary.
        /// </para>
        /// <para><b>For Beginners:</b> This method breaks down your data into low-frequency and high-frequency parts.
        /// 
        /// When decomposing a signal:
        /// - The data is filtered to separate smooth trends (approximation) from details
        /// - For long signals, the process works on manageable chunks
        /// - The result includes two sets of coefficients: approximation and detail
        /// 
        /// This is like separating a photograph into two images: one blurry image showing
        /// the main shapes and colors, and another showing just the edges and textures.
        /// Together, these two parts contain all the information from the original image.
        /// </para>
        /// </remarks>
        public override (Vector<T> approximation, Vector<T> detail) Decompose(Vector<T> input)
        {
            int n = input.Length;
            var approximation = new Vector<T>((n + 1) / 2);
            var detail = new Vector<T>((n + 1) / 2);

            for (int i = 0; i < n; i += _chunkSize)
            {
                int chunkEnd = Math.Min(i + _chunkSize, n);
                DecomposeChunk(input, approximation, detail, i, chunkEnd);
            }

            return (approximation, detail);
        }

        /// <summary>
        /// Processes a chunk of the input signal during decomposition.
        /// </summary>
        /// <param name="input">The input signal to decompose.</param>
        /// <param name="approximation">The output approximation coefficients.</param>
        /// <param name="detail">The output detail coefficients.</param>
        /// <param name="start">The starting index of the chunk to process.</param>
        /// <param name="end">The ending index of the chunk to process.</param>
        /// <remarks>
        /// <para>
        /// This method processes a chunk of the input signal during the decomposition phase of the
        /// wavelet transform. It applies the decomposition filters to the specified range of the input
        /// signal and computes the corresponding approximation and detail coefficients.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method processes a section of your data during decomposition.
        /// 
        /// When decomposing a chunk:
        /// - The method applies the wavelet filters to a specific section of your data
        /// - It computes both approximation and detail coefficients for that section
        /// - It handles the boundary conditions according to the specified method
        /// 
        /// This allows efficient processing of large datasets by breaking them into
        /// manageable pieces and processing each piece separately.
        /// </para>
        /// </remarks>
        private void DecomposeChunk(Vector<T> input, Vector<T> approximation, Vector<T> detail, int start, int end)
        {
            int lowPassLen = _decompositionLowPass.Length;
            int highPassLen = _decompositionHighPass.Length;

            for (int i = start; i < end; i += 2)
            {
                T approx = NumOps.Zero;
                T det = NumOps.Zero;

                // Apply low-pass filter for approximation
                for (int j = 0; j < lowPassLen; j++)
                {
                    int index = GetExtendedIndex(i + j - lowPassLen / 2 + 1, input.Length);
                    approx = NumOps.Add(approx, NumOps.Multiply(_decompositionLowPass[j], input[index]));
                }

                // Apply high-pass filter for detail (may have different length)
                for (int j = 0; j < highPassLen; j++)
                {
                    int index = GetExtendedIndex(i + j - highPassLen / 2 + 1, input.Length);
                    det = NumOps.Add(det, NumOps.Multiply(_decompositionHighPass[j], input[index]));
                }

                approximation[i / 2] = approx;
                detail[i / 2] = det;
            }
        }

        /// <summary>
        /// Reconstructs a signal from its approximation and detail coefficients.
        /// </summary>
        /// <param name="approximation">The approximation coefficients of the signal.</param>
        /// <param name="detail">The detail coefficients of the signal.</param>
        /// <returns>The reconstructed signal.</returns>
        /// <remarks>
        /// <para>
        /// This method reconstructs the original signal from its wavelet transform coefficients.
        /// It uses the reconstruction filters to combine the approximation and detail coefficients
        /// and produces a signal that, in the case of biorthogonal wavelets, is identical to the
        /// original signal before decomposition.
        /// </para>
        /// <para><b>For Beginners:</b> This method rebuilds your original data from its decomposed parts.
        /// 
        /// During reconstruction:
        /// - The approximation and detail coefficients are combined
        /// - The reconstruction filters are applied to convert back to the original domain
        /// - For large datasets, the process works on manageable chunks
        /// 
        /// This is like reassembling a puzzle from its pieces, where the approximation and
        /// detail coefficients are the puzzle pieces, and the reconstruction process puts
        /// them back together to form the original image.
        /// </para>
        /// </remarks>
        public Vector<T> Reconstruct(Vector<T> approximation, Vector<T> detail)
        {
            // Reconstructed signal has twice the length of the coefficient vectors
            int n = approximation.Length * 2;
            var reconstructed = new Vector<T>(n);

            for (int i = 0; i < n; i += _chunkSize)
            {
                int chunkEnd = Math.Min(i + _chunkSize, n);
                ReconstructChunk(approximation, detail, reconstructed, i, chunkEnd);
            }

            return reconstructed;
        }

        /// <summary>
        /// Processes a chunk of the signal during reconstruction.
        /// </summary>
        /// <param name="approximation">The approximation coefficients.</param>
        /// <param name="detail">The detail coefficients.</param>
        /// <param name="reconstructed">The output reconstructed signal.</param>
        /// <param name="start">The starting index of the chunk to process.</param>
        /// <param name="end">The ending index of the chunk to process.</param>
        /// <remarks>
        /// <para>
        /// This method processes a chunk of the signal during the reconstruction phase of the
        /// wavelet transform. It applies the reconstruction filters to the specified range of the
        /// approximation and detail coefficients and computes the corresponding reconstructed signal.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method processes a section of your data during reconstruction.
        /// 
        /// When reconstructing a chunk:
        /// - The method applies the reconstruction filters to specific sections of the coefficients
        /// - It computes the original signal values for that section
        /// - It handles boundary conditions and filter application efficiently
        /// 
        /// This allows efficient processing of large datasets by breaking them into
        /// manageable pieces and processing each piece separately.
        /// </para>
        /// </remarks>
        private void ReconstructChunk(Vector<T> approximation, Vector<T> detail, Vector<T> reconstructed, int start, int end)
        {
            int len = approximation.Length;
            if (len == 0) return;

            int lowPassLen = _reconstructionLowPass.Length;
            int highPassLen = _reconstructionHighPass.Length;

            for (int i = start; i < end; i++)
            {
                T value = NumOps.Zero;

                // Apply low-pass reconstruction filter
                for (int j = 0; j < lowPassLen; j++)
                {
                    int rawIndex = i / 2 - j + lowPassLen;
                    int index = ((rawIndex % len) + len) % len;

                    if ((i - j) % 2 == 0)
                    {
                        value = NumOps.Add(value, NumOps.Multiply(_reconstructionLowPass[j], approximation[index]));
                    }
                }

                // Apply high-pass reconstruction filter (may have different length)
                for (int j = 0; j < highPassLen; j++)
                {
                    int rawIndex = i / 2 - j + highPassLen;
                    int index = ((rawIndex % len) + len) % len;

                    if ((i - j) % 2 == 0)
                    {
                        value = NumOps.Add(value, NumOps.Multiply(_reconstructionHighPass[j], detail[index]));
                    }
                }

                reconstructed[i] = value;
            }
        }

        /// <summary>
        /// Performs multi-level decomposition of a signal using the wavelet transform.
        /// </summary>
        /// <param name="input">The input signal to decompose.</param>
        /// <param name="levels">The number of decomposition levels to perform.</param>
        /// <returns>A tuple containing the final approximation coefficients and a list of detail coefficients from each level.</returns>
        /// <remarks>
        /// <para>
        /// This method performs a multi-level wavelet decomposition of the input signal.
        /// It repeatedly applies the wavelet transform to the approximation coefficients,
        /// resulting in a hierarchical representation of the signal at different scales.
        /// </para>
        /// <para><b>For Beginners:</b> This method analyzes your data at multiple scales or levels of detail.
        /// 
        /// In multi-level decomposition:
        /// - The first level separates your data into approximation and detail
        /// - The next level further breaks down the approximation from the previous level
        /// - This continues for the specified number of levels
        /// - The result is a series of detail coefficients and a final approximation
        /// 
        /// This is like looking at a landscape through progressively stronger binoculars -
        /// each level reveals finer details at a different scale.
        /// </para>
        /// </remarks>
        public (Vector<T> approximation, List<Vector<T>> details) DecomposeMultiLevel(Vector<T> input, int levels)
        {
            var details = new List<Vector<T>>();
            var currentApproximation = input;

            for (int i = 0; i < levels; i++)
            {
                var (newApproximation, detail) = Decompose(currentApproximation);
                details.Add(detail);
                currentApproximation = newApproximation;

                if (currentApproximation.Length <= _decompositionLowPass.Length)
                {
                    break;
                }
            }

            return (currentApproximation, details);
        }

        /// <summary>
        /// Performs multi-level reconstruction of a signal from its wavelet transform coefficients.
        /// </summary>
        /// <param name="approximation">The final approximation coefficients.</param>
        /// <param name="details">The list of detail coefficients from each level.</param>
        /// <returns>The reconstructed signal.</returns>
        /// <remarks>
        /// <para>
        /// This method performs a multi-level wavelet reconstruction of a signal from its wavelet
        /// transform coefficients. It repeatedly applies the inverse wavelet transform, starting
        /// from the coarsest level and progressively incorporating the detail coefficients from
        /// finer levels.
        /// </para>
        /// <para><b>For Beginners:</b> This method rebuilds your original data from multi-level decomposition.
        /// 
        /// In multi-level reconstruction:
        /// - The process starts with the final approximation and the deepest level of details
        /// - It combines these to reconstruct the approximation from the previous level
        /// - This continues, working backward through all the levels
        /// - The final result is the fully reconstructed original signal
        /// 
        /// This is like reassembling a complex puzzle by first putting together the major sections,
        /// then adding progressively finer details until the complete picture emerges.
        /// </para>
        /// </remarks>
        public Vector<T> ReconstructMultiLevel(Vector<T> approximation, List<Vector<T>> details)
        {
            var currentApproximation = approximation;

            for (int i = details.Count - 1; i >= 0; i--)
            {
                currentApproximation = Reconstruct(currentApproximation, details[i]);
            }

            return currentApproximation;
        }

        /// <summary>
        /// Gets the scaling coefficients used in the Reverse Biorthogonal wavelet transform.
        /// </summary>
        /// <returns>A vector containing the scaling coefficients for signal reconstruction.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the reconstruction low-pass filter coefficients used in the Reverse
        /// Biorthogonal wavelet transform. These coefficients determine how the low-frequency components
        /// of the signal are reconstructed during the inverse transform.
        /// </para>
        /// <para><b>For Beginners:</b> This method gives you the values used to extract and reconstruct smooth features.
        /// 
        /// The scaling coefficients:
        /// - Are the reconstruction low-pass filter coefficients
        /// - Help rebuild the smooth, gradually changing parts of your data
        /// - Have specific values determined by the wavelet type you selected
        /// 
        /// These coefficients are a key part of what defines this particular wavelet's
        /// properties and how it will analyze and process your data.
        /// </para>
        /// </remarks>
        public override Vector<T> GetScalingCoefficients()
        {
            return _reconstructionLowPass;
        }

        /// <summary>
        /// Gets the wavelet coefficients used in the Reverse Biorthogonal wavelet transform.
        /// </summary>
        /// <returns>A vector containing the wavelet coefficients for signal reconstruction.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the reconstruction high-pass filter coefficients used in the Reverse
        /// Biorthogonal wavelet transform. These coefficients determine how the high-frequency components
        /// of the signal are reconstructed during the inverse transform.
        /// </para>
        /// <para><b>For Beginners:</b> This method gives you the values used to extract and reconstruct detailed features.
        /// 
        /// The wavelet coefficients:
        /// - Are the reconstruction high-pass filter coefficients
        /// - Help rebuild the detailed, rapidly changing parts of your data
        /// - Have specific values determined by the wavelet type you selected
        /// 
        /// When combined with the scaling coefficients, these define the complete reconstruction
        /// process and ensure that all important features of your data are preserved.
        /// </para>
        /// </remarks>
        public override Vector<T> GetWaveletCoefficients()
        {
            return _reconstructionHighPass;
        }

        /// <summary>
        /// Extends an index beyond the boundaries of an array according to the selected boundary handling method.
        /// </summary>
        /// <param name="index">The original index, which may be outside the array bounds.</param>
        /// <param name="length">The length of the array.</param>
        /// <returns>The extended index that is safe to use within the array bounds.</returns>
        /// <remarks>
        /// <para>
        /// This method extends an index that falls outside the boundaries of an array according to the
        /// selected boundary handling method (periodic, symmetric, or zero-padding). This allows the
        /// wavelet transform to be applied near the edges of the signal.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method handles what happens when filters reach the edges of your data.
        /// 
        /// When handling boundaries:
        /// - Periodic: Wraps around to the other side (like a loop)
        /// - Symmetric: Reflects back from the edge (like a mirror)
        /// - ZeroPadding: Assumes zero values beyond the edge
        /// 
        /// This is a crucial part of the wavelet transform process since the filters need
        /// to access values beyond the edges of your data when processing near the boundaries.
        /// The choice of boundary method can affect the quality of the transform near the edges.
        /// </para>
        /// </remarks>
        private int GetExtendedIndex(int index, int length)
        {
            if (length <= 0)
                return 0;

            switch (_boundaryMethod)
            {
                case BoundaryHandlingMethod.Periodic:
                    return ((index % length) + length) % length;

                case BoundaryHandlingMethod.Symmetric:
                    // Handle symmetric reflection properly for any index value
                    // Uses the half-sample symmetric extension
                    if (length == 1)
                        return 0;

                    int period = 2 * length - 2;
                    int normalizedIndex = ((index % period) + period) % period;

                    if (normalizedIndex >= length)
                        return period - normalizedIndex;
                    return normalizedIndex;

                case BoundaryHandlingMethod.ZeroPadding:
                    // For zero padding, clamp to valid range (boundary values get repeated)
                    // This prevents index out of range errors
                    if (index < 0)
                        return 0;
                    if (index >= length)
                        return length - 1;
                    return index;

                default:
                    throw new ArgumentException("Invalid boundary handling method");
            }
        }

        /// <summary>
        /// Gets the filter coefficients for the specified Reverse Biorthogonal wavelet type.
        /// </summary>
        /// <param name="waveletType">The type of Reverse Biorthogonal wavelet.</param>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the specified wavelet type is not supported.</exception>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the specified Reverse Biorthogonal wavelet type.
        /// Different wavelet types have different filter coefficients, which affect the properties and
        /// performance of the wavelet transform for different types of signals.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method gets the specific filter values for your chosen wavelet type.
        /// 
        /// The filter coefficients:
        /// - Define the exact mathematical properties of the wavelet
        /// - Are different for each type of Reverse Biorthogonal wavelet
        /// - Determine how effectively the wavelet can represent different features in your data
        /// 
        /// These coefficients are carefully designed to ensure exact reconstruction and to have
        /// specific mathematical properties (such as symmetry and vanishing moments) that make
        /// them suitable for different applications.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBiorthogonalCoefficients(WaveletType waveletType)
        {
            return waveletType switch
            {
                WaveletType.ReverseBior22 => GetReverseBior22Coefficients(),
                WaveletType.ReverseBior11 => GetReverseBior11Coefficients(),
                WaveletType.ReverseBior13 => GetReverseBior13Coefficients(),
                WaveletType.ReverseBior24 => GetReverseBior24Coefficients(),
                WaveletType.ReverseBior26 => GetReverseBior26Coefficients(),
                WaveletType.ReverseBior28 => GetReverseBior28Coefficients(),
                WaveletType.ReverseBior31 => GetReverseBior31Coefficients(),
                WaveletType.ReverseBior33 => GetReverseBior33Coefficients(),
                WaveletType.ReverseBior35 => GetReverseBior35Coefficients(),
                WaveletType.ReverseBior37 => GetReverseBior37Coefficients(),
                WaveletType.ReverseBior39 => GetReverseBior39Coefficients(),
                WaveletType.ReverseBior44 => GetReverseBior44Coefficients(),
                WaveletType.ReverseBior46 => GetReverseBior46Coefficients(),
                WaveletType.ReverseBior48 => GetReverseBior48Coefficients(),
                WaveletType.ReverseBior55 => GetReverseBior55Coefficients(),
                WaveletType.ReverseBior68 => GetReverseBior68Coefficients(),
                _ => throw new ArgumentOutOfRangeException(nameof(waveletType), $"Wavelet type {waveletType} is not supported for Reverse Biorthogonal wavelets."),
            };
        }

        /// <summary>
        /// Gets the filter coefficients for the ReverseBior22 wavelet.
        /// </summary>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the ReverseBior22 wavelet, which has 2 vanishing
        /// moments in the decomposition filters and 2 vanishing moments in the reconstruction filters.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides the specific coefficients for the ReverseBior22 wavelet.
        /// 
        /// The ReverseBior22 wavelet:
        /// - Has a good balance between smooth approximation and detail detection
        /// - The "22" means it has 2 vanishing moments in each direction
        /// - Is one of the most commonly used symmetric wavelets for image processing
        /// 
        /// This wavelet is a good all-purpose choice when you need symmetry and exact reconstruction,
        /// such as in image compression where visual quality is important.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior22Coefficients()
        {
            var decompositionLowPass = new Vector<T>(6);
            var decompositionHighPass = new Vector<T>(6);
            var reconstructionLowPass = new Vector<T>(6);
            var reconstructionHighPass = new Vector<T>(6);

            decompositionLowPass[0] = NumOps.FromDouble(0);
            decompositionLowPass[1] = NumOps.FromDouble(-0.1767766952966369);
            decompositionLowPass[2] = NumOps.FromDouble(0.3535533905932738);
            decompositionLowPass[3] = NumOps.FromDouble(1.0606601717798214);
            decompositionLowPass[4] = NumOps.FromDouble(0.3535533905932738);
            decompositionLowPass[5] = NumOps.FromDouble(-0.1767766952966369);

            decompositionHighPass[0] = NumOps.FromDouble(0);
            decompositionHighPass[1] = NumOps.FromDouble(0.3535533905932738);
            decompositionHighPass[2] = NumOps.FromDouble(-0.7071067811865476);
            decompositionHighPass[3] = NumOps.FromDouble(0.3535533905932738);
            decompositionHighPass[4] = NumOps.FromDouble(0);
            decompositionHighPass[5] = NumOps.FromDouble(0);

            reconstructionLowPass[0] = NumOps.FromDouble(0);
            reconstructionLowPass[1] = NumOps.FromDouble(0.3535533905932738);
            reconstructionLowPass[2] = NumOps.FromDouble(0.7071067811865476);
            reconstructionLowPass[3] = NumOps.FromDouble(0.3535533905932738);
            reconstructionLowPass[4] = NumOps.FromDouble(0);
            reconstructionLowPass[5] = NumOps.FromDouble(0);

            reconstructionHighPass[0] = NumOps.FromDouble(0);
            reconstructionHighPass[1] = NumOps.FromDouble(-0.1767766952966369);
            reconstructionHighPass[2] = NumOps.FromDouble(0.3535533905932738);
            reconstructionHighPass[3] = NumOps.FromDouble(-1.0606601717798214);
            reconstructionHighPass[4] = NumOps.FromDouble(0.3535533905932738);
            reconstructionHighPass[5] = NumOps.FromDouble(-0.1767766952966369);

            return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
        }

        /// <summary>
        /// Gets the filter coefficients for the ReverseBior11 wavelet.
        /// </summary>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the ReverseBior11 wavelet, which has 1 vanishing
        /// moment in the decomposition filters and 1 vanishing moment in the reconstruction filters.
        /// This is the simplest Reverse Biorthogonal wavelet.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides the specific coefficients for the ReverseBior11 wavelet.
        /// 
        /// The ReverseBior11 wavelet:
        /// - Is the simplest wavelet in the Reverse Biorthogonal family
        /// - The "11" means it has 1 vanishing moment in each direction
        /// - Has very short filters, making it computationally efficient
        /// 
        /// This wavelet is good for applications where computational speed is more important
        /// than having the most accurate representation of the signal.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior11Coefficients()
        {
            var decompositionLowPass = new Vector<T>(2);
            var decompositionHighPass = new Vector<T>(2);
            var reconstructionLowPass = new Vector<T>(2);
            var reconstructionHighPass = new Vector<T>(2);

            // Decomposition low-pass filter
            decompositionLowPass[0] = NumOps.FromDouble(0.7071067811865476);
            decompositionLowPass[1] = NumOps.FromDouble(0.7071067811865476);

            // Decomposition high-pass filter
            decompositionHighPass[0] = NumOps.FromDouble(-0.7071067811865476);
            decompositionHighPass[1] = NumOps.FromDouble(0.7071067811865476);

            // Reconstruction low-pass filter
            reconstructionLowPass[0] = NumOps.FromDouble(0.7071067811865476);
            reconstructionLowPass[1] = NumOps.FromDouble(0.7071067811865476);

            // Reconstruction high-pass filter
            reconstructionHighPass[0] = NumOps.FromDouble(0.7071067811865476);
            reconstructionHighPass[1] = NumOps.FromDouble(-0.7071067811865476);

            return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
        }

        /// <summary>
        /// Gets the filter coefficients for the ReverseBior13 wavelet.
        /// </summary>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the ReverseBior13 wavelet, which has 1 vanishing
        /// moment in the decomposition filters and 3 vanishing moments in the reconstruction filters.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides the specific coefficients for the ReverseBior13 wavelet.
        /// 
        /// The ReverseBior13 wavelet:
        /// - Has asymmetric properties (1 vanishing moment on one side, 3 on the other)
        /// - The "13" means it has 1 vanishing moment in decomposition and 3 in reconstruction
        /// - Provides more detail preservation in the reconstruction phase
        /// 
        /// This wavelet might be chosen when you need different characteristics for the decomposition
        /// and reconstruction phases of your analysis.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior13Coefficients()
        {
            var decompositionLowPass = new Vector<T>(6);
            var decompositionHighPass = new Vector<T>(2);
            var reconstructionLowPass = new Vector<T>(2);
            var reconstructionHighPass = new Vector<T>(6);

            // Decomposition low-pass filter
            decompositionLowPass[0] = NumOps.FromDouble(-0.0883883476483184);
            decompositionLowPass[1] = NumOps.FromDouble(0.0883883476483184);
            decompositionLowPass[2] = NumOps.FromDouble(0.7071067811865476);
            decompositionLowPass[3] = NumOps.FromDouble(0.7071067811865476);
            decompositionLowPass[4] = NumOps.FromDouble(0.0883883476483184);
            decompositionLowPass[5] = NumOps.FromDouble(-0.0883883476483184);

            // Decomposition high-pass filter
            decompositionHighPass[0] = NumOps.FromDouble(-0.7071067811865476);
            decompositionHighPass[1] = NumOps.FromDouble(0.7071067811865476);

            // Reconstruction low-pass filter
            reconstructionLowPass[0] = NumOps.FromDouble(0.7071067811865476);
            reconstructionLowPass[1] = NumOps.FromDouble(0.7071067811865476);

            // Reconstruction high-pass filter
            reconstructionHighPass[0] = NumOps.FromDouble(0.0883883476483184);
            reconstructionHighPass[1] = NumOps.FromDouble(0.0883883476483184);
            reconstructionHighPass[2] = NumOps.FromDouble(-0.7071067811865476);
            reconstructionHighPass[3] = NumOps.FromDouble(0.7071067811865476);
            reconstructionHighPass[4] = NumOps.FromDouble(-0.0883883476483184);
            reconstructionHighPass[5] = NumOps.FromDouble(-0.0883883476483184);

            return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
        }

        /// <summary>
        /// Gets the filter coefficients for the ReverseBior24 wavelet.
        /// </summary>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the ReverseBior24 wavelet, which has 2 vanishing
        /// moments in the decomposition filters and 4 vanishing moments in the reconstruction filters.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides the specific coefficients for the ReverseBior24 wavelet.
        /// 
        /// The ReverseBior24 wavelet:
        /// - Has asymmetric properties (2 vanishing moments on one side, 4 on the other)
        /// - The "24" means it has 2 vanishing moments in decomposition and 4 in reconstruction
        /// - Provides more detailed frequency analysis in the reconstruction phase
        /// 
        /// This wavelet is good for applications where you need moderately good localization in both
        /// time and frequency, with better frequency precision during reconstruction.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior24Coefficients()
        {
            var decompositionLowPass = new Vector<T>(10);
            var decompositionHighPass = new Vector<T>(6);
            var reconstructionLowPass = new Vector<T>(6);
            var reconstructionHighPass = new Vector<T>(10);

            decompositionLowPass[0] = NumOps.FromDouble(0);
            decompositionLowPass[1] = NumOps.FromDouble(-0.0331456303681194);
            decompositionLowPass[2] = NumOps.FromDouble(-0.0662912607362388);
            decompositionLowPass[3] = NumOps.FromDouble(0.1767766952966369);
            decompositionLowPass[4] = NumOps.FromDouble(0.4198446513295126);
            decompositionLowPass[5] = NumOps.FromDouble(0.9943689110435825);
            decompositionLowPass[6] = NumOps.FromDouble(0.4198446513295126);
            decompositionLowPass[7] = NumOps.FromDouble(0.1767766952966369);
            decompositionLowPass[8] = NumOps.FromDouble(-0.0662912607362388);
            decompositionLowPass[9] = NumOps.FromDouble(-0.0331456303681194);

            decompositionHighPass[0] = NumOps.FromDouble(0);
            decompositionHighPass[1] = NumOps.FromDouble(-0.1767766952966369);
            decompositionHighPass[2] = NumOps.FromDouble(0.3535533905932738);
            decompositionHighPass[3] = NumOps.FromDouble(-0.3535533905932738);
            decompositionHighPass[4] = NumOps.FromDouble(0.1767766952966369);
            decompositionHighPass[5] = NumOps.FromDouble(0);

            reconstructionLowPass[0] = NumOps.FromDouble(0);
            reconstructionLowPass[1] = NumOps.FromDouble(0.1767766952966369);
            reconstructionLowPass[2] = NumOps.FromDouble(0.3535533905932738);
            reconstructionLowPass[3] = NumOps.FromDouble(0.3535533905932738);
            reconstructionLowPass[4] = NumOps.FromDouble(0.1767766952966369);
            reconstructionLowPass[5] = NumOps.FromDouble(0);

            reconstructionHighPass[0] = NumOps.FromDouble(0);
            reconstructionHighPass[1] = NumOps.FromDouble(0.0331456303681194);
            reconstructionHighPass[2] = NumOps.FromDouble(-0.0662912607362388);
            reconstructionHighPass[3] = NumOps.FromDouble(-0.1767766952966369);
            reconstructionHighPass[4] = NumOps.FromDouble(0.4198446513295126);
            reconstructionHighPass[5] = NumOps.FromDouble(-0.9943689110435825);
            reconstructionHighPass[6] = NumOps.FromDouble(0.4198446513295126);
            reconstructionHighPass[7] = NumOps.FromDouble(-0.1767766952966369);
            reconstructionHighPass[8] = NumOps.FromDouble(-0.0662912607362388);
            reconstructionHighPass[9] = NumOps.FromDouble(0.0331456303681194);

            return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
        }

        /// <summary>
        /// Gets the filter coefficients for the ReverseBior26 wavelet.
        /// </summary>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the ReverseBior26 wavelet, which has 2 vanishing
        /// moments in the decomposition filters and 6 vanishing moments in the reconstruction filters.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides the specific coefficients for the ReverseBior26 wavelet.
        /// 
        /// The ReverseBior26 wavelet:
        /// - Has highly asymmetric properties (2 vanishing moments on one side, 6 on the other)
        /// - The "26" means it has 2 vanishing moments in decomposition and 6 in reconstruction
        /// - Provides excellent frequency resolution in the reconstruction phase
        /// 
        /// This wavelet might be chosen for applications where accurate reconstruction of frequency
        /// content is more important than precise time localization, such as in some audio processing
        /// or spectral analysis tasks.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior26Coefficients()
        {
            var decompositionLowPass = new Vector<T>(14);
            var decompositionHighPass = new Vector<T>(6);
            var reconstructionLowPass = new Vector<T>(6);
            var reconstructionHighPass = new Vector<T>(14);

            decompositionLowPass[0] = NumOps.FromDouble(0);
            decompositionLowPass[1] = NumOps.FromDouble(0.0069053396600248);
            decompositionLowPass[2] = NumOps.FromDouble(0.0138106793200496);
            decompositionLowPass[3] = NumOps.FromDouble(-0.0469563096881692);
            decompositionLowPass[4] = NumOps.FromDouble(-0.1077232986963880);
            decompositionLowPass[5] = NumOps.FromDouble(0.1697627774134332);
            decompositionLowPass[6] = NumOps.FromDouble(0.4474660099696121);
            decompositionLowPass[7] = NumOps.FromDouble(0.9667475524034829);
            decompositionLowPass[8] = NumOps.FromDouble(0.4474660099696121);
            decompositionLowPass[9] = NumOps.FromDouble(0.1697627774134332);
            decompositionLowPass[10] = NumOps.FromDouble(-0.1077232986963880);
            decompositionLowPass[11] = NumOps.FromDouble(-0.0469563096881692);
            decompositionLowPass[12] = NumOps.FromDouble(0.0138106793200496);
            decompositionLowPass[13] = NumOps.FromDouble(0.0069053396600248);

            decompositionHighPass[0] = NumOps.FromDouble(0);
            decompositionHighPass[1] = NumOps.FromDouble(0.1767766952966369);
            decompositionHighPass[2] = NumOps.FromDouble(-0.3535533905932738);
            decompositionHighPass[3] = NumOps.FromDouble(0.3535533905932738);
            decompositionHighPass[4] = NumOps.FromDouble(-0.1767766952966369);
            decompositionHighPass[5] = NumOps.FromDouble(0);

            reconstructionLowPass[0] = NumOps.FromDouble(0);
            reconstructionLowPass[1] = NumOps.FromDouble(-0.1767766952966369);
            reconstructionLowPass[2] = NumOps.FromDouble(-0.3535533905932738);
            reconstructionLowPass[3] = NumOps.FromDouble(0.3535533905932738);
            reconstructionLowPass[4] = NumOps.FromDouble(0.1767766952966369);
            reconstructionLowPass[5] = NumOps.FromDouble(0);

            reconstructionHighPass[0] = NumOps.FromDouble(0);
            reconstructionHighPass[1] = NumOps.FromDouble(-0.0069053396600248);
            reconstructionHighPass[2] = NumOps.FromDouble(0.0138106793200496);
            reconstructionHighPass[3] = NumOps.FromDouble(0.0469563096881692);
            reconstructionHighPass[4] = NumOps.FromDouble(-0.1077232986963880);
            reconstructionHighPass[5] = NumOps.FromDouble(-0.1697627774134332);
            reconstructionHighPass[6] = NumOps.FromDouble(0.4474660099696121);
            reconstructionHighPass[7] = NumOps.FromDouble(-0.9667475524034829);
            reconstructionHighPass[8] = NumOps.FromDouble(0.4474660099696121);
            reconstructionHighPass[9] = NumOps.FromDouble(-0.1697627774134332);
            reconstructionHighPass[10] = NumOps.FromDouble(-0.1077232986963880);
            reconstructionHighPass[11] = NumOps.FromDouble(0.0469563096881692);
            reconstructionHighPass[12] = NumOps.FromDouble(0.0138106793200496);
            reconstructionHighPass[13] = NumOps.FromDouble(-0.0069053396600248);

            return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
        }

        /// <summary>
        /// Gets the filter coefficients for the ReverseBior28 wavelet.
        /// </summary>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the ReverseBior28 wavelet, which has 2 vanishing
        /// moments in the decomposition filters and 8 vanishing moments in the reconstruction filters.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides the specific coefficients for the ReverseBior28 wavelet.
        /// 
        /// The ReverseBior28 wavelet:
        /// - Has extremely asymmetric properties (2 vanishing moments on one side, 8 on the other)
        /// - The "28" means it has 2 vanishing moments in decomposition and 8 in reconstruction
        /// - Provides superior frequency resolution during reconstruction
        /// 
        /// This wavelet is particularly well-suited for applications requiring very precise frequency
        /// analysis during reconstruction, such as audio feature extraction or harmonic analysis, 
        /// where the exact frequency components need to be preserved with high accuracy.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior28Coefficients()
        {
            var decompositionLowPass = new Vector<T>(16);
            var decompositionHighPass = new Vector<T>(6);
            var reconstructionLowPass = new Vector<T>(6);
            var reconstructionHighPass = new Vector<T>(16);

            decompositionLowPass[0] = NumOps.FromDouble(0);
            decompositionLowPass[1] = NumOps.FromDouble(0.0015105430506304422);
            decompositionLowPass[2] = NumOps.FromDouble(-0.0030210861012608843);
            decompositionLowPass[3] = NumOps.FromDouble(-0.012947511862546647);
            decompositionLowPass[4] = NumOps.FromDouble(0.02891610982635418);
            decompositionLowPass[5] = NumOps.FromDouble(0.052998481890690945);
            decompositionLowPass[6] = NumOps.FromDouble(-0.13491307360773608);
            decompositionLowPass[7] = NumOps.FromDouble(-0.16382918343409025);
            decompositionLowPass[8] = NumOps.FromDouble(0.4625714404759166);
            decompositionLowPass[9] = NumOps.FromDouble(0.9516421218971786);
            decompositionLowPass[10] = NumOps.FromDouble(0.4625714404759166);
            decompositionLowPass[11] = NumOps.FromDouble(-0.16382918343409025);
            decompositionLowPass[12] = NumOps.FromDouble(-0.13491307360773608);
            decompositionLowPass[13] = NumOps.FromDouble(0.052998481890690945);
            decompositionLowPass[14] = NumOps.FromDouble(0.02891610982635418);
            decompositionLowPass[15] = NumOps.FromDouble(-0.012947511862546647);

            decompositionHighPass[0] = NumOps.FromDouble(0);
            decompositionHighPass[1] = NumOps.FromDouble(0);
            decompositionHighPass[2] = NumOps.FromDouble(0.3535533905932738);
            decompositionHighPass[3] = NumOps.FromDouble(-0.7071067811865476);
            decompositionHighPass[4] = NumOps.FromDouble(0.3535533905932738);
            decompositionHighPass[5] = NumOps.FromDouble(0);

            reconstructionLowPass[0] = NumOps.FromDouble(0);
            reconstructionLowPass[1] = NumOps.FromDouble(0);
            reconstructionLowPass[2] = NumOps.FromDouble(-0.3535533905932738);
            reconstructionLowPass[3] = NumOps.FromDouble(-0.7071067811865476);
            reconstructionLowPass[4] = NumOps.FromDouble(-0.3535533905932738);
            reconstructionLowPass[5] = NumOps.FromDouble(0);

            reconstructionHighPass[0] = NumOps.FromDouble(0);
            reconstructionHighPass[1] = NumOps.FromDouble(-0.0015105430506304422);
            reconstructionHighPass[2] = NumOps.FromDouble(-0.0030210861012608843);
            reconstructionHighPass[3] = NumOps.FromDouble(0.012947511862546647);
            reconstructionHighPass[4] = NumOps.FromDouble(0.02891610982635418);
            reconstructionHighPass[5] = NumOps.FromDouble(-0.052998481890690945);
            reconstructionHighPass[6] = NumOps.FromDouble(-0.13491307360773608);
            reconstructionHighPass[7] = NumOps.FromDouble(0.16382918343409025);
            reconstructionHighPass[8] = NumOps.FromDouble(0.4625714404759166);
            reconstructionHighPass[9] = NumOps.FromDouble(-0.9516421218971786);
            reconstructionHighPass[10] = NumOps.FromDouble(0.4625714404759166);
            reconstructionHighPass[11] = NumOps.FromDouble(0.16382918343409025);
            reconstructionHighPass[12] = NumOps.FromDouble(-0.13491307360773608);
            reconstructionHighPass[13] = NumOps.FromDouble(-0.052998481890690945);
            reconstructionHighPass[14] = NumOps.FromDouble(0.02891610982635418);
            reconstructionHighPass[15] = NumOps.FromDouble(0.012947511862546647);

            return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
        }

        /// <summary>
        /// Gets the filter coefficients for the ReverseBior31 wavelet.
        /// </summary>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the ReverseBior31 wavelet, which has 3 vanishing
        /// moments in the decomposition filters and 1 vanishing moment in the reconstruction filters.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides the specific coefficients for the ReverseBior31 wavelet.
        /// 
        /// The ReverseBior31 wavelet:
        /// - Has asymmetric properties with emphasis on decomposition (3 vanishing moments on one side, 1 on the other)
        /// - The "31" means it has 3 vanishing moments in decomposition and 1 in reconstruction
        /// - Provides better frequency localization during decomposition than reconstruction
        /// 
        /// This wavelet is useful when you need more precise frequency analysis during the decomposition
        /// phase but care more about time localization during reconstruction, like in some pattern
        /// recognition or feature extraction applications.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior31Coefficients()
        {
            var decompositionLowPass = new Vector<T>(4);
            var decompositionHighPass = new Vector<T>(8);
            var reconstructionLowPass = new Vector<T>(8);
            var reconstructionHighPass = new Vector<T>(4);

            decompositionLowPass[0] = NumOps.FromDouble(-0.3535533905932738);
            decompositionLowPass[1] = NumOps.FromDouble(1.0606601717798214);
            decompositionLowPass[2] = NumOps.FromDouble(1.0606601717798214);
            decompositionLowPass[3] = NumOps.FromDouble(-0.3535533905932738);

            decompositionHighPass[0] = NumOps.FromDouble(-0.0662912607362388);
            decompositionHighPass[1] = NumOps.FromDouble(0.1988737822087164);
            decompositionHighPass[2] = NumOps.FromDouble(-0.1546796083845572);
            decompositionHighPass[3] = NumOps.FromDouble(-0.9943689110435825);
            decompositionHighPass[4] = NumOps.FromDouble(0.9943689110435825);
            decompositionHighPass[5] = NumOps.FromDouble(0.1546796083845572);
            decompositionHighPass[6] = NumOps.FromDouble(-0.1988737822087164);
            decompositionHighPass[7] = NumOps.FromDouble(0.0662912607362388);

            reconstructionLowPass[0] = NumOps.FromDouble(0.0662912607362388);
            reconstructionLowPass[1] = NumOps.FromDouble(0.1988737822087164);
            reconstructionLowPass[2] = NumOps.FromDouble(0.1546796083845572);
            reconstructionLowPass[3] = NumOps.FromDouble(-0.9943689110435825);
            reconstructionLowPass[4] = NumOps.FromDouble(-0.9943689110435825);
            reconstructionLowPass[5] = NumOps.FromDouble(0.1546796083845572);
            reconstructionLowPass[6] = NumOps.FromDouble(0.1988737822087164);
            reconstructionLowPass[7] = NumOps.FromDouble(0.0662912607362388);

            reconstructionHighPass[0] = NumOps.FromDouble(-0.3535533905932738);
            reconstructionHighPass[1] = NumOps.FromDouble(-1.0606601717798214);
            reconstructionHighPass[2] = NumOps.FromDouble(1.0606601717798214);
            reconstructionHighPass[3] = NumOps.FromDouble(0.3535533905932738);

            return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
        }

        /// <summary>
        /// Gets the filter coefficients for the ReverseBior33 wavelet.
        /// </summary>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the ReverseBior33 wavelet, which has 3 vanishing
        /// moments in both the decomposition and reconstruction filters.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides the specific coefficients for the ReverseBior33 wavelet.
        /// 
        /// The ReverseBior33 wavelet:
        /// - Has symmetric properties (3 vanishing moments on both sides)
        /// - The "33" means it has 3 vanishing moments in both decomposition and reconstruction
        /// - Provides a balance between time and frequency resolution
        /// 
        /// This wavelet is well-suited for general-purpose signal analysis and processing, offering
        /// a good compromise between localization in time and frequency domains.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior33Coefficients()
        {
            var decompositionLowPass = new Vector<T>(8);
            var decompositionHighPass = new Vector<T>(8);
            var reconstructionLowPass = new Vector<T>(8);
            var reconstructionHighPass = new Vector<T>(8);

            decompositionLowPass[0] = NumOps.FromDouble(0.0352262918857095);
            decompositionLowPass[1] = NumOps.FromDouble(-0.0854412738820267);
            decompositionLowPass[2] = NumOps.FromDouble(-0.1350110200102546);
            decompositionLowPass[3] = NumOps.FromDouble(0.4598775021184914);
            decompositionLowPass[4] = NumOps.FromDouble(0.8068915093110924);
            decompositionLowPass[5] = NumOps.FromDouble(0.3326705529500825);
            decompositionLowPass[6] = NumOps.FromDouble(-0.0279837694168599);
            decompositionLowPass[7] = NumOps.FromDouble(-0.0105974017850690);

            decompositionHighPass[0] = NumOps.FromDouble(0);
            decompositionHighPass[1] = NumOps.FromDouble(0);
            decompositionHighPass[2] = NumOps.FromDouble(-0.1767766952966369);
            decompositionHighPass[3] = NumOps.FromDouble(0.5303300858899107);
            decompositionHighPass[4] = NumOps.FromDouble(-0.5303300858899107);
            decompositionHighPass[5] = NumOps.FromDouble(0.1767766952966369);
            decompositionHighPass[6] = NumOps.FromDouble(0);
            decompositionHighPass[7] = NumOps.FromDouble(0);

            reconstructionLowPass[0] = NumOps.FromDouble(0);
            reconstructionLowPass[1] = NumOps.FromDouble(0);
            reconstructionLowPass[2] = NumOps.FromDouble(0.1767766952966369);
            reconstructionLowPass[3] = NumOps.FromDouble(0.5303300858899107);
            reconstructionLowPass[4] = NumOps.FromDouble(0.5303300858899107);
            reconstructionLowPass[5] = NumOps.FromDouble(0.1767766952966369);
            reconstructionLowPass[6] = NumOps.FromDouble(0);
            reconstructionLowPass[7] = NumOps.FromDouble(0);

            reconstructionHighPass[0] = NumOps.FromDouble(-0.0105974017850690);
            reconstructionHighPass[1] = NumOps.FromDouble(0.0279837694168599);
            reconstructionHighPass[2] = NumOps.FromDouble(0.3326705529500825);
            reconstructionHighPass[3] = NumOps.FromDouble(-0.8068915093110924);
            reconstructionHighPass[4] = NumOps.FromDouble(0.4598775021184914);
            reconstructionHighPass[5] = NumOps.FromDouble(0.1350110200102546);
            reconstructionHighPass[6] = NumOps.FromDouble(-0.0854412738820267);
            reconstructionHighPass[7] = NumOps.FromDouble(-0.0352262918857095);

            return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
        }

        /// <summary>
        /// Gets the filter coefficients for the ReverseBior35 wavelet.
        /// </summary>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the ReverseBior35 wavelet, which has 3 vanishing
        /// moments in the decomposition filters and 5 vanishing moments in the reconstruction filters.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides the specific coefficients for the ReverseBior35 wavelet.
        /// 
        /// The ReverseBior35 wavelet:
        /// - Has asymmetric properties (3 vanishing moments on one side, 5 on the other)
        /// - The "35" means it has 3 vanishing moments in decomposition and 5 in reconstruction
        /// - Offers improved frequency resolution during reconstruction compared to ReverseBior33
        /// 
        /// This wavelet is useful for applications where more precise frequency analysis is needed
        /// during reconstruction, while maintaining good time localization during decomposition.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior35Coefficients()
        {
            var decompositionLowPass = new Vector<T>(12);
            var decompositionHighPass = new Vector<T>(12);
            var reconstructionLowPass = new Vector<T>(12);
            var reconstructionHighPass = new Vector<T>(12);

            decompositionLowPass[0] = NumOps.FromDouble(-0.0130514548443985);
            decompositionLowPass[1] = NumOps.FromDouble(0.0307358671058437);
            decompositionLowPass[2] = NumOps.FromDouble(0.0686539440891211);
            decompositionLowPass[3] = NumOps.FromDouble(-0.1485354424027703);
            decompositionLowPass[4] = NumOps.FromDouble(-0.2746482511903850);
            decompositionLowPass[5] = NumOps.FromDouble(0.2746482511903850);
            decompositionLowPass[6] = NumOps.FromDouble(0.7366601814282105);
            decompositionLowPass[7] = NumOps.FromDouble(0.4976186676320155);
            decompositionLowPass[8] = NumOps.FromDouble(0.0746831846544829);
            decompositionLowPass[9] = NumOps.FromDouble(-0.0305795375195906);
            decompositionLowPass[10] = NumOps.FromDouble(-0.0126815724766769);
            decompositionLowPass[11] = NumOps.FromDouble(0.0010131419871576);

            decompositionHighPass[0] = NumOps.FromDouble(0);
            decompositionHighPass[1] = NumOps.FromDouble(0);
            decompositionHighPass[2] = NumOps.FromDouble(0);
            decompositionHighPass[3] = NumOps.FromDouble(0.0662912607362388);
            decompositionHighPass[4] = NumOps.FromDouble(-0.1988737822087164);
            decompositionHighPass[5] = NumOps.FromDouble(0.1546796083845572);
            decompositionHighPass[6] = NumOps.FromDouble(0.9943689110435825);
            decompositionHighPass[7] = NumOps.FromDouble(-0.1546796083845572);
            decompositionHighPass[8] = NumOps.FromDouble(-0.1988737822087164);
            decompositionHighPass[9] = NumOps.FromDouble(0.0662912607362388);
            decompositionHighPass[10] = NumOps.FromDouble(0);
            decompositionHighPass[11] = NumOps.FromDouble(0);

            reconstructionLowPass[0] = NumOps.FromDouble(0);
            reconstructionLowPass[1] = NumOps.FromDouble(0);
            reconstructionLowPass[2] = NumOps.FromDouble(0);
            reconstructionLowPass[3] = NumOps.FromDouble(-0.0662912607362388);
            reconstructionLowPass[4] = NumOps.FromDouble(-0.1988737822087164);
            reconstructionLowPass[5] = NumOps.FromDouble(0.1546796083845572);
            reconstructionLowPass[6] = NumOps.FromDouble(0.9943689110435825);
            reconstructionLowPass[7] = NumOps.FromDouble(0.1546796083845572);
            reconstructionLowPass[8] = NumOps.FromDouble(-0.1988737822087164);
            reconstructionLowPass[9] = NumOps.FromDouble(-0.0662912607362388);
            reconstructionLowPass[10] = NumOps.FromDouble(0);
            reconstructionLowPass[11] = NumOps.FromDouble(0);

            reconstructionHighPass[0] = NumOps.FromDouble(0.0010131419871576);
            reconstructionHighPass[1] = NumOps.FromDouble(0.0126815724766769);
            reconstructionHighPass[2] = NumOps.FromDouble(-0.0305795375195906);
            reconstructionHighPass[3] = NumOps.FromDouble(-0.0746831846544829);
            reconstructionHighPass[4] = NumOps.FromDouble(0.4976186676320155);
            reconstructionHighPass[5] = NumOps.FromDouble(-0.7366601814282105);
            reconstructionHighPass[6] = NumOps.FromDouble(0.2746482511903850);
            reconstructionHighPass[7] = NumOps.FromDouble(0.2746482511903850);
            reconstructionHighPass[8] = NumOps.FromDouble(-0.1485354424027703);
            reconstructionHighPass[9] = NumOps.FromDouble(-0.0686539440891211);
            reconstructionHighPass[10] = NumOps.FromDouble(0.0307358671058437);
            reconstructionHighPass[11] = NumOps.FromDouble(0.0130514548443985);

            return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
        }

        /// <summary>
        /// Gets the filter coefficients for the ReverseBior37 wavelet.
        /// </summary>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the ReverseBior37 wavelet, which has 3 vanishing
        /// moments in the decomposition filters and 7 vanishing moments in the reconstruction filters.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides the specific coefficients for the ReverseBior37 wavelet.
        /// 
        /// The ReverseBior37 wavelet:
        /// - Has highly asymmetric properties (3 vanishing moments on one side, 7 on the other)
        /// - The "37" means it has 3 vanishing moments in decomposition and 7 in reconstruction
        /// - Provides even better frequency resolution during reconstruction than ReverseBior35
        /// 
        /// This wavelet is particularly useful for applications requiring very accurate frequency
        /// analysis during reconstruction, while still maintaining good time localization in decomposition.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior37Coefficients()
        {
            var decompositionLowPass = new Vector<T>(12);
            var decompositionHighPass = new Vector<T>(8);
            var reconstructionLowPass = new Vector<T>(8);
            var reconstructionHighPass = new Vector<T>(12);

            decompositionLowPass[0] = NumOps.FromDouble(0.0030210861012608843);
            decompositionLowPass[1] = NumOps.FromDouble(-0.009063258303782653);
            decompositionLowPass[2] = NumOps.FromDouble(-0.01683176542131064);
            decompositionLowPass[3] = NumOps.FromDouble(0.074663985074019);
            decompositionLowPass[4] = NumOps.FromDouble(0.03133297870736289);
            decompositionLowPass[5] = NumOps.FromDouble(-0.301159125922835);
            decompositionLowPass[6] = NumOps.FromDouble(-0.026499240945345472);
            decompositionLowPass[7] = NumOps.FromDouble(0.9516421218971786);
            decompositionLowPass[8] = NumOps.FromDouble(0.9516421218971786);
            decompositionLowPass[9] = NumOps.FromDouble(-0.026499240945345472);
            decompositionLowPass[10] = NumOps.FromDouble(-0.301159125922835);
            decompositionLowPass[11] = NumOps.FromDouble(0.03133297870736289);

            decompositionHighPass[0] = NumOps.FromDouble(0);
            decompositionHighPass[1] = NumOps.FromDouble(0);
            decompositionHighPass[2] = NumOps.FromDouble(-0.1767766952966369);
            decompositionHighPass[3] = NumOps.FromDouble(0.5303300858899107);
            decompositionHighPass[4] = NumOps.FromDouble(-0.5303300858899107);
            decompositionHighPass[5] = NumOps.FromDouble(0.1767766952966369);
            decompositionHighPass[6] = NumOps.FromDouble(0);
            decompositionHighPass[7] = NumOps.FromDouble(0);

            reconstructionLowPass[0] = NumOps.FromDouble(0);
            reconstructionLowPass[1] = NumOps.FromDouble(0);
            reconstructionLowPass[2] = NumOps.FromDouble(0.1767766952966369);
            reconstructionLowPass[3] = NumOps.FromDouble(0.5303300858899107);
            reconstructionLowPass[4] = NumOps.FromDouble(0.5303300858899107);
            reconstructionLowPass[5] = NumOps.FromDouble(0.1767766952966369);
            reconstructionLowPass[6] = NumOps.FromDouble(0);
            reconstructionLowPass[7] = NumOps.FromDouble(0);

            reconstructionHighPass[0] = NumOps.FromDouble(0.0030210861012608843);
            reconstructionHighPass[1] = NumOps.FromDouble(0.009063258303782653);
            reconstructionHighPass[2] = NumOps.FromDouble(-0.01683176542131064);
            reconstructionHighPass[3] = NumOps.FromDouble(-0.074663985074019);
            reconstructionHighPass[4] = NumOps.FromDouble(0.03133297870736289);
            reconstructionHighPass[5] = NumOps.FromDouble(0.301159125922835);
            reconstructionHighPass[6] = NumOps.FromDouble(-0.026499240945345472);
            reconstructionHighPass[7] = NumOps.FromDouble(-0.9516421218971786);
            reconstructionHighPass[8] = NumOps.FromDouble(0.9516421218971786);
            reconstructionHighPass[9] = NumOps.FromDouble(0.026499240945345472);
            reconstructionHighPass[10] = NumOps.FromDouble(-0.301159125922835);
            reconstructionHighPass[11] = NumOps.FromDouble(-0.03133297870736289);

            return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
        }

        /// <summary>
        /// Gets the filter coefficients for the ReverseBior39 wavelet.
        /// </summary>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the ReverseBior39 wavelet, which has 3 vanishing
        /// moments in the decomposition filters and 9 vanishing moments in the reconstruction filters.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides the specific coefficients for the ReverseBior39 wavelet.
        /// 
        /// The ReverseBior39 wavelet:
        /// - Has extremely asymmetric properties (3 vanishing moments on one side, 9 on the other)
        /// - The "39" means it has 3 vanishing moments in decomposition and 9 in reconstruction
        /// - Offers superior frequency resolution during reconstruction
        /// 
        /// This wavelet is ideal for applications requiring extremely precise frequency analysis
        /// during reconstruction, such as high-fidelity audio processing or detailed spectral analysis,
        /// while still maintaining reasonable time localization during decomposition.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior39Coefficients()
        {
            var decompositionLowPass = new Vector<T>(16);
            var decompositionHighPass = new Vector<T>(10);
            var reconstructionLowPass = new Vector<T>(10);
            var reconstructionHighPass = new Vector<T>(16);

            decompositionLowPass[0] = NumOps.FromDouble(-0.000679744372783699);
            decompositionLowPass[1] = NumOps.FromDouble(0.002039233118351097);
            decompositionLowPass[2] = NumOps.FromDouble(0.005060319219611981);
            decompositionLowPass[3] = NumOps.FromDouble(-0.020618912641105536);
            decompositionLowPass[4] = NumOps.FromDouble(-0.014112787930175846);
            decompositionLowPass[5] = NumOps.FromDouble(0.09913478249423216);
            decompositionLowPass[6] = NumOps.FromDouble(0.012300136269419315);
            decompositionLowPass[7] = NumOps.FromDouble(-0.32019196836077857);
            decompositionLowPass[8] = NumOps.FromDouble(0.0020500227115698858);
            decompositionLowPass[9] = NumOps.FromDouble(0.9421257006782068);
            decompositionLowPass[10] = NumOps.FromDouble(0.9421257006782068);
            decompositionLowPass[11] = NumOps.FromDouble(0.0020500227115698858);
            decompositionLowPass[12] = NumOps.FromDouble(-0.32019196836077857);
            decompositionLowPass[13] = NumOps.FromDouble(0.012300136269419315);
            decompositionLowPass[14] = NumOps.FromDouble(0.09913478249423216);
            decompositionLowPass[15] = NumOps.FromDouble(-0.014112787930175846);

            decompositionHighPass[0] = NumOps.FromDouble(0);
            decompositionHighPass[1] = NumOps.FromDouble(0);
            decompositionHighPass[2] = NumOps.FromDouble(0);
            decompositionHighPass[3] = NumOps.FromDouble(0.0662912607362388);
            decompositionHighPass[4] = NumOps.FromDouble(0.1988737822087164);
            decompositionHighPass[5] = NumOps.FromDouble(0.1988737822087164);
            decompositionHighPass[6] = NumOps.FromDouble(0.0662912607362388);
            decompositionHighPass[7] = NumOps.FromDouble(0);
            decompositionHighPass[8] = NumOps.FromDouble(0);
            decompositionHighPass[9] = NumOps.FromDouble(0);

            reconstructionLowPass[0] = NumOps.FromDouble(0);
            reconstructionLowPass[1] = NumOps.FromDouble(0);
            reconstructionLowPass[2] = NumOps.FromDouble(0);
            reconstructionLowPass[3] = NumOps.FromDouble(-0.0662912607362388);
            reconstructionLowPass[4] = NumOps.FromDouble(0.1988737822087164);
            reconstructionLowPass[5] = NumOps.FromDouble(-0.1988737822087164);
            reconstructionLowPass[6] = NumOps.FromDouble(0.0662912607362388);
            reconstructionLowPass[7] = NumOps.FromDouble(0);
            reconstructionLowPass[8] = NumOps.FromDouble(0);
            reconstructionLowPass[9] = NumOps.FromDouble(0);

            reconstructionHighPass[0] = NumOps.FromDouble(-0.014112787930175846);
            reconstructionHighPass[1] = NumOps.FromDouble(-0.09913478249423216);
            reconstructionHighPass[2] = NumOps.FromDouble(0.012300136269419315);
            reconstructionHighPass[3] = NumOps.FromDouble(0.32019196836077857);
            reconstructionHighPass[4] = NumOps.FromDouble(0.0020500227115698858);
            reconstructionHighPass[5] = NumOps.FromDouble(-0.9421257006782068);
            reconstructionHighPass[6] = NumOps.FromDouble(0.9421257006782068);
            reconstructionHighPass[7] = NumOps.FromDouble(-0.0020500227115698858);
            reconstructionHighPass[8] = NumOps.FromDouble(-0.32019196836077857);
            reconstructionHighPass[9] = NumOps.FromDouble(-0.012300136269419315);
            reconstructionHighPass[10] = NumOps.FromDouble(0.09913478249423216);
            reconstructionHighPass[11] = NumOps.FromDouble(0.014112787930175846);
            reconstructionHighPass[12] = NumOps.FromDouble(-0.020618912641105536);
            reconstructionHighPass[13] = NumOps.FromDouble(-0.005060319219611981);
            reconstructionHighPass[14] = NumOps.FromDouble(0.002039233118351097);
            reconstructionHighPass[15] = NumOps.FromDouble(0.000679744372783699);

            return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
        }

        /// <summary>
        /// Gets the filter coefficients for the ReverseBior44 wavelet.
        /// </summary>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the ReverseBior44 wavelet, which has 4 vanishing
        /// moments in both the decomposition and reconstruction filters.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides the specific coefficients for the ReverseBior44 wavelet.
        /// 
        /// The ReverseBior44 wavelet:
        /// - Has symmetric properties (4 vanishing moments on both sides)
        /// - The "44" means it has 4 vanishing moments in both decomposition and reconstruction
        /// - Provides a good balance between time and frequency resolution, with improved smoothness
        /// 
        /// This wavelet is well-suited for applications requiring smooth analysis and reconstruction,
        /// offering better frequency localization than ReverseBior33 while maintaining symmetry.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior44Coefficients()
        {
            var decompositionLowPass = new Vector<T>(10);
            var decompositionHighPass = new Vector<T>(10);
            var reconstructionLowPass = new Vector<T>(10);
            var reconstructionHighPass = new Vector<T>(10);

            decompositionLowPass[0] = NumOps.FromDouble(0);
            decompositionLowPass[1] = NumOps.FromDouble(0.03782845550699535);
            decompositionLowPass[2] = NumOps.FromDouble(-0.023849465019380);
            decompositionLowPass[3] = NumOps.FromDouble(-0.11062440441842);
            decompositionLowPass[4] = NumOps.FromDouble(0.37740285561265);
            decompositionLowPass[5] = NumOps.FromDouble(0.85269867900940);
            decompositionLowPass[6] = NumOps.FromDouble(0.37740285561265);
            decompositionLowPass[7] = NumOps.FromDouble(-0.11062440441842);
            decompositionLowPass[8] = NumOps.FromDouble(-0.023849465019380);
            decompositionLowPass[9] = NumOps.FromDouble(0.03782845550699535);

            decompositionHighPass[0] = NumOps.FromDouble(0);
            decompositionHighPass[1] = NumOps.FromDouble(-0.06453888262893856);
            decompositionHighPass[2] = NumOps.FromDouble(0.04068941760955867);
            decompositionHighPass[3] = NumOps.FromDouble(0.41809227322161724);
            decompositionHighPass[4] = NumOps.FromDouble(-0.7884856164056651);
            decompositionHighPass[5] = NumOps.FromDouble(0.4180922732216172);
            decompositionHighPass[6] = NumOps.FromDouble(0.040689417609558675);
            decompositionHighPass[7] = NumOps.FromDouble(-0.06453888262893856);
            decompositionHighPass[8] = NumOps.FromDouble(0);
            decompositionHighPass[9] = NumOps.FromDouble(0);

            reconstructionLowPass[0] = NumOps.FromDouble(0);
            reconstructionLowPass[1] = NumOps.FromDouble(-0.06453888262893856);
            reconstructionLowPass[2] = NumOps.FromDouble(-0.04068941760955867);
            reconstructionLowPass[3] = NumOps.FromDouble(0.41809227322161724);
            reconstructionLowPass[4] = NumOps.FromDouble(0.7884856164056651);
            reconstructionLowPass[5] = NumOps.FromDouble(0.4180922732216172);
            reconstructionLowPass[6] = NumOps.FromDouble(-0.040689417609558675);
            reconstructionLowPass[7] = NumOps.FromDouble(-0.06453888262893856);
            reconstructionLowPass[8] = NumOps.FromDouble(0);
            reconstructionLowPass[9] = NumOps.FromDouble(0);

            reconstructionHighPass[0] = NumOps.FromDouble(0);
            reconstructionHighPass[1] = NumOps.FromDouble(0.03782845550699535);
            reconstructionHighPass[2] = NumOps.FromDouble(0.023849465019380);
            reconstructionHighPass[3] = NumOps.FromDouble(-0.11062440441842);
            reconstructionHighPass[4] = NumOps.FromDouble(-0.37740285561265);
            reconstructionHighPass[5] = NumOps.FromDouble(0.85269867900940);
            reconstructionHighPass[6] = NumOps.FromDouble(-0.37740285561265);
            reconstructionHighPass[7] = NumOps.FromDouble(-0.11062440441842);
            reconstructionHighPass[8] = NumOps.FromDouble(0.023849465019380);
            reconstructionHighPass[9] = NumOps.FromDouble(0.03782845550699535);

            return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
        }

        /// <summary>
        /// Gets the filter coefficients for the ReverseBior46 wavelet.
        /// </summary>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the ReverseBior46 wavelet, which has 4 vanishing
        /// moments in the decomposition filters and 6 vanishing moments in the reconstruction filters.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides the specific coefficients for the ReverseBior46 wavelet.
        /// 
        /// The ReverseBior46 wavelet:
        /// - Has asymmetric properties (4 vanishing moments on one side, 6 on the other)
        /// - The "46" means it has 4 vanishing moments in decomposition and 6 in reconstruction
        /// - Offers improved frequency resolution during reconstruction compared to ReverseBior44
        /// 
        /// This wavelet is useful for applications requiring smooth decomposition and more precise
        /// frequency analysis during reconstruction, balancing time-frequency localization asymmetrically.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior46Coefficients()
        {
            var decompositionLowPass = new Vector<T>(14);
            var decompositionHighPass = new Vector<T>(10);
            var reconstructionLowPass = new Vector<T>(10);
            var reconstructionHighPass = new Vector<T>(14);

            decompositionLowPass[0] = NumOps.FromDouble(0.0019088317364812906);
            decompositionLowPass[1] = NumOps.FromDouble(-0.0019142861290887667);
            decompositionLowPass[2] = NumOps.FromDouble(-0.016990639867602342);
            decompositionLowPass[3] = NumOps.FromDouble(0.01193456527972926);
            decompositionLowPass[4] = NumOps.FromDouble(0.04973290349094079);
            decompositionLowPass[5] = NumOps.FromDouble(-0.07726317316720414);
            decompositionLowPass[6] = NumOps.FromDouble(-0.09405920349573646);
            decompositionLowPass[7] = NumOps.FromDouble(0.4207962846098268);
            decompositionLowPass[8] = NumOps.FromDouble(0.8259229974584023);
            decompositionLowPass[9] = NumOps.FromDouble(0.4207962846098268);
            decompositionLowPass[10] = NumOps.FromDouble(-0.09405920349573646);
            decompositionLowPass[11] = NumOps.FromDouble(-0.07726317316720414);
            decompositionLowPass[12] = NumOps.FromDouble(0.04973290349094079);
            decompositionLowPass[13] = NumOps.FromDouble(0.01193456527972926);

            decompositionHighPass[0] = NumOps.FromDouble(0);
            decompositionHighPass[1] = NumOps.FromDouble(0);
            decompositionHighPass[2] = NumOps.FromDouble(0.03782845550699535);
            decompositionHighPass[3] = NumOps.FromDouble(-0.023849465019380);
            decompositionHighPass[4] = NumOps.FromDouble(-0.11062440441842);
            decompositionHighPass[5] = NumOps.FromDouble(0.37740285561265);
            decompositionHighPass[6] = NumOps.FromDouble(-0.85269867900940);
            decompositionHighPass[7] = NumOps.FromDouble(0.37740285561265);
            decompositionHighPass[8] = NumOps.FromDouble(-0.11062440441842);
            decompositionHighPass[9] = NumOps.FromDouble(-0.023849465019380);

            reconstructionLowPass[0] = NumOps.FromDouble(0);
            reconstructionLowPass[1] = NumOps.FromDouble(0);
            reconstructionLowPass[2] = NumOps.FromDouble(0.03782845550699535);
            reconstructionLowPass[3] = NumOps.FromDouble(0.023849465019380);
            reconstructionLowPass[4] = NumOps.FromDouble(-0.11062440441842);
            reconstructionLowPass[5] = NumOps.FromDouble(-0.37740285561265);
            reconstructionLowPass[6] = NumOps.FromDouble(0.85269867900940);
            reconstructionLowPass[7] = NumOps.FromDouble(-0.37740285561265);
            reconstructionLowPass[8] = NumOps.FromDouble(-0.11062440441842);
            reconstructionLowPass[9] = NumOps.FromDouble(0.023849465019380);

            reconstructionHighPass[0] = NumOps.FromDouble(0.0019088317364812906);
            reconstructionHighPass[1] = NumOps.FromDouble(0.0019142861290887667);
            reconstructionHighPass[2] = NumOps.FromDouble(-0.016990639867602342);
            reconstructionHighPass[3] = NumOps.FromDouble(-0.01193456527972926);
            reconstructionHighPass[4] = NumOps.FromDouble(0.04973290349094079);
            reconstructionHighPass[5] = NumOps.FromDouble(0.07726317316720414);
            reconstructionHighPass[6] = NumOps.FromDouble(-0.09405920349573646);
            reconstructionHighPass[7] = NumOps.FromDouble(-0.4207962846098268);
            reconstructionHighPass[8] = NumOps.FromDouble(0.8259229974584023);
            reconstructionHighPass[9] = NumOps.FromDouble(-0.4207962846098268);
            reconstructionHighPass[10] = NumOps.FromDouble(-0.09405920349573646);
            reconstructionHighPass[11] = NumOps.FromDouble(0.07726317316720414);
            reconstructionHighPass[12] = NumOps.FromDouble(0.04973290349094079);
            reconstructionHighPass[13] = NumOps.FromDouble(-0.01193456527972926);

            return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
        }

        /// <summary>
        /// Gets the filter coefficients for the ReverseBior48 wavelet.
        /// </summary>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the ReverseBior48 wavelet, which has 4 vanishing
        /// moments in the decomposition filters and 8 vanishing moments in the reconstruction filters.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides the specific coefficients for the ReverseBior48 wavelet.
        /// 
        /// The ReverseBior48 wavelet:
        /// - Has highly asymmetric properties (4 vanishing moments on one side, 8 on the other)
        /// - The "48" means it has 4 vanishing moments in decomposition and 8 in reconstruction
        /// - Provides superior frequency resolution during reconstruction
        /// 
        /// This wavelet is particularly well-suited for applications requiring smooth decomposition and
        /// very precise frequency analysis during reconstruction, such as detailed signal analysis where
        /// preserving high-frequency components during reconstruction is crucial.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior48Coefficients()
        {
            var decompositionLowPass = new Vector<T>(18);
            var decompositionHighPass = new Vector<T>(10);
            var reconstructionLowPass = new Vector<T>(10);
            var reconstructionHighPass = new Vector<T>(18);

            decompositionLowPass[0] = NumOps.FromDouble(0);
            decompositionLowPass[1] = NumOps.FromDouble(-0.0001174767841);
            decompositionLowPass[2] = NumOps.FromDouble(-0.0002349535682);
            decompositionLowPass[3] = NumOps.FromDouble(0.0013925484327);
            decompositionLowPass[4] = NumOps.FromDouble(0.0030931751602);
            decompositionLowPass[5] = NumOps.FromDouble(-0.0138017446325);
            decompositionLowPass[6] = NumOps.FromDouble(-0.0457246565401);
            decompositionLowPass[7] = NumOps.FromDouble(0.0687510184021);
            decompositionLowPass[8] = NumOps.FromDouble(0.3848874557553);
            decompositionLowPass[9] = NumOps.FromDouble(0.8525720202122);
            decompositionLowPass[10] = NumOps.FromDouble(0.3848874557553);
            decompositionLowPass[11] = NumOps.FromDouble(0.0687510184021);
            decompositionLowPass[12] = NumOps.FromDouble(-0.0457246565401);
            decompositionLowPass[13] = NumOps.FromDouble(-0.0138017446325);
            decompositionLowPass[14] = NumOps.FromDouble(0.0030931751602);
            decompositionLowPass[15] = NumOps.FromDouble(0.0013925484327);
            decompositionLowPass[16] = NumOps.FromDouble(-0.0002349535682);
            decompositionLowPass[17] = NumOps.FromDouble(-0.0001174767841);

            decompositionHighPass[0] = NumOps.FromDouble(0);
            decompositionHighPass[1] = NumOps.FromDouble(-0.0645388826289);
            decompositionHighPass[2] = NumOps.FromDouble(0.0406894176091);
            decompositionHighPass[3] = NumOps.FromDouble(0.4180922732222);
            decompositionHighPass[4] = NumOps.FromDouble(-0.7884856164057);
            decompositionHighPass[5] = NumOps.FromDouble(0.4180922732222);
            decompositionHighPass[6] = NumOps.FromDouble(0.0406894176091);
            decompositionHighPass[7] = NumOps.FromDouble(-0.0645388826289);
            decompositionHighPass[8] = NumOps.FromDouble(0);
            decompositionHighPass[9] = NumOps.FromDouble(0);

            reconstructionLowPass[0] = NumOps.FromDouble(0);
            reconstructionLowPass[1] = NumOps.FromDouble(0.0645388826289);
            reconstructionLowPass[2] = NumOps.FromDouble(0.0406894176091);
            reconstructionLowPass[3] = NumOps.FromDouble(-0.4180922732222);
            reconstructionLowPass[4] = NumOps.FromDouble(-0.7884856164057);
            reconstructionLowPass[5] = NumOps.FromDouble(-0.4180922732222);
            reconstructionLowPass[6] = NumOps.FromDouble(0.0406894176091);
            reconstructionLowPass[7] = NumOps.FromDouble(0.0645388826289);
            reconstructionLowPass[8] = NumOps.FromDouble(0);
            reconstructionLowPass[9] = NumOps.FromDouble(0);

            reconstructionHighPass[0] = NumOps.FromDouble(0);
            reconstructionHighPass[1] = NumOps.FromDouble(-0.0001174767841);
            reconstructionHighPass[2] = NumOps.FromDouble(0.0002349535682);
            reconstructionHighPass[3] = NumOps.FromDouble(0.0013925484327);
            reconstructionHighPass[4] = NumOps.FromDouble(-0.0030931751602);
            reconstructionHighPass[5] = NumOps.FromDouble(-0.0138017446325);
            reconstructionHighPass[6] = NumOps.FromDouble(0.0457246565401);
            reconstructionHighPass[7] = NumOps.FromDouble(0.0687510184021);
            reconstructionHighPass[8] = NumOps.FromDouble(-0.3848874557553);
            reconstructionHighPass[9] = NumOps.FromDouble(0.8525720202122);
            reconstructionHighPass[10] = NumOps.FromDouble(-0.3848874557553);
            reconstructionHighPass[11] = NumOps.FromDouble(0.0687510184021);
            reconstructionHighPass[12] = NumOps.FromDouble(0.0457246565401);
            reconstructionHighPass[13] = NumOps.FromDouble(-0.0138017446325);
            reconstructionHighPass[14] = NumOps.FromDouble(-0.0030931751602);
            reconstructionHighPass[15] = NumOps.FromDouble(0.0013925484327);
            reconstructionHighPass[16] = NumOps.FromDouble(0.0002349535682);
            reconstructionHighPass[17] = NumOps.FromDouble(-0.0001174767841);

            return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
        }

        /// <summary>
        /// Gets the filter coefficients for the ReverseBior55 wavelet.
        /// </summary>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the ReverseBior55 wavelet, which has 5 vanishing
        /// moments in both the decomposition and reconstruction filters.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides the specific coefficients for the ReverseBior55 wavelet.
        /// 
        /// The ReverseBior55 wavelet:
        /// - Has symmetric properties (5 vanishing moments on both sides)
        /// - The "55" means it has 5 vanishing moments in both decomposition and reconstruction
        /// - Provides a good balance between time and frequency resolution, with improved smoothness compared to ReverseBior44
        /// 
        /// This wavelet is well-suited for applications requiring smooth analysis and reconstruction,
        /// offering better frequency localization than ReverseBior44 while maintaining symmetry.
        /// It's particularly useful for signals with both smooth regions and sharp transitions.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior55Coefficients()
        {
            var decompositionLowPass = new Vector<T>(12);
            var decompositionHighPass = new Vector<T>(12);
            var reconstructionLowPass = new Vector<T>(12);
            var reconstructionHighPass = new Vector<T>(12);

            decompositionLowPass[0] = NumOps.FromDouble(0.0);
            decompositionLowPass[1] = NumOps.FromDouble(0.013456709459118716);
            decompositionLowPass[2] = NumOps.FromDouble(-0.002694966880111507);
            decompositionLowPass[3] = NumOps.FromDouble(-0.13670658466432914);
            decompositionLowPass[4] = NumOps.FromDouble(-0.09350469740093886);
            decompositionLowPass[5] = NumOps.FromDouble(0.47680326579848425);
            decompositionLowPass[6] = NumOps.FromDouble(0.8995061097486484);
            decompositionLowPass[7] = NumOps.FromDouble(0.47680326579848425);
            decompositionLowPass[8] = NumOps.FromDouble(-0.09350469740093886);
            decompositionLowPass[9] = NumOps.FromDouble(-0.13670658466432914);
            decompositionLowPass[10] = NumOps.FromDouble(-0.002694966880111507);
            decompositionLowPass[11] = NumOps.FromDouble(0.013456709459118716);

            decompositionHighPass[0] = NumOps.FromDouble(0.013456709459118716);
            decompositionHighPass[1] = NumOps.FromDouble(-0.002694966880111507);
            decompositionHighPass[2] = NumOps.FromDouble(-0.13670658466432914);
            decompositionHighPass[3] = NumOps.FromDouble(-0.09350469740093886);
            decompositionHighPass[4] = NumOps.FromDouble(0.47680326579848425);
            decompositionHighPass[5] = NumOps.FromDouble(-0.8995061097486484);
            decompositionHighPass[6] = NumOps.FromDouble(0.47680326579848425);
            decompositionHighPass[7] = NumOps.FromDouble(-0.09350469740093886);
            decompositionHighPass[8] = NumOps.FromDouble(-0.13670658466432914);
            decompositionHighPass[9] = NumOps.FromDouble(-0.002694966880111507);
            decompositionHighPass[10] = NumOps.FromDouble(0.013456709459118716);
            decompositionHighPass[11] = NumOps.FromDouble(0.0);

            reconstructionLowPass[0] = NumOps.FromDouble(0.013456709459118716);
            reconstructionLowPass[1] = NumOps.FromDouble(0.002694966880111507);
            reconstructionLowPass[2] = NumOps.FromDouble(-0.13670658466432914);
            reconstructionLowPass[3] = NumOps.FromDouble(0.09350469740093886);
            reconstructionLowPass[4] = NumOps.FromDouble(0.47680326579848425);
            reconstructionLowPass[5] = NumOps.FromDouble(0.8995061097486484);
            reconstructionLowPass[6] = NumOps.FromDouble(0.47680326579848425);
            reconstructionLowPass[7] = NumOps.FromDouble(0.09350469740093886);
            reconstructionLowPass[8] = NumOps.FromDouble(-0.13670658466432914);
            reconstructionLowPass[9] = NumOps.FromDouble(0.002694966880111507);
            reconstructionLowPass[10] = NumOps.FromDouble(0.013456709459118716);
            reconstructionLowPass[11] = NumOps.FromDouble(0.0);

            reconstructionHighPass[0] = NumOps.FromDouble(0.0);
            reconstructionHighPass[1] = NumOps.FromDouble(0.013456709459118716);
            reconstructionHighPass[2] = NumOps.FromDouble(-0.002694966880111507);
            reconstructionHighPass[3] = NumOps.FromDouble(-0.13670658466432914);
            reconstructionHighPass[4] = NumOps.FromDouble(-0.09350469740093886);
            reconstructionHighPass[5] = NumOps.FromDouble(0.47680326579848425);
            reconstructionHighPass[6] = NumOps.FromDouble(-0.8995061097486484);
            reconstructionHighPass[7] = NumOps.FromDouble(0.47680326579848425);
            reconstructionHighPass[8] = NumOps.FromDouble(-0.09350469740093886);
            reconstructionHighPass[9] = NumOps.FromDouble(-0.13670658466432914);
            reconstructionHighPass[10] = NumOps.FromDouble(-0.002694966880111507);
            reconstructionHighPass[11] = NumOps.FromDouble(0.013456709459118716);

            return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
        }

        /// <summary>
        /// Gets the filter coefficients for the ReverseBior68 wavelet.
        /// </summary>
        /// <returns>A tuple containing the decomposition and reconstruction filter coefficients.</returns>
        /// <remarks>
        /// <para>
        /// This method returns the filter coefficients for the ReverseBior68 wavelet, which has 6 vanishing
        /// moments in the decomposition filters and 8 vanishing moments in the reconstruction filters.
        /// </para>
        /// <para><b>For Beginners:</b> This helper method provides the specific coefficients for the ReverseBior68 wavelet.
        /// 
        /// The ReverseBior68 wavelet:
        /// - Has highly asymmetric properties (6 vanishing moments on one side, 8 on the other)
        /// - The "68" means it has 6 vanishing moments in decomposition and 8 in reconstruction
        /// - Provides superior frequency resolution during reconstruction while maintaining good time localization in decomposition
        /// 
        /// This wavelet is particularly well-suited for applications requiring smooth decomposition and
        /// very precise frequency analysis during reconstruction. It's useful for detailed signal analysis
        /// where preserving high-frequency components during reconstruction is crucial, while still
        /// maintaining good time localization during decomposition.
        /// </para>
        /// </remarks>
        private (Vector<T>, Vector<T>, Vector<T>, Vector<T>) GetReverseBior68Coefficients()
        {
            var decompositionLowPass = new Vector<T>(18);
            var decompositionHighPass = new Vector<T>(10);
            var reconstructionLowPass = new Vector<T>(10);
            var reconstructionHighPass = new Vector<T>(18);

            // Decomposition low-pass filter
            decompositionLowPass[0] = NumOps.FromDouble(0);
            decompositionLowPass[1] = NumOps.FromDouble(0.0001490583487665);
            decompositionLowPass[2] = NumOps.FromDouble(-0.0003179695108439);
            decompositionLowPass[3] = NumOps.FromDouble(-0.0018118519793764);
            decompositionLowPass[4] = NumOps.FromDouble(0.0047314665272548);
            decompositionLowPass[5] = NumOps.FromDouble(0.0087901063101452);
            decompositionLowPass[6] = NumOps.FromDouble(-0.0297451247861220);
            decompositionLowPass[7] = NumOps.FromDouble(-0.0736365678679802);
            decompositionLowPass[8] = NumOps.FromDouble(0.1485354836691763);
            decompositionLowPass[9] = NumOps.FromDouble(0.4675630700319812);
            decompositionLowPass[10] = NumOps.FromDouble(0.9667475524034829);
            decompositionLowPass[11] = NumOps.FromDouble(0.4675630700319812);
            decompositionLowPass[12] = NumOps.FromDouble(0.1485354836691763);
            decompositionLowPass[13] = NumOps.FromDouble(-0.0736365678679802);
            decompositionLowPass[14] = NumOps.FromDouble(-0.0297451247861220);
            decompositionLowPass[15] = NumOps.FromDouble(0.0087901063101452);
            decompositionLowPass[16] = NumOps.FromDouble(0.0047314665272548);
            decompositionLowPass[17] = NumOps.FromDouble(-0.0018118519793764);

            // Decomposition high-pass filter
            decompositionHighPass[0] = NumOps.FromDouble(0);
            decompositionHighPass[1] = NumOps.FromDouble(-0.0107148249460572);
            decompositionHighPass[2] = NumOps.FromDouble(0.0328921202609630);
            decompositionHighPass[3] = NumOps.FromDouble(0.0308560115845869);
            decompositionHighPass[4] = NumOps.FromDouble(-0.1870348117190931);
            decompositionHighPass[5] = NumOps.FromDouble(0.0279837694169839);
            decompositionHighPass[6] = NumOps.FromDouble(0.6308807679295904);
            decompositionHighPass[7] = NumOps.FromDouble(-0.7148465705525415);
            decompositionHighPass[8] = NumOps.FromDouble(0.2303778133088552);
            decompositionHighPass[9] = NumOps.FromDouble(0.0279837694169839);

            // Reconstruction low-pass filter
            reconstructionLowPass[0] = NumOps.FromDouble(0);
            reconstructionLowPass[1] = NumOps.FromDouble(0.0279837694169839);
            reconstructionLowPass[2] = NumOps.FromDouble(0.2303778133088552);
            reconstructionLowPass[3] = NumOps.FromDouble(0.7148465705525415);
            reconstructionLowPass[4] = NumOps.FromDouble(0.6308807679295904);
            reconstructionLowPass[5] = NumOps.FromDouble(0.0279837694169839);
            reconstructionLowPass[6] = NumOps.FromDouble(-0.1870348117190931);
            reconstructionLowPass[7] = NumOps.FromDouble(0.0308560115845869);
            reconstructionLowPass[8] = NumOps.FromDouble(0.0328921202609630);
            reconstructionLowPass[9] = NumOps.FromDouble(-0.0107148249460572);

            // Reconstruction high-pass filter
            reconstructionHighPass[0] = NumOps.FromDouble(0);
            reconstructionHighPass[1] = NumOps.FromDouble(-0.0018118519793764);
            reconstructionHighPass[2] = NumOps.FromDouble(0.0047314665272548);
            reconstructionHighPass[3] = NumOps.FromDouble(0.0087901063101452);
            reconstructionHighPass[4] = NumOps.FromDouble(-0.0297451247861220);
            reconstructionHighPass[5] = NumOps.FromDouble(-0.0736365678679802);
            reconstructionHighPass[6] = NumOps.FromDouble(0.1485354836691763);
            reconstructionHighPass[7] = NumOps.FromDouble(0.4675630700319812);
            reconstructionHighPass[8] = NumOps.FromDouble(-0.9667475524034829);
            reconstructionHighPass[9] = NumOps.FromDouble(0.4675630700319812);
            reconstructionHighPass[10] = NumOps.FromDouble(0.1485354836691763);
            reconstructionHighPass[11] = NumOps.FromDouble(-0.0736365678679802);
            reconstructionHighPass[12] = NumOps.FromDouble(-0.0297451247861220);
            reconstructionHighPass[13] = NumOps.FromDouble(0.0087901063101452);
            reconstructionHighPass[14] = NumOps.FromDouble(0.0047314665272548);
            reconstructionHighPass[15] = NumOps.FromDouble(-0.0018118519793764);
            reconstructionHighPass[16] = NumOps.FromDouble(-0.0003179695108439);
            reconstructionHighPass[17] = NumOps.FromDouble(0.0001490583487665);

            return (decompositionLowPass, decompositionHighPass, reconstructionLowPass, reconstructionHighPass);
        }
    }
}
