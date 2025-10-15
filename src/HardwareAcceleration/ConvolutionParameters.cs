using AiDotNet.LinearAlgebra;

namespace AiDotNet.HardwareAcceleration
{
    /// <summary>
    /// Parameters for convolution operations
    /// </summary>
    public class ConvolutionParameters
    {
        /// <summary>
        /// Gets or sets the stride for each dimension
        /// </summary>
        public Vector<int> Stride { get; set; } = new Vector<int>(new[] { 1, 1 });

        /// <summary>
        /// Gets or sets the padding for each dimension
        /// </summary>
        public Vector<int> Padding { get; set; } = new Vector<int>(new[] { 0, 0 });

        /// <summary>
        /// Gets or sets the dilation for each dimension
        /// </summary>
        public Vector<int> Dilation { get; set; } = new Vector<int>(new[] { 1, 1 });

        /// <summary>
        /// Gets or sets the number of groups for grouped convolution
        /// </summary>
        public int Groups { get; set; } = 1;

        /// <summary>
        /// Gets or sets whether to use bias
        /// </summary>
        public bool UseBias { get; set; } = true;

        /// <summary>
        /// Gets or sets the data format (NCHW or NHWC)
        /// </summary>
        public string DataFormat { get; set; } = "NCHW";

        /// <summary>
        /// Initializes a new instance of the ConvolutionParameters class
        /// </summary>
        public ConvolutionParameters()
        {
        }

        /// <summary>
        /// Initializes a new instance of the ConvolutionParameters class with specified stride and padding
        /// </summary>
        /// <param name="stride">The stride value</param>
        /// <param name="padding">The padding value</param>
        public ConvolutionParameters(int stride, int padding)
        {
            Stride = new Vector<int>(new[] { stride, stride });
            Padding = new Vector<int>(new[] { padding, padding });
        }

        /// <summary>
        /// Initializes a new instance of the ConvolutionParameters class with specified parameters
        /// </summary>
        /// <param name="stride">The stride vector</param>
        /// <param name="padding">The padding vector</param>
        /// <param name="dilation">The dilation vector</param>
        public ConvolutionParameters(Vector<int> stride, Vector<int> padding, Vector<int>? dilation = null)
        {
            Stride = stride;
            Padding = padding;
            Dilation = dilation ?? new Vector<int>(new[] { 1, 1 });
        }

        /// <summary>
        /// Initializes a new instance of the ConvolutionParameters class with specified parameters (backward compatibility)
        /// </summary>
        /// <param name="stride">The stride array</param>
        /// <param name="padding">The padding array</param>
        /// <param name="dilation">The dilation array</param>
        public ConvolutionParameters(int[] stride, int[] padding, int[]? dilation = null)
        {
            Stride = new Vector<int>(stride);
            Padding = new Vector<int>(padding);
            Dilation = dilation != null ? new Vector<int>(dilation) : new Vector<int>(new[] { 1, 1 });
        }
    }
}