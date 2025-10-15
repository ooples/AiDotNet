using System.Collections.Generic;

namespace AiDotNet.Models
{
    /// <summary>
    /// Helper class for building multimodal inputs in a fluent manner.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    public class MultimodalInput<T>
        where T : struct
    {
        private readonly Dictionary<string, object> _modalityData = new();

        /// <summary>
        /// Adds text data to the multimodal input.
        /// </summary>
        /// <param name="texts">Array of text samples.</param>
        /// <returns>This instance for method chaining.</returns>
        public MultimodalInput<T> AddTextData(string[] texts)
        {
            _modalityData["text"] = texts;
            return this;
        }

        /// <summary>
        /// Adds image data to the multimodal input.
        /// </summary>
        /// <param name="imagePaths">Array of image file paths.</param>
        /// <returns>This instance for method chaining.</returns>
        public MultimodalInput<T> AddImageData(string[] imagePaths)
        {
            _modalityData["image"] = imagePaths;
            return this;
        }

        /// <summary>
        /// Adds audio data to the multimodal input.
        /// </summary>
        /// <param name="audioPaths">Array of audio file paths.</param>
        /// <returns>This instance for method chaining.</returns>
        public MultimodalInput<T> AddAudioData(string[] audioPaths)
        {
            _modalityData["audio"] = audioPaths;
            return this;
        }

        /// <summary>
        /// Adds video data to the multimodal input.
        /// </summary>
        /// <param name="videoPaths">Array of video file paths.</param>
        /// <returns>This instance for method chaining.</returns>
        public MultimodalInput<T> AddVideoData(string[] videoPaths)
        {
            _modalityData["video"] = videoPaths;
            return this;
        }

        /// <summary>
        /// Adds custom modality data to the multimodal input.
        /// </summary>
        /// <param name="modalityName">Name of the modality.</param>
        /// <param name="data">The modality data.</param>
        /// <returns>This instance for method chaining.</returns>
        public MultimodalInput<T> AddModalityData(string modalityName, object data)
        {
            _modalityData[modalityName] = data;
            return this;
        }

        /// <summary>
        /// Gets the modality data dictionary.
        /// </summary>
        /// <returns>Dictionary mapping modality names to their data.</returns>
        public Dictionary<string, object> GetModalityData()
        {
            return new Dictionary<string, object>(_modalityData);
        }

        /// <summary>
        /// Implicit conversion to Dictionary for use with IMultimodalModel.
        /// </summary>
        public static implicit operator Dictionary<string, object>(MultimodalInput<T> input)
        {
            return input.GetModalityData();
        }
    }
}