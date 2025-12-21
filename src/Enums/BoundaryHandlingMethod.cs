namespace AiDotNet.Enums;

/// <summary>
/// Specifies how to handle boundaries when processing data that extends beyond the available range.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> When working with operations like convolutions, filtering, or sampling, 
/// you often need to access data points outside the boundaries of your dataset. 
/// 
/// For example, if you have an image and want to apply a filter to every pixel, what happens 
/// at the edges where the filter would need pixels that don't exist? Boundary handling methods 
/// provide different ways to solve this problem.
/// 
/// Think of it like trying to read a word at the edge of a page - you need to decide what to do 
/// when part of the word would be off the page.
/// </para>
/// </remarks>
public enum BoundaryHandlingMethod
{
    /// <summary>
    /// Treats the data as if it repeats infinitely in all directions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> With Periodic boundary handling, the data is treated as if it wraps around 
    /// from one edge to the opposite edge, like a loop.
    /// 
    /// Imagine a photo where walking off the right edge makes you reappear at the left edge, 
    /// or walking off the bottom makes you reappear at the top.
    /// 
    /// Example: For a 1D array [A,B,C,D,E]:
    /// - If you need a value one position to the left of A, you get E
    /// - If you need a value two positions to the right of E, you get B
    /// 
    /// This method is useful when your data represents something naturally cyclical, like time series 
    /// with seasonal patterns or images that should tile seamlessly.
    /// </para>
    /// </remarks>
    Periodic,

    /// <summary>
    /// Reflects the data at boundaries as if there were a mirror at each edge.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> With Symmetric boundary handling, the data is reflected at the edges, 
    /// as if there were mirrors placed at the boundaries.
    /// 
    /// Imagine standing between two mirrors and seeing your reflection repeated - the data 
    /// is reflected back in the opposite direction when it hits a boundary.
    /// 
    /// Example: For a 1D array [A,B,C,D,E]:
    /// - If you need a value one position to the left of A, you get B
    /// - If you need a value two positions to the right of E, you get C
    /// 
    /// This method preserves continuity at the edges and is often used in image processing 
    /// because it avoids creating artificial edges or discontinuities.
    /// </para>
    /// </remarks>
    Symmetric,

    /// <summary>
    /// Fills values outside the boundaries with zeros.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> With ZeroPadding, any attempt to access data beyond the boundaries 
    /// returns a zero value.
    /// 
    /// Imagine your data is surrounded by an infinite sea of zeros in all directions.
    /// 
    /// Example: For a 1D array [A,B,C,D,E]:
    /// - If you need a value one position to the left of A, you get 0
    /// - If you need a value one position to the right of E, you get 0
    /// 
    /// This method is simple to implement and understand. It's commonly used in signal processing 
    /// and neural networks, especially in convolutional layers. However, it can create artificial 
    /// edges at the boundaries where your data suddenly transitions to zeros.
    /// </para>
    /// </remarks>
    ZeroPadding
}
