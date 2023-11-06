namespace AiDotNet.Enums;

/// <summary>
/// Matrix layout will either be column arrays or row arrays. Default is column arrays.
///
/// Column arrays example
/// { x1, x2, x3 }, { y1, y2, y3 }
/// 
/// Row arrays example
/// { x1, y1 }, { x2, y2 }, { x3, y3 }
/// </summary>
public enum MatrixLayout
{
    ColumnArrays,
    RowArrays
}