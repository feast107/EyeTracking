using System.Globalization;
using Avalonia.Data.Converters;

namespace EyeTracking.Desktop.Converters;

public class StringFormatConverter : IValueConverter
{
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture) => value is null ? null : string.Format(culture, (string)parameter!, value);

    public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture) => throw new NotImplementedException();
}