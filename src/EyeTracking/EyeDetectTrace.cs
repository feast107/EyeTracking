using CommunityToolkit.Mvvm.ComponentModel;
using System;
using System.Collections.Generic;
using System.Text;

namespace EyeTracking
{
    public partial class EyeDetectTrace : ObservableObject
    {
        [ObservableProperty]
        public partial string? Content { get; set; }
    }
}
