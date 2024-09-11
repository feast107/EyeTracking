﻿using Avalonia.Controls;
using EyeTracking.Desktop.ViewModels;

namespace EyeTracking.Desktop;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        DataContext = new EyeTrackViewModel(this);
    }
}