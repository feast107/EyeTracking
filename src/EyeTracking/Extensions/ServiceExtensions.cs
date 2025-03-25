using Microsoft.Extensions.DependencyInjection;

namespace EyeTracking.Extensions;

public static class ServiceExtensions
{
    public static IServiceCollection AddEyeTrackContext(this IServiceCollection collection) =>
        collection.AddSingleton<EyeTrackContext<EyeDetectResult>, LatestTrackContext>();
}