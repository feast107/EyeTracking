#pragma once

using namespace System;

namespace EyeTracking
{
    namespace Windows
    {
        namespace CLR
        {
            namespace Detect
            {
                public ref class DetectResult
                {
                public:
                    int left_x;
                    int left_y;
                    int right_x;
                    int right_y;
                };
                
                public ref class Detector
                {
                    // TODO: 在此处为此类添加方法。
                public:
                    static void Detect(void* cv_ptr, DetectResult^ result);
                };
                
            }
        }
    }
}
