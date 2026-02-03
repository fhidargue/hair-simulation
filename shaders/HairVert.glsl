#version 410 core

layout(location = 0) in vec4 inPosRadius;
uniform mat4 MVP;
uniform int render_points;

void main()
{
    gl_Position = MVP * vec4(inPosRadius.xyz, 1.0);

    if (render_points == 1)
    {
        gl_PointSize = max(1.0, inPosRadius.w * 1000.0);
    }
}
