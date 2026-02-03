#version 410 core

out vec4 fragColor;
uniform vec3 color;
uniform int render_points;

void main()
{
    // If Braids is selected, draw points
    if (render_points == 1)
    {
        vec2 uv = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(uv, uv);
        if (r2 > 1.0) discard;

        fragColor = vec4(color, 0.85);
    }
    else
    {
        fragColor = vec4(color, 1.0);
    }
}
