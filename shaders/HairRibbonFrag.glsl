#version 410 core

in vec3 vWorldPos;
in vec3 vTangent;
in vec3 vSide;
in float vV;

out vec4 frag_color;

uniform vec3 color;
uniform vec3 light_position;
uniform vec3 camera_position;
uniform float rim_strength;
uniform float spec_strength;

void main()
{
    vec3 L = normalize(light_position - vWorldPos);
    vec3 V = normalize(camera_position - vWorldPos);
    vec3 N = normalize(cross(vTangent, vSide));

    if (dot(N, V) < 0.0) N = -N;

    float diff = max(dot(N, L), 0.0);

    float rim = pow(1.0 - max(dot(N, V), 0.0), 2.0);

    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 40.0);

    // Edge alpha
    float edge = 1.0 - smoothstep(0.7, 1.0, abs(vV));
    float alpha = clamp(edge * 0.85 + 0.15, 0.0, 1.0);

    vec3 shaded =
        color * (0.25 + 0.75 * diff) +
            color * rim * rim_strength +
            vec3(1.0) * spec * spec_strength;

    frag_color = vec4(shaded, alpha);
}
