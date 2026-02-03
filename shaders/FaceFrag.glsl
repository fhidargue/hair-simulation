#version 410 core

in vec3 vNormal;
in vec3 vWorldPos;

out vec4 frag_color;

uniform vec3 color;
uniform vec3 light_position;

void main()
{
    vec3 N = normalize(vNormal);
    vec3 L = normalize(light_position - vWorldPos);

    float diff = max(dot(N, L), 0.0);
    float lighting = 0.25 + 0.75 * diff;

    frag_color = vec4(color * lighting, 1.0);
}
