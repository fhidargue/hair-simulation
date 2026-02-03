#version 410 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

uniform mat4 MVP;
uniform mat4 M;

out vec3 vNormal;
out vec3 vWorldPos;

void main()
{
    vec4 worldPos = M * vec4(position, 1.0);

    vWorldPos = worldPos.xyz;
    vNormal = mat3(transpose(inverse(M))) * normal;

    gl_Position = MVP * vec4(position, 1.0);
}
