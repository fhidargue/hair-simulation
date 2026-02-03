#version 410 core

uniform mat4 MVP;
uniform vec3 camera_position;
uniform float thickness;

// Follower controls
uniform int followers_per_segment;
uniform float follower_spread;
uniform int follower_seed;
uniform float max_root_radius;

// Segment buffer (TBO)
uniform samplerBuffer segmentTex;

out vec3 vWorldPos;
out vec3 vTangent;
out vec3 vSide;
out float vV;

// Helpers
float hash11(float value)
{
    value = fract(value * 0.1031);
    value *= value + 33.33;
    value *= value + value;
    return fract(value);
}

vec2 hash21(float value)
{
    float hash_x = hash11(value);
    float hash_y = hash11(value + 19.19);
    return vec2(hash_x, hash_y);
}

vec3 safe_normalize(vec3 v)
{
    float len = length(v);
    if (len < 1e-8) return vec3(0.0, 1.0, 0.0);
    return v / len;
}

void build_frame(vec3 tangent_dir, out vec3 frame_side_dir, out vec3 frame_binormal_dir)
{
    vec3 world_up = vec3(0.0, 1.0, 0.0);
    if (abs(dot(tangent_dir, world_up)) > 0.9)
        world_up = vec3(1.0, 0.0, 0.0);

    frame_side_dir = safe_normalize(cross(tangent_dir, world_up));
    frame_binormal_dir = safe_normalize(cross(tangent_dir, frame_side_dir));
}

void main()
{
    int followers_count = max(followers_per_segment, 1);
    int instance_index = gl_InstanceID;
    int segment_index = instance_index / followers_count;
    int follower_index = instance_index - segment_index * followers_count;

    // Fetch segment endpoints plus radii from the TBO
    vec4 segmentA_packed = texelFetch(segmentTex, segment_index * 2 + 0);
    vec4 segmentB_packed = texelFetch(segmentTex, segment_index * 2 + 1);

    vec3 segment_start_world = segmentA_packed.xyz;
    vec3 segment_end_world = segmentB_packed.xyz;
    float radius_start = segmentA_packed.w;
    float radius_end = segmentB_packed.w;

    // Quad vertex index inside the 4-vertex strip (0..3)
    int quad_vertex_index = gl_VertexID % 4;
    float along_segment_t = (quad_vertex_index >= 2) ? 1.0 : 0.0;
    float ribbon_side_sign = (quad_vertex_index == 0 || quad_vertex_index == 2) ? 1.0 : -1.0;

    vec3 centerline_world = mix(segment_start_world, segment_end_world, along_segment_t);
    float strand_radius = mix(radius_start, radius_end, along_segment_t);
    vec3 tangent_world = safe_normalize(segment_end_world - segment_start_world);

    // Cross-section frame (perpendicular to tangent)
    vec3 frame_side_world, frame_binormal_world;
    build_frame(tangent_world, frame_side_world, frame_binormal_world);
    vec3 follower_offset_world = vec3(0.0);

    if (follower_index != 0 && followers_count > 1)
    {
        // Deterministic random seed per segment
        float random_seed = float(follower_seed) + float(segment_index) * 0.001 + float(follower_index) * 13.37;
        vec2 random2 = hash21(random_seed);

        float angle_radians = 6.2831853 * random2.x;
        float follower_radius = (0.35 + 0.65 * random2.y) * follower_spread;

        float radius_normalized = clamp(strand_radius / max(max_root_radius, 1e-6), 0.0, 1.0);
        float taper_factor = pow(radius_normalized, 1.2);

        vec3 follower_dir_world = cos(angle_radians) * frame_side_world + sin(angle_radians) * frame_binormal_world;
        follower_offset_world = follower_dir_world * follower_radius * taper_factor;
    }

    // Apply follower offset to centerline
    centerline_world += follower_offset_world;

    vec3 view_dir_world = safe_normalize(camera_position - centerline_world);

    // Billboard side direction in view plane
    vec3 ribbon_side_dir_world = cross(view_dir_world, tangent_world);
    if (length(ribbon_side_dir_world) < 1e-6)
    {
        ribbon_side_dir_world = cross(vec3(0.0, 1.0, 0.0), tangent_world);
        if (length(ribbon_side_dir_world) < 1e-6)
            ribbon_side_dir_world = cross(vec3(1.0, 0.0, 0.0), tangent_world);
    }
    ribbon_side_dir_world = safe_normalize(ribbon_side_dir_world);

    // Final ribbon vertex position
    vec3 ribbon_vertex_world =
        centerline_world +
            ribbon_side_dir_world * ribbon_side_sign * strand_radius * thickness;

    // Outs
    vWorldPos = ribbon_vertex_world;
    vTangent = tangent_world;
    vSide = ribbon_side_dir_world * ribbon_side_sign;
    vV = ribbon_side_sign;

    gl_Position = MVP * vec4(ribbon_vertex_world, 1.0);
}
