#version 130

uniform mat4 mvp;

out vec3 pos_fragment;


void main()
{
    /* Transform vertex directly from depth image space to clip space: */
    gl_Position= gl_ModelViewProjectionMatrix*gl_Vertex;
    pos_fragment = gl_Vertex.xyz;
}