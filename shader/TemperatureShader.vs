#version 140

#extension GL_ARB_texture_rectangle : enable

varying vec3 pos_fragment;

void main()
{
    /* Transform vertex directly from depth image space to clip space: */
    gl_Position=gl_ModelViewProjectionMatrix*gl_Vertex;
    pos_fragment = vec3(gl_Vertex.xyz);
}