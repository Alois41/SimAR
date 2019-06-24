#extension GL_ARB_texture_rectangle : enable

varying  vec4 v_color;
varying vec3 pos_fragment;

void main()
{
    /* Transform vertex directly from depth image space to clip space: */
    gl_Position=gl_ModelViewProjectionMatrix*gl_Vertex;
    pos_fragment = vec3(gl_Vertex.xyz);
    v_color = vec4(0, 0, 1, 1);
}