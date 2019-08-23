#version 130


varying vec3 pos_fragment;
varying vec4 vTexCoord;

// uniform mat4 modelViewMatrix;

void main(void)
{
    vTexCoord = gl_MultiTexCoord0;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    pos_fragment = vec3(gl_Vertex.xyz);
}