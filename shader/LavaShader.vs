#version 140

varying vec4 vTexCoord;


void main(void)
{
    vTexCoord = gl_MultiTexCoord0;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}