#version 140

uniform sampler2D myTexture;

varying vec4 vTexCoord;

uniform float Time;

void main(void)

{
    gl_FragColor = texture2D(myTexture, vec2(vTexCoord.x + Time, vTexCoord.y + Time)).rgba;
}