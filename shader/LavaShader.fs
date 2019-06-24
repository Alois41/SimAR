uniform sampler2D myTexture;

varying vec2 vTexCoord;

uniform float Time;

void main(void)

{
    gl_FragColor = texture2D(myTexture, vec2(vTexCoord.x + Time, vTexCoord.y + Time)).rgba;
}