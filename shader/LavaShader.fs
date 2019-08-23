#version 130

varying vec3 pos_fragment;
uniform sampler2D myTexture;

varying vec4 vTexCoord;

uniform float offset_x;
uniform float offset_y;
uniform float delta_t;
uniform float height;

void main(void)
{
    gl_FragColor = texture2D(myTexture, vec2(vTexCoord.x, vTexCoord.y)).rgba;
    if(pos_fragment.y > height + 0.006 *sin(10*pos_fragment.x + 500*offset_x)
                            + 0.005 * sin(15*pos_fragment.x - 200*offset_x)
                            + 0.002 * sin(30*pos_fragment.x - 300*offset_x)
                            + 0.002 * sin(5*pos_fragment.x + 100*offset_x))
    {
//        gl_FragColor = vec4(0,0,0,0);
    }

}