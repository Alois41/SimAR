varying vec4 v_color;
varying vec3 pos_fragment;

uniform float Tinside;
uniform float Toutside;
uniform float lenght;


void main()
{
    float y = 1-(0.5*(pos_fragment.x + 1.0));// coordonnée y du pixel entre 0 et 1
    float t_y = (y * (Tinside-Toutside)) / lenght;//equation chaleur linéaire (pas de +Toutside car relatif)
    gl_FragColor = mix(v_color, vec4(1, 0, 0, 1), max(0, t_y / 1600.0));
    gl_FragColor.a = 0.7;
}