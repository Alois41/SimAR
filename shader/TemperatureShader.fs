#version 440
#define MAX_SIZE 4000
varying vec3 pos_fragment;

uniform float brick_dim[2];
uniform float grid_pos;
uniform float Corrosion;
uniform float T[MAX_SIZE];
uniform float step;
uniform float border;

void main()
{
    float y = 1 - pos_fragment.y;// coordonnée y du pixel entre 0 et 1
    float x = pos_fragment.x;// coordonnée x du pixel entre 0 et 1
    int i = int(trunc(x * brick_dim[0]));
    int j = int(trunc(y * brick_dim[1]));
    float temperature = 0;

    int index = int(trunc(grid_pos + i + j * step));// + brick_dim[0]*j);

    if (border == 1 || border == 3)
    {
        float mix_y = T[index], mix_y_next = T[index+1];
        if (border == 2 || border == 3)
        {
            float ratio_y = mod(y * brick_dim[1], 1);
            mix_y = mix(T[index], T[int(index + step)], ratio_y);
            mix_y_next = mix(T[index + 1], T[int(index + 1 + step)], ratio_y);
        }
        float ratio_x = mod(x * brick_dim[0], 1);
        temperature = mix(mix_y, mix_y_next, ratio_x);
    }
    else
    {
        temperature = T[index];
    }

    float c = max(0, min(1, 1-Corrosion));
    gl_FragColor = mix(vec4(0, 0, 1, 1), vec4(1, 0, 0, 1), max(0, (temperature - 293.0) / 1600.0));
    bool condition = (x) * (x) + (y) * (y) > (c*c)*2;
    gl_FragColor.a = condition ? 1 : 0.3;
}