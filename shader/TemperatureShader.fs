#version 430

in vec3 pos_fragment;

uniform float brick_dim[2];
uniform int grid_pos;
uniform float Corrosion;

layout(std430, binding = 0) buffer  temp_buffer {
  int T[];
};
uniform float step;
uniform float border;

out vec4 color;


void main()
{
    float y = 1 - pos_fragment.y;// coordonnée y du pixel entre 0 et 1
    float x = pos_fragment.x;// coordonnée x du pixel entre 0 et 1
    int i = int((x * brick_dim[1]));
    int j = int((y * brick_dim[0]));
    float temperature = 0;

    int index = int(trunc(grid_pos + i + j * step));

//     Mixing colors

    if ((border == 1 || border == 3) && T[index + 1] >= 0 && T[index + 1] < 2000)
    {
        float mix_y = T[index], mix_y_next = T[index+1];
        if ((border == 2 || border == 3) && T[int(index + 1 + step)] >=0 && T[int(index + step)] >=0)
        {
            float ratio_y = mod(y * brick_dim[0], 1);
            mix_y = mix(T[index], T[int(index + step)], ratio_y);
            mix_y_next = mix(T[index + 1], T[int(index + 1 + step)], ratio_y);
        }
        float ratio_x = mod(x * brick_dim[1], 1);
        temperature = mix(mix_y, mix_y_next, ratio_x);
    }
    else
    {
        temperature = T[index];
    }

    float c = max(0, min(1, 1-Corrosion));
    //pseudo random
    float random = fract(sin(dot(pos_fragment.xy, vec2(12.9898,78.233)))*43758.5453123);
    color = mix(vec4(0, 0, 1, 1), vec4(1, 0, 0, 1), max(0, (temperature) / 1600.0));
//    bool condition = x > c;  // corrosive hole shape
    bool condition = (random > c || c == 0) || x < 0.01 || x > 0.99;//|| x > c;
    color = condition ? color : vec4(0,0,0,1);

}