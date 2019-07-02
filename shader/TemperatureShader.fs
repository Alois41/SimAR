#version 440

varying vec3 pos_fragment;

uniform float Corrosion;
uniform float T[10*10];

void main()
{
    float y = pos_fragment.y;// coordonnée y du pixel entre 0 et 1
    float x = pos_fragment.x;// coordonnée x du pixel entre 0 et 1
    int i = int(x * 10);
    int j = int(y * 10);
    float temperature = 0;

    int index = int(i+10*j);

    if (i < 9)
    {
        float ratio = mod(x*10, 1);
        temperature = mix(T[index], T[index+1], ratio);
    }
    else {
        temperature = T[index];
    }

    //if(y<=Corrosion)
    {
        gl_FragColor = mix(vec4(0, 0, 1, 1), vec4(1, 0, 0, 1), max(0, temperature / 1500.0));
        //gl_FragColor = mix(vec4(0, 0, 1, 1), vec4(1, 0, 0, 1), max(0, Temperature / 1600.0));
    }
    //    else
    //    {
    //        gl_FragColor = vec4(0,0,0,0.3);
    //    }

    //gl_FragColor = mix(vec4(0, 0, 1, 1), vec4(1, 0, 0, 1), int(x*10)/10.0);

}