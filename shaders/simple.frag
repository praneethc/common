//In values
in vec4 gl_FragColor;
in vec4 gl_FragCoord;
in vec2 UV;
in vec4 f_color;
	
//Out values
out vec4 out_color;

// Static
uniform float alpha;

//main shader
void main(void)
{
    out_color = f_color;
}

