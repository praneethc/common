//In values
varying in vec4 gl_FragColor;
varying in vec4 gl_FragCoord;
varying in vec2 UV;
varying in vec4 f_color;
	
//Out values
varying out vec4 out_color;

// Static
uniform float alpha;

//main shader
void main(void)
{
    out_color = f_color;
}

