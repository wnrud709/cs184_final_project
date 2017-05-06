using UnityEngine;
using System.Collections;

public class SlideFluid : MonoBehaviour
{
	public float densityFactor = 1.0f;
	public int fluidResolution = 256;
	public float mouseSpeed = 0.4f;

	// Default liquid color
	public Color defaultColor = Color.grey;
	public Color inkColor = Color.green;

	Texture2D tex;
	int width, height;
	int numRows;
	int size;
	float[] density;
	Color[] textureTempArray;


	public float dt = 0.8f;
	public float visc = 0f;
	public float diffCoeff = 0f;
	public int iterations = 10;

	float[] u, v, prev_u, prev_v;
	float[] prev_density;



	void Start()
	{
		// Texture size Initialization
		width = fluidResolution;
		height = fluidResolution;
		numRows = width + 2;
		size = (width+2)*(height+2);
		textureTempArray = new Color[width * height];

		// Init Texture
		tex = new Texture2D (width, height, TextureFormat.ARGB32, false);

		tex.Resize (width, height); 
		GetComponent<Renderer>().material.mainTexture = tex;
		// get grid dimensions from texture


		// initialize fluid arrays
		density = new float[size];
		prev_density = new float[size];
		u = new float[size];
		prev_u = new float[size];
		v = new float[size];
		prev_v = new float[size];
		for (int i = 0; i < size; i++) {
			
			int y = i / numRows;
			int x = i % numRows;
			if (i < size / 3 && x != 0) {	// Filled with fluid bottom half
				density [i] = prev_density [i] = 0.9f;
			} else {
				density [i] = prev_density [i] = 0f;
			}
			prev_u[i] = prev_v[i] = u[i] = v[i] = 0;
		}



	}



	void Update()
	{
		// prev_u and prev_v and prev_density are added to u and v and density, so reset them so they're not added every time step.
		for (int i = 0; i < size; i++) {
			prev_density[i] = 0;
			prev_u[i] = 0;
			prev_v[i] = 0;
		}

		UserInput ();

		//Add source velocity, apply diffusion, movement in velocity field, and apply Chopin's projection method using conservation of mass.
		for (int i=0; i<size ; i++ ) {
			u[i] += dt*prev_u[i];
		}
		for (int i=0; i<size ; i++ ) {
			v[i] += dt*prev_v[i];
		}
		float[] temp;
		temp = prev_u; prev_u = u; u = temp;
		temp = prev_v; prev_v = v; v = temp;
		gauss_seidel (u, prev_u, visc);
		gauss_seidel (v, prev_v, visc);
		project(u, v, prev_u, prev_v);
		temp = prev_u; prev_u = u; u = temp;
		temp = prev_v; prev_v = v; v = temp;
		advect(u, prev_u, prev_u, prev_v, dt);
		advect(v, prev_v, prev_u, prev_v, dt);
		project(u, v, prev_u, prev_v);

		//Add source velocity, apply diffusion, and movement in velocity field.
		for (int i=0; i<size ; i++ ) {
			density[i] += dt*prev_density[i];
		}
		gauss_seidel(prev_density, density, diffCoeff);
		advect(density, prev_density, u, v, dt);
		Draw();
	}


	void UserInput()
	{
		// draw on the water
		if (Input.GetMouseButton(0)) {

			Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
			RaycastHit hit;
			if (Physics.Raycast(ray, out hit)) {
				// determine indices where the user clicked
				int x = (int)(hit.point.x * width);
				int y = (int)(hit.point.y * height);
				int i = (x + 1) + (y + 1) * numRows;

				if (x >= 0 && x <= width && y >= 0 && y <= height) {
					prev_u [i] += mouseSpeed * Input.GetAxis ("Mouse X");
					prev_v [i] += mouseSpeed * Input.GetAxis ("Mouse Y");
				}

			} 
		}
	}


	/** Solves linear system for diffusion by Gauss-Seidel relaxation. */
	void gauss_seidel(float[] x, float[] x0, float coeff)
	{
		float cons = 1 + 4 * coeff;
		for (int i = 0 ; i < iterations; i++) {
			for (int j = 1 ; j <= height; j++) {
				for (int k = 1; k <= width; k++) {
					int lastRow = (j - 1) * numRows + k;
					int ind = j * numRows + k;
					int nextRow = (j + 1) * numRows + k;
					x [ind] = (x0 [ind] + coeff * (x[lastRow] + x [nextRow] + x [ind-1] + x [ind+1])) / cons;
				}
			}
		}
	}

	/** For every cell, subtract the velocity to get the location of the fluid previous time step and linearly interpolate to get the new density.*/
	void advect(float[] d, float[] d0, float[] u, float[] v, float dt)
	{
		float l0, l1, l2, l3;
		for (int j = 1; j<= height; j++) {
			for (int i = 1; i <= width; i++) {
				float x = i - dt * width * u[j * numRows + i]; 
				float y = j - dt * width * v[j * numRows + i];
				if (x < .5f) {
					x = .5f;
				} else if (x > width + .5) {
					x = width + .5f;
				}
				if (y < .5f) {
					y = .5f;
				} else if (y > height + .5f) {
					y = height + .5f;
				}
				l0 = x - (int)x;
				l1 = 1 - l0;
				l2 = y - (int)y;
				l3 = 1 - l2;
				d[j * numRows + i] =
					l1 * (l3 * d0[(int)x + numRows * (int)y] + l2 * d0[(int)x + numRows * ((int)y + 1)]) +
					l0 * (l3 * d0[((int)x + 1) + numRows * (int)y] + l2 * d0[((int)x+1) + numRows *((int)y+1)]);
			}
		}
	}

	/** Applies conservation of mass to momentum using Chopin's projection on intermediate velocity.. */
	void project(float[] u, float[] v, float[] p, float[] div)
	{
		float derivX;
		float derivY;

		//Calculate divergence of velocity.
		for (int i = 1; i < height; i++) {
			for (int j = 1; j < width; j++) {
				derivX = -0.5f / Mathf.Sqrt (width * height) * (u [i * numRows + j + 1] - u [i * numRows + j - 1]);
				derivY = -0.5f / Mathf.Sqrt (width * height) * (v [(i + 1) * numRows + j] - v [(i - 1) * numRows + j]);
				div [i * numRows + j] = (derivX + derivY);
				p [i * numRows + j] = 0;
			}
		}
		gauss_seidel (p, div, 1);

		for (int i = 1; i < height; i++) {
			for (int j = 1; j < width; j++) {
				u [i * numRows + j] -= width * (p [i * numRows + j + 1] - p [i * numRows + j - 1]);
				v [i * numRows + j] -= width * (p [(i + 1) * numRows + j] - p [(i - 1) * numRows + j]);
			}
		}
	}



	void Draw()
	{
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int i = (x + 1) + (y + 1) * numRows;
				float d = density [i] * densityFactor;
				Color flowColor = new Color ();
				flowColor = Color.Lerp (defaultColor, inkColor, d);
				textureTempArray [y * width + x] = new Color (flowColor.r, flowColor.g, flowColor.b, flowColor.a);
			}
		}
		tex.SetPixels (textureTempArray);
		tex.Apply(false);
	}


}

