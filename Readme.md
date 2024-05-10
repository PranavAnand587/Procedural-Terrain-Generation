# 3D Terrain Generator

This Python script generates a 3D terrain using Perlin noise and displays it using OpenGL.

## Dependencies

- `noise`
- `numpy`
- `OpenGL`
- `matplotlib`

You can install the dependencies via pip:

```bash
pip install noise numpy pyopengl matplotlib
```

## Usage
1. Ensure all dependencies are installed.
2. Run the script using Python:
```bash
python render.py
```

3. Use the following keys for navigation:
W: Move forward
A: Move left
S: Move backward
D: Move right

## Description

The script utilizes Perlin noise to generate a 3D terrain. It employs OpenGL for rendering and navigation. The terrain is generated using multi-layered noise with customizable parameters like octaves, persistence, and lacunarity. Navigation within the generated terrain is possible by adjusting the offset.

