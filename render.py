import noise
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import matplotlib.pyplot as plt

class Vector(object):
    """
    Vector class meant to store 3D coordinates.
    """

    def __init__(self, x, y, z=0):
        """
        init method, z cordinate is optional as it's sometimes not used.
        :param x: float coordinate
        :param y: float coordinate
        :param z: float coordinate
        """
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


# scale variable meant for noise heightmap calculation
scale = 15.0

# height scale variable meant for as a mesh height multiplier.
height_scale = 10

# octaves, octaveOffsets, persistence and lacunarity are all used for multi-layered noisemap
octaves = 5
octaveOffsets = []

persistence = 0.5
lacunarity = 2

# if mesh height is less than flattening threshold, mesh height will be 0
flattening_threshold = 0.0125

# generating octave offsets
for i in range(octaves):
    octaveOffsets.append(Vector(np.random.randint(-10000, 10000), np.random.randint(-10000, 10000)))


terrain_size = 150

# stores terrain height data.
terrain = []

# movement offset
offset = Vector(0, 0, 50)

def extract_frustum_planes(projection_matrix, model_view_matrix):

    clip_matrix = np.dot(model_view_matrix, projection_matrix)
    frustum_planes = [np.zeros(4, dtype=np.float32) for _ in range(6)]

    # Extract the LEFT plane
    frustum_planes[0] = np.array([clip_matrix[0][3] + clip_matrix[0][0],
                                   clip_matrix[1][3] + clip_matrix[1][0],
                                   clip_matrix[2][3] + clip_matrix[2][0],
                                   clip_matrix[3][3] + clip_matrix[3][0]], dtype=np.float32)

    # Extract the RIGHT plane
    frustum_planes[1] = np.array([clip_matrix[0][3] - clip_matrix[0][0],
                                   clip_matrix[1][3] - clip_matrix[1][0],
                                   clip_matrix[2][3] - clip_matrix[2][0],
                                   clip_matrix[3][3] - clip_matrix[3][0]], dtype=np.float32)

    # Extract the BOTTOM plane
    frustum_planes[2] = np.array([clip_matrix[0][3] + clip_matrix[0][1],
                                   clip_matrix[1][3] + clip_matrix[1][1],
                                   clip_matrix[2][3] + clip_matrix[2][1],
                                   clip_matrix[3][3] + clip_matrix[3][1]], dtype=np.float32)

    # Extract the TOP plane
    frustum_planes[3] = np.array([clip_matrix[0][3] - clip_matrix[0][1],
                                   clip_matrix[1][3] - clip_matrix[1][1],
                                   clip_matrix[2][3] - clip_matrix[2][1],
                                   clip_matrix[3][3] - clip_matrix[3][1]], dtype=np.float32)

    # Extract the NEAR plane
    frustum_planes[4] = np.array([clip_matrix[0][3] + clip_matrix[0][2],
                                   clip_matrix[1][3] + clip_matrix[1][2],
                                   clip_matrix[2][3] + clip_matrix[2][2],
                                   clip_matrix[3][3] + clip_matrix[3][2]], dtype=np.float32)

    # Extract the FAR plane
    frustum_planes[5] = np.array([clip_matrix[0][3] - clip_matrix[0][2],
                                   clip_matrix[1][3] - clip_matrix[1][2],
                                   clip_matrix[2][3] - clip_matrix[2][2],
                                   clip_matrix[3][3] - clip_matrix[3][2]], dtype=np.float32)

    # Normalize the planes
    for i in range(6):
        frustum_planes[i] = frustum_planes[i] / np.linalg.norm(frustum_planes[i][:3])

    return frustum_planes


def point_in_frustum(point, frustum_planes):
    """
    Check if a 3D point is inside the viewing frustum defined by the six frustum planes.
    """
    for plane in frustum_planes:
        distance = np.dot(plane[:3], point) + plane[3]
        if distance < 0:
            return False
    return True



def calculate_terrain():

    # calculates terrain via multi-layered noise
    global terrain
    terrain = []
    for y in range(terrain_size):
        terrain.append([])
        for x in range(terrain_size):
            terrain[-1].append(0)

    for y in range(terrain_size):
        for x in range(terrain_size):
            amplitude = 1
            frequency = 1
            noiseHeight = 0
            for i in range(octaves):
                sampleX = frequency * (x + octaveOffsets[i].x + offset.x) / scale
                sampleY = frequency * (y + octaveOffsets[i].y + offset.y) / scale
                noiseHeight += amplitude * noise.pnoise2(sampleX, sampleY)
                amplitude *= persistence
                frequency *= lacunarity
            terrain[x][y] = noiseHeight

    min_val = np.min(terrain)
    max_val = np.max(terrain)

    terrain = np.array(terrain)
    np.clip(terrain, -0.7, max_val)


# manually calculated values to for coloring vertices.
color_heights = [-0.7078, -0.6518, -0.5057, -0.27, -0.07, 0.1765, 0.3725, 0.5686, 0.9608]


def keyboard(bkey, x, y):
    key = bkey.decode("utf-8")
    if key == 'w':
        offset.y += offset.z
        calculate_terrain()
    if key == 'a':
        offset.x -= offset.z
        calculate_terrain()
    if key == 's':
        offset.y -= offset.z
        calculate_terrain()
    if key == 'd':
        offset.x += offset.z
        calculate_terrain()

    glutPostRedisplay()


def initGL():

    calculate_terrain()
    glClear(GL_COLOR_BUFFER_BIT)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glShadeModel(GL_SMOOTH)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)


def getColor(value):

    global color_heights
    if value < color_heights[0]:
        return 42 / 255.0, 93 / 255.0, 186 / 255.0
    elif color_heights[0] <= value < color_heights[1]:
        return 51 / 255.0, 102 / 255.0, 195 / 255.0
    elif color_heights[1] <= value < color_heights[2]:
        return 207 / 255.0, 215 / 255.0, 127 / 255.0
    elif color_heights[2] <= value < color_heights[3]:
        return 91 / 255.0, 169 / 255.0, 24 / 255.0
    elif color_heights[3] <= value < color_heights[4]:
        return 63 / 255.0, 119 / 255.0, 17 / 255.0
    elif color_heights[4] <= value < color_heights[5]:
        return 89 / 255.0, 68 / 255.0, 61 / 255.0
    elif color_heights[5] <= value < color_heights[6]:
        return 74 / 255.0, 59 / 255.0, 55 / 255.0
    elif color_heights[6] <= value < color_heights[7]:
        return 250 / 255.0, 250 / 255.0, 250 / 255.0
    elif value >= color_heights[7]:
        return 1, 1, 1

def draw_vertex(x, y, height):
    if height < flattening_threshold:
        color = getColor(height)
        glColor3f(*color)
        glVertex3f(x, y, 0)
    else:
        color = getColor(height)
        glColor3f(*color)
        glVertex3f(x, y, height * height_scale)

def display():
    # standard OpenGL function
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # translation and rotation for terrain display
    glTranslatef(-terrain_size / 2.0, -25, -6)
    glRotate(60, -1, 0, 0)
    
    projection_matrix = np.array(glGetDoublev(GL_PROJECTION_MATRIX), dtype=np.float32)
    model_view_matrix = np.array(glGetDoublev(GL_MODELVIEW_MATRIX), dtype=np.float32)

    # Extract the frustum planes
    frustum_planes = extract_frustum_planes(projection_matrix, model_view_matrix)

    # drawing triangles
    glBegin(GL_TRIANGLES)
    for y in range(1, terrain_size - 1):
        for x in range(1, terrain_size - 1):
            # Check if the triangle vertices are inside the frustum
            p1 = np.array([x, y + 1, terrain[x][y + 1] * height_scale], dtype=np.float32)
            p2 = np.array([x, y, terrain[x][y] * height_scale], dtype=np.float32)
            p3 = np.array([x + 1, y + 1, terrain[x + 1][y + 1] * height_scale], dtype=np.float32)

            if point_in_frustum(p1, frustum_planes) or point_in_frustum(p2, frustum_planes) or point_in_frustum(p3, frustum_planes):
                draw_vertex(x, y + 1, terrain[x][y + 1])
                draw_vertex(x, y, terrain[x][y])
                draw_vertex(x + 1, y + 1, terrain[x + 1][y + 1])

    glEnd()

    # drawing more triangles
    glBegin(GL_TRIANGLES)
    for y in range(1, terrain_size - 1):
        for x in range(1, terrain_size - 1):
            # Check if the triangle vertices are inside the frustum
            p1 = np.array([x + 1, y + 1, terrain[x + 1][y + 1] * height_scale], dtype=np.float32)
            p2 = np.array([x + 1, y, terrain[x + 1][y] * height_scale], dtype=np.float32)
            p3 = np.array([x, y, terrain[x][y] * height_scale], dtype=np.float32)

            if point_in_frustum(p1, frustum_planes) or point_in_frustum(p2, frustum_planes) or point_in_frustum(p3, frustum_planes):
                draw_vertex(x + 1, y + 1, terrain[x + 1][y + 1])
                draw_vertex(x + 1, y, terrain[x + 1][y])
                draw_vertex(x, y, terrain[x][y])

    glEnd()

    glutSwapBuffers()


def reshape(width, height):

    if height == 0:
        height = 1
    aspect = width / height

    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    gluPerspective(45.0, aspect, 0.1, 100.0)


if __name__ == '__main__':
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE)
    glutInitWindowSize(800, 600)
    glutInitWindowPosition(50, 50)
    glutCreateWindow(b"3D Terrain")
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    initGL()
    plt.imshow(terrain, cmap='gray')
    plt.colorbar()
    plt.show()
    glutMainLoop()