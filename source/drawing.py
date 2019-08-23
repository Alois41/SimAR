from OpenGL.GL import *
from OpenGL.GL.shaders import *
from OpenGL.GLUT import *
from source.configuration import Config as Conf
import numpy as np
import cv2


def draw_rectangle(x_s, y_s, w, h, r=1, g=1, b=1, a=1):
    glPushMatrix()
    glColor4f(r, g, b, a)
    glTranslatef(x_s, y_s, 0)
    glScalef(w, h, 1)
    glBegin(GL_QUADS)  # start drawing a rectangle
    glVertex2f(0, 0)  # bottom left point
    glVertex2f(1, 0)  # bottom right point
    glVertex2f(1, 1)  # top right point
    glVertex2f(0, 1)  # top left point
    glEnd()
    glPopMatrix()


def draw_textured_rectangle(x, y, l, h):

    glPushMatrix()
    glColor3f(1.0, 1.0, 1.0)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

    glTranslatef(x, y, 0)
    glScalef(l, h, 1)

    glBegin(GL_QUADS)
    glTexCoord2f(0.0, 0.0)
    glVertex2f(0.0, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex2f(1.0, 0.0)
    glTexCoord2f(1.0, 1.0)
    glVertex2f(1.0, 1.0)
    glTexCoord2f(0.0, 1.0)
    glVertex2f(0.0, 1.0)
    glEnd()
    glPopMatrix()


def draw_rectangle_empty(x_s, y_s, w, h, r, g, b, thickness=0.2):
    glUseProgram(0)
    glPushMatrix()
    glColor3f(r, g, b)
    glTranslatef(x_s, y_s, 0)
    glScalef(w, h, 1)
    glLineWidth(thickness)

    glBegin(GL_LINE_LOOP)  # start drawing a rectangle
    glVertex2f(0, 0)  # bottom left point
    glVertex2f(1, 0)  # bottom right point
    glVertex2f(1, 1)  # top right point
    glVertex2f(0, 1)  # top left point
    glEnd()
    glPopMatrix()


def get_file_content(file):
    content = open(file, 'r').read()
    return content


def glut_print(x, y, font, text, r, g, b):
    blending = False
    if glIsEnabled(GL_BLEND):
        blending = True

    # glEnable(GL_BLEND)
    glPushMatrix()
    glColor3f(r, g, b)
    glRasterPos2f(x, y)

    for ch in text:
        glutBitmapCharacter(font, ctypes.c_int(ord(ch)))

    if not blending:
        glDisable(GL_BLEND)
    glPopMatrix()


class TextureHandler:
    """ Handle texture for OpenGL """

    def __init__(self) -> void:
        """ load fixed textures and prepare all textures location in OPENGL"""
        self.texture_array = glGenTextures(4)
        molten_steel_texture = cv2.imread("./texture/molten_steel_diff.png")
        im = cv2.cvtColor(molten_steel_texture, cv2.COLOR_BGR2RGBA)

        self.bind_texture(0, None, Conf.height, Conf.width)
        self.bind_texture(1, None, Conf.width, Conf.height)
        self.bind_texture(2, im, im.shape[0], im.shape[1])
        self.bind_texture(3, None, Conf.width, Conf.height)

    def bind_texture(self, index: int, texture: np.ndarray or None, width: int, height: int) -> void:
        """ bind and create a texture for the first time in this loc"""
        glBindTexture(GL_TEXTURE_2D, self.texture_array[index])
        glShadeModel(GL_SMOOTH)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, texture)

    def bind_update_texture(self, index: int, texture: np.ndarray or None, width: int, height: int) -> void:
        """ bind and update a texture in this loc, faster"""
        glBindTexture(GL_TEXTURE_2D, self.texture_array[index])
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, height, width,
                        GL_RGBA, GL_UNSIGNED_BYTE, texture)

    def use_texture(self, index):
        """ just bind texture for next draw"""
        glBindTexture(GL_TEXTURE_2D, self.texture_array[index])

    @staticmethod
    def next_power_of_2(_x):
        return 1 if _x == 0 else 2 ** (_x - 1).bit_length()


class ShaderHandler:
    """ Handle shader for OpenGL """

    def __init__(self, path: str, data_names: dict) -> void:
        """ Load a given vertex and fragment shader from files """
        f_shader = compileShader(get_file_content(path + ".fs"), GL_FRAGMENT_SHADER)
        v_shader = compileShader(get_file_content(path + ".vs"), GL_VERTEX_SHADER)
        self.shaderProgram = glCreateProgram()
        self.buffer_invalid = True
        glAttachShader(self.shaderProgram, v_shader)
        glAttachShader(self.shaderProgram, f_shader)
        glLinkProgram(self.shaderProgram)

        self.data_location = {}
        self.data_types = {}

        # count = glGetProgramiv(self.shaderProgram, GL_ACTIVE_UNIFORMS)
        # print("Active Uniforms: %d\n", count)
        #
        # for i in range(count):
        #     print(glGetActiveUniform(self.shaderProgram, i, 16)[0])

        for _name in data_names.keys():
            if data_names[_name] == GL_SHADER_STORAGE_BUFFER:
                self.data_location[_name] = glGenBuffers(1)
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.data_location[_name])
                glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(ctypes.c_float) * 350000, None, usage=GL_STATIC_DRAW)
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

            else:
                self.data_location[_name] = glGetUniformLocation(self.shaderProgram, _name)

            self.data_types[_name] = data_names[_name]

    def bind(self, data: dict) -> void:
        """ bind the shader and send data to it """
        glUseProgram(self.shaderProgram)
        self.update(data)

    def invalidate_buffer(self):
        self.buffer_invalid = True

    def fix_buffer(self):
        self.buffer_invalid = False

    def update(self, data: dict) -> void:
        if len(data) != len(self.data_location):
            raise IndexError("Not enough data")

        for d in data.keys():
            if d is not None:
                if self.data_types[d] == int:
                    glUniform1i(self.data_location[d], int(data[d]))

                elif self.data_types[d] == float:
                    glUniform1f(self.data_location[d], float(data[d]))

                elif self.data_types[d] == GL_SHADER_STORAGE_BUFFER and self.buffer_invalid:
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.data_location[d])
                    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, data=data[d])
                    # myArrayBlockIdx = glGetUniformBlockIndex(self.shaderProgram, d)
                    # glUniformBlockBinding(self.shaderProgram, myArrayBlockIdx,  0)
                    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.data_location[d])
                    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

                elif self.data_types[d] == np.ndarray:
                    glUniform1fv(self.data_location[d], len(data[d]), data[d])

    @staticmethod
    def unbind() -> void:
        """ unbind shader, no shader will be used after that"""
        glUseProgram(0)
