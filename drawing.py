from OpenGL.GL import *
from OpenGL.GL.shaders import *
from OpenGL.GLUT import *


def draw_rectangle(x_s, y_s, w, h, r, g, b):
    glPushMatrix()
    glColor3f(r, g, b)
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

    # Draw textured Quads
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

    glBegin(GL_LINE_STRIP)  # start drawing a rectangle
    glVertex2f(0, 0)  # bottom left point
    glVertex2f(1, 0)  # bottom right point
    glVertex2f(1, 1)  # top right point
    glVertex2f(0, 1)  # top left point
    glEnd()
    glPopMatrix()
