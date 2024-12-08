import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

def rot2d(_x, _y, angle):
    """Поворот заданных координат на угол angle."""
    return (_x*np.cos(angle)-_y*np.sin(angle),
            _x*np.sin(angle)+_y*np.cos(angle))

def mdl(data_x, data_y, i):
    """Вычисление модуля вектора в точке i."""
    return round((data_x[i]**2+data_y[i]**2)**0.5,5)

# Символьная переменная
t = sp.Symbol('t', real=True)

# Данные варианта 22:
# z(t) = 5 - 0.5t
# phi(t) = 2t
z = 5 - 0.5*t
phi = 2*t

# Перевод в декартовы координаты
x = z*sp.cos(phi)
y = z*sp.sin(phi)

# Скорости
Vx = sp.diff(x, t)
Vy = sp.diff(y, t)
V = sp.sqrt(Vx**2 + Vy**2)

# Ускорения
Wx = sp.diff(Vx, t)
Wy = sp.diff(Vy, t)
W = sp.sqrt(Wx**2 + Wy**2)

# Тангенциальное ускорение
Wt = sp.diff(V, t)

# Нормальное ускорение
Wn = sp.sqrt(W**2 - Wt**2)

# Проекции тангенциального ускорения
Wtx = Vx/V*Wt
Wty = Vy/V*Wt

# Проекции нормального ускорения
Wnx = Wx - Wtx
Wny = Wy - Wty

# Единичные направления нормального ускорения Nx, Ny
Nx = sp.Piecewise((0, sp.Eq(Wn,0)), (Wnx/Wn, True))
Ny = sp.Piecewise((0, sp.Eq(Wn,0)), (Wny/Wn, True))

# Радиус кривизны CR = V² / Wn, при Wn=0 задаём 0 чтобы избежать деления на ноль
curvature_module = sp.Piecewise((0, sp.Eq(Wn,0)), ((V**2)/Wn, True))
curvature_x = curvature_module*Nx
curvature_y = curvature_module*Ny

# Временной массив
T = np.linspace(0,6.28,1000)

# Численные функции
Fx = sp.lambdify(t, x, 'numpy')
Fy = sp.lambdify(t, y, 'numpy')
FVx = sp.lambdify(t, Vx, 'numpy')
FVy = sp.lambdify(t, Vy, 'numpy')
FWx = sp.lambdify(t, Wx, 'numpy')
FWy = sp.lambdify(t, Wy, 'numpy')
Fcx = sp.lambdify(t, curvature_x, 'numpy')
Fcy = sp.lambdify(t, curvature_y, 'numpy')

X = Fx(T)
Y = Fy(T)

# Проверка на скалярность и преобразование в массивы
vx_val = FVx(T)*0.2
if np.isscalar(vx_val):
    vx_val = np.full_like(T, vx_val)
vy_val = FVy(T)*0.2
if np.isscalar(vy_val):
    vy_val = np.full_like(T, vy_val)

WX_val = FWx(T)*0.05
if np.isscalar(WX_val):
    WX_val = np.full_like(T, WX_val)
WY_val = FWy(T)*0.05
if np.isscalar(WY_val):
    WY_val = np.full_like(T, WY_val)

CX = Fcx(T)
if np.isscalar(CX):
    CX = np.full_like(T, CX)
CY = Fcy(T)
if np.isscalar(CY):
    CY = np.full_like(T, CY)

VX = vx_val
VY = vy_val
WX = WX_val
WY = WY_val

fig = plt.figure()
fig.canvas.manager.set_window_title('Вариант 22')
ax = fig.add_subplot(1,1,1)
ax.axis('equal')
ax.set_xlim([min(X)-1, max(X)+1])
ax.set_ylim([min(Y)-1, max(Y)+1])

ax.plot(X,Y,'black')
point = ax.plot(X[0],Y[0],marker='o',markerfacecolor='grey',markeredgecolor='red',markersize=9)[0]

# Вектор скорости
v_line = ax.plot([X[0],X[0]+VX[0]],[Y[0],Y[0]+VY[0]],'r',label='Скорость (V)')[0]
x_v_arr = np.array([-0.05,0,-0.05])
y_v_arr = np.array([0.05,0,-0.05])
r_x_v,r_y_v = rot2d(x_v_arr,y_v_arr,math.atan2(VY[0],VX[0]))
v_arrow = ax.plot(r_x_v+X[0]+VX[0], r_y_v+Y[0]+VY[0],'r')[0]

# Вектор ускорения
w_line = ax.plot([X[0],X[0]+WX[0]],[Y[0],Y[0]+WY[0]],'g',label='Ускорение (W)')[0]
x_w_arr = np.array([-0.05,0,-0.05])
y_w_arr = np.array([0.05,0,-0.05])
r_x_w,r_y_w = rot2d(x_w_arr,y_w_arr,math.atan2(WY[0],WX[0]))
w_arrow = ax.plot(r_x_w+X[0]+WX[0], r_y_w+Y[0]+WY[0],'g')[0]

# Радиус-вектор
r_line = ax.plot([0,X[0]],[0,Y[0]],'b',label='Радиус-вектор (R)')[0]
x_r_arr = np.array([-0.05,0,-0.05])
y_r_arr = np.array([0.05,0,-0.05])
r_x_r,r_y_r = rot2d(x_r_arr,y_r_arr,math.atan2(Y[0],X[0]))
r_arrow = ax.plot(r_x_r+X[0], r_y_r+Y[0],'b')[0]

# Радиус кривизны
curvature_radius = ax.plot([X[0],X[0]+CX[0]],[Y[0],Y[0]+CY[0]],
                           'black',linestyle='--',label='Радиус кривизны (CR)')[0]

raw_text = 'R   = {r}\nV   = {v}\nW   = {w}\nCR = {cr}'
text = ax.text(0.03,0.03,raw_text.format(r=mdl(X,Y,0), v=mdl(VX,VY,0), w=mdl(WX,WY,0), cr=mdl(CX,CY,0)),
               transform=ax.transAxes,fontsize=8)

def animate(i):
    point.set_data([X[i]], [Y[i]])

    
    v_line.set_data([X[i],X[i]+VX[i]],[Y[i],Y[i]+VY[i]])
    r_x_v,r_y_v = rot2d(x_v_arr,y_v_arr,math.atan2(VY[i],VX[i]))
    v_arrow.set_data(r_x_v+X[i]+VX[i],r_y_v+Y[i]+VY[i])

    w_line.set_data([X[i],X[i]+WX[i]],[Y[i],Y[i]+WY[i]])
    r_x_w,r_y_w = rot2d(x_w_arr,y_w_arr,math.atan2(WY[i],WX[i]))
    w_arrow.set_data(r_x_w+X[i]+WX[i], r_y_w+Y[i]+WY[i])

    r_line.set_data([0,X[i]],[0,Y[i]])
    r_x_r,r_y_r = rot2d(x_r_arr,y_r_arr,math.atan2(Y[i],X[i]))
    r_arrow.set_data(r_x_r+X[i], r_y_r+Y[i])

    curvature_radius.set_data([X[i],X[i]+CX[i]],[Y[i],Y[i]+CY[i]])

    text.set_text(raw_text.format(r=mdl(X,Y,i),
                                  v=mdl(VX,VY,i),
                                  w=mdl(WX,WY,i),
                                  cr=mdl(CX,CY,i)))
    return point,v_line,v_arrow,curvature_radius

animation = FuncAnimation(fig, animate, frames=len(T), interval=1)
ax.legend(fontsize=7)
plt.show()
