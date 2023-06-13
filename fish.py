import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
import time

def swim(flipping_frequency, flipping_amplitude, phase_shift, target, amplitude_controller, elementRatio, plot=False):

  rho = 997  # water density in kg/m^3
  fork_length = 1
  H = 0.285*fork_length
  L = 0.1*H
  H_fish = 0.27*fork_length
  L_fish = fork_length
  SA_fish_front = H_fish**2
  CD_fish = 0.75  # drag coefficient
  CD_rect = 1.2
  C_rd = 1 # coeff of rotational drag

  m = 18
  N = 100  # number of fin elements
  ## Dont forget to define dr = L/N

  # Random Values defined here
  target = 2
  amplitude_controller = 1

  omega = flipping_frequency * 2 * np.pi

  # L1 + L2 = L
  # L1 + L1 * element_ratio = L
  L1 = L / (1 + elementRatio)
  L2 = L - L1

  I_fish = (1/12) * m * (L_fish**2 + H_fish**2)
  Ma = rho * np.pi / 4 * H**2

  # time
  total_time = 3  # total time for the journey in seconds
  num = total_time * 50
  t = np.linspace(0, total_time, num)

  xf = np.zeros((3, num))
  theta = np.zeros((3, num)) 
  alpha1 = np.zeros((3, num))
  alpha2 = np.zeros((3, num))

  # xf[0], xf[1], xf[2] => Fish position, Velocity, Acceleration
  # alpha1[0], alpha1[1], alpha1[2] => Fin angle, angular velocity, angular acceleration
  # alpha2[0], alpha2[1], alpha2[2] => Fin angle, angular velocity, angular acceleration
  # theta[0], theta[1], theta[2] => Fish angle, angular velocity, angular acceleration
  # v_n => normal velocity
  # dv_n => dv_n/dt

  def v_normal_1(xf, r, alpha1, theta):
    v_n = xf[1] * np.sin(alpha1[0]) + r * (alpha1[1] + theta[1])
    return v_n

  def v_normal_2(xf, r, alpha1, alpha2, theta):
    v_n = xf[1] * np.sin(alpha2[0]) + r * (alpha2[1] + theta[1]) + L1 * (theta[1] + alpha1[1]) * np.cos(alpha2[0]-alpha1[0])
    return v_n

  def vd_normal_1(xf, r, alpha1, theta):
    dv_n = r * (theta[2] + alpha1[2]) + alpha1[1] * xf[1] * np.cos(alpha1[0]) + xf[2] * np.sin(alpha1[0])
    return dv_n

  def vd_normal_2(xf, r, alpha1, alpha2, theta):
    dv_n = r * alpha2[2] + alpha2[1] * xf[1] * np.cos(alpha2[0]) + L1 * alpha1[2] * np.cos(alpha2[0] + alpha1[0]) - L1 * (theta[1] + alpha1[1]) * (alpha2[1] - alpha1[1]) * np.sin(alpha2[0] - alpha1[0])
    return dv_n

  def dFd_a(dv_n, dr):
    dFd_a = Ma * dv_n * dr
    return dFd_a

  def dFd_s(v_n, dr):
    dFd_s = 0.5 * rho * v_n * np.abs(v_n) * CD_rect * H * dr
    return dFd_s

  def dFd_net(xf, r, alpha1, alpha2, theta, dr, fin):
    if fin == 1:
      dFd = dFd_a(vd_normal_1(xf, r, alpha1, theta), dr) + dFd_s(v_normal_1(xf, r, alpha1, theta), dr)
    else:
      dFd = dFd_a(vd_normal_2(xf, r, alpha1, alpha2, theta), dr) + dFd_s(v_normal_2(xf, r, alpha1, alpha2, theta), dr)
    return dFd

  def Fd_fin(t, xf, theta, phase_shift, fin):

    dFd = np.zeros(num)
    alpha1 = get_alpha(t, omega, 0)
    alpha2 = get_alpha(t, omega, phase_shift)

    if fin == 1:
      dr = L1/N
    else:
      dr = L2/N
    
    for i in range(N):
      r = i * dr
      dFd[i] = dFd_net(xf, r, alpha1, alpha2, theta, dr, fin)

    Fd = np.sum(dFd)
    
    return Fd

  def Ft_fun(t, xf, theta, phase_shift):

    alpha1 = get_alpha(t, omega, 0)
    alpha2 = get_alpha(t, omega, phase_shift)

    Ft1 = - np.sin(alpha1[0]) * Fd_fin(t, xf, theta, 0, 1)
    Ft2 = - np.sin(alpha2[0]) * Fd_fin(t, xf, theta, phase_shift, 2)

    Ft = Ft1 + Ft2

    return Ft

  def dT_fun(xf, r, alpha1, alpha2, theta, dr, fin):

    if fin == 1:
      dT = - dFd_net(xf, r, alpha1, alpha2, theta, dr, fin) * (np.cos(alpha1[0] * L_fish/2 + r))
    else:
      dT = - dFd_net(xf, r, alpha1, alpha2, theta, dr, fin) * (np.cos(alpha2[0] * L_fish/2 + r + L1 * np.cos(alpha2[0]-alpha1[0])))
    return dT

  def T_tot(t, xf, theta, phase_shift):

    dT1 = np.zeros(N)
    dT2 = np.zeros(N)

    alpha1 = get_alpha(t, omega, 0)
    alpha2 = get_alpha(t, omega, phase_shift)

    for i in range(N):
      dT1[i] = dT_fun(xf, i*L1/N, alpha1, alpha2, theta, L1/N, 1)
      dT2[i] = dT_fun(xf, i*L2/N, alpha1, alpha2, theta, L2/N, 2)

    T1 = np.sum(dT1)
    T2 = np.sum(dT2)

    T = T1 + T2

    return T

  def fish_drag(xf):
    drag = 0.5 * rho * CD_fish * SA_fish_front * xf[1]**2
    return drag

  def rotation_drag(alpha):
    drag = 0.5 * rho * (alpha[1] * L_fish/4) * np.abs(alpha[1] * L_fish/4) * L_fish * H_fish * C_rd
    return drag


  def get_alpha(t, omega, phase_shift):
    alpha = flipping_amplitude * np.sin(omega * t + phase_shift)
    alphad = omega * np.pi/6 * np.cos(omega * t + phase_shift)
    alphadd = - omega**2 * np.pi/6 * np.sin(omega * t + phase_shift)
    return [alpha, alphad, alphadd]

  dt = total_time / num # differential time

  Ft = np.zeros(num)

  def equations_of_motion(t, z, ax, thetadd, phase_shift):
    x, vx, theta, thetad = z
    theta_temp = [theta, thetad, thetadd]
    x_temp = [x, vx, ax]
    rot_drag = rotation_drag(theta_temp)
    if np.abs(rot_drag) > np.abs(T_tot(t, x_temp, theta_temp, phase_shift)):
      rot_drag = T_tot(t, x_temp, theta_temp, phase_shift)
    thetadd = (T_tot(t, x_temp, theta_temp, phase_shift) - rot_drag)/I_fish
    dvx_dt =  (Ft_fun(t, x_temp, theta_temp, phase_shift) - fish_drag(x_temp))/m

    return [vx, dvx_dt, thetad, thetadd]

  sol = solve_ivp(equations_of_motion, [0, total_time], [xf[0, 0], xf[1, 0], theta[0, 0], theta[1, 0]], args=( 0, 0, phase_shift), t_eval=t)  # args: ax, thetadd, phase_shift

  
  xf[0] = sol.y[0]
  xf[1] = sol.y[1]
  theta[0] = sol.y[2]
  theta[1] = sol.y[3]
  alpha = np.pi/6 * np.sin(omega * t)

  # plt.subplot(4,1,1)
  # plt.plot(t, alpha, label = "Fin angle")
  # plt.legend()

  # plt.subplot(4, 1, 2)
  # plt.plot(t, theta[0], label = "Fish angle")
  # plt.legend()

  # plt.subplot(4,1,3)
  # plt.plot(t, xf[1], label = "Fish velocity")
  # plt.legend()

  # plt.subplot(4,1,4)
  # plt.plot(t, xf[0], label = "Fish position")
  # plt.legend()
  # plt.show
  
  maxVelocity = np.max(xf[1])
  timeToTarget = 0
  energyUsedToTarget = 0
  
  return timeToTarget, energyUsedToTarget, maxVelocity

# start = time.time()
# vals = swim(np.pi, np.pi/6, np.pi/4, 1, 1, 1, False)
# end = time.time()

# print(vals)
# print("Time taken: " + str(end - start) + " seconds")