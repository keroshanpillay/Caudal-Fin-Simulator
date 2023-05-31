import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# constants
rho = 997  # water density in kg/m^3
H = 0.1  # height of the fin in m
L = 0.1  # length of the fin in m
theta0 = 0.1  # max angle in radians
CD = 0.2  # drag coefficient
CD_rect = 1.2
m = 4
N = 100  # number of fin elements
dr = L/N  # length of each element
phi = np.pi/6


# initial conditions
initial_position = 0
initial_velocity = 0
initial_conditions = [initial_position, initial_velocity]

# target position
target = 1 # in m

# time
total_time = 100.0  # total time for the journey in seconds
t = np.linspace(0, total_time, num=1000)

omega = 2*np.pi

def get_theta0(x, target):
    #if we are not there, swim with all our might
    if x < target: 
        theta0 = 0.1
    #if we are there, don't swim  
    else: 
        theta0 = 0
    return theta0

def blade_element_thrust(t, theta0, U0, omega):
    # initialize arrays to store the results
    Fn = np.zeros(N)  # normal force on each element
    v_n = np.zeros(N)  # velocity normal to each element
    v_ndot = np.zeros(N)
    Ma = np.zeros(N)  # added mass force on each element

    # loop over each element
    for i in range(N):
        r = i * dr  # position along the fin
        
        theta = theta0 * np.cos(omega*t)
        thetad = -omega * theta0 * np.sin(omega*t)
        thetadd = -omega**2 * theta0 * np.cos(omega*t)
        
        # theta2 = theta0 * np.cos(omega*t+phi)
        # thetad2 = -omega *phi * theta0 * np.sin(omega*t)
        # thetadd2 = -omega**2 * phi**2 * theta0 * np.cos(omega*t)
        
        alpha = theta
        # alpha2 = theta+theta2
        
        # calculate the normal velocity for this element
        v_n[i] = U0 * np.sin(theta) + r * thetad

        # calculate the normal force for this element
        Ma[i] = rho * alpha * (dr) * (np.pi/4) * H**2
        v_ndot[i] = U0*np.cos(theta)*thetad + r*thetadd  # derivative of the normal velocity

    for i in range(N):
        Fn_qs = 0.5 * rho * CD_rect * v_n[i] * abs(v_n[i]) * H * dr
        Fn_a = Ma[i] * v_ndot[i]
        Fn[i] = -Fn_qs - Fn_a

    # integrate the normal forces to get the total thrust
    FT = np.sum(Fn)
    
    return FT


def thrust(t, x, U0, omega, target):
    #Calculate theta0 - a fish's brain would do this
    theta0 = get_theta0(x, target)
    # calculate the thrust using blade element theory
    FT = blade_element_thrust(t, theta0, U0, omega)
    return FT

def thrust_analytical(t, x, U0, omega, target):
    
    #Calculate theta0 - a fish's brain would do this
    theta0 = get_theta0(x, target)
    
    theta = theta0*np.cos(omega*t)
    thetad = -omega * theta0 *np.sin(omega*t)
    thetadd = -omega**2 * theta0 * np.cos(omega*t)

    #Thrust equation from class
    FT = -rho * (np.pi/4) * H**2 * (0.5 * L**2 * thetadd + U0 * L * np.cos(theta) * thetad)*np.sin(theta) - 0.5 * rho * CD * H * (0.5 * L**3 * thetadd**2 + L**2 * U0 * thetad * np.sin(theta) + U0**2 * L * np.sin(theta)**2)*np.sin(theta)

    return FT


def equations_of_motion(y, t, target, omega):
    x, v = y
    a = (thrust(t, x, v, omega, target))/m - (0.5 * rho * CD * H * L * v**2)/m
    return [v, a]

def equations_of_motion_analytical(y, t, target, omega):
    x, v = y
    a = (thrust(t, x, v, omega, target))/m - (0.5 * rho * CD * H * L * v**2)/m
    return [v, a]

# solve the system of equations
solution = odeint(equations_of_motion, initial_conditions, t, args=(target, omega))
analytical_solution = odeint(equations_of_motion_analytical, initial_conditions, t, args=(target, omega))

# extract position and velocity
position = solution[:, 0]
velocity = solution[:, 1]

position_analytical = analytical_solution[:, 0]
velocity_analytical = analytical_solution[:, 1]

# calculate acceleration
acceleration = np.gradient(velocity, t)
analytical_acceleration = np.gradient(velocity_analytical, t)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse_value = rmse(position, position_analytical)
print("RMSE:", rmse_value)


# create the plots
fig1 = plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, position)
plt.plot(t, position_analytical)
plt.title('Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')

plt.subplot(3, 1, 2)
plt.plot(t, velocity)
plt.plot(t, velocity_analytical)
plt.title('Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')

plt.subplot(3, 1, 3)
plt.plot(t, acceleration)
plt.plot(t, analytical_acceleration)
plt.title('Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')

plt.tight_layout()
plt.show()
