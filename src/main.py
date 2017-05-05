'''
Created on 03.05.2017
Main program module
@author: raqu
'''

import math
import random

def G():
    return 6.67408e-011

class Planet():
    def __init__(self, r=6378, d=149e6, m=5.98e24, alfa=0, t=365):
        self.r = r
        self.d = d
        self.m = m
        self.alfa = alfa
        self.t = t
        self.x = d * math.cos(alfa)
        self.y = d * math.sin(alfa)
    
    def distance(self, x, y):
        x -= self.x
        y -= self.y
        return math.sqrt(x*x + y*y)
    
    def move(self, dt):
        self.alfa += 2 * math.pi * (dt / self.t)
        if self.alfa > 2 * math.pi:
            self.alfa -= 2 * math.pi
        self.x = self.d * math.cos(self.alfa)
        self.y = self.d * math.sin(self.alfa)
    
    def collides(self, x, y):
        return (self.x - x)*(self.x - x) + (self.y - y)*(self.y - y) <= self.r*self.r

    def gravity(self, x, y):
        x -= self.x
        y -= self.y
        d = math.sqrt(x*x + y*y)
        if d == 0:
            return (0, 0)
        grav = G() * self.m / d*d
        gravx = grav * x / d
        gravy = grav * y / d
        return (-gravx, -gravy)
    
class Sun():
    def __init__(self, r=696342, m=1.98e30):
        self.r = r
        self.m = m
    
    def collides(self, x, y):
        return y*y + x*x <= self.r*self.r
    
    def gravity(self, x, y):
        d = math.sqrt(x*x + y*y)
        grav = G() * self.m / d*d
        gravx = grav * x / d
        gravy = grav * y / d
        return (-gravx, -gravy)
    
class SolarSystem():
    def __init__(self, planets, sun):
        self.planets = planets
        self.sun = sun
    
    def move(self, dt):
        for planet in self.planets:
            planet.move(dt)
    
        
class Satelite():
    def __init__(self, planet, destplanet, state):
        self.startplanet = planet
        self.destplanet = destplanet
        self.collided = False
        self.x = planet.x
        self.y = planet.y
        (self.t0, self.a0, self.v0) = state
        self.v0x = self.v0 * math.cos(self.a0)
        self.v0y = self.v0 * math.sin(self.a0)
        
        vinx, viny = self.inertiaStart()
        self.v0x += vinx
        self.v0y += viny
        self.inSpace = False
    
    def inertiaStart(self):
        vin = 2 * math.pi * self.startplanet.d / self.startplanet.t
        vinx = -vin * math.sin(self.startplanet.alfa)
        viny = vin * math.cos(self.startplanet.alfa)
        return vinx, viny
    
    def getDistanceToDestination(self):
        return self.destplanet.distance(self.x, self.y)
        
    def fly(self, solarsystem, dt):
        if self.collided:
            return False
        self.t0 -= dt
        if self.t0 <= 0:
            (gx, gy) = self.gravity(solarsystem)
            print("gravity: {}".format((gx, gy)))
            dx = self.v0x*dt + 0.5 * gx * dt*dt
            dy = self.v0y*dt + 0.5 * gy * dt*dt
            dvx = gx * dt
            dvy = gy * dt
            #update
            self.x += dx
            self.y += dy
            self.v0x += dvx
            self.v0y += dvy
        
            if not self.inSpace:
                if not self.startplanet.collides(self.x, self.y):
                    self.inSpace = True
        
            if solarsystem.sun.collides(self.x, self.y):
                self.collided = True
                return False
            for planet in solarsystem.planets:
                if planet.collides(self.x, self.y) and (planet != self.startplanet or self.inSpace):
                    self.collided = True
                    return False
        solarsystem.move(dt)
        return True
        
    def gravity(self, solarsys):
        gravx = 0.0
        gravy = 0.0
        
        (gx, gy) = solarsys.sun.gravity(self.x, self.y)
        gravx += gx
        gravy += gy
        
        for planet in solarsys.planets:
            (gx, gy) = planet.gravity(self.x, self.y)
            gravx += gx
            gravy += gy
        return (gravx, gravy)


def createWorld(satstate):
    earth = Planet(r=3, d=40, m=400000000, alfa=0, t=365)
    venus = Planet(r=4, d=72, m=400000000, alfa=3.2, t=243)
    sun = Sun(r=10, m=1000000000)
    solarsys = SolarSystem([earth, venus], sun)
    satelite = Satelite(earth, venus, satstate)
    return solarsys, satelite


def simulate(startstate, dt, tmax):
    solarsyst, satelite = createWorld(startstate)
    t = 0
    result = satelite.getDistanceToDestination()
    while t < tmax and satelite.fly(solarsyst, dt):
        print("x={} y={}".format(satelite.x, satelite.y))
        dist = satelite.getDistanceToDestination()
        if dist < result:
            result = dist
        t += dt
    return result

class SimulatedAnnealing():
    def __init__(self):
        self.state = (0, 0.3*math.pi, 0.1)
        self.dt = 0.1
        self.da = 0.1
        self.dv = 0.1
        
        self.dtime = 0.1
        self.tmax = 50
        self.vmax = 200000
        self.amax = 2*math.pi
    
        self.temperature = 100
        
    def neighbourhood(self, state):
        result = []
        (t, a, v) = state
        if t + self.dt <= self.tmax:
            result.append((t + self.dt, a, v))
        if t - self.dt >= 0:
            result.append((t - self.dt, a, v))
        if a + self.da <= self.amax:
            result.append((t, a + self.da, v))
        if a - self.da >= 0:
            result.append((t, a - self.da, v))
        if v + self.dv <= self.vmax:
            result.append((t, a, v + self.dv))
        if v - self.dv >= 0:
            result.append((t, a, v - self.dv))
        
        return result
    
    def probability(self, resx, resy):
        if self.temperature == 0:
            return 0
        exponent = - math.fabs(resx-resy) / self.temperature
        return math.exp(exponent)
        
    def start(self):
        x = self.state
        for  i in range(1000):
            y = random.choice(self.neighbourhood(x))
            resultx = simulate(x, self.dtime, self.tmax)
            resulty = simulate(y, self.dtime, self.tmax)
            print("x: {} {}".format(x, resultx))
            print("y: {} {}".format(y, resulty))

            if resulty < resultx:
                x = y
            elif random.uniform(0, 1) < self.probability(resultx, resulty):
                x = y
            self.temperature *= 0.99
        return x
            
        
#anneal = SimulatedAnnealing()

#print(simulate((0, 45, 200050000), 0.1, 50))

# animation
import numpy as np
import sys
from matplotlib import pyplot as plt
from matplotlib import animation

s, sat = createWorld((0, 1.57, 1))
DT = 0.2
print(sat.x, sat.y)

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 7)

RANG = 1e2

ax = plt.axes(xlim=(-RANG, RANG), ylim=(-RANG, RANG))

satGraph = plt.Circle((sat.x, sat.y), 0.8, fc='r')
sunGraph = plt.Circle((0, 0), s.sun.r, fc='yellow', color='black')
planetGraphs = []
for p in s.planets:
    plaGraph = plt.Circle((p.x, p.y), p.r, fc='c', color='black')
    planetGraphs.append((p, plaGraph))

print(sat.x, sat.y)
def init():
    pgraphs = []
    for (p, pgraph) in planetGraphs:
        pgraph.center = (p.x, p.y)
        ax.add_patch(pgraph)
        pgraphs.append(pgraph)
        
    satGraph.center = (sat.x, sat.y)
    ax.add_patch(satGraph)
    ax.add_patch(sunGraph)
    return pgraphs + [sunGraph, satGraph]

def animate(i):
    print(sat.x, sat.y)
    sat.fly(s, DT)
    #sys.stdin.read(1)
    pgraphs = []
    for (p, pgraph) in planetGraphs:
        pgraph.center = (p.x, p.y)
        pgraphs.append(pgraph)
    
    satGraph.center = (sat.x, sat.y)
    
    return pgraphs + [sunGraph, satGraph]


def main(dt, tmax):
    f = int(tmax/dt)
    anim = animation.FuncAnimation(fig, animate, 
                                   init_func=init, 
                                   frames=f, 
                                   interval=20,
                                   blit=True)

    plt.show()

main(DT, 1)


      
            