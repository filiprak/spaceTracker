'''
Created on 10.06.2017
Module contains classes that represents universe elements
@author: raqu
'''

import math

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

    def __init__(self, str):
        params = str.strip().split(" ")
        self.r = int(params[0])
        self.d = int(params[1])
        self.m = float(params[2])
        self.alfa = float(params[3])
        self.t = int(params[4])
        self.x = self.d * math.cos(self.alfa)
        self.y = self.d * math.sin(self.alfa)

    
    def distance(self, x, y):
        x -= self.x
        y -= self.y
        return math.sqrt(x*x + y*y)
    
    def move(self, dt):
        self.alfa += 2 * math.pi * (float(dt) / self.t)
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

    def __init__(self, str):
        params = str.strip().split(" ")
        self.r = int(params[0])
        self.m = float(params[1])

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
            #print("gravity: {}".format((gx, gy)))
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
        else:
            # satelite have to be ready for start
            if self.t0 - dt <= 0:
                vinx, viny = self.inertiaStart()
                self.v0x += vinx
                self.v0y += viny
            self.x = self.startplanet.x
            self.y = self.startplanet.y
        
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
    
    
    