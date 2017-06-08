#!/usr/bin/python

'''
Created on 03.05.2017
Main program module
@author: raqu
@author: gepard5
'''

#file containing planets configuration
config_file = 'planets'

show_every_step = False
create_steps_graph = False
steps_results_list = []

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


def createWorld(satstate):
	f = open(config_file, 'r')
	f.readline()
	world_params = f.readline().strip().split(" ")
	planets_number = int(world_params[0])
	start_planet = int(world_params[1])
	dest_planet = int(world_params[2])
	sun = Sun(f.readline())
	planets = []
	for i in range(planets_number):
		planets.append(Planet(f.readline()))
	solarsys = SolarSystem(planets, sun)
	satelite = Satelite(planets[start_planet], planets[dest_planet], satstate)
	return solarsys, satelite


def simulate(startstate, dt, tmax):
	solarsyst, satelite = createWorld(startstate)
	t = 0
	result = satelite.getDistanceToDestination()
	while t < tmax and satelite.fly(solarsyst, dt):
		#print("x={} y={}".format(satelite.x, satelite.y))
		dist = satelite.getDistanceToDestination()
		if dist < result:
			result = dist
		t += dt
	return result

class SimulatedAnnealing():
	def __init__(self):
		self.state = None
		self.dt = None
		self.da = None
		self.dv = None

		self.tmax = None
		self.vmax = None
		self.amax = None

		self.steps = None

		self.temperature = None
		self.freezing = None

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
		if self.temperature < 0.001:
			self.temperature= 0.0
			return 0.0
		diff = float(math.fabs(resx-resy)) * 100.0
		if diff  < 0.1:
			diff = 0.1
		exponent = float( - diff / self.temperature )
		return float(math.exp(exponent))

	def start(self):
		self.read_configuration()
		archieve_results = []

		x = (random.uniform(0, self.tmax), random.uniform(0, self.amax), random.uniform(0, self.vmax)) 
		current_best_result = simulate(x, self.dt, self.tmax)
		i = 0
		for  i in range(self.steps):
			y = random.choice(self.neighbourhood(x))
			new_result = simulate(y, self.dt, self.tmax)
			if show_every_step == True:
				print "Steps and current result: "
				print i
				print new_result
			if new_result < current_best_result:
				x = y
				current_best_result = new_result
			elif float(random.uniform(0, 1))  < self.probability(current_best_result, new_result):
				x = y
				current_best_result = new_result
			if create_steps_graph == True:
				archieve_results.append(current_best_result)
			if current_best_result <= self.minimal_result :
				break
			self.temperature *= self.freezing

		global steps_results_list
		steps_results_list = archieve_results
		return (current_best_result, i, x)

	def read_configuration(self):
		f = open(config_file, 'r')
		anneal_config = f.readline().strip().split(" ")
		self.dt = float(anneal_config[0])
		self.da = float(anneal_config[1])
		self.dv = float(anneal_config[2])
		self.tmax = float(anneal_config[3])
		self.amax = float(anneal_config[4])
		self.vmax = float(anneal_config[5])
		self.steps = int(anneal_config[6])
		self.temperature = float(anneal_config[7])
		self.freezing = float(anneal_config[8])
		self.minimal_result = float(anneal_config[9])


#annealing
import sys, getopt

show_chart = False
algorithm_runs = 1
show_every_annealing = False
only_graph = False
sat_state = None

try:
	opts, args = getopt.getopt(sys.argv[1:],"hvi:r:ag:s",["ifile=","show_every_step"])
except getopt.GetoptError:
	print 'main.py -i <inputfile> -o <outputfile>'
	sys.exit(2)

for opt, arg in opts:
	if opt == '-h':
		print 'main.py [-ha] [-r <anneal_tries] [-i <config_file>] [-v] [-g 0, 0, 0]'
		sys.exit(2)
	elif opt in ("-i", "--ifile"):
		config_file = arg
	elif opt == '-v':
		show_chart = True
	elif opt == '-r':
		algorithm_runs = int(arg)
	elif opt == "--show_every_step":
		show_every_step = True
	elif opt == '-a':
		show_every_annealing = True
	elif opt == '-g':
		only_graph = True
		sat_state = tuple(map(float, arg.split(',')))
	elif opt == '-s':
		create_steps_graph = True


# annealing algorithm runs
from matplotlib import pyplot as plt

# world plot sizes
RANG = 1e3

anneal = SimulatedAnnealing()
best_score = sys.float_info.max
best_state = []
best_steps = 0

if not only_graph:
	for i in range(algorithm_runs):
		(score, steps, result_state) = anneal.start()
		if show_every_annealing == True:
			print "Numer: " + str(i)
			print "Wynik: " + str(score)
			print "Ilosc krokow: " + str(steps)
			print "Stan satelity: " + str(result_state)
		if score < best_score :
			best_score = score
			best_state = result_state
			best_steps = steps
		
		# save steps results plot
		plt.clf()
		plt.plot(range(len(steps_results_list)), steps_results_list, 'ro', markersize=1)
		plt.axis([0, 1.1*anneal.steps, 0, 2*RANG])
		plt.grid(True)
		plt.savefig('run{}_{}_steps.png'.format(i, config_file),
				bbox_inches='tight', dpi=300)
		
	print "Najlepszy wynik: " + str(best_score)
	print "Ilosc krokow: " + str(best_steps)
	print "Stan satelity: " + str(best_state)
	
else:
	anneal.read_configuration()
	best_state = sat_state

# animation
import numpy as np
import sys
from matplotlib import animation

if show_chart == False :
	sys.exit(0)

s, sat = createWorld( best_state )
DT = anneal.dt

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7,7)

TIME_LABEL = 'world time: {}'
DIST_LABEL = 'far: {}'

ax = plt.axes(xlim=(-RANG, RANG), ylim=(-RANG, RANG))
time_text = ax.text(-RANG+100, -RANG+150, TIME_LABEL.format(0), fontsize=10)
dist_text = ax.text(-RANG+100, -RANG+75, DIST_LABEL.format(0), fontsize=10)

satGraph = plt.Circle((sat.x, sat.y), 8, fc='r')
sunGraph = plt.Circle((0, 0), s.sun.r, fc='yellow', color='black')
planetGraphs = []
for p in s.planets:
	if p == sat.startplanet:
		plaGraph = plt.Circle((p.x, p.y), p.r, fc='m', color='red')
	elif p == sat.destplanet:
		plaGraph = plt.Circle((p.x, p.y), p.r, fc='m', color='red')
	else:
		plaGraph = plt.Circle((p.x, p.y), p.r, fc='c', color='black')
	planetGraphs.append((p, plaGraph))

pgraphs = []
def init():
	global pgraphs
	for (p, pgraph) in planetGraphs:
		pgraph.center = (p.x, p.y)
		ax.add_patch(pgraph)
		pgraphs.append(pgraph)

	satGraph.center = (sat.x, sat.y)
	ax.add_patch(satGraph)
	ax.add_patch(sunGraph)
	return pgraphs + [sunGraph, satGraph, time_text, dist_text]

stopped = False
def animate(i):
	global stopped, pgraphs
	if i == int(anneal.tmax/DT) - 1:
		stopped = True
		
	if not stopped:
		sat.fly(s, DT)
		pgraphs = []
		for (p, pgraph) in planetGraphs:
			pgraph.center = (p.x, p.y)
			pgraphs.append(pgraph)
	
		satGraph.center = (sat.x, sat.y)
		time_text.set_text(TIME_LABEL.format((i+1)*float(DT)))
		
		dist = sat.destplanet.distance(sat.x, sat.y)
		dist_text.set_text(DIST_LABEL.format(float(dist)))

	return pgraphs + [sunGraph, satGraph, time_text, dist_text]


def main(tmax):
	f = int(tmax/DT)
	anim = animation.FuncAnimation(fig, animate, 
								   init_func=init, 
								   frames=f, 
								   interval=30*DT,
								   blit=True)

	plt.show()

main(anneal.tmax)

