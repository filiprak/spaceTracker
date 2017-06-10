#!/usr/bin/python

'''
Created on 03.05.2017
Main program module
@author: raqu
@author: gepard5
'''

from universe import Planet, Satelite, SolarSystem, Sun

# default file containing planets configuration
config_file = 'planets'

# run flags
show_every_step = False
create_steps_graph = False
steps_results_list = []

import math
import random


def createWorld(satstate):
	'''
	Creates universe from config file params
	:param satstate: satelite state at the begining
	'''
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
	'''
	Runs world simulation
	:param startstate: begining state of satelite
	:param dt: time quantum
	:param tmax: maximum run time
	'''
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
	'''
	Implements simulated annealing algorithm
	'''
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
		'''
		Returns list of neighbours of given state
		:param state:
		'''
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
		'''
		Starts simulated annealing search
		'''
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
		'''
		Reads all the neccessary params from config file
		'''
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





# handle with command line options
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





# annealing algorithm run
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
		
		# save steps results to a plot
		if create_steps_graph:
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




# result world animation show
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

