import matplotlib.pyplot as plt
#### Analyse du corpus ####

# Nombre de jugements dans le corpus
def count_lines(fichier):
	corpus = open(fichier)
	nb_lines = 0
	for line in corpus:
		nb_lines = nb_lines + 1
	print "Nombre de jugements = ", nb_lines

count_lines("movie_lens.csv")

# Nombre d'utilisateurs
def count_users(fichier):
	nb_users = 0
	tab_users = []
	corpus = open(fichier)
	for line in corpus:
		aut,film,rate = line.split('|')
		if aut not in tab_users:
			tab_users.append(aut)
			nb_users = nb_users + 1
	print "Nombre d'utilisateurs = ", nb_users

count_users("movie_lens.csv")

# Nombre de films differents
def count_films(fichier):
	nb_films = 0
	tab_films = []
	corpus = open(fichier)
	for line in corpus:
		aut,film,rate = line.split('|')
		if film not in tab_films:
			tab_films.append(film)
			nb_films = nb_films + 1
	print "Nombre de films = ", nb_films

count_films("movie_lens.csv")


# Nombre de films differents
def det_oldest_film(fichier):
	oldest_film_date = 9999
	oldest_film_name = ""
	oldest_l = []
	tab_films = []
	corpus = open(fichier)
	for line in corpus:
		aut,film,rate = line.split('|')
		if film not in tab_films:
			tab_films.append(film)
	for f in tab_films:
		if f <> "unknown":
			titre = f[0:-6]
			date = f[-5:-1]
			if int(date) < oldest_film_date:
				oldest_film_date = int(date)
				oldest_film_name = titre
	oldest_l.append(oldest_film_name)
	for f in tab_films:
		if f <> "unknown":
			titre = f[0:-6]
			date = f[-5:-1]
			if int(date) == oldest_film_date and titre not in oldest_l:
				oldest_film_name = oldest_film_name + " ; " + titre
	print "Film(s) le(s) plus ancien = ", oldest_film_name, "(", oldest_film_date, ")"

det_oldest_film("movie_lens.csv")

# Nombre de films differents
def det_newest_film(fichier):
	newest_film_date = 0
	newest_film_name = ""
	newest_l = []
	tab_films = []
	corpus = open(fichier)
	for line in corpus:
		aut,film,rate = line.split('|')
		if film not in tab_films:
			tab_films.append(film)
	for f in tab_films:
		if f <> "unknown":
			titre = f[0:-6]
			date = f[-5:-1]
			if int(date) > newest_film_date:
				newest_film_date = int(date)
				newest_film_name = titre
	newest_l.append(newest_film_name)
	for f in tab_films:
		if f <> "unknown":
			titre = f[0:-6]
			date = f[-5:-1]
			if int(date) == newest_film_date and titre not in newest_l:
				newest_film_name = newest_film_name + " ; " + titre
	print "Film(s) le(s) plus recent = ", newest_film_name, "(", newest_film_date, ")"

det_newest_film("movie_lens.csv")

# Distribution des notes
def rep_note(fichier):
	tab_notes = []
	corpus = open(fichier)
	for line in corpus:
		aut,film,rate = line.split('|')
		tab_notes.append(int(rate))
	return tab_notes

repartition_notes = rep_note("movie_lens.csv")

#plt.hist(repartition_notes, bins=5)
#plt.savefig("plot.png")

# Nombre de jugement par utilisateur
def count_jug_users(fichier):
	tab_jug = [0] * 943
	corpus = open(fichier)
	for line in corpus:
		aut,film,rate = line.split('|')
		tab_jug[int(aut)-1] = tab_jug[int(aut)-1] + 1
	return tab_jug

repartition_jug = count_jug_users("movie_lens.csv")
y = repartition_jug
x = range(0,len(y))
plt.plot(x,y)
plt.savefig("plot2.png")

