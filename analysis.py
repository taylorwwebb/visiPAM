import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import pandas as pd
import pingouin as pg
import csv
import argparse

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--diptest', action='store_true')
args = parser.parse_args()

if args.diptest:
	data_fname = './human_visiPAM_data_diptest.csv'
else:
	data_fname = './human_visiPAM_data.csv'

# Load data
file = open(data_fname, 'r')
csvreader = csv.reader(file)
# Get header
header = next(csvreader)
# Get rows
rows = []
for row in csvreader:
	rows.append(row)
# Close file
file.close()

# Get data for PAM
all_PAM_SSC_targ_animal = []
all_PAM_SSC_targ_chair = []
all_PAM_DSC_targ_animal = []
all_PAM_DSC_targ_chair = []
for r in range(len(rows)):
	if rows[r][0] == 'pam':
		if rows[r][8] != '-':
			dist_to_mean = float(rows[r][8])
			if rows[r][3].upper() == 'TRUE':
				if rows[r][5] == 'animal':
					all_PAM_SSC_targ_animal.append(dist_to_mean)
				else:
					all_PAM_SSC_targ_chair.append(dist_to_mean)
			else:
				if rows[r][5] == 'animal':
					all_PAM_DSC_targ_animal.append(dist_to_mean)
				else:
					all_PAM_DSC_targ_chair.append(dist_to_mean)
PAM_SSC_targ_animal = np.mean(all_PAM_SSC_targ_animal)
PAM_DSC_targ_animal = np.mean(all_PAM_DSC_targ_animal)
PAM_SSC_targ_chair = np.mean(all_PAM_SSC_targ_chair)
PAM_DSC_targ_chair = np.mean(all_PAM_DSC_targ_chair)

# Get subject IDs
subjIDs = []
for r in range(len(rows)):
	subjIDs.append(rows[r][0])
unique_subjIDs = np.unique(np.array(subjIDs)[np.array(subjIDs) != 'pam'].astype(int))

# Get data for each subject
all_SSC_targ_animal = []
all_SSC_targ_chair = []
all_DSC_targ_animal = []
all_DSC_targ_chair = []
for s in range(len(unique_subjIDs)):
	# Loop through trials and get subject data
	SSC_targ_animal = []
	SSC_targ_chair = []
	DSC_targ_animal = []
	DSC_targ_chair = []
	for r in range(len(rows)):
		if rows[r][0] == str(unique_subjIDs[s]):
			dist_to_mean = float(rows[r][8])
			if rows[r][3].upper() == 'TRUE':
				if rows[r][5] == 'animal':
					SSC_targ_animal.append(dist_to_mean)
				else:
					SSC_targ_chair.append(dist_to_mean)
			else:
				if rows[r][5] == 'animal':
					DSC_targ_animal.append(dist_to_mean)
				else:
					DSC_targ_chair.append(dist_to_mean)
	all_SSC_targ_animal.append(np.mean(SSC_targ_animal))
	all_SSC_targ_chair.append(np.mean(SSC_targ_chair))
	all_DSC_targ_animal.append(np.mean(DSC_targ_animal))
	all_DSC_targ_chair.append(np.mean(DSC_targ_chair))
# Convert to arrays
all_SSC_targ_animal = np.array(all_SSC_targ_animal)
all_SSC_targ_chair = np.array(all_SSC_targ_chair)
all_DSC_targ_animal = np.array(all_DSC_targ_animal)
all_DSC_targ_chair = np.array(all_DSC_targ_chair)
# Compute stats
all_SSC_targ_animal_mn = all_SSC_targ_animal.mean()
all_SSC_targ_animal_se = sem(all_SSC_targ_animal)
all_SSC_targ_chair_mn = all_SSC_targ_chair.mean()
all_SSC_targ_chair_se = sem(all_SSC_targ_chair)
all_DSC_targ_animal_mn = all_DSC_targ_animal.mean()
all_DSC_targ_animal_se = sem(all_DSC_targ_animal)
all_DSC_targ_chair_mn = all_DSC_targ_chair.mean()
all_DSC_targ_chair_se = sem(all_DSC_targ_chair)

# Report number of subjects included
print('N subjects = ' + str(len(all_SSC_targ_animal)))

# Function for creating combined violin and strip plot
def violin_strip_plot(x_loc, y_data, color, jitter=0.08, mean_bar_width=0.2, violin_width=0.4):
	# Exclude outliers from violin plot
	outlier_criterion = 2.5
	include_in_violin = np.logical_and((y_data < (y_data.mean() + (y_data.std() * 2.5))), (y_data > (y_data.mean() - (y_data.std() * 2.5))))
	# Violin plot
	parts = ax.violinplot([y_data[include_in_violin]], positions=[x_loc], widths=violin_width, showextrema=False)
	for pc in parts['bodies']:
		pc.set_facecolor(color)
		pc.set_alpha(0.5)
	# Plot standard deviation
	plt.errorbar([x_loc], [y_data.mean()], yerr=np.expand_dims([y_data.std(),y_data.std()],1),color='black',capsize=10)
	# Plot mean
	plt.plot([x_loc-(mean_bar_width/2),x_loc+(mean_bar_width/2)],[y_data.mean(),y_data.mean()],color='black')
	# Plot individual data points
	ind_x_loc = (np.random.rand(y_data.shape[0]) * jitter) - (jitter/2) + x_loc
	plt.scatter(ind_x_loc, np.array(y_data), color=color, s=2)
	return ind_x_loc

# Violin plot
ax = plt.subplot(111)
# Dummy figures for legend
plt.fill([100,101],[100,101],color='salmon',alpha=0.5)
plt.fill([100,101],[100,101],color='turquoise',alpha=0.5)
plt.scatter([100,101],[100,101],color='limegreen',s=40,edgecolors='black')
plt.scatter([100,101],[100,101],color='gray',s=40,edgecolors='black')
plt.scatter([100,101],[100,101],color='red',s=40,edgecolors='black')
# Source = animal, target = animal
SSC_targ_animal_ind_x_loc = violin_strip_plot(-0.2, all_SSC_targ_animal, 'salmon')
# Source = chair, target = animal
DSC_targ_animal_ind_x_loc = violin_strip_plot(0.2, all_DSC_targ_animal, 'turquoise')
# Source = chair, target = chair
SSC_targ_chair_ind_x_loc = violin_strip_plot(0.8, all_SSC_targ_chair, 'salmon')
# Source = animal, target = chair
DSC_targ_chair_ind_x_loc = violin_strip_plot(1.2, all_DSC_targ_chair, 'turquoise')
# Plot pair-wise comparisons
for s in range(len(all_SSC_targ_animal)):
	plt.plot([SSC_targ_animal_ind_x_loc[s], DSC_targ_animal_ind_x_loc[s]], [all_SSC_targ_animal[s], all_DSC_targ_animal[s]], color='gray', alpha=0.2, linewidth=0.5)
	plt.plot([SSC_targ_chair_ind_x_loc[s], DSC_targ_chair_ind_x_loc[s]], [all_SSC_targ_chair[s], all_DSC_targ_chair[s]], color='gray', alpha=0.2, linewidth=0.5)
# Plot PAM peformance
plt.scatter([-0.2,0.2,0.8,1.2],[PAM_SSC_targ_animal,PAM_DSC_targ_animal,PAM_SSC_targ_chair,PAM_DSC_targ_chair],color='limegreen',s=40,zorder=103,edgecolors='black')
plt.plot([-0.2,0.2],[PAM_SSC_targ_animal,PAM_DSC_targ_animal],color='black',zorder=100)
plt.plot([0.8,1.2],[PAM_SSC_targ_chair,PAM_DSC_targ_chair],color='black',zorder=100)
# Axes
plt.xlim([-.5,1.5])
plt.ylim([0,70])
plt.xticks([0,1], ['Animal', 'Chair'], fontsize=14)
plt.xlabel('Target category', fontsize=14)
plt.yticks([0,20,40,60],['0','20','40','60'], fontsize=14)
plt.ylabel('Distance from mean placement (pixels)', fontsize=14)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
# Legend
plt.legend(['Same category (human)', 'Different category (human)', 'VisiPAM'], frameon=False, fontsize=14, loc=2)
# Save
if args.diptest:
	plot_fname = './human_visiPAM_results_diptest.pdf'
else:
	plot_fname = './human_visiPAM_results.pdf'
plt.savefig(plot_fname, dpi=300, bbox_inches="tight")
plt.close()

# ANOVA
# Create dataframe
dist = np.array([all_SSC_targ_animal, all_SSC_targ_chair, all_DSC_targ_animal, all_DSC_targ_chair]).flatten()
subjID = np.tile(np.expand_dims(np.arange(len(all_SSC_targ_animal)),0),[4,1]).flatten()
SSC_DSC = np.concatenate([np.zeros(len(all_SSC_targ_animal) * 2), np.ones(len(all_SSC_targ_animal) * 2)]).astype(int)
targ_animal_chair = np.concatenate([np.zeros(len(all_SSC_targ_animal)), np.ones(len(all_SSC_targ_animal)), np.zeros(len(all_SSC_targ_animal)), np.ones(len(all_SSC_targ_animal))]).astype(int)
df = pd.DataFrame({'dist': dist, 'subjID': subjID, 'SSC_DSC': SSC_DSC, 'targ_animal_chair': targ_animal_chair})
# Run 2 X 2 repeated-measures anova
aov = pg.rm_anova(dv='dist', within=['SSC_DSC', 'targ_animal_chair'], subject='subjID', data=df)
print(aov)
# Save
if args.diptest:
	results_fname = './aov_results_diptest.txt'
else:
	results_fname = './aov_results.txt'
fid = open(results_fname,'w')
fid.write(str(aov))	
fid.close()

# Report average distance from human mean in within- vs. between-category conditions
print('Distance to average human mappings')
human_SSC_mn = np.mean(np.concatenate([all_SSC_targ_animal,all_SSC_targ_chair]))
human_DSC_mn = np.mean(np.concatenate([all_DSC_targ_animal,all_DSC_targ_chair]))
overall_human_mn = np.mean(np.concatenate([all_SSC_targ_animal,all_SSC_targ_chair,all_DSC_targ_animal,all_DSC_targ_chair]))
print('human:')
print('same superordinate category = ' + str(np.around(human_SSC_mn,4)))
print('different superordinate category = ' + str(np.around(human_DSC_mn,4)))
print('overall = ' + str(np.around(overall_human_mn,4)))
PAM_SSC_mn = np.mean(np.concatenate([all_PAM_SSC_targ_animal,all_PAM_SSC_targ_chair]))
PAM_DSC_mn = np.mean(np.concatenate([all_PAM_DSC_targ_animal,all_PAM_DSC_targ_chair]))
overall_PAM_mn = np.mean(np.concatenate([all_PAM_SSC_targ_animal,all_PAM_SSC_targ_chair,all_PAM_DSC_targ_animal,all_PAM_DSC_targ_chair]))
print('PAM:')
print('same superordinate category = ' + str(np.around(PAM_SSC_mn,4)))
print('different superordinate category = ' + str(np.around(PAM_DSC_mn,4)))
print('overall = ' + str(np.around(overall_PAM_mn,4)))




