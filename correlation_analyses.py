import numpy as np
import csv
import argparse

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='combined', choices=['combined', 'node_only', 'edge_only'])
args = parser.parse_args()

# Load data
if args.model == 'combined':
	data_fname = './human_visiPAM_data.csv'
elif args.model == 'node_only':
	data_fname = './human_visiPAM_node_only_data.csv'
elif args.model == 'edge_only':
	data_fname = './human_visiPAM_edge_only_data.csv'
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

# Get image filenames
all_im_fnames = []
for r in range(len(rows)):
	all_im_fnames.append(rows[r][1])
all_im_fnames = np.unique(np.array(all_im_fnames))

# Get part names for each problem
all_part_names = []
for f in range(len(all_im_fnames)):
	part_names = []
	for r in range(len(rows)):
		if rows[r][1] == str(all_im_fnames[f]):
			part_names.append(rows[r][2])
	part_names = np.unique(part_names)
	all_part_names.append(part_names)

# Get average distance to mean for each problem
all_human_dist = []
all_PAM_dist = []
N_PAM_missing = 0
for f in range(len(all_im_fnames)):
	part_names = all_part_names[f]
	for p in range(len(part_names)):
		# Get PAM data
		PAM_dist = []
		for r in range(len(rows)):
			if rows[r][1] == str(all_im_fnames[f]) and rows[r][2] == part_names[p] and rows[r][0] == 'pam':
				if rows[r][8] != '-':
					PAM_dist.append(float(rows[r][8]))
		if len(PAM_dist) > 0:
			human_dist = []
			for r in range(len(rows)):
				if rows[r][1] == str(all_im_fnames[f]) and rows[r][2] == part_names[p] and rows[r][0] != 'pam':
					human_dist.append(float(rows[r][8]))
			all_human_dist.append(np.mean(human_dist))
			all_PAM_dist.append(PAM_dist[0])
		else:
			N_PAM_missing += 1

print(str(N_PAM_missing) + ' missing PAM responses')

# Correlation analysis
PAM_human_r = np.corrcoef(all_PAM_dist, all_human_dist)[0][1]
print('model: ' + args.model)
print('r = ' + str(np.around(PAM_human_r,4)))
fid = open('./corr_analysis_' + args.model + '.txt', 'w')
fid.write('r = ' + str(np.around(PAM_human_r,4)))
fid.close()
