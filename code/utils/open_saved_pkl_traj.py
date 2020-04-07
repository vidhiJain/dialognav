import pickle as pkl

max_count = 20  # from debug.py
savedir = '../pickled_raw_traj_data'


def get_yaw(savedir, max_count):
	
	yaw = []
	for count in range(2, max_count):
		with open(savedir + '/MC_{:03d}.pkl'.format(count), 'rb') as f:
			msg = pkl.load(f)
			print(msg['timestamp'], msg['observations']['Yaw'])
			yaw.append(msg['observations']['Yaw'])

	return yaw


def open_pickled_trajectory_files(savedir, max_count):
	for count in range(2, max_count):
		with open(savedir + '/MC_{:03d}.pkl'.format(count), 'rb') as f:
			msg = pkl.load(f)
	return msg

