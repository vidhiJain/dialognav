import numpy as np
import matplotlib.pyplot as plt


def get_visual_attention_probabilities(env_obs, player_parameters):

	yaw = env_obs['yaw']
	distance = env_obs['distance']	
	resolution = env_obs['resolution']
	angle = env_obs['angle']
	envsize = env_obs['envsize']
	
	theta_sigma, theta_mu = player_parameters['theta_sigma'],  np.pi*yaw/180
	r_sigma, r_mu = player_parameters['r_sigma'], 1.0

	agent_pos = {'z': envsize//2, 'x': envsize//2}
	
	r, theta_in_deg = np.meshgrid(np.linspace(1,distance,resolution), np.linspace(yaw-angle,yaw+angle,resolution), sparse=True)
	theta = np.pi*theta_in_deg/180
	gtheta = np.exp(-( (theta-theta_mu)**2 / ( 2.0 * theta_sigma**2 ) ) )
	gr = np.exp(-( (r-r_mu)**2 / ( 2.0 * r_sigma**2 ) ) )
	z = (r*np.cos(theta)) + agent_pos['z']
	x = -(r*np.sin(theta)) + agent_pos['x'] 
	g = gr * gtheta

	visibility_prob_grid = np.zeros((envsize, envsize))
	for i in range(resolution):
		for j in range(resolution):
			visibility_prob_grid[int(z[i][j])][int(x[i][j])] = g[i][j]
	return visibility_prob_grid


def inview2D_with_opaque_objects(grid, yaw, distance, angle=60):
    """
    Using polar coordinates to return the blocks in the 
    field of view of minecraft player

    Parameters:
        the grid and player's yaw, 
        line of sight distance, and 
        angle for field of view
    Returns: 
        visible grid: 2D numpy array

    Note: 
    envsize should be same as the grid size.
    Assuming that the player has a headlight with them
    This function is independent of environment's visibility

    TODO: Extend to 3D visibility by 3D polar coordinates check
    """

    envsize = grid.shape[0]
    visible_grid = np.zeros((envsize, envsize)) 
    agent_pos = {'z': envsize//2, 'x':envsize//2}
    for theta_in_deg in range(int(yaw-angle), int(yaw+angle)):
        # print('theta_in_deg', theta_in_deg)
        for r in range(1, int(distance)):
            theta_in_rad = np.pi*theta_in_deg/180
            z = int(r*np.cos(theta_in_rad)) + agent_pos['z']
            x = - int(r*np.sin(theta_in_rad)) + agent_pos['x']
            # debug
            # print('grid[z][x]', z, x, grid[z][x])
            # is_opaque is env specific function
            if self.is_opaque(grid[z][x]):
                visible_grid[z][x] = 1
                break
            else:
                visible_grid[z][x] = 1
    return visible_grid        



# h = plt.contourf(z,x,g)
# zz, xx = np.meshgrid(z,x)
def main():
	yaw = -22.0
	distance = 60
	resolution = 120
	angle = 70 
	envsize = 120
	
	theta_sigma, theta_mu = 0.9,  np.pi*yaw/180
	r_sigma, r_mu = 20.0, 1.0
	
	env_obs = {
		'yaw': yaw,
		'distance': distance,
		'resolution': resolution,
		'angle': angle,
		'envsize': envsize,
	}

	player_parameters = {
		'theta_sigma': theta_sigma,
		'r_sigma': r_sigma,
	# 'r_mu': 1.0
	}

	visibility_prob_grid = get_visual_attention_probabilities(env_obs, player_parameters)
	print(visibility_prob_grid)
	plt.imshow(visibility_prob_grid)
	plt.colorbar()
	plt.savefig(f'../figures/visibility_prob_at_yaw_{int(yaw)}')

if __name__ == '__main__':
	main()
